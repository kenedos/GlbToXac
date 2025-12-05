import os
import io
import math
import base64
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from pygltflib import GLTF2
import pyrr

from xac_writer import XacModelData, XacNodeData, XacMaterialData, XacMeshData, XacSubMeshData
from xsm_writer import XsmAnimationData, XsmTrack, XsmKeyframe


@dataclass
class TextureData:
    """Extracted texture data from GLB."""
    name: str
    data: np.ndarray  # Image data as numpy array (H, W, C)
    mime_type: str


def load_glb(filepath: str) -> Tuple[XacModelData, List[XsmAnimationData], List[TextureData]]:
    """
    Load a GLB file and convert it to XAC model data and XSM animation data.

    Returns:
        Tuple of (XacModelData, List[XsmAnimationData], List[TextureData])
    """
    gltf = GLTF2().load(filepath)

    model_name = os.path.splitext(os.path.basename(filepath))[0]

    # Extract model data
    model_data = _extract_model(gltf, model_name)

    # Extract animations
    animations = _extract_animations(gltf, model_data.nodes)

    # Extract textures
    textures = _extract_textures(gltf)

    return model_data, animations, textures


def _get_accessor_data(gltf: GLTF2, accessor_index: int) -> np.ndarray:
    """Get data from a GLTF accessor."""
    if accessor_index is None or accessor_index < 0:
        return None

    accessor = gltf.accessors[accessor_index]
    buffer_view = gltf.bufferViews[accessor.bufferView]

    # Get the binary blob
    if gltf.binary_blob():
        data = gltf.binary_blob()
    else:
        # Handle external buffer
        buffer = gltf.buffers[buffer_view.buffer]
        if buffer.uri:
            # For now, only support embedded data
            return None
        data = gltf.binary_blob()

    # Calculate offset
    offset = buffer_view.byteOffset or 0
    if accessor.byteOffset:
        offset += accessor.byteOffset

    # Determine numpy dtype
    component_type_map = {
        5120: np.int8,
        5121: np.uint8,
        5122: np.int16,
        5123: np.uint16,
        5125: np.uint32,
        5126: np.float32,
    }
    dtype = component_type_map.get(accessor.componentType, np.float32)

    # Determine number of components
    type_count_map = {
        'SCALAR': 1,
        'VEC2': 2,
        'VEC3': 3,
        'VEC4': 4,
        'MAT4': 16,
    }
    num_components = type_count_map.get(accessor.type, 1)

    # Calculate byte length
    byte_length = accessor.count * num_components * np.dtype(dtype).itemsize

    # Extract and reshape data
    raw_data = data[offset:offset + byte_length]
    arr = np.frombuffer(raw_data, dtype=dtype)

    if num_components > 1:
        arr = arr.reshape(-1, num_components)

    return arr


def _extract_model(gltf: GLTF2, model_name: str) -> XacModelData:
    """Extract model data from GLTF."""
    model = XacModelData(actor_name=model_name)

    # Build node hierarchy
    node_index_map = {}  # GLTF node index -> XAC node index

    # Find skin to get the skeleton nodes
    skin = gltf.skins[0] if gltf.skins else None
    joint_nodes = set(skin.joints) if skin else set()

    # First pass: identify which nodes are skeleton nodes
    skeleton_nodes = set()
    if skin:
        for joint_idx in skin.joints:
            skeleton_nodes.add(joint_idx)

    # Process nodes that are part of the skeleton
    def process_node(gltf_node_idx: int, parent_xac_idx: int):
        gltf_node = gltf.nodes[gltf_node_idx]

        # Get local transform
        if gltf_node.matrix:
            # Decompose matrix
            mat = np.array(gltf_node.matrix).reshape(4, 4).T
            translation, rotation, scale = pyrr.matrix44.decompose(mat)
            # Handle NaN values from degenerate matrices
            if np.any(np.isnan(translation)):
                translation = np.array([0, 0, 0])
            if np.any(np.isnan(rotation)):
                rotation = np.array([0, 0, 0, 1])
            if np.any(np.isnan(scale)) or np.any(scale == 0):
                scale = np.array([1, 1, 1])
        else:
            translation = np.array(gltf_node.translation or [0, 0, 0])
            rotation = np.array(gltf_node.rotation or [0, 0, 0, 1])
            scale = np.array(gltf_node.scale or [1, 1, 1])

        # Ensure scale has no zero values
        scale = np.where(scale == 0, 1.0, scale)

        # Normalize quaternion
        rot_len = np.linalg.norm(rotation)
        if rot_len > 0:
            rotation = rotation / rot_len
        else:
            rotation = np.array([0, 0, 0, 1])

        # Create XAC node
        xac_node = XacNodeData(
            name=gltf_node.name or f"Node_{gltf_node_idx}",
            local_pos=tuple(translation),
            local_rot=tuple(rotation),
            local_scale=tuple(scale),
            parent_index=parent_xac_idx
        )

        xac_idx = len(model.nodes)
        model.nodes.append(xac_node)
        node_index_map[gltf_node_idx] = xac_idx

        # Process children
        if gltf_node.children:
            for child_idx in gltf_node.children:
                if child_idx in skeleton_nodes:
                    process_node(child_idx, xac_idx)

    # Find and process skeleton roots
    if skin and skin.skeleton is not None:
        process_node(skin.skeleton, -1)
    elif skin:
        # Find root joints (joints with no parent in the joint list)
        for joint_idx in skin.joints:
            is_root = True
            for other_joint_idx in skin.joints:
                if other_joint_idx == joint_idx:
                    continue
                other_node = gltf.nodes[other_joint_idx]
                if other_node.children and joint_idx in other_node.children:
                    is_root = False
                    break
            if is_root and joint_idx not in node_index_map:
                process_node(joint_idx, -1)

    # If no skeleton, create a default root node
    if not model.nodes:
        model.nodes.append(XacNodeData(
            name="Root",
            local_pos=(0, 0, 0),
            local_rot=(0, 0, 0, 1),
            local_scale=(1, 1, 1),
            parent_index=-1
        ))
        node_index_map[0] = 0

    # Extract materials
    material_map = {}  # GLTF material index -> XAC material index
    if gltf.materials:
        for i, mat in enumerate(gltf.materials):
            texture_name = None

            # Try to get base color texture
            if mat.pbrMetallicRoughness and mat.pbrMetallicRoughness.baseColorTexture:
                tex_idx = mat.pbrMetallicRoughness.baseColorTexture.index
                if tex_idx is not None and gltf.textures:
                    texture = gltf.textures[tex_idx]
                    if texture.source is not None and gltf.images:
                        image = gltf.images[texture.source]
                        if image.name:
                            texture_name = image.name
                        elif image.uri:
                            texture_name = os.path.basename(image.uri)

            xac_mat = XacMaterialData(
                name=mat.name or f"Material_{i}",
                texture_name=texture_name
            )
            model.materials.append(xac_mat)
            material_map[i] = len(model.materials) - 1

    # Add default material if none exist
    if not model.materials:
        model.materials.append(XacMaterialData(name="DefaultMaterial"))
        material_map[0] = 0

    # Extract meshes
    for gltf_node_idx, gltf_node in enumerate(gltf.nodes):
        if gltf_node.mesh is None:
            continue

        gltf_mesh = gltf.meshes[gltf_node.mesh]

        for prim in gltf_mesh.primitives:
            # Get vertex data
            positions = _get_accessor_data(gltf, prim.attributes.POSITION)
            if positions is None:
                continue

            normals = _get_accessor_data(gltf, prim.attributes.NORMAL)
            uvs = _get_accessor_data(gltf, prim.attributes.TEXCOORD_0)

            # Get skinning data
            joints = _get_accessor_data(gltf, prim.attributes.JOINTS_0)
            weights = _get_accessor_data(gltf, prim.attributes.WEIGHTS_0)

            # Get indices
            indices = _get_accessor_data(gltf, prim.indices)
            if indices is None:
                # Generate indices for non-indexed geometry
                indices = np.arange(len(positions), dtype=np.uint32)
            else:
                indices = indices.flatten().astype(np.uint32)

            # Map joint indices from GLTF to XAC
            bone_ids = None
            bone_weights = None
            if joints is not None and weights is not None and skin:
                bone_ids = np.zeros_like(joints, dtype=np.int32)
                for i in range(len(joints)):
                    for j in range(4):
                        gltf_joint_idx = skin.joints[int(joints[i, j])]
                        if gltf_joint_idx in node_index_map:
                            bone_ids[i, j] = node_index_map[gltf_joint_idx]
                        else:
                            bone_ids[i, j] = 0
                bone_weights = weights.astype(np.float32)

            # Determine material
            mat_idx = material_map.get(prim.material, 0) if prim.material is not None else 0

            # Create sub-mesh
            sub_mesh = XacSubMeshData(
                indices=indices,
                material_index=mat_idx,
                num_vertices=len(positions)
            )

            # Create mesh
            xac_mesh = XacMeshData(
                positions=positions.astype(np.float32),
                normals=normals.astype(np.float32) if normals is not None else None,
                uvs=uvs.astype(np.float32) if uvs is not None else None,
                sub_meshes=[sub_mesh],
                node_index=0,
                bone_ids=bone_ids,
                bone_weights=bone_weights
            )
            model.meshes.append(xac_mesh)

    return model


def _extract_animations(gltf: GLTF2, xac_nodes: List[XacNodeData]) -> List[XsmAnimationData]:
    """Extract animations from GLTF."""
    animations = []

    if not gltf.animations:
        return animations

    # Build node name to XAC index map
    node_name_to_idx = {node.name: i for i, node in enumerate(xac_nodes)}

    # Build GLTF node index to name map
    gltf_node_to_name = {}
    for i, node in enumerate(gltf.nodes):
        gltf_node_to_name[i] = node.name or f"Node_{i}"

    for gltf_anim in gltf.animations:
        anim = XsmAnimationData(
            name=gltf_anim.name or "Animation",
            tracks=[]
        )

        # Group channels by target node
        node_channels: Dict[int, Dict[str, int]] = {}  # node_idx -> {path: channel_idx}

        for i, channel in enumerate(gltf_anim.channels):
            node_idx = channel.target.node
            if node_idx not in node_channels:
                node_channels[node_idx] = {}
            node_channels[node_idx][channel.target.path] = i

        # Process each node's animation
        for gltf_node_idx, channels in node_channels.items():
            node_name = gltf_node_to_name.get(gltf_node_idx, f"Node_{gltf_node_idx}")

            # Skip if this node isn't in our skeleton
            if node_name not in node_name_to_idx:
                continue

            xac_node = xac_nodes[node_name_to_idx[node_name]]

            track = XsmTrack(
                node_name=node_name,
                bind_pos=xac_node.local_pos,
                bind_rot=xac_node.local_rot,
                bind_scale=xac_node.local_scale
            )

            # Process translation
            if 'translation' in channels:
                channel = gltf_anim.channels[channels['translation']]
                sampler = gltf_anim.samplers[channel.sampler]
                times = _get_accessor_data(gltf, sampler.input)
                values = _get_accessor_data(gltf, sampler.output)

                if times is not None and values is not None:
                    for t, v in zip(times.flatten(), values):
                        track.pos_keys.append(XsmKeyframe(time=float(t), value=tuple(v)))
                        anim.duration = max(anim.duration, float(t))

            # Process rotation
            if 'rotation' in channels:
                channel = gltf_anim.channels[channels['rotation']]
                sampler = gltf_anim.samplers[channel.sampler]
                times = _get_accessor_data(gltf, sampler.input)
                values = _get_accessor_data(gltf, sampler.output)

                if times is not None and values is not None:
                    for t, v in zip(times.flatten(), values):
                        track.rot_keys.append(XsmKeyframe(time=float(t), value=tuple(v)))
                        anim.duration = max(anim.duration, float(t))

            # Process scale
            if 'scale' in channels:
                channel = gltf_anim.channels[channels['scale']]
                sampler = gltf_anim.samplers[channel.sampler]
                times = _get_accessor_data(gltf, sampler.input)
                values = _get_accessor_data(gltf, sampler.output)

                if times is not None and values is not None:
                    for t, v in zip(times.flatten(), values):
                        track.scale_keys.append(XsmKeyframe(time=float(t), value=tuple(v)))
                        anim.duration = max(anim.duration, float(t))

            # Only add track if it has animation data
            if track.pos_keys or track.rot_keys or track.scale_keys:
                anim.tracks.append(track)

        if anim.tracks:
            animations.append(anim)

    return animations


def _extract_textures(gltf: GLTF2) -> List[TextureData]:
    """Extract embedded textures from GLTF."""
    textures = []

    if not gltf.images:
        return textures

    try:
        import imageio.v3 as iio
    except ImportError:
        print("Warning: imageio not installed, cannot extract textures")
        return textures

    for i, image in enumerate(gltf.images):
        try:
            image_data = None
            mime_type = image.mimeType or "image/png"

            # Get image name
            if image.name:
                name = image.name
            elif image.uri and not image.uri.startswith('data:'):
                name = os.path.splitext(os.path.basename(image.uri))[0]
            else:
                name = f"texture_{i}"

            # Extract image data
            if image.bufferView is not None:
                # Image is embedded in buffer
                buffer_view = gltf.bufferViews[image.bufferView]
                blob = gltf.binary_blob()
                if blob:
                    offset = buffer_view.byteOffset or 0
                    length = buffer_view.byteLength
                    image_bytes = blob[offset:offset + length]
                    image_data = iio.imread(io.BytesIO(image_bytes))
            elif image.uri:
                if image.uri.startswith('data:'):
                    # Data URI
                    header, encoded = image.uri.split(',', 1)
                    image_bytes = base64.b64decode(encoded)
                    image_data = iio.imread(io.BytesIO(image_bytes))
                # External URI not supported in GLB

            if image_data is not None:
                textures.append(TextureData(
                    name=name,
                    data=image_data,
                    mime_type=mime_type
                ))

        except Exception as e:
            print(f"Warning: Failed to extract texture {i}: {e}")

    return textures
