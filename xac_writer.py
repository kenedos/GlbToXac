import io
import os
import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

from binary_writer import BinaryWriter


# ==========================================================================
# XAC Chunk IDs and Constants
# ==========================================================================
class XacChunk:
    Node = 0
    Mesh = 1
    SkinningInfo = 2
    StdMaterial = 3
    StdMaterialLayer = 4
    FxMaterial = 5
    Limit = 6
    Info = 7
    MeshLodLevels = 8
    StdProgMorphTarget = 9
    NodeGroups = 10
    Nodes = 11
    StdPMorphTargets = 12
    MaterialInfo = 13
    NodeMotionSources = 14
    AttachmentNodes = 15


class XacAttribute:
    Positions = 0
    Normals = 1
    Tangents = 2
    UVCoords = 3
    Colors32 = 4
    OrgVtxNumbers = 5
    Colors128 = 6
    Bitangents = 7


# XAC File Magic: 'XAC ' in little-endian
XAC_MAGIC = 0x20434158  # ' CAX' when read as u32 LE


# ==========================================================================
# Data Classes for XAC Structure
# ==========================================================================
@dataclass
class XacNodeData:
    name: str
    local_pos: Tuple[float, float, float]
    local_rot: Tuple[float, float, float, float]  # quaternion (x, y, z, w)
    local_scale: Tuple[float, float, float]
    parent_index: int  # -1 or 0xFFFFFFFF for root nodes


@dataclass
class XacMaterialData:
    name: str
    texture_name: Optional[str] = None
    ambient: Tuple[float, float, float, float] = (0.2, 0.2, 0.2, 1.0)
    diffuse: Tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0)
    specular: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    emissive: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    shine: float = 25.0
    shine_strength: float = 1.0
    opacity: float = 1.0


@dataclass
class XacSubMeshData:
    indices: np.ndarray  # uint32 indices
    material_index: int
    num_vertices: int


@dataclass
class XacMeshData:
    positions: np.ndarray  # (N, 3) float32
    normals: Optional[np.ndarray]  # (N, 3) float32
    uvs: Optional[np.ndarray]  # (N, 2) float32
    sub_meshes: List[XacSubMeshData]
    node_index: int = 0
    # Skinning data
    bone_ids: Optional[np.ndarray] = None  # (N, 4) int32
    bone_weights: Optional[np.ndarray] = None  # (N, 4) float32


@dataclass
class XacModelData:
    """Complete model data for XAC export."""
    nodes: List[XacNodeData] = field(default_factory=list)
    materials: List[XacMaterialData] = field(default_factory=list)
    meshes: List[XacMeshData] = field(default_factory=list)
    actor_name: str = "Exported Model"


# ==========================================================================
# XAC Writer
# ==========================================================================
class XACWriter:
    def __init__(self, model_data: XacModelData):
        self.model = model_data

    def write(self, filepath: str):
        """Write the model to an XAC file."""
        with open(filepath, 'wb') as f:
            writer = BinaryWriter(f, endian="<")
            self._write_header(writer)
            self._write_info_chunk(writer)
            self._write_nodes_chunk(writer)
            self._write_material_info_chunk(writer)
            for mat in self.model.materials:
                self._write_std_material_chunk(writer, mat)
            for i, mesh in enumerate(self.model.meshes):
                self._write_mesh_chunk(writer, mesh, i)
                if mesh.bone_ids is not None and mesh.bone_weights is not None:
                    self._write_skinning_info_chunk(writer, mesh, i)

    def _write_header(self, writer: BinaryWriter):
        """Write the 8-byte XAC header."""
        writer.write_u32(XAC_MAGIC)  # 'XAC '
        writer.write_u8(1)  # hi_version
        writer.write_u8(0)  # lo_version
        writer.write_u8(0)  # endian_type (0 = little endian)
        writer.write_u8(1)  # mul_order (1 = 3DS Max style R*S*T)

    def _write_chunk_header(self, writer: BinaryWriter, chunk_id: int, version: int, data: bytes):
        """Write a chunk with its header and data."""
        writer.write_u32(chunk_id)
        writer.write_u32(len(data))
        writer.write_u32(version)
        writer.write_bytes(data)

    def _write_info_chunk(self, writer: BinaryWriter):
        """Write the Info chunk (chunk ID 7)."""
        buf = io.BytesIO()
        w = BinaryWriter(buf, endian="<")

        # Version 1 format
        w.write_u32(0)  # repositioning_mask
        w.write_u32(0xFFFFFFFF)  # repositioning_node_index
        w.write_u8(1)  # exporter_high_version
        w.write_u8(0)  # exporter_low_version
        w.write_u16(0)  # padding

        w.write_string("GlbToXac Exporter")  # source_app
        w.write_string("")  # original_filename
        w.write_string("")  # compilation_date
        w.write_string(self.model.actor_name)  # actor_name

        self._write_chunk_header(writer, XacChunk.Info, 1, buf.getvalue())

    def _write_nodes_chunk(self, writer: BinaryWriter):
        """Write the Nodes chunk (chunk ID 11)."""
        buf = io.BytesIO()
        w = BinaryWriter(buf, endian="<")

        num_nodes = len(self.model.nodes)
        num_root_nodes = sum(1 for n in self.model.nodes if n.parent_index == -1 or n.parent_index == 0xFFFFFFFF)

        w.write_u32(num_nodes)
        w.write_u32(num_root_nodes)

        for node in self.model.nodes:
            self._write_node(w, node)

        self._write_chunk_header(writer, XacChunk.Nodes, 1, buf.getvalue())

    def _write_node(self, w: BinaryWriter, node: XacNodeData):
        """Write a single node (version 4 format embedded in Nodes chunk)."""
        # Get GLTF values (already in swizzled Y-up format)
        gltf_pos = node.local_pos
        gltf_rot = node.local_rot
        gltf_scale = node.local_scale

        # Reverse swizzle: GLTF(-x, z, y) -> XAC(x, y, z)
        # If GLTF has (a, b, c) = (-x, z, y), then x=-a, y=c, z=b
        # So XAC = (-a, c, b)
        xac_pos = (-gltf_pos[0], gltf_pos[2], gltf_pos[1])

        # Quaternion reverse swizzle: GLTF(-x, z, y, -w) -> XAC(x, y, z, w)
        # If GLTF has (a, b, c, d) = (-x, z, y, -w), then x=-a, y=c, z=b, w=-d
        # So XAC = (-a, c, b, -d)
        xac_rot = (-gltf_rot[0], gltf_rot[2], gltf_rot[1], -gltf_rot[3])

        # Normalize quaternion
        quat_len = (xac_rot[0]**2 + xac_rot[1]**2 + xac_rot[2]**2 + xac_rot[3]**2) ** 0.5
        if quat_len > 0:
            xac_rot = tuple(q / quat_len for q in xac_rot)
        else:
            xac_rot = (0, 0, 0, 1)

        # Scale is NOT swizzled, but ensure no zero values
        xac_scale = tuple(max(abs(s), 0.0001) if s == 0 else s for s in gltf_scale)

        w.write_quat(xac_rot)  # local_quat
        w.write_quat((0, 0, 0, 1))  # scale_rot (identity)
        w.write_vec3(xac_pos)  # local_pos
        w.write_vec3(xac_scale)  # local_scale
        w.write_vec3((0, 0, 0))  # shear
        w.write_u32(0xFFFFFFFF)  # skeletal_lods
        w.write_u32(0)  # motion_lods (version 4)

        parent_idx = node.parent_index if node.parent_index >= 0 else 0xFFFFFFFF
        w.write_u32(parent_idx)  # parent_index
        w.write_u32(0)  # num_children (version 4)
        w.write_u8(0)  # node_flags (version >= 2)

        # OBB matrix (version >= 3) - 16 floats, identity
        for i in range(4):
            for j in range(4):
                w.write_f32(1.0 if i == j else 0.0)

        w.write_f32(1.0)  # importance_factor (version 4)
        w.write_bytes(b'\x00' * 3)  # padding
        w.write_string(node.name)

    def _write_material_info_chunk(self, writer: BinaryWriter):
        """Write the MaterialInfo chunk (chunk ID 13)."""
        buf = io.BytesIO()
        w = BinaryWriter(buf, endian="<")

        num_materials = len(self.model.materials)
        w.write_u32(num_materials)  # num_total_materials
        w.write_u32(num_materials)  # num_standard_materials
        w.write_u32(0)  # num_fx_materials

        self._write_chunk_header(writer, XacChunk.MaterialInfo, 1, buf.getvalue())

    def _write_std_material_chunk(self, writer: BinaryWriter, mat: XacMaterialData):
        """Write a Standard Material chunk (chunk ID 3)."""
        buf = io.BytesIO()
        w = BinaryWriter(buf, endian="<")

        w.write_color(mat.ambient)
        w.write_color(mat.diffuse)
        w.write_color(mat.specular)
        w.write_color(mat.emissive)
        w.write_f32(mat.shine)
        w.write_f32(mat.shine_strength)
        w.write_f32(mat.opacity)
        w.write_f32(1.0)  # ior

        w.write_u8(1)  # double_sided
        w.write_u8(0)  # wireframe
        w.write_u8(0)  # transparency_type

        # Number of layers
        num_layers = 1 if mat.texture_name else 0
        w.write_u8(num_layers)

        w.write_string(mat.name)

        # Write texture layer if present
        if mat.texture_name:
            self._write_material_layer(w, mat.texture_name)

        self._write_chunk_header(writer, XacChunk.StdMaterial, 2, buf.getvalue())

    def _write_material_layer(self, w: BinaryWriter, texture_name: str):
        """Write a material layer (embedded in StdMaterial, version 2)."""
        w.write_f32(1.0)  # amount
        w.write_f32(0.0)  # u_offset
        w.write_f32(0.0)  # v_offset
        w.write_f32(1.0)  # u_tiling
        w.write_f32(1.0)  # v_tiling
        w.write_f32(0.0)  # rotation_radians
        w.write_u16(0)  # material_number
        w.write_u8(2)  # map_type (2 = diffuse)
        w.write_u8(0)  # blend_mode (version 2)
        w.write_string(texture_name)

    def _write_mesh_chunk(self, writer: BinaryWriter, mesh: XacMeshData, mesh_index: int):
        """Write a Mesh chunk (chunk ID 1)."""
        buf = io.BytesIO()
        w = BinaryWriter(buf, endian="<")

        num_verts = len(mesh.positions)
        total_indices = sum(len(sm.indices) for sm in mesh.sub_meshes)
        num_sub_meshes = len(mesh.sub_meshes)

        # Apply inverse swizzle to positions and normals
        positions = mesh.positions.copy()
        swizzled_pos = np.zeros_like(positions)
        swizzled_pos[:, 0] = -positions[:, 0]
        swizzled_pos[:, 1] = positions[:, 2]
        swizzled_pos[:, 2] = positions[:, 1]

        normals = None
        if mesh.normals is not None:
            normals = mesh.normals.copy()
            swizzled_norm = np.zeros_like(normals)
            swizzled_norm[:, 0] = -normals[:, 0]
            swizzled_norm[:, 1] = normals[:, 2]
            swizzled_norm[:, 2] = normals[:, 1]
            normals = swizzled_norm

        # Count layers
        num_layers = 2  # positions + org_vtx_numbers (required)
        if normals is not None:
            num_layers += 1
        if mesh.uvs is not None:
            num_layers += 1

        w.write_u32(mesh.node_index)  # node_index
        w.write_u32(num_verts)  # num_org_verts
        w.write_u32(num_verts)  # total_verts
        w.write_u32(total_indices)  # total_indices
        w.write_u32(num_sub_meshes)  # num_sub_meshes
        w.write_u32(num_layers)  # num_layers
        w.write_u8(0)  # is_collision_mesh
        w.write_bytes(b'\x00' * 3)  # padding

        # Write vertex attribute layers
        # Positions
        self._write_vertex_layer(w, XacAttribute.Positions, swizzled_pos.astype(np.float32).tobytes(), 12)

        # Normals
        if normals is not None:
            self._write_vertex_layer(w, XacAttribute.Normals, normals.astype(np.float32).tobytes(), 12)

        # UVs
        if mesh.uvs is not None:
            # Flip V coordinate back
            uvs = mesh.uvs.copy()
            uvs[:, 1] = 1.0 - uvs[:, 1]
            self._write_vertex_layer(w, XacAttribute.UVCoords, uvs.astype(np.float32).tobytes(), 8)

        # Original vertex numbers (required for skinning)
        org_vtx = np.arange(num_verts, dtype=np.uint32)
        self._write_vertex_layer(w, XacAttribute.OrgVtxNumbers, org_vtx.tobytes(), 4)

        # Write sub-meshes
        for sm in mesh.sub_meshes:
            w.write_u32(len(sm.indices))  # num_indices
            w.write_u32(sm.num_vertices)  # num_verts
            w.write_u32(sm.material_index)  # material_index
            w.write_u32(0)  # num_bones

            # Write indices
            for idx in sm.indices:
                w.write_u32(int(idx))

            # No bones array (num_bones = 0)

        self._write_chunk_header(writer, XacChunk.Mesh, 1, buf.getvalue())

    def _write_vertex_layer(self, w: BinaryWriter, layer_type: int, data: bytes, attrib_size: int):
        """Write a vertex attribute layer."""
        w.write_u32(layer_type)
        w.write_u32(attrib_size)
        w.write_u8(1)  # enable_deformations
        w.write_u8(0)  # is_scale
        w.write_bytes(b'\x00' * 2)  # padding
        w.write_bytes(data)

    def _write_skinning_info_chunk(self, writer: BinaryWriter, mesh: XacMeshData, mesh_index: int):
        """Write a SkinningInfo chunk (chunk ID 2)."""
        buf = io.BytesIO()
        w = BinaryWriter(buf, endian="<")

        num_verts = len(mesh.positions)

        # Build influence list and table
        influences = []
        table = []

        for i in range(num_verts):
            start_idx = len(influences)
            num_influences = 0

            for j in range(4):
                weight = mesh.bone_weights[i, j]
                if weight > 0.0001:
                    influences.append((weight, int(mesh.bone_ids[i, j])))
                    num_influences += 1

            table.append((start_idx, num_influences))

        w.write_u32(mesh_index)  # node_index
        w.write_u32(len(self.model.nodes))  # num_local_bones (version >= 3)
        w.write_u32(len(influences))  # num_total_influences (version >= 2)
        w.write_u8(0)  # is_for_collision_mesh
        w.write_bytes(b'\x00' * 3)  # padding

        # Write influences
        for weight, bone_id in influences:
            w.write_f32(weight)
            w.write_u32(bone_id)

        # Write table
        for start_idx, num_elem in table:
            w.write_u32(start_idx)
            w.write_u32(num_elem)

        self._write_chunk_header(writer, XacChunk.SkinningInfo, 3, buf.getvalue())


def write_xac(model_data: XacModelData, filepath: str):
    """Convenience function to write an XAC file."""
    writer = XACWriter(model_data)
    writer.write(filepath)
