import io
import os
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

from binary_writer import BinaryWriter


# ==========================================================================
# XSM Chunk IDs and Constants
# ==========================================================================
class XsmChunk:
    Info = 201  # 0xC9
    SkeletalMotion = 202  # 0xCA


class SharedChunk:
    MotionEventTable = 50
    Timestamp = 51


# XSM File Magic: 'XSM ' in little-endian
XSM_MAGIC = 0x204D5358  # ' MSX' when read as u32 LE


# ==========================================================================
# Data Classes for XSM Structure
# ==========================================================================
@dataclass
class XsmKeyframe:
    time: float
    value: Tuple  # Vec3 for pos/scale, Quat for rotation


@dataclass
class XsmTrack:
    node_name: str
    pos_keys: List[XsmKeyframe] = field(default_factory=list)
    rot_keys: List[XsmKeyframe] = field(default_factory=list)
    scale_keys: List[XsmKeyframe] = field(default_factory=list)
    # Bind pose values (used when no animation keys exist)
    bind_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    bind_rot: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    bind_scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass
class XsmAnimationData:
    """Complete animation data for XSM export."""
    name: str
    tracks: List[XsmTrack] = field(default_factory=list)
    duration: float = 0.0


# ==========================================================================
# XSM Writer
# ==========================================================================
class XSMWriter:
    def __init__(self, animation: XsmAnimationData):
        self.animation = animation

    def write(self, filepath: str):
        """Write the animation to an XSM file."""
        with open(filepath, 'wb') as f:
            writer = BinaryWriter(f, endian="<")
            self._write_header(writer)
            self._write_skeletal_motion_chunk(writer)

    def _write_header(self, writer: BinaryWriter):
        """Write the 8-byte XSM header."""
        writer.write_u32(XSM_MAGIC)  # 'XSM '
        writer.write_u8(2)  # hi_version
        writer.write_u8(0)  # lo_version
        writer.write_u8(0)  # endian_type (0 = little endian)
        writer.write_u8(0)  # padding

    def _write_chunk_header(self, writer: BinaryWriter, chunk_id: int, version: int, data: bytes):
        """Write a chunk with its header and data."""
        writer.write_u32(chunk_id)
        writer.write_u32(len(data))
        writer.write_u32(version)
        writer.write_bytes(data)

    def _write_skeletal_motion_chunk(self, writer: BinaryWriter):
        """Write the SkeletalMotion chunk (chunk ID 202)."""
        buf = io.BytesIO()
        w = BinaryWriter(buf, endian="<")

        w.write_u32(len(self.animation.tracks))  # num_sub_motions

        for track in self.animation.tracks:
            self._write_skeletal_sub_motion(w, track)

        self._write_chunk_header(writer, XsmChunk.SkeletalMotion, 2, buf.getvalue())

    def _write_skeletal_sub_motion(self, w: BinaryWriter, track: XsmTrack):
        """Write a skeletal sub-motion (one bone's animation data)."""
        # Apply inverse swizzle to bind pose values
        # GLTF -> XAC: position (-x, z, y), rotation (-x, z, y, -w)
        bind_pos = track.bind_pos
        xac_bind_pos = (-bind_pos[0], bind_pos[2], bind_pos[1])

        bind_rot = track.bind_rot
        xac_bind_rot = (-bind_rot[0], bind_rot[2], bind_rot[1], -bind_rot[3])

        # Normalize quaternion for quat16
        def normalize_quat(q):
            length = (q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2) ** 0.5
            if length > 0:
                return (q[0]/length, q[1]/length, q[2]/length, q[3]/length)
            return (0, 0, 0, 1)

        xac_bind_rot = normalize_quat(xac_bind_rot)

        # Write pose values (current pose, usually same as bind)
        w.write_quat16(xac_bind_rot)  # pose_rot
        w.write_quat16(xac_bind_rot)  # bind_pose_rot
        w.write_quat16((0, 0, 0, 1))  # pose_scale_rot
        w.write_quat16((0, 0, 0, 1))  # bind_pose_scale_rot
        w.write_vec3(xac_bind_pos)  # pose_pos
        w.write_vec3(track.bind_scale)  # pose_scale
        w.write_vec3(xac_bind_pos)  # bind_pose_pos
        w.write_vec3(track.bind_scale)  # bind_pose_scale

        # Key counts
        w.write_u32(len(track.pos_keys))  # num_pos_keys
        w.write_u32(len(track.rot_keys))  # num_rot_keys
        w.write_u32(len(track.scale_keys))  # num_scale_keys
        w.write_u32(0)  # num_scale_rot_keys (not used)

        w.write_f32(0.0)  # max_error
        w.write_string(track.node_name)

        # Write position keys
        for key in track.pos_keys:
            # Apply inverse swizzle
            pos = key.value
            xac_pos = (-pos[0], pos[2], pos[1])
            w.write_vec3(xac_pos)
            w.write_f32(key.time)

        # Write rotation keys
        for key in track.rot_keys:
            # Apply inverse swizzle
            rot = key.value
            xac_rot = (-rot[0], rot[2], rot[1], -rot[3])
            xac_rot = normalize_quat(xac_rot)
            w.write_quat16(xac_rot)
            w.write_f32(key.time)

        # Write scale keys
        for key in track.scale_keys:
            w.write_vec3(key.value)  # Scale is not swizzled
            w.write_f32(key.time)

        # No scale_rot keys


def write_xsm(animation: XsmAnimationData, filepath: str):
    """Convenience function to write an XSM file."""
    writer = XSMWriter(animation)
    writer.write(filepath)
