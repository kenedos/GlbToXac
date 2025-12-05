import struct
import os


class BinaryWriter:
    """
    A helper class to write structured binary data to a stream.
    """

    def __init__(self, stream, endian="<"):
        self.stream = stream
        self.endian = endian

    def write_bytes(self, data: bytes):
        self.stream.write(data)

    def write_struct(self, fmt: str, *values):
        self.stream.write(struct.pack(self.endian + fmt, *values))

    # --- Basic Data Types ---
    def write_u8(self, value: int):
        self.write_struct("B", value)

    def write_i8(self, value: int):
        self.write_struct("b", value)

    def write_u16(self, value: int):
        self.write_struct("H", value)

    def write_i16(self, value: int):
        self.write_struct("h", value)

    def write_u32(self, value: int):
        self.write_struct("I", value)

    def write_i32(self, value: int):
        self.write_struct("i", value)

    def write_f32(self, value: float):
        self.write_struct("f", value)

    # --- 3D Graphics Data Types ---
    def write_vec3(self, vec):
        self.write_struct("fff", vec[0], vec[1], vec[2])

    def write_quat(self, quat):
        self.write_struct("ffff", quat[0], quat[1], quat[2], quat[3])

    def write_color(self, color):
        self.write_struct("ffff", color[0], color[1], color[2], color[3])

    def write_quat16(self, quat):
        """Write quaternion as 4 signed 16-bit integers (normalized to -32767..32767)."""
        def to_i16(val):
            return max(-32767, min(32767, int(val * 32767.0)))
        self.write_struct("hhhh", to_i16(quat[0]), to_i16(quat[1]), to_i16(quat[2]), to_i16(quat[3]))

    # --- String Writing ---
    def write_string(self, s: str, encoding: str = 'utf-8'):
        """Write a U32-prefixed string (XAC style)."""
        data = s.encode(encoding)
        self.write_u32(len(data))
        if len(data) > 0:
            self.write_bytes(data)

    # --- Stream Control ---
    def tell(self) -> int:
        return self.stream.tell()

    def seek(self, offset: int, whence: int = 0) -> int:
        return self.stream.seek(offset, whence)

    def write_padding(self, alignment: int = 4):
        """Write padding bytes to align to the specified boundary."""
        pos = self.tell()
        padding = (alignment - (pos % alignment)) % alignment
        if padding > 0:
            self.write_bytes(b'\x00' * padding)
