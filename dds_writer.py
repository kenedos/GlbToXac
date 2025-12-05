"""
DDS (DirectDraw Surface) file writer.

Writes uncompressed RGBA DDS files from image data.
"""

import struct
import numpy as np
from typing import Tuple


# DDS Header constants
DDS_MAGIC = 0x20534444  # 'DDS '

# dwFlags
DDSD_CAPS = 0x1
DDSD_HEIGHT = 0x2
DDSD_WIDTH = 0x4
DDSD_PITCH = 0x8
DDSD_PIXELFORMAT = 0x1000
DDSD_MIPMAPCOUNT = 0x20000
DDSD_LINEARSIZE = 0x80000

# dwCaps
DDSCAPS_TEXTURE = 0x1000

# Pixel format flags
DDPF_ALPHAPIXELS = 0x1
DDPF_RGB = 0x40


def write_dds(filepath: str, image_data: np.ndarray):
    """
    Write image data to a DDS file (uncompressed RGBA).

    Args:
        filepath: Output file path
        image_data: NumPy array of shape (height, width, channels) with uint8 data
    """
    if len(image_data.shape) == 2:
        # Grayscale - convert to RGBA
        height, width = image_data.shape
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        rgba[:, :, 0] = image_data
        rgba[:, :, 1] = image_data
        rgba[:, :, 2] = image_data
        rgba[:, :, 3] = 255
        image_data = rgba
    elif image_data.shape[2] == 3:
        # RGB - add alpha channel
        height, width = image_data.shape[:2]
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        rgba[:, :, :3] = image_data
        rgba[:, :, 3] = 255
        image_data = rgba
    elif image_data.shape[2] == 4:
        # Already RGBA
        pass
    else:
        raise ValueError(f"Unsupported number of channels: {image_data.shape[2]}")

    height, width = image_data.shape[:2]

    with open(filepath, 'wb') as f:
        # Write DDS magic
        f.write(struct.pack('<I', DDS_MAGIC))

        # Write DDS header (124 bytes)
        dwSize = 124
        dwFlags = DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PITCH | DDSD_PIXELFORMAT
        dwHeight = height
        dwWidth = width
        dwPitchOrLinearSize = width * 4  # 4 bytes per pixel (RGBA)
        dwDepth = 0
        dwMipMapCount = 0
        dwReserved1 = [0] * 11

        f.write(struct.pack('<I', dwSize))
        f.write(struct.pack('<I', dwFlags))
        f.write(struct.pack('<I', dwHeight))
        f.write(struct.pack('<I', dwWidth))
        f.write(struct.pack('<I', dwPitchOrLinearSize))
        f.write(struct.pack('<I', dwDepth))
        f.write(struct.pack('<I', dwMipMapCount))
        for _ in range(11):
            f.write(struct.pack('<I', 0))  # dwReserved1

        # Write pixel format (32 bytes)
        _write_pixel_format(f)

        # Write caps
        dwCaps = DDSCAPS_TEXTURE
        dwCaps2 = 0
        dwCaps3 = 0
        dwCaps4 = 0
        dwReserved2 = 0

        f.write(struct.pack('<I', dwCaps))
        f.write(struct.pack('<I', dwCaps2))
        f.write(struct.pack('<I', dwCaps3))
        f.write(struct.pack('<I', dwCaps4))
        f.write(struct.pack('<I', dwReserved2))

        # Write pixel data (BGRA order for DDS)
        # Convert RGBA to BGRA
        bgra = image_data.copy()
        bgra[:, :, 0] = image_data[:, :, 2]  # B
        bgra[:, :, 2] = image_data[:, :, 0]  # R

        f.write(bgra.tobytes())


def _write_pixel_format(f):
    """Write the DDS_PIXELFORMAT structure (32 bytes)."""
    dwSize = 32
    dwFlags = DDPF_RGB | DDPF_ALPHAPIXELS
    dwFourCC = 0  # Not compressed
    dwRGBBitCount = 32
    dwRBitMask = 0x00FF0000
    dwGBitMask = 0x0000FF00
    dwBBitMask = 0x000000FF
    dwABitMask = 0xFF000000

    f.write(struct.pack('<I', dwSize))
    f.write(struct.pack('<I', dwFlags))
    f.write(struct.pack('<I', dwFourCC))
    f.write(struct.pack('<I', dwRGBBitCount))
    f.write(struct.pack('<I', dwRBitMask))
    f.write(struct.pack('<I', dwGBitMask))
    f.write(struct.pack('<I', dwBBitMask))
    f.write(struct.pack('<I', dwABitMask))
