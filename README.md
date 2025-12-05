# GLB to XAC/XSM Converter

Converts GLB (GLTF 2.0 binary) files to XAC (Actor), XSM (Skeletal Motion), and DDS (texture) files used by the EMotion FX engine. Preserves IPF folder structure in output.

## Credits

All credits go to [SalmanTKhan](https://github.com/SalmanTKhan) for the original XAC/XSM format research and implementation.

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Place GLB files in the `input/` folder and run:

```bash
python convert.py
```

Output XAC, XSM, and DDS files will be created in the `output/` folder with IPF folder structure.

### Command Line Options

```bash
python convert.py -i <input> -o <output> [-v]
```

- `-i, --input`: Input GLB file or directory (default: `input/`)
- `-o, --output`: Output directory for XAC/XSM/DDS files (default: `output/`)
- `-v, --verbose`: Print verbose output

### Examples

```bash
# Convert all GLB files in input folder
python convert.py

# Convert a single file
python convert.py -i model.glb

# Convert with verbose output
python convert.py -i input/ -o output/ -v
```

## IPF Folder Structure

The converter automatically generates the proper IPF folder structure based on the input filename.

### Input

```
input/
└── monster_popolion_set.glb      # GLB file with embedded animations and textures
```

### Output

```
output/
├── char_hi.ipf/
│   └── monster/
│       └── monster_popolion_set.xac      # Model file
├── animation.ipf/
│   └── monster/
│       └── popolion/
│           ├── popolion_idle.xsm         # Animation files
│           ├── popolion_run.xsm
│           ├── popolion_atk1.xsm
│           └── popolion_hit.xsm
└── char_texture.ipf/
    └── monster/
        └── popolion/
            └── popolion.dds              # Texture files
```

The converter will:
1. Detect the model type from filename prefix (monster_, npc_, pc_)
2. Extract the model name (e.g., `monster_popolion_set` -> `popolion`)
3. Create `char_hi.ipf/<type>/` for the XAC file
4. Create `animation.ipf/<type>/<model_name>/` for XSM files
5. Create `char_texture.ipf/<type>/<model_name>/` for DDS texture files

### Preserving Input Structure

If your input already has folder structure, it will be preserved:

```
input/
└── char_hi.ipf/
    └── monster/
        └── monster_popolion_set.glb
```

Output:
```
output/
├── char_hi.ipf/
│   └── monster/
│       └── monster_popolion_set.xac
├── animation.ipf/
│   └── monster/
│       └── popolion/
│           └── popolion_*.xsm
└── char_texture.ipf/
    └── monster/
        └── popolion/
            └── *.dds
```

## File Formats

### XAC (Actor)

The XAC format stores:
- Node hierarchy (skeleton)
- Mesh geometry (positions, normals, UVs)
- Materials with texture references
- Skinning information (bone weights)

### XSM (Skeletal Motion)

The XSM format stores:
- Animation tracks per bone
- Position keyframes
- Rotation keyframes (quaternion)
- Scale keyframes

### DDS (DirectDraw Surface)

The DDS format stores:
- Uncompressed RGBA texture data
- Compatible with game engines and image editors

## Coordinate System

The converter handles the coordinate system transformation between:
- **GLTF**: Y-up, right-handed
- **XAC/XSM**: Z-up coordinate system

Transformation applied:
- Position: `(-x, z, y)`
- Quaternion: `(-x, z, y, -w)`

## Limitations

- Only supports skinned meshes with up to 4 bone influences per vertex
- DDS textures are exported as uncompressed RGBA (no DXT compression)
- Morph targets are not supported
- Only linear interpolation is used for animations

## Project Structure

```
GlbToXac/
├── convert.py          # Main conversion script
├── glb_importer.py     # GLB file reader
├── xac_writer.py       # XAC file writer
├── xsm_writer.py       # XSM animation file writer
├── dds_writer.py       # DDS texture file writer
├── binary_writer.py    # Binary writing utilities
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── input/              # Place GLB files here
└── output/             # Converted files appear here (with IPF structure)
```

## See Also

- [XacToGlb](../XacToGlb/) - The reverse converter (XAC/XSM to GLB)
