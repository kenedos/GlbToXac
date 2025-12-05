#!/usr/bin/env python3
"""
GLB to XAC/XSM Converter

Converts GLB (GLTF 2.0 binary) files to XAC (Actor) and XSM (Skeletal Motion) files.
Preserves IPF folder structure in output.
"""

import os
import sys
import argparse
from pathlib import Path

from glb_importer import load_glb
from xac_writer import write_xac
from xsm_writer import write_xsm
from dds_writer import write_dds


def extract_model_name(glb_file: Path) -> str:
    """
    Extract base model name from GLB filename.
    e.g., 'monster_popolion_set.glb' -> 'popolion'
    """
    name = glb_file.stem.lower()

    # Remove common prefixes
    for prefix in ['monster_', 'npc_', 'pc_', 'char_']:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break

    # Remove common suffixes
    for suffix in ['_set', '_model', '_mesh']:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break

    return name


def get_ipf_output_structure(glb_file: Path, input_dir: Path) -> dict:
    """
    Determine the IPF folder structure for output based on input file location.

    Returns dict with paths for:
    - xac_dir: Directory for the XAC file (char_hi.ipf/...)
    - xsm_dir: Directory for XSM files (animation.ipf/...)
    - texture_dir: Directory for texture files (char_texture.ipf/...)
    """
    # Get relative path from input directory
    try:
        rel_path = glb_file.relative_to(input_dir)
    except ValueError:
        rel_path = Path(glb_file.name)

    # Get the parent directory structure (without the filename)
    parent_parts = list(rel_path.parent.parts)

    model_name = extract_model_name(glb_file)

    # Determine the subfolder (e.g., 'monster', 'npc', etc.)
    subfolder = None
    if parent_parts:
        # Check if we're in a char_hi.ipf structure already
        for i, part in enumerate(parent_parts):
            if part.endswith('.ipf'):
                # Use remaining parts as subfolder
                if i + 1 < len(parent_parts):
                    subfolder = '/'.join(parent_parts[i + 1:])
                break

        # If no .ipf found, use the directory structure as-is
        if subfolder is None and parent_parts:
            subfolder = '/'.join(parent_parts)

    # Default subfolder based on filename prefix
    if not subfolder:
        filename = glb_file.stem.lower()
        if filename.startswith('monster_'):
            subfolder = 'monster'
        elif filename.startswith('npc_'):
            subfolder = 'npc'
        elif filename.startswith('pc_'):
            subfolder = 'pc'
        else:
            subfolder = 'misc'

    return {
        'xac_dir': Path('char_hi.ipf') / subfolder,
        'xsm_dir': Path('animation.ipf') / subfolder / model_name,
        'texture_dir': Path('char_texture.ipf') / subfolder / model_name,
        'model_name': model_name
    }


def convert_glb(input_path: Path, input_dir: Path, output_dir: Path, verbose: bool = False):
    """
    Convert a GLB file to XAC, XSM, and DDS files with IPF folder structure.

    Args:
        input_path: Path to the input GLB file
        input_dir: Base input directory (for relative path calculation)
        output_dir: Base output directory
        verbose: Print verbose output

    Returns:
        Tuple of (num_animations, num_textures)
    """
    if verbose:
        print(f"Loading: {input_path}")

    # Load GLB file
    model_data, animations, textures = load_glb(str(input_path))

    # Get IPF output structure
    ipf_structure = get_ipf_output_structure(input_path, input_dir)

    # Create output directories
    xac_output_dir = output_dir / ipf_structure['xac_dir']
    xsm_output_dir = output_dir / ipf_structure['xsm_dir']
    texture_output_dir = output_dir / ipf_structure['texture_dir']

    xac_output_dir.mkdir(parents=True, exist_ok=True)
    if animations:
        xsm_output_dir.mkdir(parents=True, exist_ok=True)
    if textures:
        texture_output_dir.mkdir(parents=True, exist_ok=True)

    # Create output filename based on input
    base_name = input_path.stem

    # Write XAC file
    xac_path = xac_output_dir / f"{base_name}.xac"
    write_xac(model_data, str(xac_path))
    if verbose:
        print(f"  Created: {xac_path}")
        print(f"    Nodes: {len(model_data.nodes)}")
        print(f"    Meshes: {len(model_data.meshes)}")
        print(f"    Materials: {len(model_data.materials)}")

    # Write XSM files for each animation
    for anim in animations:
        # Sanitize animation name for filename
        anim_name = anim.name.replace(" ", "_").replace("/", "_").replace("\\", "_")

        # Use model_name prefix for animation files (matching original structure)
        model_name = ipf_structure['model_name']
        xsm_path = xsm_output_dir / f"{model_name}_{anim_name}.xsm"
        write_xsm(anim, str(xsm_path))
        if verbose:
            print(f"  Created: {xsm_path}")
            print(f"    Tracks: {len(anim.tracks)}")
            print(f"    Duration: {anim.duration:.3f}s")

    # Write DDS texture files
    model_name = ipf_structure['model_name']
    for i, tex in enumerate(textures):
        # Use model_name for the texture filename (matching original structure)
        # If multiple textures, append index
        if len(textures) == 1:
            tex_name = model_name
        else:
            tex_name = f"{model_name}_{i}" if i > 0 else model_name

        dds_path = texture_output_dir / f"{tex_name}.dds"
        write_dds(str(dds_path), tex.data)
        if verbose:
            print(f"  Created: {dds_path}")
            print(f"    Size: {tex.data.shape[1]}x{tex.data.shape[0]}")

    return len(animations), len(textures)


def main():
    parser = argparse.ArgumentParser(
        description="Convert GLB files to XAC, XSM, and DDS format with IPF folder structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert.py                           # Convert all GLB files in input/
  python convert.py -i model.glb              # Convert a single file
  python convert.py -i input/ -o output/ -v   # Convert folder with verbose output
        """
    )

    parser.add_argument(
        "-i", "--input",
        default="input",
        help="Input GLB file or directory (default: input/)"
    )

    parser.add_argument(
        "-o", "--output",
        default="output",
        help="Output directory for XAC/XSM/DDS files (default: output/)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print verbose output"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    # Collect GLB files to process
    glb_files = []
    input_dir = input_path

    if input_path.is_file():
        if input_path.suffix.lower() == ".glb":
            glb_files.append(input_path)
            input_dir = input_path.parent
        else:
            print(f"Error: {input_path} is not a GLB file")
            sys.exit(1)
    elif input_path.is_dir():
        glb_files = list(input_path.rglob("*.glb"))
        if not glb_files:
            print(f"No GLB files found in {input_path}")
            sys.exit(1)
    else:
        print(f"Error: {input_path} does not exist")
        sys.exit(1)

    print(f"Found {len(glb_files)} GLB file(s) to convert")
    print()

    total_animations = 0
    total_textures = 0
    successful = 0
    failed = 0

    for glb_file in glb_files:
        try:
            print(f"Converting: {glb_file.name}")
            num_anims, num_textures = convert_glb(glb_file, input_dir, output_dir, args.verbose)
            total_animations += num_anims
            total_textures += num_textures
            successful += 1
            print(f"  Success: 1 XAC + {num_anims} XSM + {num_textures} DDS file(s)")
        except Exception as e:
            failed += 1
            print(f"  Error: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
        print()

    print("=" * 50)
    print(f"Conversion complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total XAC files: {successful}")
    print(f"  Total XSM files: {total_animations}")
    print(f"  Total DDS files: {total_textures}")
    print(f"  Output directory: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
