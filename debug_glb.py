#!/usr/bin/env python3
"""Debug script to check GLB node data."""

from pygltflib import GLTF2

def debug_glb(filepath):
    print(f"\n=== Debugging GLB: {filepath} ===\n")
    gltf = GLTF2().load(filepath)

    print(f"Number of nodes: {len(gltf.nodes)}")

    # Find skin to get skeleton info
    if gltf.skins:
        skin = gltf.skins[0]
        print(f"Skin joints: {len(skin.joints)} joints")
        print(f"Skeleton root: {skin.skeleton}")

    print()
    for i, node in enumerate(gltf.nodes[:10]):  # First 10 nodes
        print(f"Node {i}: {node.name}")
        print(f"  translation: {node.translation}")
        print(f"  rotation: {node.rotation}")
        print(f"  scale: {node.scale}")
        print(f"  matrix: {node.matrix}")
        print(f"  children: {node.children}")
        print()

if __name__ == "__main__":
    # Check the GLB that was converted from original XAC
    debug_glb(r"E:\Melia\XacToGlb\output\char_hi.ipf\monster\monster_popolion_set.glb")
