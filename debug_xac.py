#!/usr/bin/env python3
"""Debug script to compare XAC files."""

import sys
sys.path.insert(0, r'E:\Melia\XacToGlb')

from xac_parser import XACParser, XACNodes
import numpy as np

def debug_xac(filepath):
    print(f"\n=== Debugging: {filepath} ===\n")
    parser = XACParser(filepath)
    chunks = parser.parse()

    print(f"Header mul_order: {parser.header.mul_order}")

    nodes_chunk = next((c for c in chunks if isinstance(c, XACNodes)), None)
    if nodes_chunk:
        print(f"Number of nodes: {nodes_chunk.num_nodes}")
        print(f"Number of root nodes: {nodes_chunk.num_root_nodes}")
        print()

        for i, node in enumerate(nodes_chunk.nodes[:5]):  # First 5 nodes
            print(f"Node {i}: {node.node_name}")
            print(f"  local_pos: {node.local_pos}")
            print(f"  local_quat: {node.local_quat}")
            print(f"  local_scale: {node.local_scale}")
            print(f"  parent_index: {node.parent_index}")

            # Check quaternion length
            q = node.local_quat
            qlen = (q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2) ** 0.5
            print(f"  quat length: {qlen}")
            print()

if __name__ == "__main__":
    # Compare original vs generated
    print("ORIGINAL XAC (from game):")
    debug_xac(r"C:\MeliaIPF\char_hi.ipf\monster\monster_popolion_set.xac")

    print("\n" + "="*60 + "\n")

    print("GENERATED XAC (from GlbToXac):")
    debug_xac(r"E:\Melia\XacToGlb\input\char_hi.ipf\monster\monster_popolion_set.xac")
