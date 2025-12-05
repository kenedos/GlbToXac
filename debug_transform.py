#!/usr/bin/env python3
"""Debug the transform chain."""

import numpy as np
import pyrr

# Original XAC values for Bip01
xac_pos = (-4.6586197299802734e-07, 15.113086700439453, 7.466019630432129)
xac_quat = (-0.4999997317790985, 0.4999997317790985, -0.5000003576278687, 0.5000003576278687)
xac_scale = (1.0, 1.0, 1.0)

print("=== Original XAC values ===")
print(f"pos: {xac_pos}")
print(f"quat: {xac_quat}")
print(f"scale: {xac_scale}")

# XacToGlb get_local_transform
print("\n=== XacToGlb transformation ===")
transformed_pos = pyrr.Vector3([-xac_pos[0], xac_pos[2], xac_pos[1]])
transformed_quat = pyrr.Quaternion([-xac_quat[0], xac_quat[2], xac_quat[1], -xac_quat[3]])
transformed_quat = pyrr.quaternion.normalize(transformed_quat)

print(f"swizzled pos: {transformed_pos}")
print(f"swizzled quat: {transformed_quat}")

translation = pyrr.matrix44.create_from_translation(transformed_pos)
rotation = pyrr.matrix44.create_from_quaternion(transformed_quat)
scale_mat = pyrr.matrix44.create_from_scale(xac_scale)

# mul_order = 1
combined_mat = rotation @ scale_mat @ translation

print(f"\nCombined matrix:\n{combined_mat}")

# Decompose
t, r, s = pyrr.matrix44.decompose(combined_mat)
print(f"\n=== Decomposed (what goes into GLB) ===")
print(f"translation: {t}")
print(f"rotation: {r}")
print(f"scale: {s}")

# Now try to reverse it
print("\n=== Trying to reverse ===")

# Rebuild the matrix from decomposed values
t_mat = pyrr.matrix44.create_from_translation(t)
r_mat = pyrr.matrix44.create_from_quaternion(r)
s_mat = pyrr.matrix44.create_from_scale(s)
rebuilt_mat = r_mat @ s_mat @ t_mat

print(f"Rebuilt matrix:\n{rebuilt_mat}")
print(f"Matrices equal: {np.allclose(combined_mat, rebuilt_mat)}")

# Apply reverse swizzle to position and quaternion
# reverse of (-x, z, y) is (-x, z, y) (it's its own inverse!)
reversed_pos = (-t[0], t[2], t[1])
reversed_quat = (-r[0], r[2], r[1], -r[3])

print(f"\n=== Reversed values ===")
print(f"reversed pos: {reversed_pos}")
print(f"reversed quat: {reversed_quat}")

print(f"\n=== Comparison ===")
print(f"original pos: {xac_pos}")
print(f"reversed pos: {reversed_pos}")
print(f"pos match: {np.allclose(xac_pos, reversed_pos)}")

print(f"\noriginal quat: {xac_quat}")
print(f"reversed quat: {reversed_quat}")
