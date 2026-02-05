#!/usr/bin/env python
"""Test script for dual sampling methods."""

from spirals_mlp import generate_intertwined_spirals

# Test with BOTH sampling methods - should show side-by-side plots
print("Testing BOTH sampling methods (RND and CTR)...")
result = generate_intertwined_spirals(
    n=500,
    r_inner=1.0,
    r_outer=2.0,
    shift=1.0,
    seed=7,
    plot=True,
    sampling_method='BOTH'
)

if isinstance(result, tuple):
    rnd_data, ctr_data = result
    print(f"✓ RND dataset: {len(rnd_data)} points")
    print(f"✓ CTR dataset: {len(ctr_data)} points")
    print("✓ Side-by-side plots displayed successfully!")
else:
    print("ERROR: Expected tuple result for BOTH method")
