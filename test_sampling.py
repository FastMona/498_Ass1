#!/usr/bin/env python
"""Test script for all three sampling methods (RND, CTR, and EDGE)."""

from data import generate_intertwined_spirals

# Test with ALL sampling methods - should show side-by-side plots with all three
print("Testing ALL sampling methods (RND, CTR, and EDGE)...")
result = generate_intertwined_spirals(
    n=500,
    seed=7,
    plot=True,
    sampling_method='ALL'
)

if isinstance(result, tuple) and len(result) == 3:
    rnd_data, ctr_data, edge_data = result
    print(f"✓ RND dataset: {len(rnd_data)} points")
    print(f"✓ CTR dataset: {len(ctr_data)} points")
    print(f"✓ EDGE dataset: {len(edge_data)} points")
    print("✓ Three-way comparison plot displayed successfully!")
else:
    print(f"ERROR: Expected tuple of 3 datasets, got {type(result)}")
    if isinstance(result, tuple):
        print(f"  Tuple length: {len(result)}")
