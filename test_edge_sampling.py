#!/usr/bin/env python
"""Test script for edge-weighted sampling method - compares all three methods."""

import matplotlib.pyplot as plt

from data import generate_intertwined_spirals

def compare_all_three_methods():
    """Generate and visualize all three sampling methods side by side."""
    n = 600
    seed = 7
    
    print("Generating data with three sampling methods...")
    
    # Generate data for each method
    rnd_data = generate_intertwined_spirals(
        n=n, seed=seed, sampling_method='RND', plot=False
    )
    
    ctr_data = generate_intertwined_spirals(
        n=n, seed=seed, sampling_method='CTR', plot=False
    )
    
    edge_data = generate_intertwined_spirals(
        n=n, seed=seed, sampling_method='EDGE', plot=False
    )
    
    def extract_data(data):
        x_c1 = [p[0] for p in data if p[2] == 0]
        y_c1 = [p[1] for p in data if p[2] == 0]
        x_c2 = [p[0] for p in data if p[2] == 1]
        y_c2 = [p[1] for p in data if p[2] == 1]
        return x_c1, y_c1, x_c2, y_c2
    
    rnd = extract_data(rnd_data)
    ctr = extract_data(ctr_data)
    edge = extract_data(edge_data)
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Plot RND
    ax1.plot(rnd[0], rnd[1], '.', label='C1', alpha=0.6, markersize=3, color='blue')
    ax1.plot(rnd[2], rnd[3], '.', label='C2', alpha=0.6, markersize=3, color='orange')
    ax1.set_xlabel('$x_1$', fontsize=11)
    ax1.set_ylabel('$x_2$', fontsize=11)
    ax1.set_title('RND: Uniform Random Sampling', fontsize=12, fontweight='bold')
    ax1.set_aspect('equal', 'box')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-4, 2.5)
    
    # Plot CTR
    ax2.plot(ctr[0], ctr[1], '.', label='C1', alpha=0.6, markersize=3, color='blue')
    ax2.plot(ctr[2], ctr[3], '.', label='C2', alpha=0.6, markersize=3, color='orange')
    ax2.set_xlabel('$x_1$', fontsize=11)
    ax2.set_ylabel('$x_2$', fontsize=11)
    ax2.set_title('CTR: Center-Weighted at (0, -0.5)', fontsize=12, fontweight='bold')
    ax2.set_aspect('equal', 'box')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-4, 2.5)
    
    # Plot EDGE
    ax3.plot(edge[0], edge[1], '.', label='C1', alpha=0.6, markersize=3, color='blue')
    ax3.plot(edge[2], edge[3], '.', label='C2', alpha=0.6, markersize=3, color='orange')
    ax3.set_xlabel('$x_1$', fontsize=11)
    ax3.set_ylabel('$x_2$', fontsize=11)
    ax3.set_title('EDGE: Boundary-Attracted Sampling', fontsize=12, fontweight='bold')
    ax3.set_aspect('equal', 'box')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_xlim(-3, 3)
    ax3.set_ylim(-4, 2.5)
    
    plt.suptitle('Comparison of Three Sampling Methods', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
    
    print(f"✓ RND dataset: {len(rnd_data)} points")
    print(f"✓ CTR dataset: {len(ctr_data)} points")
    print(f"✓ EDGE dataset: {len(edge_data)} points")
    print("✓ Three-way comparison plot displayed successfully!")


def main():
    print("Comparing all three sampling methods...")
    print("=" * 60)
    compare_all_three_methods()


if __name__ == "__main__":
    main()
