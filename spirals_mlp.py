import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


# ============================
# Region Definitions
# ============================
def _c1_predicate(x, y, r_inner, r_outer, shift):
    """C1: Left half-annulus + right half disk centered at (0, -shift)."""
    r2_origin = x ** 2 + y ** 2
    r2_shifted = x ** 2 + (y + shift) ** 2
    in_left_annulus = (x <= 0) & (r2_origin >= r_inner ** 2) & (r2_origin <= r_outer ** 2)
    in_right_inner_disk = (x >= 0) & (r2_shifted <= r_inner ** 2)
    return in_left_annulus | in_right_inner_disk


def _c2_predicate(x, y, r_inner, r_outer, shift):
    """C2: Right half-annulus centered at (0, -shift) + left half disk at origin."""
    r2_origin = x ** 2 + y ** 2
    r2_shifted = x ** 2 + (y + shift) ** 2
    in_right_annulus = (x >= 0) & (r2_shifted >= r_inner ** 2) & (r2_shifted <= r_outer ** 2)
    in_left_inner_disk = (x <= 0) & (r2_origin <= r_inner ** 2)
    return in_right_annulus | in_left_inner_disk


# ============================
# Data Generation Function
# ============================
def _plot_interlocked_region_boundaries(
    ax,
    c1_predicate,
    c2_predicate,
    bounds,
    grid_res=400,
    remove_x0_ranges=None
):
    xmin, xmax, ymin, ymax = bounds
    xs = np.linspace(xmin, xmax, grid_res)
    ys = np.linspace(ymin, ymax, grid_res)
    xx, yy = np.meshgrid(xs, ys)

    c1_mask = c1_predicate(xx, yy).astype(float)
    c2_mask = c2_predicate(xx, yy).astype(float)

    # Create combined region map for shading (before masking)
    region_map = np.zeros_like(xx)
    region_map[c1_mask > 0.5] = 1  # C1
    region_map[c2_mask > 0.5] = 2  # C2

    # Shade regions with light colors
    from matplotlib.colors import ListedColormap
    colors_list = ['white', '#CCE5FF', '#FFE5CC']  # C3 (white), C1 (light blue), C2 (light orange)
    cmap = ListedColormap(colors_list)
    ax.contourf(xx, yy, region_map, levels=[0, 0.5, 1.5, 2.5], cmap=cmap, alpha=0.7)

    # Apply masking only for boundary lines
    if remove_x0_ranges:
        dx = xs[1] - xs[0]
        near_x0 = np.abs(xx) <= (1.5 * dx)
        for y_min, y_max in remove_x0_ranges:
            in_range = (yy >= y_min) & (yy <= y_max)
            c1_mask[near_x0 & in_range] = np.nan
            c2_mask[near_x0 & in_range] = np.nan

    ax.contour(xx, yy, c1_mask, levels=[0.5], colors='k', linewidths=3)
    ax.contour(xx, yy, c2_mask, levels=[0.5], colors='k', linewidths=3)


def _plot_both_sampling_methods(rnd_data, ctr_data, r_inner, r_outer, shift):
    """Plot side-by-side comparison of RND and CTR sampling methods."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Extract data for RND
    x_c1_rnd = [p[0] for p in rnd_data if p[2] == 0]
    y_c1_rnd = [p[1] for p in rnd_data if p[2] == 0]
    x_c2_rnd = [p[0] for p in rnd_data if p[2] == 1]
    y_c2_rnd = [p[1] for p in rnd_data if p[2] == 1]
    
    # Extract data for CTR
    x_c1_ctr = [p[0] for p in ctr_data if p[2] == 0]
    y_c1_ctr = [p[1] for p in ctr_data if p[2] == 0]
    x_c2_ctr = [p[0] for p in ctr_data if p[2] == 1]
    y_c2_ctr = [p[1] for p in ctr_data if p[2] == 1]
    
    # Create predicates for boundaries
    def c1_pred(x, y):
        return _c1_predicate(x, y, r_inner, r_outer, shift)
    def c2_pred(x, y):
        return _c2_predicate(x, y, r_inner, r_outer, shift)
    
    bounds = (-r_outer - 0.5, r_outer + 0.5, -shift - r_outer - 0.5, r_outer + 0.5)
    
    # Plot RND (left subplot)
    ax1.plot(x_c1_rnd, y_c1_rnd, '.', label='C1', alpha=0.6, markersize=3)
    ax1.plot(x_c2_rnd, y_c2_rnd, '.', label='C2', alpha=0.6, markersize=3)
    _plot_interlocked_region_boundaries(
        ax1, c1_pred, c2_pred, bounds, remove_x0_ranges=[(0, r_inner)]
    )
    ax1.set_xlabel('$x_1$')
    ax1.set_ylabel('$x_2$')
    ax1.set_title('RND (Uniform Random Sampling)')
    ax1.set_aspect('equal', 'box')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot CTR (right subplot)
    ax2.plot(x_c1_ctr, y_c1_ctr, '.', label='C1', alpha=0.6, markersize=3)
    ax2.plot(x_c2_ctr, y_c2_ctr, '.', label='C2', alpha=0.6, markersize=3)
    _plot_interlocked_region_boundaries(
        ax2, c1_pred, c2_pred, bounds, remove_x0_ranges=[(0, r_inner)]
    )
    ax2.set_xlabel('$x_1$')
    ax2.set_ylabel('$x_2$')
    ax2.set_title('CTR (Center-Weighted Sampling at (0, -0.5))')
    ax2.set_aspect('equal', 'box')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


def _plot_three_sampling_methods(rnd_data, ctr_data, edge_data, r_inner, r_outer, shift):
    """Plot side-by-side comparison of RND, CTR, and EDGE sampling methods."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Extract data for RND
    x_c1_rnd = [p[0] for p in rnd_data if p[2] == 0]
    y_c1_rnd = [p[1] for p in rnd_data if p[2] == 0]
    x_c2_rnd = [p[0] for p in rnd_data if p[2] == 1]
    y_c2_rnd = [p[1] for p in rnd_data if p[2] == 1]
    
    # Extract data for CTR
    x_c1_ctr = [p[0] for p in ctr_data if p[2] == 0]
    y_c1_ctr = [p[1] for p in ctr_data if p[2] == 0]
    x_c2_ctr = [p[0] for p in ctr_data if p[2] == 1]
    y_c2_ctr = [p[1] for p in ctr_data if p[2] == 1]
    
    # Extract data for EDGE
    x_c1_edge = [p[0] for p in edge_data if p[2] == 0]
    y_c1_edge = [p[1] for p in edge_data if p[2] == 0]
    x_c2_edge = [p[0] for p in edge_data if p[2] == 1]
    y_c2_edge = [p[1] for p in edge_data if p[2] == 1]
    
    # Create predicates for boundaries
    def c1_pred(x, y):
        return _c1_predicate(x, y, r_inner, r_outer, shift)
    def c2_pred(x, y):
        return _c2_predicate(x, y, r_inner, r_outer, shift)
    
    bounds = (-r_outer - 0.5, r_outer + 0.5, -shift - r_outer - 0.5, r_outer + 0.5)
    
    # Plot RND (left subplot)
    ax1.plot(x_c1_rnd, y_c1_rnd, '.', label='C1', alpha=0.6, markersize=3)
    ax1.plot(x_c2_rnd, y_c2_rnd, '.', label='C2', alpha=0.6, markersize=3)
    _plot_interlocked_region_boundaries(
        ax1, c1_pred, c2_pred, bounds, remove_x0_ranges=[(0, r_inner)]
    )
    ax1.set_xlabel('$x_1$')
    ax1.set_ylabel('$x_2$')
    ax1.set_title('RND (Uniform Random Sampling)')
    ax1.set_aspect('equal', 'box')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot CTR (middle subplot)
    ax2.plot(x_c1_ctr, y_c1_ctr, '.', label='C1', alpha=0.6, markersize=3)
    ax2.plot(x_c2_ctr, y_c2_ctr, '.', label='C2', alpha=0.6, markersize=3)
    _plot_interlocked_region_boundaries(
        ax2, c1_pred, c2_pred, bounds, remove_x0_ranges=[(0, r_inner)]
    )
    ax2.set_xlabel('$x_1$')
    ax2.set_ylabel('$x_2$')
    ax2.set_title('CTR (Center-Weighted Sampling at (0, -0.5))')
    ax2.set_aspect('equal', 'box')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot EDGE (right subplot)
    ax3.plot(x_c1_edge, y_c1_edge, '.', label='C1', alpha=0.6, markersize=3)
    ax3.plot(x_c2_edge, y_c2_edge, '.', label='C2', alpha=0.6, markersize=3)
    _plot_interlocked_region_boundaries(
        ax3, c1_pred, c2_pred, bounds, remove_x0_ranges=[(0, r_inner)]
    )
    ax3.set_xlabel('$x_1$')
    ax3.set_ylabel('$x_2$')
    ax3.set_title('EDGE (Boundary-Attracted Sampling)')
    ax3.set_aspect('equal', 'box')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    plt.show()


def _plot_three_sampling_methods(rnd_data, ctr_data, edge_data, r_inner, r_outer, shift):
    """Plot side-by-side comparison of RND, CTR, and EDGE sampling methods."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Extract data for RND
    x_c1_rnd = [p[0] for p in rnd_data if p[2] == 0]
    y_c1_rnd = [p[1] for p in rnd_data if p[2] == 0]
    x_c2_rnd = [p[0] for p in rnd_data if p[2] == 1]
    y_c2_rnd = [p[1] for p in rnd_data if p[2] == 1]
    
    # Extract data for CTR
    x_c1_ctr = [p[0] for p in ctr_data if p[2] == 0]
    y_c1_ctr = [p[1] for p in ctr_data if p[2] == 0]
    x_c2_ctr = [p[0] for p in ctr_data if p[2] == 1]
    y_c2_ctr = [p[1] for p in ctr_data if p[2] == 1]
    
    # Extract data for EDGE
    x_c1_edge = [p[0] for p in edge_data if p[2] == 0]
    y_c1_edge = [p[1] for p in edge_data if p[2] == 0]
    x_c2_edge = [p[0] for p in edge_data if p[2] == 1]
    y_c2_edge = [p[1] for p in edge_data if p[2] == 1]
    
    # Create predicates for boundaries
    def c1_pred(x, y):
        return _c1_predicate(x, y, r_inner, r_outer, shift)
    def c2_pred(x, y):
        return _c2_predicate(x, y, r_inner, r_outer, shift)
    
    bounds = (-r_outer - 0.5, r_outer + 0.5, -shift - r_outer - 0.5, r_outer + 0.5)
    
    # Plot RND (left subplot)
    ax1.plot(x_c1_rnd, y_c1_rnd, '.', label='C1', alpha=0.6, markersize=3)
    ax1.plot(x_c2_rnd, y_c2_rnd, '.', label='C2', alpha=0.6, markersize=3)
    _plot_interlocked_region_boundaries(
        ax1, c1_pred, c2_pred, bounds, remove_x0_ranges=[(0, r_inner)]
    )
    ax1.set_xlabel('$x_1$')
    ax1.set_ylabel('$x_2$')
    ax1.set_title('RND (Uniform Random Sampling)')
    ax1.set_aspect('equal', 'box')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot CTR (middle subplot)
    ax2.plot(x_c1_ctr, y_c1_ctr, '.', label='C1', alpha=0.6, markersize=3)
    ax2.plot(x_c2_ctr, y_c2_ctr, '.', label='C2', alpha=0.6, markersize=3)
    _plot_interlocked_region_boundaries(
        ax2, c1_pred, c2_pred, bounds, remove_x0_ranges=[(0, r_inner)]
    )
    ax2.set_xlabel('$x_1$')
    ax2.set_ylabel('$x_2$')
    ax2.set_title('CTR (Center-Weighted Sampling at (0, -0.5))')
    ax2.set_aspect('equal', 'box')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot EDGE (right subplot)
    ax3.plot(x_c1_edge, y_c1_edge, '.', label='C1', alpha=0.6, markersize=3)
    ax3.plot(x_c2_edge, y_c2_edge, '.', label='C2', alpha=0.6, markersize=3)
    _plot_interlocked_region_boundaries(
        ax3, c1_pred, c2_pred, bounds, remove_x0_ranges=[(0, r_inner)]
    )
    ax3.set_xlabel('$x_1$')
    ax3.set_ylabel('$x_2$')
    ax3.set_title('EDGE (Boundary-Attracted Sampling)')
    ax3.set_aspect('equal', 'box')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    plt.show()


def generate_intertwined_spirals(
    n=600,
    r_inner=1.0,
    r_outer=2.0,
    shift=1.0,
    noise_std=0.0,
    seed=None,
    include_inner_caps=False,
    plot=False,
    sampling_method="RND"
):
    """
    Generate two interlocked regions that form symmetric spiraling shapes.

    Region C1 (blue): Left half-annulus [r_inner, r_outer] centered at origin
                      + right half disk [0, r_inner] centered at (0, -shift)
    Region C2 (orange): Right half-annulus [r_inner, r_outer] centered at (0, -shift)
                        + left half disk [0, r_inner] centered at origin

    The regions interlock symmetrically around the point (0, -shift/2).

    Parameters:
    -----------
    n : int
        Number of points per region (default 600)
    r_inner : float
        Inner radius of annuli (default 1.0)
    r_outer : float
        Outer radius of annuli (default 2.0)
    shift : float
        Downward shift of right half regions (default 1.0)
    noise_std : float
        Standard deviation of Gaussian noise (default 0.0)
    seed : int or None
        Random seed for reproducibility (default None)
    include_inner_caps : bool
        Unused legacy parameter (default False)
    plot : bool
        If True, plot the regions with boundaries (default False)
    sampling_method : str
        "RND" for uniform random sampling
        "CTR" for center-weighted (shotgun) sampling around (0, -0.5)
        "EDGE" for edge-weighted sampling (concentrates points near boundaries)
        "BOTH" to generate both and show side-by-side plots

    Returns:
    --------
    list of tuples or tuple of three lists
        If sampling_method is "RND", "CTR", or "EDGE": list of 2n tuples (x, y, label)
        If sampling_method is "BOTH": tuple of (rnd_data, ctr_data, edge_data)
    """
    rng = np.random.default_rng(seed)

    # Use predicate helper functions with partial application
    def c1_predicate(x, y):
        return _c1_predicate(x, y, r_inner, r_outer, shift)

    def c2_predicate(x, y):
        return _c2_predicate(x, y, r_inner, r_outer, shift)

    def _sample_half_annulus(n_samples, theta_min, theta_max, y_shift=0):
        """Sample uniformly from a half-annulus in polar coordinates."""
        theta = rng.uniform(theta_min, theta_max, n_samples)
        r = np.sqrt(rng.uniform(r_inner ** 2, r_outer ** 2, n_samples))
        return r * np.cos(theta), r * np.sin(theta) + y_shift

    def _sample_half_disk(n_samples, theta_min, theta_max, y_shift=0):
        """Sample uniformly from a half-disk in polar coordinates."""
        theta = rng.uniform(theta_min, theta_max, n_samples)
        r = np.sqrt(rng.uniform(0, r_inner ** 2, n_samples))
        return r * np.cos(theta), r * np.sin(theta) + y_shift

    def _sample_c1(n_samples):
        """Sample C1: left annulus + right half disk at (0, -shift)."""
        # Proportional split by area
        area_annulus = np.pi * (r_outer ** 2 - r_inner ** 2) / 2
        area_disk = np.pi * r_inner ** 2 / 2
        n_annulus = int(n_samples * area_annulus / (area_annulus + area_disk))
        n_disk = n_samples - n_annulus
        
        x_ann, y_ann = _sample_half_annulus(n_annulus, np.pi / 2, 3 * np.pi / 2)
        x_disk, y_disk = _sample_half_disk(n_disk, -np.pi / 2, np.pi / 2, -shift)
        
        return np.concatenate([x_ann, x_disk]), np.concatenate([y_ann, y_disk])

    def _sample_c2(n_samples):
        """Sample C2: right annulus at (0, -shift) + left half disk at origin."""
        # Proportional split by area
        area_annulus = np.pi * (r_outer ** 2 - r_inner ** 2) / 2
        area_disk = np.pi * r_inner ** 2 / 2
        n_annulus = int(n_samples * area_annulus / (area_annulus + area_disk))
        n_disk = n_samples - n_annulus
        
        x_ann, y_ann = _sample_half_annulus(n_annulus, -np.pi / 2, np.pi / 2, -shift)
        x_disk, y_disk = _sample_half_disk(n_disk, np.pi / 2, 3 * np.pi / 2)
        
        return np.concatenate([x_ann, x_disk]), np.concatenate([y_ann, y_disk])

    # Center-weighted (shotgun) sampling functions
    def _sample_c1_centered(n_samples, center_x=0, center_y=-0.5, sigma=0.8):
        """Sample C1 using center-weighted Gaussian distribution with rejection sampling."""
        x_samples, y_samples = [], []
        max_attempts = n_samples * 100  # Safety limit
        attempts = 0
        
        while len(x_samples) < n_samples and attempts < max_attempts:
            # Generate candidates from 2D Gaussian centered at (center_x, center_y)
            batch_size = (n_samples - len(x_samples)) * 3
            x_cand = rng.normal(center_x, sigma, batch_size)
            y_cand = rng.normal(center_y, sigma, batch_size)
            
            # Keep only points that fall within C1 region
            mask = c1_predicate(x_cand, y_cand)
            x_samples.extend(x_cand[mask])
            y_samples.extend(y_cand[mask])
            attempts += batch_size
        
        return np.array(x_samples[:n_samples]), np.array(y_samples[:n_samples])

    def _sample_c2_centered(n_samples, center_x=0, center_y=-0.5, sigma=0.8):
        """Sample C2 using center-weighted Gaussian distribution with rejection sampling."""
        x_samples, y_samples = [], []
        max_attempts = n_samples * 100  # Safety limit
        attempts = 0
        
        while len(x_samples) < n_samples and attempts < max_attempts:
            # Generate candidates from 2D Gaussian centered at (center_x, center_y)
            batch_size = (n_samples - len(x_samples)) * 3
            x_cand = rng.normal(center_x, sigma, batch_size)
            y_cand = rng.normal(center_y, sigma, batch_size)
            
            # Keep only points that fall within C2 region
            mask = c2_predicate(x_cand, y_cand)
            x_samples.extend(x_cand[mask])
            y_samples.extend(y_cand[mask])
            attempts += batch_size
        
        return np.array(x_samples[:n_samples]), np.array(y_samples[:n_samples])

    # Edge-weighted sampling functions
    def _sample_half_annulus_edge(n_samples, theta_min, theta_max, y_shift=0, edge_bias=3.0):
        """Sample from a half-annulus with concentration near inner and outer edges.
        
        Uses Beta distribution to create U-shaped radial distribution.
        edge_bias: lower values (e.g., 0.5) = stronger edge concentration
        """
        theta = rng.uniform(theta_min, theta_max, n_samples)
        # Beta distribution with alpha=beta < 1 creates U-shape (concentration at edges)
        u = rng.beta(edge_bias, edge_bias, n_samples)
        # Transform to radius, concentrating near r_inner and r_outer
        r_squared = r_inner**2 + u * (r_outer**2 - r_inner**2)
        r = np.sqrt(r_squared)
        return r * np.cos(theta), r * np.sin(theta) + y_shift

    def _sample_half_disk_edge(n_samples, theta_min, theta_max, y_shift=0, edge_bias=2.0):
        """Sample from a half-disk with concentration near the outer edge.
        
        Uses power distribution to push points toward r_inner boundary.
        edge_bias: higher values (e.g., 2-3) = stronger edge concentration
        """
        theta = rng.uniform(theta_min, theta_max, n_samples)
        # Power distribution: u^(1/edge_bias) concentrates near 1 (outer edge)
        u = rng.uniform(0, 1, n_samples)
        u_transformed = u ** (1.0 / edge_bias)
        r = np.sqrt(u_transformed * r_inner**2)
        return r * np.cos(theta), r * np.sin(theta) + y_shift

    def _sample_c1_edge(n_samples, edge_bias_annulus=0.5, edge_bias_disk=2.5):
        """Sample C1: left annulus + right half disk with edge concentration."""
        # Proportional split by area
        area_annulus = np.pi * (r_outer ** 2 - r_inner ** 2) / 2
        area_disk = np.pi * r_inner ** 2 / 2
        n_annulus = int(n_samples * area_annulus / (area_annulus + area_disk))
        n_disk = n_samples - n_annulus
        
        x_ann, y_ann = _sample_half_annulus_edge(n_annulus, np.pi / 2, 3 * np.pi / 2, 0, edge_bias_annulus)
        x_disk, y_disk = _sample_half_disk_edge(n_disk, -np.pi / 2, np.pi / 2, -shift, edge_bias_disk)
        
        return np.concatenate([x_ann, x_disk]), np.concatenate([y_ann, y_disk])

    def _sample_c2_edge(n_samples, edge_bias_annulus=0.5, edge_bias_disk=2.5):
        """Sample C2: right annulus at (0, -shift) + left half disk with edge concentration."""
        # Proportional split by area
        area_annulus = np.pi * (r_outer ** 2 - r_inner ** 2) / 2
        area_disk = np.pi * r_inner ** 2 / 2
        n_annulus = int(n_samples * area_annulus / (area_annulus + area_disk))
        n_disk = n_samples - n_annulus
        
        x_ann, y_ann = _sample_half_annulus_edge(n_annulus, -np.pi / 2, np.pi / 2, -shift, edge_bias_annulus)
        x_disk, y_disk = _sample_half_disk_edge(n_disk, np.pi / 2, 3 * np.pi / 2, 0, edge_bias_disk)
        
        return np.concatenate([x_ann, x_disk]), np.concatenate([y_ann, y_disk])

    # Choose sampling method
    if sampling_method == "BOTH":
        # Generate all three datasets
        # Random sampling
        x_left_rnd, y_left_rnd = _sample_c1(n)
        x_right_rnd, y_right_rnd = _sample_c2(n)
        
        # Center-weighted sampling
        x_left_ctr, y_left_ctr = _sample_c1_centered(n)
        x_right_ctr, y_right_ctr = _sample_c2_centered(n)
        
        # Edge-weighted sampling
        x_left_edge, y_left_edge = _sample_c1_edge(n)
        x_right_edge, y_right_edge = _sample_c2_edge(n)
        
        if noise_std > 0:
            x_left_rnd += noise_std * rng.standard_normal(n)
            y_left_rnd += noise_std * rng.standard_normal(n)
            x_right_rnd += noise_std * rng.standard_normal(n)
            y_right_rnd += noise_std * rng.standard_normal(n)
            
            x_left_ctr += noise_std * rng.standard_normal(n)
            y_left_ctr += noise_std * rng.standard_normal(n)
            x_right_ctr += noise_std * rng.standard_normal(n)
            y_right_ctr += noise_std * rng.standard_normal(n)
            
            x_left_edge += noise_std * rng.standard_normal(n)
            y_left_edge += noise_std * rng.standard_normal(n)
            x_right_edge += noise_std * rng.standard_normal(n)
            y_right_edge += noise_std * rng.standard_normal(n)
        
        rnd_data = [(x_left_rnd[i], y_left_rnd[i], 0) for i in range(n)] + \
                   [(x_right_rnd[i], y_right_rnd[i], 1) for i in range(n)]
        ctr_data = [(x_left_ctr[i], y_left_ctr[i], 0) for i in range(n)] + \
                   [(x_right_ctr[i], y_right_ctr[i], 1) for i in range(n)]
        edge_data = [(x_left_edge[i], y_left_edge[i], 0) for i in range(n)] + \
                    [(x_right_edge[i], y_right_edge[i], 1) for i in range(n)]
        
        if plot:
            _plot_three_sampling_methods(rnd_data, ctr_data, edge_data, r_inner, r_outer, shift)
        
        return rnd_data, ctr_data, edge_data
    
    elif sampling_method == "CTR":
        # Center-weighted sampling only
        x_left, y_left = _sample_c1_centered(n)
        x_right, y_right = _sample_c2_centered(n)
    elif sampling_method == "EDGE":
        # Edge-weighted sampling only
        x_left, y_left = _sample_c1_edge(n)
        x_right, y_right = _sample_c2_edge(n)
    else:  # "RND" or default
        # Uniform random sampling
        x_left, y_left = _sample_c1(n)
        x_right, y_right = _sample_c2(n)

    if noise_std > 0:
        x_left += noise_std * rng.standard_normal(n)
        y_left += noise_std * rng.standard_normal(n)
        x_right += noise_std * rng.standard_normal(n)
        y_right += noise_std * rng.standard_normal(n)

    c1_tuples = [(x_left[i], y_left[i], 0) for i in range(n)]
    c2_tuples = [(x_right[i], y_right[i], 1) for i in range(n)]
    data = c1_tuples + c2_tuples

    if plot:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(x_left, y_left, '.', label='C1', alpha=0.6)
        ax.plot(x_right, y_right, '.', label='C2', alpha=0.6)
        bounds = (-r_outer - 0.5, r_outer + 0.5, -shift - r_outer - 0.5, r_outer + 0.5)
        _plot_interlocked_region_boundaries(
            ax,
            c1_predicate,
            c2_predicate,
            bounds,
            remove_x0_ranges=[(0, r_inner)]
        )
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_title('Interlocked Annulus Regions')
        ax.set_aspect('equal', 'box')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.show()

    return data


# ============================
# Data Generation Configuration (easily modifiable)
# ============================
DATA_PARAMS = {
    "n": 1000,            # Number of points per region
    "r_inner": 1.0,        # Inner radius
    "r_outer": 2.0,        # Outer radius
    "shift": 1.0,          # Downward shift of right half
    "noise_std": 0.0,      # Gaussian noise (0 for clean boundaries)
    "seed": 7,
    "include_inner_caps": False,  # No special fill
    "sampling_method": "BOTH",  # "RND" (uniform random), "CTR" (center-weighted), "EDGE" (boundary-attracted), or "BOTH" (generate all three)
}

# ============================
# MLP Architectures (easily modifiable)
# ============================
MLP_ARCHITECTURES = {
    "MLP_0": [],                        # No hidden layers (logistic regression)
    "MLP_1": [8],                  # 16 node hidden layer
    "MLP_2": [8, 8],                  # 64 node hidden layers
    "MLP_3": [16],                # 3 hidden layers
    "MLP_4": [16, 16 ],                # 4
    "MLP_5": [16, 16, 16],                # 5
}


class MLP(nn.Module):
    """
    Flexible Multi-Layer Perceptron that supports 1-5 hidden layers.
    
    Parameters:
    -----------
    hidden_layers : list
        List of hidden layer sizes (e.g., [128, 128, 64])
        Length can be 0 to 5; [] means no hidden layers
    input_size : int
        Input dimension (default 2 for 2D coordinates)
    output_size : int
        Output dimension (default 1 for binary classification)
    """
    def __init__(self, hidden_layers, input_size=2, output_size=1):
        super(MLP, self).__init__()
        
        # Validate hidden layers
        if not isinstance(hidden_layers, list):
            raise ValueError("hidden_layers must be a list")
        if len(hidden_layers) > 5:
            raise ValueError("Maximum 5 hidden layers supported")
        
        # Build layers
        layers = []
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def prepare_data(spirals_data, test_split=0.2, val_split=0.2, batch_size=32):
    """
    Prepare data for training with 3-way split: train/validation/test.
    
    Parameters:
    -----------
    spirals_data : list of tuples
        Output from generate_intertwined_spirals()
    test_split : float
        Fraction of data for testing (default 0.2, i.e., 20%)
    val_split : float
        Fraction of REMAINING data for validation after test split (default 0.2, i.e., 20% of 80% = 16%)
        Results in: 64% train, 16% validation, 20% test
    batch_size : int
        Batch size for DataLoader
    
    Returns:
    --------
    train_loader, val_loader, test_loader : DataLoader objects
    """
    # Extract x, y, and labels
    x_data = np.array([[x, y] for x, y, _ in spirals_data], dtype=np.float32)
    y_data = np.array([label for _, _, label in spirals_data], dtype=np.float32).reshape(-1, 1)
    
    # Convert to tensors
    x_tensor = torch.from_numpy(x_data)
    y_tensor = torch.from_numpy(y_data)
    
    # Split into train+val and test
    n_samples = len(spirals_data)
    n_test = int(n_samples * test_split)
    n_trainval = n_samples - n_test
    
    indices = np.random.permutation(n_samples)
    trainval_indices = indices[:n_trainval]
    test_indices = indices[n_trainval:]
    
    # Further split trainval into train and validation
    n_val = int(n_trainval * val_split)
    n_train = n_trainval - n_val
    
    train_indices = trainval_indices[:n_train]
    val_indices = trainval_indices[n_train:]
    
    train_dataset = TensorDataset(x_tensor[train_indices], y_tensor[train_indices])
    val_dataset = TensorDataset(x_tensor[val_indices], y_tensor[val_indices])
    test_dataset = TensorDataset(x_tensor[test_indices], y_tensor[test_indices])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def train_model(model, train_loader, val_loader, test_loader, epochs=100, lr=0.001, verbose=True):
    """
    Train an MLP model with validation set for early stopping.
    
    Parameters:
    -----------
    model : nn.Module
        The model to train
    train_loader : DataLoader
        Training data loader (60% of data)
    val_loader : DataLoader
        Validation data loader (20% of data, used for early stopping)
    test_loader : DataLoader
        Testing data loader (20% of data, for final evaluation)
    epochs : int
        Number of training epochs
    lr : float
        Learning rate
    verbose : bool
        Print progress
    
    Returns:
    --------
    model : trained model
    train_losses : list of training losses
    val_losses : list of validation losses
    test_losses : list of test losses
    """
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    test_losses = []
    
    # Early stopping parameters (using validation set)
    patience = 10
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase (for early stopping)
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Early stopping check based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Test phase (final evaluation)
        test_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                test_loss += loss.item()
        
        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}, Test: {test_loss:.4f}")
        
        # Stop if validation loss hasn't improved
        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch + 1} (best epoch: {best_epoch + 1}, Val Loss: {best_val_loss:.4f})")
            break
    
    # Restore best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        # Truncate loss lists to best epoch
        train_losses = train_losses[:best_epoch + 1]
        val_losses = val_losses[:best_epoch + 1]
        test_losses = test_losses[:best_epoch + 1]
    
    return model, train_losses, val_losses, test_losses


def predict(model, x, y):
    """
    Predict class for a single point.
    
    Parameters:
    -----------
    model : nn.Module
        Trained model
    x, y : float
        Coordinates of the point
    
    Returns:
    --------
    prediction : 0 (C1) or 1 (C2)
    confidence : float between 0 and 1
    """
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor([[x, y]], dtype=torch.float32)
        output = model(input_tensor).item()
    
    prediction = 1 if output > 0.5 else 0
    confidence = output if prediction == 1 else (1 - output)
    
    return prediction, confidence


def evaluate_model(model, test_loader):
    """
    Evaluate model on test set and return accuracy.
    
    Parameters:
    -----------
    model : nn.Module
        Trained model
    test_loader : DataLoader
        Testing data loader
    
    Returns:
    --------
    accuracy : float between 0 and 1
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            outputs = model(x_batch)
            predicted = (outputs > 0.5).float()
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    
    return correct / total


def plot_learning_curves(train_histories, title="Learning Curves", fig=None):
    """
    Plot learning curves (loss vs epochs) for one or more models.

    Parameters:
    -----------
    train_histories : dict
        {model_name: {"train_losses": [...], "test_losses": [...]}}
    title : str
        Plot title
    """
    if not train_histories:
        return

    if fig is None or not plt.fignum_exists(fig.number):
        fig = plt.figure(figsize=(9, 6))
    else:
        fig.clf()
        plt.figure(fig.number)
    for model_name, history in train_histories.items():
        train_losses = history.get("train_losses", [])
        if train_losses:
            epochs = range(1, len(train_losses) + 1)
            # Get architecture from MLP_ARCHITECTURES dictionary
            arch = MLP_ARCHITECTURES.get(model_name, [])
            plt.plot(epochs, train_losses, label=f"{arch}")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)
    return fig


if __name__ == "__main__":
    print("="*60)
    print("INTERTWINED SPIRALS CLASSIFICATION WITH MLPs")
    print("="*60)
    
    # Generate spiral data
    print("\nGenerating interlocked region data...")
    sampling_method = DATA_PARAMS.get("sampling_method", "RND")
    
    # Storage for all datasets
    dataset_storage = {}
    active_dataset = "RND"  # Default active dataset
    
    if sampling_method == "BOTH":
        # Generate all three datasets
        result = generate_intertwined_spirals(
            n=DATA_PARAMS["n"],
            r_inner=DATA_PARAMS["r_inner"],
            r_outer=DATA_PARAMS["r_outer"],
            shift=DATA_PARAMS["shift"],
            noise_std=DATA_PARAMS["noise_std"],
            seed=DATA_PARAMS["seed"],
            sampling_method="BOTH",
            plot=False
        )
        dataset_storage["RND"] = result[0]
        dataset_storage["CTR"] = result[1]
        dataset_storage["EDGE"] = result[2]
        spirals = dataset_storage[active_dataset]  # Default to RND
        print(f"Generated RND, CTR, and EDGE datasets")
        print(f"Active dataset for training: {active_dataset}")
    else:
        spirals = generate_intertwined_spirals(**DATA_PARAMS)
        dataset_storage[sampling_method] = spirals
        active_dataset = sampling_method
    
    total_points = len(spirals)
    points_per_spiral = DATA_PARAMS["n"]
    print(f"Total points: {total_points} ({points_per_spiral} per spiral)")
    print(f"Data parameters: {DATA_PARAMS}")
    print(f"Sampling method: {sampling_method}")
    
    # Extract data for plotting
    x_data = np.array([[x, y] for x, y, _ in spirals])
    c1_mask = np.array([label == 0 for _, _, label in spirals])
    c2_mask = np.array([label == 1 for _, _, label in spirals])
    
    # Prepare data
    print("Preparing data (60% train, 20% validation, 20% test)...")
    train_loader, val_loader, test_loader = prepare_data(spirals, test_split=0.2, val_split=0.2, batch_size=32)
    
    # Create models dictionary - organized by dataset
    trained_models = {}  # {dataset_name: {model_name: model}}
    train_histories = {}  # {dataset_name: {model_name: history}}
    test_loaders = {}  # {dataset_name: test_loader}
    val_loaders = {}  # {dataset_name: val_loader}
    train_loaders = {}  # {dataset_name: train_loader}
    
    # Initialize loaders for all available datasets
    for ds_name in dataset_storage.keys():
        train_loaders[ds_name], val_loaders[ds_name], test_loaders[ds_name] = prepare_data(
            dataset_storage[ds_name], test_split=0.2, val_split=0.2, batch_size=32
        )
        trained_models[ds_name] = {}
        train_histories[ds_name] = {}
    
    plot_fig = None
    learning_curve_fig = None
    plt.ion()
    
    while True:
        print("\n" + "="*60)
        print("MAIN MENU")
        print("="*60)
        print("1. Train MLPs (single dataset)")
        print("2. Batch Test (evaluate on test set)")
        print("3. Single Point Test (classify one point)")
        print("4. Display Data Plot")
        if len(dataset_storage) > 1:
            print(f"5. Switch Dataset (current: {active_dataset})")
            print("6. Train on ALL Datasets")
            print("7. Cross-Dataset Comparison")
        print("0. Exit")
        print("="*60)
        
        max_option = "7" if len(dataset_storage) > 1 else "4"
        choice = input(f"Select option (0-{max_option}): ").strip()
        
        if choice == "5" and len(dataset_storage) > 1:
            print("\n" + "="*60)
            print("SELECT DATASET")
            print("="*60)
            datasets = list(dataset_storage.keys())
            for i, ds_name in enumerate(datasets, 1):
                marker = " (current)" if ds_name == active_dataset else ""
                print(f"  {i}. {ds_name}{marker}")
            print("  0. Cancel")
            
            ds_choice = input(f"Select dataset (0-{len(datasets)}): ").strip()
            if ds_choice.isdigit() and 1 <= int(ds_choice) <= len(datasets):
                new_dataset = datasets[int(ds_choice) - 1]
                active_dataset = new_dataset
                spirals = dataset_storage[active_dataset]
                
                # Update references to active loaders
                print(f"\nSwitching to {active_dataset} dataset...")
                train_loader = train_loaders[active_dataset]
                val_loader = val_loaders[active_dataset]
                test_loader = test_loaders[active_dataset]
                
                # Update plot data
                x_data = np.array([[x, y] for x, y, _ in spirals])
                c1_mask = np.array([label == 0 for _, _, label in spirals])
                c2_mask = np.array([label == 1 for _, _, label in spirals])
                
                print(f"✓ Now using {active_dataset} dataset for training")
                print(f"  Total points: {len(spirals)}")
            elif ds_choice != "0":
                print("Invalid selection.")
        
        elif choice == "1":
            print("\n" + "="*60)
            print("TRAINING MENU")
            print(f"Active dataset: {active_dataset}")
            print("="*60)
            print("Available architectures:")
            for i, (name, arch) in enumerate(MLP_ARCHITECTURES.items(), 1):
                print(f"  {i}. {name}: {arch}")
            
            train_choice = input("\nTrain which architecture(s)? (default 'all', or '1', '1,2,3'): ").strip().lower()
            
            if train_choice == "all" or train_choice == "":
                architectures_to_train = list(MLP_ARCHITECTURES.items())
            else:
                try:
                    indices = [int(x.strip()) - 1 for x in train_choice.split(",")]
                    architectures_to_train = [
                        (name, arch) for i, (name, arch) in enumerate(MLP_ARCHITECTURES.items())
                        if i in indices
                    ]
                except (ValueError, IndexError):
                    print("Invalid input. Using all architectures.")
                    architectures_to_train = list(MLP_ARCHITECTURES.items())
            
            epochs_str = input("Number of epochs (default 100): ").strip()
            epochs = int(epochs_str) if epochs_str else 100
            
            lr_str = input("Learning rate (default 0.001): ").strip()
            lr = float(lr_str) if lr_str else 0.001
            
            train_rows = []
            for arch_name, hidden_layers in architectures_to_train:
                print(f"\n{'='*60}")
                print(f"Training {hidden_layers} on {active_dataset} dataset")
                print(f"{'='*60}")
                
                model = MLP(hidden_layers)
                trained_model, train_losses, val_losses, test_losses = train_model(
                    model, train_loader, val_loader, test_loader, epochs=epochs, lr=lr, verbose=True
                )
                
                trained_models[active_dataset][arch_name] = trained_model
                train_histories[active_dataset][arch_name] = {
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "test_losses": test_losses
                }
                
                final_acc = evaluate_model(trained_model, test_loader)
                print(f"Final Test Accuracy: {final_acc*100:.2f}%")
                train_rows.append((arch_name, hidden_layers, final_acc))
            
            if train_rows:
                print("\n" + "="*60)
                print("TRAINING SUMMARY (Structure & Accuracy)")
                print("="*60)
                print(f"{'Architecture':24s} | {'Accuracy':>9s}")
                print("-"*40)
                for name, structure, acc in train_rows:
                    print(f"{str(structure):24s} | {acc*100:8.2f}%")

                # Plot learning curves for trained models (separate window)
                if train_histories[active_dataset]:
                    learning_curve_fig = plot_learning_curves(
                        train_histories[active_dataset],
                        title=f"Training Learning Curves ({active_dataset} Dataset)",
                        fig=learning_curve_fig
                    )
        
        elif choice == "2":
            if not any(trained_models.values()):
                print("\nNo trained models yet. Please train models first (option 1 or 6).")
                continue
            
            print("\n" + "="*60)
            print("BATCH TEST MENU")
            print(f"Testing on {active_dataset} dataset")
            print("="*60)
            print("Trained models:")
            all_models = list(trained_models[active_dataset].keys())
            for i, model_name in enumerate(all_models, 1):
                print(f"  {i}. {model_name}")
            
            test_choice = input("Test which model(s)? (default 'all', or '1', '1,2'): ").strip().lower()
            
            if test_choice == "all" or test_choice == "":
                models_to_test = all_models
            else:
                try:
                    indices = [int(x.strip()) - 1 for x in test_choice.split(",")]
                    models_to_test = [all_models[i] for i in indices if 0 <= i < len(all_models)]
                except (ValueError, IndexError):
                    print("Invalid input. Testing all models.")
                    models_to_test = all_models
            
            print("\n" + "-"*60)
            print(f"Test Results on {active_dataset} dataset:")
            print("-"*60)
            test_rows = []
            for model_name in models_to_test:
                model = trained_models[active_dataset][model_name]
                accuracy = evaluate_model(model, test_loaders[active_dataset])
                print(f"{model_name:20s} - Test Accuracy: {accuracy*100:.2f}%")
                structure = MLP_ARCHITECTURES.get(model_name, "N/A")
                test_rows.append((model_name, structure, accuracy))

            if test_rows:
                print("\n" + "-"*60)
                print("Summary (Structure & Accuracy)")
                print("-"*60)
                print(f"{'Architecture':24s} | {'Accuracy':>9s}")
                print("-"*40)
                for name, structure, acc in test_rows:
                    print(f"{str(structure):24s} | {acc*100:8.2f}%")
                
                # Save to test_results.txt
                with open('test_results.txt', 'a') as f:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"\n{'='*60}\n")
                    f.write(f"Test Results - {timestamp}\n")
                    f.write(f"Dataset: {active_dataset}\n")
                    f.write(f"{'='*60}\n")
                    f.write(f"{'Model':12s} | {'Structure':24s} | {'Accuracy':>9s}\n")
                    f.write(f"{'-'*60}\n")
                    for name, structure, acc in test_rows:
                        f.write(f"{name:12s} | {str(structure):24s} | {acc*100:8.2f}%\n")
                
                print(f"\nResults saved to test_results.txt")
        
        elif choice == "3":
            if not any(trained_models.values()):
                print("\nNo trained models yet. Please train models first (option 1 or 6).")
                continue
            
            print("\n" + "="*60)
            print("SINGLE POINT TEST MENU")
            print(f"Using models from {active_dataset} dataset")
            print("="*60)
            print("Trained models:")
            all_models = list(trained_models[active_dataset].keys())
            for i, model_name in enumerate(all_models, 1):
                print(f"  {i}. {model_name}")
            
            test_choice = input("Test which model(s)? (default 'all', or '1', '1,2'): ").strip().lower()
            
            if test_choice == "all" or test_choice == "":
                models_to_test = all_models
            else:
                try:
                    indices = [int(x.strip()) - 1 for x in test_choice.split(",")]
                    models_to_test = [all_models[i] for i in indices if 0 <= i < len(all_models)]
                except (ValueError, IndexError):
                    print("Invalid input. Testing all models.")
                    models_to_test = all_models
            
            try:
                x_str = input("Enter x coordinate: ").strip()
                y_str = input("Enter y coordinate: ").strip()
                x = float(x_str)
                y = float(y_str)
                
                print("\n" + "-"*60)
                print(f"Predictions for point ({x:.4f}, {y:.4f}):")
                print("-"*60)
                for model_name in models_to_test:
                    model = trained_models[active_dataset][model_name]
                    pred, conf = predict(model, x, y)
                    class_name = "C1" if pred == 0 else "C2"
                    print(f"{model_name:20s} -> {class_name} (confidence: {conf:.4f})")
            
            except ValueError:
                print("Invalid input. Please enter valid numbers.")
        
        elif choice == "4":
            print("\n" + "="*60)
            print("DISPLAY DATA PLOT")
            print(f"Active dataset: {active_dataset}")
            print("="*60)
            
            # Check if we have multiple datasets available
            if len(dataset_storage) > 1:
                # Regenerate with plot=True to show all three side-by-side
                _ = generate_intertwined_spirals(**{**DATA_PARAMS, "plot": True})
            else:
                # Single plot for current dataset
                if plot_fig is None or not plt.fignum_exists(plot_fig.number):
                    plot_fig = plt.figure(figsize=(8, 8))
                else:
                    plot_fig.clf()
                    plt.figure(plot_fig.number)
                plt.plot(x_data[c1_mask, 0], x_data[c1_mask, 1], '.', label='C1', alpha=0.6)
                plt.plot(x_data[c2_mask, 0], x_data[c2_mask, 1], '.', label='C2', alpha=0.6)
                # Define region predicates using helper functions
                def _c1_pred(x, y):
                    return _c1_predicate(x, y, DATA_PARAMS["r_inner"], DATA_PARAMS["r_outer"], DATA_PARAMS["shift"])

                def _c2_pred(x, y):
                    return _c2_predicate(x, y, DATA_PARAMS["r_inner"], DATA_PARAMS["r_outer"], DATA_PARAMS["shift"])

                bounds = (
                    -DATA_PARAMS["r_outer"] - 0.5,
                    DATA_PARAMS["r_outer"] + 0.5,
                    -DATA_PARAMS["shift"] - DATA_PARAMS["r_outer"] - 0.5,
                    DATA_PARAMS["r_outer"] + 0.5,
                )
                _plot_interlocked_region_boundaries(
                    plt.gca(),
                    _c1_pred,
                    _c2_pred,
                    bounds,
                    remove_x0_ranges=[(0, DATA_PARAMS["r_inner"])],
                )
                plt.xlabel('$x_1$')
                plt.ylabel('$x_2$')
                plt.title(f'Interlocked Annulus Regions Dataset ({active_dataset})')
                plt.axis('equal')
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.show(block=False)
                plt.pause(0.001)
        
        elif choice == "6" and len(dataset_storage) > 1:
            print("\n" + "="*60)
            print("TRAIN ON ALL DATASETS")
            print("="*60)
            print("Available architectures:")
            for i, (name, arch) in enumerate(MLP_ARCHITECTURES.items(), 1):
                print(f"  {i}. {name}: {arch}")
            
            train_choice = input("\nTrain which architecture(s)? (default 'all', or '1', '1,2,3'): ").strip().lower()
            
            if train_choice == "all" or train_choice == "":
                architectures_to_train = list(MLP_ARCHITECTURES.items())
            else:
                try:
                    indices = [int(x.strip()) - 1 for x in train_choice.split(",")]
                    architectures_to_train = [
                        (name, arch) for i, (name, arch) in enumerate(MLP_ARCHITECTURES.items())
                        if i in indices
                    ]
                except (ValueError, IndexError):
                    print("Invalid input. Using all architectures.")
                    architectures_to_train = list(MLP_ARCHITECTURES.items())
            
            epochs_str = input("Number of epochs (default 100): ").strip()
            epochs = int(epochs_str) if epochs_str else 100
            
            lr_str = input("Learning rate (default 0.001): ").strip()
            lr = float(lr_str) if lr_str else 0.001
            
            # Train on all datasets
            all_results = {}  # {dataset: {model_name: accuracy}}
            for ds_name in dataset_storage.keys():
                print(f"\n{'='*60}")
                print(f"Training on {ds_name} dataset")
                print(f"{'='*60}")
                all_results[ds_name] = {}
                
                for arch_name, hidden_layers in architectures_to_train:
                    print(f"\n  Training {hidden_layers}...")
                    model = MLP(hidden_layers)
                    trained_model, train_losses, val_losses, test_losses = train_model(
                        model, train_loaders[ds_name], val_loaders[ds_name], test_loaders[ds_name], 
                        epochs=epochs, lr=lr, verbose=False
                    )
                    
                    trained_models[ds_name][arch_name] = trained_model
                    train_histories[ds_name][arch_name] = {
                        "train_losses": train_losses,
                        "val_losses": val_losses,
                        "test_losses": test_losses
                    }
                    
                    final_acc = evaluate_model(trained_model, test_loaders[ds_name])
                    all_results[ds_name][arch_name] = final_acc
                    print(f"    Test Accuracy: {final_acc*100:.2f}%")
            
            # Display summary comparison
            print("\n" + "="*60)
            print("TRAINING SUMMARY - ALL DATASETS")
            print("="*60)
            print(f"{'Architecture':24s} | ", end="")
            for ds_name in dataset_storage.keys():
                print(f"{ds_name:^10s} | ", end="")
            print()
            print("-"*70)
            
            for arch_name, hidden_layers in architectures_to_train:
                print(f"{str(hidden_layers):24s} | ", end="")
                for ds_name in dataset_storage.keys():
                    acc = all_results[ds_name][arch_name]
                    print(f"{acc*100:9.2f}% | ", end="")
                print()
            
            # Save to file
            with open('test_results.txt', 'a') as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"\n{'='*60}\n")
                f.write(f"Multi-Dataset Training Results - {timestamp}\n")
                f.write(f"{'='*60}\n")
                f.write(f"{'Architecture':24s} | ")
                for ds_name in dataset_storage.keys():
                    f.write(f"{ds_name:^10s} | ")
                f.write("\n" + "-"*70 + "\n")
                for arch_name, hidden_layers in architectures_to_train:
                    f.write(f"{str(hidden_layers):24s} | ")
                    for ds_name in dataset_storage.keys():
                        acc = all_results[ds_name][arch_name]
                        f.write(f"{acc*100:9.2f}% | ")
                    f.write("\n")
            
            print(f"\nResults saved to test_results.txt")
        
        elif choice == "7" and len(dataset_storage) > 1:
            if not any(any(models.values()) for models in trained_models.values()):
                print("\nNo trained models yet. Please train models first (option 1 or 6).")
                continue
            
            print("\n" + "="*60)
            print("CROSS-DATASET COMPARISON")
            print("="*60)
            print("This tests models trained on one dataset against all datasets")
            print()
            
            # Find which datasets have trained models
            available_sources = [ds for ds in dataset_storage.keys() if trained_models[ds]]
            if not available_sources:
                print("No trained models available.")
                continue
            
            print("Select source dataset (models trained on):")
            for i, ds_name in enumerate(available_sources, 1):
                num_models = len(trained_models[ds_name])
                print(f"  {i}. {ds_name} ({num_models} models)")
            
            source_choice = input(f"Choice (1-{len(available_sources)}): ").strip()
            if not source_choice.isdigit() or not (1 <= int(source_choice) <= len(available_sources)):
                print("Invalid choice.")
                continue
            
            source_ds = available_sources[int(source_choice) - 1]
            available_models = list(trained_models[source_ds].keys())
            
            if not available_models:
                print(f"No models trained on {source_ds} dataset.")
                continue
            
            print(f"\nTesting models trained on {source_ds} across all datasets...")
            print("\n" + "="*60)
            print("CROSS-DATASET PERFORMANCE")
            print("="*60)
            print(f"{'Architecture':24s} | ", end="")
            for test_ds in dataset_storage.keys():
                print(f"{test_ds:^10s} | ", end="")
            print()
            print("-"*70)
            
            cross_results = {}  # {model_name: {test_dataset: accuracy}}
            for model_name in available_models:
                # Get the hidden layers for this model from MLP_ARCHITECTURES
                hidden_layers = MLP_ARCHITECTURES.get(model_name, "N/A")
                cross_results[model_name] = {}
                print(f"{str(hidden_layers):24s} | ", end="")
                model = trained_models[source_ds][model_name]
                
                for test_ds in dataset_storage.keys():
                    acc = evaluate_model(model, test_loaders[test_ds])
                    cross_results[model_name][test_ds] = acc
                    print(f"{acc*100:9.2f}% | ", end="")
                print()
            
            # Save to file
            with open('test_results.txt', 'a') as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"\n{'='*60}\n")
                f.write(f"Cross-Dataset Comparison - {timestamp}\n")
                f.write(f"Models trained on: {source_ds}\n")
                f.write(f"{'='*60}\n")
                f.write(f"{'Architecture':24s} | ")
                for test_ds in dataset_storage.keys():
                    f.write(f"{test_ds:^10s} | ")
                f.write("\n" + "-"*70 + "\n")
                for model_name in available_models:
                    # Get the hidden layers for this model
                    hidden_layers = MLP_ARCHITECTURES.get(model_name, "N/A")
                    f.write(f"{str(hidden_layers):24s} | ")
                    for test_ds in dataset_storage.keys():
                        acc = cross_results[model_name][test_ds]
                        f.write(f"{acc*100:9.2f}% | ")
                    f.write("\n")
            
            print(f"\nResults saved to test_results.txt")
        
        elif choice == "0":
            print("\nExiting. Goodbye!")
            break
        
        else:
            max_valid = 7 if len(dataset_storage) > 1 else 4
            print(f"Invalid option. Please select 0-{max_valid}.")
