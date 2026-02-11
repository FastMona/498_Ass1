import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path

from regions import classify_points, get_region_paths, build_c1_boundary


# ============================
# Region Definitions
# ============================

_REGION_PATHS = get_region_paths()
_C1_PATH = _REGION_PATHS["C1"]
_C2_PATH = _REGION_PATHS["C2"]


def _mask_from_path(path: Path, x, y, include_boundary: bool = True):
    points = np.column_stack([np.ravel(x), np.ravel(y)])
    radius = 1e-9 if include_boundary else 0.0
    mask = path.contains_points(points, radius=radius)
    return mask.reshape(np.shape(x))


def _c1_predicate(x, y):
    """C1: Region from regions.py."""
    return _mask_from_path(_C1_PATH, x, y, include_boundary=True)


def _c2_predicate(x, y):
    """C2: Region from regions.py."""
    c2_mask = _mask_from_path(_C2_PATH, x, y, include_boundary=True)
    c1_mask = _mask_from_path(_C1_PATH, x, y, include_boundary=True)
    return c2_mask & ~c1_mask


# ============================
# Data Generation Helpers
# ============================

def _apply_normalization(x, y, norm_stats):
    if not norm_stats:
        return x, y
    mean = np.asarray(norm_stats["mean"], dtype=float)
    std = np.asarray(norm_stats["std"], dtype=float)
    return (x - mean[0]) / std[0], (y - mean[1]) / std[1]


def _plot_interlocked_region_boundaries(
    ax,
    bounds,
    grid_res=400,
    remove_x0_ranges=None,
    norm_stats=None,
):
    c1_x, c1_y = build_c1_boundary()
    c2_x, c2_y = -c1_x, -c1_y - 1.0

    c1_x, c1_y = _apply_normalization(c1_x, c1_y, norm_stats)
    c2_x, c2_y = _apply_normalization(c2_x, c2_y, norm_stats)

    ax.fill(c1_x, c1_y, color="#CCE5FF", alpha=0.7, zorder=0)
    ax.fill(c2_x, c2_y, color="#FFE5CC", alpha=0.7, zorder=0)
    ax.plot(c1_x, c1_y, color="tab:blue", lw=2.5, zorder=1)


def _plot_three_sampling_methods(rnd_data, ctr_data, edge_data, norm_stats_by_dataset=None):
    """Plot side-by-side comparison of RND, CTR, and EDGE sampling methods."""
    _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))

    rnd_count = len(rnd_data)
    ctr_count = len(ctr_data)
    edge_count = len(edge_data)

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

    bounds = (-2.5, 2.5, -3.5, 2.5)

    # Plot RND (left subplot)
    ax1.plot(x_c1_rnd, y_c1_rnd, '.', label='C1', alpha=0.6, markersize=3)
    ax1.plot(x_c2_rnd, y_c2_rnd, '.', label='C2', alpha=0.6, markersize=3)
    _plot_interlocked_region_boundaries(
        ax1,
        bounds,
        remove_x0_ranges=[(-1, 0), (1, 2)],
        norm_stats=(norm_stats_by_dataset or {}).get("RND"),
    )
    ax1.set_xlabel('$x_1$')
    ax1.set_ylabel('$x_2$')
    ax1.set_title(f'RND (Uniform Random Sampling) - {rnd_count} points')
    ax1.set_aspect('equal', 'box')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot CTR (middle subplot)
    ax2.plot(x_c1_ctr, y_c1_ctr, '.', label='C1', alpha=0.6, markersize=3)
    ax2.plot(x_c2_ctr, y_c2_ctr, '.', label='C2', alpha=0.6, markersize=3)
    _plot_interlocked_region_boundaries(
        ax2,
        bounds,
        remove_x0_ranges=[(-1, 0), (1, 2)],
        norm_stats=(norm_stats_by_dataset or {}).get("CTR"),
    )
    ax2.set_xlabel('$x_1$')
    ax2.set_ylabel('$x_2$')
    ax2.set_title(f'CTR (Center-Weighted Sampling at (0, -0.5)) - {ctr_count} points')
    ax2.set_aspect('equal', 'box')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot EDGE (right subplot)
    ax3.plot(x_c1_edge, y_c1_edge, '.', label='C1', alpha=0.6, markersize=3)
    ax3.plot(x_c2_edge, y_c2_edge, '.', label='C2', alpha=0.6, markersize=3)
    _plot_interlocked_region_boundaries(
        ax3,
        bounds,
        remove_x0_ranges=[(-1, 0), (1, 2)],
        norm_stats=(norm_stats_by_dataset or {}).get("EDGE"),
    )
    ax3.set_xlabel('$x_1$')
    ax3.set_ylabel('$x_2$')
    ax3.set_title(f'EDGE (Boundary-Attracted Sampling) - {edge_count} points')
    ax3.set_aspect('equal', 'box')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    plt.tight_layout()
    plt.show()


def plot_three_datasets_with_fp_fn(dataset_storage, fp_fn_by_dataset, norm_stats_by_dataset=None):
    """Plot datasets with FP/FN points overlaid in a new window."""
    _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    panels = [
        ("RND", dataset_storage.get("RND"), ax1),
        ("CTR", dataset_storage.get("CTR"), ax2),
        ("EDGE", dataset_storage.get("EDGE"), ax3),
    ]

    bounds = (-2.5, 2.5, -3.5, 2.5)

    for ds_name, data, ax in panels:
        if not data:
            ax.set_axis_off()
            continue

        x_c1 = [p[0] for p in data if p[2] == 0]
        y_c1 = [p[1] for p in data if p[2] == 0]
        x_c2 = [p[0] for p in data if p[2] == 1]
        y_c2 = [p[1] for p in data if p[2] == 1]

        ax.plot(x_c1, y_c1, '.', label='C1', alpha=0.6, markersize=3)
        ax.plot(x_c2, y_c2, '.', label='C2', alpha=0.6, markersize=3)
        _plot_interlocked_region_boundaries(
            ax,
            bounds,
            remove_x0_ranges=[(-1, 0), (1, 2)],
            norm_stats=(norm_stats_by_dataset or {}).get(ds_name),
        )

        fp_fn = fp_fn_by_dataset.get(ds_name, {})
        fp_points = fp_fn.get("fp", [])
        fn_points = fp_fn.get("fn", [])
        if fp_points:
            fp_x, fp_y = zip(*fp_points)
            ax.scatter(fp_x, fp_y, s=30, marker='o', c='#2ECC71', label='FP')
        if fn_points:
            fn_x, fn_y = zip(*fn_points)
            ax.scatter(fn_x, fn_y, s=30, marker='o', c='#E74C3C', label='FN')

        model_name = fp_fn.get("model_name")
        acc = fp_fn.get("acc")
        suffix = ""
        if model_name is not None and acc is not None:
            suffix = f" | {model_name} ({acc * 100:.2f}%)"
        ax.set_title(f"{ds_name} with FP/FN{suffix}")
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_aspect('equal', 'box')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)
    return True


# ============================
# Data Generation Function
# ============================

def generate_intertwined_spirals(
    n=600,
    noise_std=0.0,
    seed=None,
    plot=False,
    sampling_method="RND"
):
    """
    Generate datasets for regions defined in regions.py.

    Parameters:
    -----------
    n : int
        Number of points per region (default 600)
    """
    rng = np.random.default_rng(seed)

    def _region_bounds(path: Path) -> tuple[float, float, float, float]:
        vertices = np.asarray(path.vertices)
        min_x, min_y = vertices.min(axis=0)
        max_x, max_y = vertices.max(axis=0)
        return min_x, max_x, min_y, max_y

    def _sample_region(region: str, sampler, n_samples: int, max_batches: int = 200) -> np.ndarray:
        points: list[np.ndarray] = []
        for _ in range(max_batches):
            batch = max(256, n_samples)
            candidates = sampler(batch)
            mask = classify_points(candidates)[region]
            if np.any(mask):
                points.append(candidates[mask])
            if sum(len(p) for p in points) >= n_samples:
                break
        if not points or sum(len(p) for p in points) < n_samples:
            raise RuntimeError(f"Unable to sample {n_samples} points for {region}.")
        return np.vstack(points)[:n_samples]

    def _uniform_sampler(region: str):
        path = _REGION_PATHS[region]
        min_x, max_x, min_y, max_y = _region_bounds(path)

        def sampler(batch: int) -> np.ndarray:
            x = rng.uniform(min_x, max_x, batch)
            y = rng.uniform(min_y, max_y, batch)
            return np.column_stack([x, y])

        return sampler

    def _center_sampler(region: str, sigma: float):
        path = _REGION_PATHS[region]
        center = np.asarray(path.vertices).mean(axis=0)

        def sampler(batch: int) -> np.ndarray:
            x = rng.normal(center[0], sigma, batch)
            y = rng.normal(center[1], sigma, batch)
            return np.column_stack([x, y])

        return sampler

    def _edge_sampler(region: str, sigma: float):
        path = _REGION_PATHS[region]
        vertices = np.asarray(path.vertices)

        def sampler(batch: int) -> np.ndarray:
            boundary_batch = max(1, int(0.15 * batch))
            vertex_batch = batch - boundary_batch

            if vertex_batch > 0:
                idx = rng.integers(0, len(vertices), size=vertex_batch)
                base = vertices[idx]
                noise = rng.normal(scale=sigma, size=base.shape)
                vertex_points = base + noise
            else:
                vertex_points = np.empty((0, 2))

            shared_count = max(1, boundary_batch // 2)
            region_count = boundary_batch - shared_count

            y_shared = rng.uniform(-1.0, 0.0, shared_count)
            x_shared = rng.normal(0.0, sigma, shared_count)
            shared_points = np.column_stack([x_shared, y_shared])

            if region == "C1":
                y_region = rng.uniform(1.0, 2.0, region_count)
            elif region == "C2":
                y_region = rng.uniform(-3.0, -2.0, region_count)
            else:
                y_region = rng.uniform(-1.0, 0.0, region_count)
            x_region = rng.normal(0.0, sigma, region_count)
            region_points = np.column_stack([x_region, y_region])

            boundary_points = np.vstack([shared_points, region_points])

            return np.vstack([vertex_points, boundary_points])

        return sampler

    def _sample_pair(method: str) -> tuple[np.ndarray, np.ndarray]:
        if method == "CTR":
            c1 = _sample_region("C1", _center_sampler("C1", sigma=0.8), n)
            c2 = _sample_region("C2", _center_sampler("C2", sigma=0.8), n)
        elif method == "EDGE":
            c1 = _sample_region("C1", _edge_sampler("C1", sigma=0.08), n)
            c2 = _sample_region("C2", _edge_sampler("C2", sigma=0.08), n)
        else:  # "RND" or default
            c1 = _sample_region("C1", _uniform_sampler("C1"), n)
            c2 = _sample_region("C2", _uniform_sampler("C2"), n)
        return c1, c2

    # Choose sampling method
    if sampling_method == "ALL":
        # Generate all three datasets
        c1_rnd, c2_rnd = _sample_pair("RND")
        c1_ctr, c2_ctr = _sample_pair("CTR")
        c1_edge, c2_edge = _sample_pair("EDGE")

        if noise_std > 0:
            c1_rnd += noise_std * rng.standard_normal(c1_rnd.shape)
            c2_rnd += noise_std * rng.standard_normal(c2_rnd.shape)

            c1_ctr += noise_std * rng.standard_normal(c1_ctr.shape)
            c2_ctr += noise_std * rng.standard_normal(c2_ctr.shape)

            c1_edge += noise_std * rng.standard_normal(c1_edge.shape)
            c2_edge += noise_std * rng.standard_normal(c2_edge.shape)

        rnd_data = [(c1_rnd[i, 0], c1_rnd[i, 1], 0) for i in range(n)] + \
                   [(c2_rnd[i, 0], c2_rnd[i, 1], 1) for i in range(n)]
        ctr_data = [(c1_ctr[i, 0], c1_ctr[i, 1], 0) for i in range(n)] + \
                   [(c2_ctr[i, 0], c2_ctr[i, 1], 1) for i in range(n)]
        edge_data = [(c1_edge[i, 0], c1_edge[i, 1], 0) for i in range(n)] + \
                    [(c2_edge[i, 0], c2_edge[i, 1], 1) for i in range(n)]

        if plot:
            _plot_three_sampling_methods(rnd_data, ctr_data, edge_data)

        return rnd_data, ctr_data, edge_data

    c1, c2 = _sample_pair(sampling_method)

    if noise_std > 0:
        c1 += noise_std * rng.standard_normal(c1.shape)
        c2 += noise_std * rng.standard_normal(c2.shape)

    c1_tuples = [(c1[i, 0], c1[i, 1], 0) for i in range(n)]
    c2_tuples = [(c2[i, 0], c2[i, 1], 1) for i in range(n)]
    data = c1_tuples + c2_tuples

    if plot:
        _, ax = plt.subplots(figsize=(8, 8))
        ax.plot(c1[:, 0], c1[:, 1], '.', label='C1', alpha=0.6)
        ax.plot(c2[:, 0], c2[:, 1], '.', label='C2', alpha=0.6)
        bounds = (-2.5, 2.5, -3.5, 2.5)
        _plot_interlocked_region_boundaries(
            ax,
            bounds,
            remove_x0_ranges=[(-1, 0), (1, 2)]
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
    "noise_std": 0.0,      # Gaussian noise (0 for clean boundaries)
    "seed": 7,
    "sampling_method": "ALL",  # "RND" (uniform random), "CTR" (center-weighted), "EDGE" (boundary-attracted), or "ALL" (generate all three)
    "normalize": False,   # Normalize to zero mean/unit variance
}


def normalize_spirals(spirals):
    """Normalize spiral coordinates to zero mean and unit variance."""
    if not spirals:
        return spirals, {"mean": np.zeros(2), "std": np.ones(2)}

    x_data = np.array([[x, y] for x, y, _ in spirals], dtype=float)
    mean = x_data.mean(axis=0)
    std = x_data.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)

    normalized = [
        ((x - mean[0]) / std[0], (y - mean[1]) / std[1], label)
        for x, y, label in spirals
    ]

    return normalized, {"mean": mean, "std": std}


def plot_dataset(spirals, data_params, active_dataset, plot_fig=None, norm_stats=None):
    """Plot a single dataset with boundaries and return the figure handle."""
    if plot_fig is None or not plt.fignum_exists(plot_fig.number):
        plot_fig = plt.figure(figsize=(8, 8))
    else:
        plot_fig.clf()
        plt.figure(plot_fig.number)

    x_data = np.array([[x, y] for x, y, _ in spirals])
    c1_mask = np.array([label == 0 for _, _, label in spirals])
    c2_mask = np.array([label == 1 for _, _, label in spirals])

    plt.plot(x_data[c1_mask, 0], x_data[c1_mask, 1], '.', label='C1', alpha=0.6)
    plt.plot(x_data[c2_mask, 0], x_data[c2_mask, 1], '.', label='C2', alpha=0.6)

    bounds = (-2.5, 2.5, -3.5, 2.5)

    _plot_interlocked_region_boundaries(
        plt.gca(),
        bounds,
        remove_x0_ranges=[(-1, 0), (1, 2)],
        norm_stats=norm_stats,
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
    return plot_fig


def plot_three_datasets(dataset_storage, data_params, norm_stats_by_dataset=None):
    """Plot RND, CTR, and EDGE datasets side-by-side in one window."""
    rnd_data = dataset_storage.get("RND")
    ctr_data = dataset_storage.get("CTR")
    edge_data = dataset_storage.get("EDGE")
    if not (rnd_data and ctr_data and edge_data):
        return False
    _plot_three_sampling_methods(
        rnd_data,
        ctr_data,
        edge_data,
        norm_stats_by_dataset=norm_stats_by_dataset,
    )
    return True


__all__ = [
    "DATA_PARAMS",
    "generate_intertwined_spirals",
    "normalize_spirals",
    "plot_dataset",
    "plot_three_datasets",
    "plot_three_datasets_with_fp_fn",
    "_c1_predicate",
    "_c2_predicate",
]
