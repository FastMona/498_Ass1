"""Sketch the described regions.
regions are C1 and C2 with no common points.
C1 has priority: any shared boundary belongs to C1, and C2 excludes points in C1.
C1 is defined piecewise by:
- less than the outer semicircle: x^2 + y^2 <= 4
- greater than the inner semicircle: x^2 + y^2 >= 1
- less than the straight line segment: (0,1):(0,2)
- greater than the straight line segment: (0,-1):(0,0)
- less than the small semicircle: x^2 + (y+1)^2 <= 1

C2 is defined piecewise by:
- less than the outer semicircle: x^2 + (y+1)^2 <= 4
- greater than the inner semicircle: x^2 + (y+1)^2 >= 1
- less than the straight line segment: (0,-1):(0,0)
- greater than the straight line segment: (0,-2):(0,-3)
- less than the small semicircle: x^2 + y^2 <= 1
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path


def build_c1_boundary(num_outer: int = 400, num_inner: int = 400, num_circle: int = 300) -> tuple[np.ndarray, np.ndarray]:
	"""Return a closed boundary for C1 as (x, y) arrays."""
	outer_theta = np.linspace(np.pi / 2, 3 * np.pi / 2, num_outer)
	inner_theta = np.linspace(3 * np.pi / 2, np.pi / 2, num_inner)
	outer_x = 2.0 * np.cos(outer_theta)
	outer_y = 2.0 * np.sin(outer_theta)
	inner_x = 1.0 * np.cos(inner_theta)
	inner_y = 1.0 * np.sin(inner_theta)

	semi_t = np.linspace(-np.pi / 2, np.pi / 2, num_circle)
	semi_x = np.cos(semi_t)
	semi_y = -1.0 + np.sin(semi_t)

	line1_x = np.zeros(2)
	line1_y = np.array([0.0, -1.0])
	line2_x = np.zeros(2)
	line2_y = np.array([1.0, 2.0])

	c1_x = np.concatenate([outer_x, semi_x, line1_x, inner_x, line2_x])
	c1_y = np.concatenate([outer_y, semi_y, line1_y, inner_y, line2_y])
	return c1_x, c1_y


def get_region_paths() -> dict[str, Path]:
	"""Return Path objects for C1 and C2."""
	c1_x, c1_y = build_c1_boundary()
	c1_path = Path(np.column_stack([c1_x, c1_y]), closed=True)
	c2_path = Path(np.column_stack([-c1_x, -c1_y - 1.0]), closed=True)
	return {"C1": c1_path, "C2": c2_path}


def classify_points(points: np.ndarray) -> dict[str, np.ndarray]:
	"""Return boolean masks for points inside each region.

	C1 is primary: boundary points belong to C1, and any overlap is assigned to C1.
	"""
	paths = get_region_paths()
	# Small positive radius treats boundary points as inside.
	c1_mask = paths["C1"].contains_points(points, radius=1e-9)
	c2_mask = paths["C2"].contains_points(points, radius=1e-9)
	return {"C1": c1_mask, "C2": np.logical_and(c2_mask, ~c1_mask)}


def sample_points_in_region(region: str, count: int, rng: np.random.Generator | None = None) -> np.ndarray:
	"""Sample points uniformly from a region using rejection sampling."""
	paths = get_region_paths()
	if region not in paths:
		raise ValueError(f"Unknown region: {region}")
	path = paths[region]
	vertices = path.vertices
	min_x, min_y = vertices.min(axis=0)
	max_x, max_y = vertices.max(axis=0)

	if rng is None:
		rng = np.random.default_rng()

	points: list[np.ndarray] = []
	while sum(len(p) for p in points) < count:
		batch = max(256, count)
		x = rng.uniform(min_x, max_x, batch)
		y = rng.uniform(min_y, max_y, batch)
		candidates = np.column_stack([x, y])
		mask = path.contains_points(candidates)
		points.append(candidates[mask])

	return np.vstack(points)[:count]


def plot_regions() -> None:
	fig, ax = plt.subplots(figsize=(6, 6))

	c1_x, c1_y = build_c1_boundary()
	ax.fill(c1_x, c1_y, color="tab:blue", alpha=0.3, zorder=1)
	ax.plot(c1_x, c1_y, color="tab:blue", lw=2.5, zorder=2)
	ax.fill(-c1_x, -c1_y - 1.0, color="tab:red", alpha=0.3, zorder=1)

	ax.axhline(0.0, color="0.7", lw=0.8)
	ax.axvline(0.0, color="0.7", lw=0.8)
	ax.set_aspect("equal", "box")
	ax.set_xlim(-2.2, 2.2)
	ax.set_ylim(-3.2, 2.2)
	ax.set_xlabel("x")
	ax.set_ylabel("y")
	ax.set_title("Region Sketch")

	plt.show(block=True)
	plt.close(fig)


if __name__ == "__main__":
	plot_regions()
