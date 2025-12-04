import os
from typing import List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def _extract_points(solutions: List[dict]):
    Z1 = [s.get('Z1', np.nan) for s in solutions]
    Z2 = [s.get('Z2', np.nan) for s in solutions]
    Z3 = [s.get('Z3', np.nan) for s in solutions]
    return np.array(Z1), np.array(Z2), np.array(Z3)


def plot_solutions_3d(
    sols1: List[dict],
    sols2: List[dict],
    sols3p: List[dict],
    outfile: str = "solutions_3d.png",
    title: str = "Solution Pool: Z1 (Cost), Z2 (Emissions), Z3 (Revenue)",
):
    """
    Plot the 3D scatter of (Z1, Z2, Z3) values from sol1, sol2 and sol3p and save figure.
    """
    z1_a, z2_a, z3_a = _extract_points(sols1)
    z1_b, z2_b, z3_b = _extract_points(sols2)
    z1_c, z2_c, z3_c = _extract_points(sols3p)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(z1_a, z2_a, z3_a, c='tab:blue', label='sol1 (Min Cost)', s=30)
    ax.scatter(z1_b, z2_b, z3_b, c='tab:orange', label='sol2 (Min Emissions)', s=30)
    ax.scatter(z1_c, z2_c, z3_c, c='tab:green', label='sol3p (Max Revenue)', s=30)

    ax.set_xlabel('Z1: Cost')
    ax.set_ylabel('Z2: Emissions')
    ax.set_zlabel('Z3: Revenue')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Save figure
    os.makedirs(os.path.dirname(outfile) or '.', exist_ok=True)
    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    plt.close(fig)


def plot_with_cuboid(
    sols1: List[dict],
    sols2: List[dict],
    sols3p: List[dict],
    rec: Tuple[float, float, float, float],
    outfile: str = "solutions_3d_with_cuboid.png",
    title: Optional[str] = None,
):
    """
    Overlay a cuboid on top of the 3D scatter using bounds from `rec`.

    Parameters
    - rec: tuple (z2_min, z2_max, z3p_min, z3p_max) where z3p = -Z3.
      We convert to Z3 bounds as Z3 in [-z3p_max, -z3p_min]. Z1 bounds are
      taken from the data min/max to span the cube across observed range.
    """
    z1_a, z2_a, z3_a = _extract_points(sols1)
    z1_b, z2_b, z3_b = _extract_points(sols2)
    z1_c, z2_c, z3_c = _extract_points(sols3p)

    z1_all = np.concatenate([z1_a, z1_b, z1_c])
    z2_all = np.concatenate([z2_a, z2_b, z2_c])
    z3_all = np.concatenate([z3_a, z3_b, z3_c])

    # Data-driven Z1 bounds (span across observed values with slight padding)
    z1_min = float(np.nanmin(z1_all))
    z1_max = float(np.nanmax(z1_all))
    pad1 = 0.05 * (z1_max - z1_min if (z1_max - z1_min) != 0 else 1.0)
    z1_min -= pad1
    z1_max += pad1

    # Bounds from rec for Z2 and Z3 (convert z3p bounds to Z3)
    z2_min, z2_max, z3p_min, z3p_max = rec
    z3_min = -float(z3p_max)
    z3_max = -float(z3p_min)

    # Build cuboid vertices
    # 8 vertices of the rectangular prism spanning [z1_min,z1_max] x [z2_min,z2_max] x [z3_min,z3_max]
    vertices = np.array([
        [z1_min, z2_min, z3_min],
        [z1_max, z2_min, z3_min],
        [z1_max, z2_max, z3_min],
        [z1_min, z2_max, z3_min],
        [z1_min, z2_min, z3_max],
        [z1_max, z2_min, z3_max],
        [z1_max, z2_max, z3_max],
        [z1_min, z2_max, z3_max],
    ])

    # Faces defined by lists of vertex indices
    faces = [
        [vertices[i] for i in [0, 1, 2, 3]],  # bottom
        [vertices[i] for i in [4, 5, 6, 7]],  # top
        [vertices[i] for i in [0, 1, 5, 4]],  # front
        [vertices[i] for i in [2, 3, 7, 6]],  # back
        [vertices[i] for i in [1, 2, 6, 5]],  # right
        [vertices[i] for i in [0, 3, 7, 4]],  # left
    ]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter points
    ax.scatter(z1_a, z2_a, z3_a, c='tab:blue', label='sol1 (Min Cost)', s=30)
    ax.scatter(z1_b, z2_b, z3_b, c='tab:orange', label='sol2 (Min Emissions)', s=30)
    ax.scatter(z1_c, z2_c, z3_c, c='tab:green', label='sol3p (Max Revenue)', s=30)

    # Add translucent cuboid
    poly = Poly3DCollection(faces, alpha=0.15, facecolor='tab:red', edgecolor='tab:red')
    ax.add_collection3d(poly)

    # Axis labeling
    ax.set_xlabel('Z1: Cost')
    ax.set_ylabel('Z2: Emissions')
    ax.set_zlabel('Z3: Revenue')
    if title is None:
        title = (
            f"Solutions with Cuboid: Z2∈[{z2_min:.2f},{z2_max:.2f}], "
            f"Z3∈[{z3_min:.2f},{z3_max:.2f}]"
        )
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Save figure
    os.makedirs(os.path.dirname(outfile) or '.', exist_ok=True)
    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    plt.close(fig)


def plot_with_cuboid_and_hyperplane(
    sols1: List[dict],
    sols2: List[dict],
    sols3p: List[dict],
    rec: Tuple[float, float, float, float],
    curr_z2: Optional[float] = None,
    curr_z3p: Optional[float] = None,
    outfile: str = "solutions_3d_with_cuboid_plane.png",
    title: Optional[str] = None,
):
    """
    Plot 3D scatter, overlay the cuboid defined by `rec`, and add one or two
    cutting hyperplanes derived from the selected lines in the rectangular
    splitting logic:

    - If `curr_z2` is provided, draw the plane Z2 = curr_z2.
    - If `curr_z3p` is provided, draw the plane Z3 = -curr_z3p.

    Parameters
    - rec: (z2_min, z2_max, z3p_min, z3p_max)
    - curr_z2: value used in splitting; draws a vertical plane along Z2.
    - curr_z3p: value used in splitting; converts to Z3 plane at -curr_z3p.
    """
    z1_a, z2_a, z3_a = _extract_points(sols1)
    z1_b, z2_b, z3_b = _extract_points(sols2)
    z1_c, z2_c, z3_c = _extract_points(sols3p)

    z1_all = np.concatenate([z1_a, z1_b, z1_c])
    z2_all = np.concatenate([z2_a, z2_b, z2_c])
    z3_all = np.concatenate([z3_a, z3_b, z3_c])

    # Data-driven Z1 bounds with padding
    z1_min = float(np.nanmin(z1_all))
    z1_max = float(np.nanmax(z1_all))
    pad1 = 0.05 * (z1_max - z1_min if (z1_max - z1_min) != 0 else 1.0)
    z1_min -= pad1
    z1_max += pad1

    # Bounds from rec for Z2 and Z3 (convert z3p bounds to Z3)
    z2_min, z2_max, z3p_min, z3p_max = rec
    z3_min = -float(z3p_max)
    z3_max = -float(z3p_min)

    # Cuboid vertices and faces
    vertices = np.array([
        [z1_min, z2_min, z3_min],
        [z1_max, z2_min, z3_min],
        [z1_max, z2_max, z3_min],
        [z1_min, z2_max, z3_min],
        [z1_min, z2_min, z3_max],
        [z1_max, z2_min, z3_max],
        [z1_max, z2_max, z3_max],
        [z1_min, z2_max, z3_max],
    ])

    faces = [
        [vertices[i] for i in [0, 1, 2, 3]],  # bottom
        [vertices[i] for i in [4, 5, 6, 7]],  # top
        [vertices[i] for i in [0, 1, 5, 4]],  # front
        [vertices[i] for i in [2, 3, 7, 6]],  # back
        [vertices[i] for i in [1, 2, 6, 5]],  # right
        [vertices[i] for i in [0, 3, 7, 4]],  # left
    ]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter points
    ax.scatter(z1_a, z2_a, z3_a, c='tab:blue', label='sol1 (Min Cost)', s=30)
    ax.scatter(z1_b, z2_b, z3_b, c='tab:orange', label='sol2 (Min Emissions)', s=30)
    ax.scatter(z1_c, z2_c, z3_c, c='tab:green', label='sol3p (Max Revenue)', s=30)

    # Add translucent cuboid
    poly = Poly3DCollection(faces, alpha=0.15, facecolor='tab:red', edgecolor='tab:red')
    ax.add_collection3d(poly)

    # Add hyperplanes if specified
    # Plane Z2 = curr_z2
    if curr_z2 is not None and np.isfinite(curr_z2):
        # Create a rectangle spanning Z1 in [z1_min,z1_max] and Z3 in [z3_min,z3_max]
        plane_z2 = float(curr_z2)
        Xp = np.array([[z1_min, z1_max], [z1_min, z1_max]])
        Yp = np.array([[plane_z2, plane_z2], [plane_z2, plane_z2]])
        Zp = np.array([[z3_min, z3_min], [z3_max, z3_max]])
        ax.plot_surface(Xp, Yp, Zp, color='tab:purple', alpha=0.25, linewidth=0, shade=True)
        ax.text(z1_max, plane_z2, z3_max, f"Z2={plane_z2:.2f}", color='tab:purple')

    # Plane Z3 = -curr_z3p
    if curr_z3p is not None and np.isfinite(curr_z3p):
        plane_z3 = -float(curr_z3p)
        Xp = np.array([[z1_min, z1_max], [z1_min, z1_max]])
        Yp = np.array([[z2_min, z2_min], [z2_max, z2_max]])
        Zp = np.array([[plane_z3, plane_z3], [plane_z3, plane_z3]])
        ax.plot_surface(Xp, Yp, Zp, color='tab:brown', alpha=0.25, linewidth=0, shade=True)
        ax.text(z1_max, z2_max, plane_z3, f"Z3={plane_z3:.2f}", color='tab:brown')

    # Axis labeling
    ax.set_xlabel('Z1: Cost')
    ax.set_ylabel('Z2: Emissions')
    ax.set_zlabel('Z3: Revenue')
    if title is None:
        title = "Solutions with Cuboid and Cutting Hyperplane(s)"
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Save figure
    os.makedirs(os.path.dirname(outfile) or '.', exist_ok=True)
    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    plt.close(fig)
