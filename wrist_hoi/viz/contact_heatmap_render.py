"""
Pyrender-based hand contact heatmap rendering (extracted from ECCV verification contact_heatmap).
Used by scene3d_text for the right-hand MANO heatmap column.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
except Exception:
    pass

try:
    import matplotlib.cm as matplotlib_cm_module
except Exception:
    matplotlib_cm_module = None

try:
    import pyrender
    import trimesh
except Exception:
    pyrender = None
    trimesh = None

MANO_NUM_VERTICES = 778
HEATMAP_CMAP = "jet"


def _vertex_counts_to_colors(
    vertex_counts: np.ndarray,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap_name: str = HEATMAP_CMAP,
) -> np.ndarray:
    """Map per-vertex scalars to vertex colors (n, 4) uint8 RGBA."""
    if matplotlib_cm_module is None:
        raise RuntimeError("matplotlib is required for contact heatmap coloring")
    vals = vertex_counts.astype(np.float64)
    if vmin is None:
        vmin = float(vals.min())
    if vmax is None:
        vmax = float(vals.max())
    if vmax <= vmin:
        vmax = vmin + 1e-6
    norm = (vals - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0.0, 1.0)
    try:
        cmap = matplotlib.colormaps.get_cmap(cmap_name)
    except AttributeError:
        cmap = matplotlib_cm_module.get_cmap(cmap_name)
    colors_float = cmap(norm)[:, :4]
    return (np.clip(colors_float * 255, 0, 255)).astype(np.uint8)


def _apply_view_rotation(
    verts: np.ndarray,
    view_angles: Tuple[float, float, float],
) -> np.ndarray:
    """Rotate vertices by view_angles (degrees)."""
    rx, ry, rz = np.deg2rad(view_angles[0]), np.deg2rad(view_angles[1]), np.deg2rad(view_angles[2])
    R_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, math.cos(rx), -math.sin(rx)], [0.0, math.sin(rx), math.cos(rx)]],
        dtype=np.float64,
    )
    R_y = np.array(
        [[math.cos(ry), 0.0, math.sin(ry)], [0.0, 1.0, 0.0], [-math.sin(ry), 0.0, math.cos(ry)]],
        dtype=np.float64,
    )
    R_z = np.array(
        [[math.cos(rz), -math.sin(rz), 0.0], [math.sin(rz), math.cos(rz), 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return (R_z @ R_y @ R_x @ verts.T).T


def _render_hand_heatmap_pyrender(
    verts: np.ndarray,
    faces: np.ndarray,
    vertex_values: np.ndarray,
    view_angles: Tuple[float, float, float],
    img_size: Tuple[int, int] = (480, 480),
    cam_dist: float = 0.45,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> np.ndarray:
    """
    Render a lit hand heatmap with pyrender.
    Returns RGBA (H, W, 4).
    """
    if pyrender is None or trimesh is None:
        raise RuntimeError("pyrender and trimesh are required")
    verts_rot = _apply_view_rotation(verts, view_angles)
    vertex_colors = _vertex_counts_to_colors(vertex_values, vmin=vmin, vmax=vmax)

    mesh_tri = trimesh.Trimesh(
        vertices=verts_rot,
        faces=faces,
        process=False,
        vertex_colors=vertex_colors,
    )

    w, h = img_size
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.35, 0.35, 0.35])
    mesh_pr = pyrender.Mesh.from_trimesh(mesh_tri, material=None, smooth=True)
    scene.add(mesh_pr)

    yfov = np.deg2rad(35.0)
    camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=1.0)
    cam_pose = np.eye(4, dtype=np.float32)
    cam_pose[2, 3] = cam_dist
    scene.add(camera, pose=cam_pose)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    light2 = pyrender.DirectionalLight(color=np.ones(3), intensity=1.2)
    scene.add(light, pose=cam_pose)
    scene.add(light2, pose=cam_pose)

    renderer = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
    color, _ = renderer.render(
        scene,
        flags=pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SKIP_CULL_FACES,
    )
    renderer.delete()
    return color
