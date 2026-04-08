"""公开格式 WristHOI 序列的可视化与几何工具（由原 ECCV 校验脚本迁入）。

多相机拼图（固定机位 RGB/Depth/映射 + 手腕动态视角 + 接触面板）由 ``PublicSequenceVisualizer`` 完成。
若仅需按帧枚举所有相机文件路径与标定，请使用 ``wrist_hoi.dataset.PublicMultiviewLoader``（与本文同源路径规则）。
"""

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANO_MODEL_DIR = str(_REPO_ROOT / "models" / "manov1.2")

import cv2
import numpy as np

try:
    import torch
    import smplx
except Exception:
    torch = None
    smplx = None

try:
    import pyrender
    import trimesh
except Exception:
    pyrender = None
    trimesh = None

try:
    import open3d as o3d
except Exception:
    o3d = None


STATE_COLORS_BGR = {
    "Approach": (255, 180, 0),
    "Contact-Start": (0, 200, 255),
    "In-Contact": (0, 255, 0),
    "Release": (0, 120, 255),
}
STATE_ORDER = ["Approach", "Contact-Start", "In-Contact", "Release"]

IMAGE_EXTS = (".png", ".jpg", ".jpeg")


@dataclass
class SequencePaths:
    # 将一个公开版序列涉及的核心目录统一收拢，避免后续反复拼路径。
    dataset_root: str
    subject_id: str
    sequence_id: str
    sensor_dir: str
    label_dir: str
    ann_dir: str
    calib_dir: str


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load and visualize the normalized ECCV public dataset."
    )
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--subject_id", type=str, required=True)
    parser.add_argument("--sequence_id", type=str, required=True)
    parser.add_argument("--mano_model_dir", type=str, default=DEFAULT_MANO_MODEL_DIR)
    parser.add_argument("--fixed_cam", type=str, default="09")
    parser.add_argument("--dynamic_cams", nargs="*", default=["02", "08"])
    parser.add_argument("--frame_step", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--cell_width", type=int, default=640)
    parser.add_argument("--cell_height", type=int, default=360)
    parser.add_argument("--save_video", type=str, default="")
    parser.add_argument("--save_frames_dir", type=str, default="")
    parser.add_argument("--mano_side", type=str, default="", choices=["", "left", "right"])
    parser.add_argument("--mano_flat_hand_mean", action="store_true")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    # 动态相机（如 02/08）无 world 外参时：用 3D 场景把 RGB 作为远景平面、手物体为前景（非像素投影）。
    parser.add_argument(
        "--dynamic_view_mode",
        type=str,
        default="mask",
        choices=["mask", "scene3d"],
        help="mask: RGB+hand_mask 融合（旧）；scene3d: 白底+RGB 背景平面+前景 MANO/物体网格（无相机外参对齐）。",
    )
    parser.add_argument(
        "--scene_az_deg",
        type=float,
        default=38.0,
        help="scene3d：虚拟相机绕场景中心的方位角（度），Y 轴朝上。",
    )
    parser.add_argument(
        "--scene_el_deg",
        type=float,
        default=20.0,
        help="scene3d：虚拟相机仰角（度），略俯视更易与手腕视角接近。",
    )
    parser.add_argument(
        "--scene_cam_dist_scale",
        type=float,
        default=3.2,
        help="scene3d：相机距离 ≈ 场景半径×该系数；≤0 时用内置默认。",
    )
    parser.add_argument(
        "--scene_orbit_deg_per_frame",
        type=float,
        default=0.0,
        help="scene3d：每帧方位角增量（度），用于轻微旋转视频。",
    )
    return parser


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sequence_paths(dataset_root: str, subject_id: str, sequence_id: str) -> SequencePaths:
    # 公开版数据按 subjects/<subject>/sensor_data 与 labels 两棵树组织，
    # 这里一次性解析出当前序列所需的所有根路径。
    dataset_root = os.path.abspath(dataset_root)
    sensor_dir = os.path.join(dataset_root, "subjects", subject_id, "sensor_data", sequence_id)
    label_dir = os.path.join(dataset_root, "subjects", subject_id, "labels", sequence_id)
    if not os.path.isdir(sensor_dir):
        raise FileNotFoundError(f"sensor_dir not found: {sensor_dir}")
    if not os.path.isdir(label_dir):
        raise FileNotFoundError(f"label_dir not found: {label_dir}")
    return SequencePaths(
        dataset_root=dataset_root,
        subject_id=subject_id,
        sequence_id=sequence_id,
        sensor_dir=sensor_dir,
        label_dir=label_dir,
        ann_dir=os.path.join(label_dir, "annotations"),
        calib_dir=os.path.join(label_dir, "calibration"),
    )


def load_frame_index_csv(path: str) -> Dict[str, dict]:
    # frame_index.csv 用于描述每帧有哪些模态/标签可用，
    # 这里转成 frame_id -> row 的字典，便于后面按帧快速查询。
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for row in rows:
        frame_id = str(row.get("frame_id", "")).strip()
        if not frame_id:
            continue
        out[frame_id] = row
    return out


def frame_path(root_dir: str, frame_id: str) -> Optional[str]:
    for ext in IMAGE_EXTS:
        path = os.path.join(root_dir, f"{frame_id}{ext}")
        if os.path.isfile(path):
            return path
    return None


def read_rgb_image(path: Optional[str], fallback_shape: Tuple[int, int] = (720, 1280)) -> np.ndarray:
    if path:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is not None:
            return image
    h, w = fallback_shape
    return np.zeros((h, w, 3), dtype=np.uint8)


def read_depth_vis(path: Optional[str], fallback_shape: Tuple[int, int]) -> np.ndarray:
    # 深度图只用于可视化预览，不参与几何计算。
    # 因此这里做百分位归一化，再转成伪彩色方便观察。
    h, w = fallback_shape
    if not path:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(canvas, "No depth", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        return canvas
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(canvas, "Depth load failed", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        return canvas
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    depth = np.asarray(depth)
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        canvas = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
        cv2.putText(canvas, "Empty depth", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        return canvas
    values = depth[valid].astype(np.float32)
    lo = np.percentile(values, 5.0)
    hi = np.percentile(values, 95.0)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(values.min())
        hi = float(values.max() + 1.0)
    depth_norm = np.zeros_like(depth, dtype=np.uint8)
    scaled = (np.clip(depth.astype(np.float32), lo, hi) - lo) / max(hi - lo, 1e-6)
    depth_norm[valid] = np.clip((1.0 - scaled[valid]) * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)


def draw_points(
    image: np.ndarray,
    uv: np.ndarray,
    valid_mask: np.ndarray,
    color: Tuple[int, int, int],
    radius: int,
) -> None:
    h, w = image.shape[:2]
    pts = np.round(uv).astype(np.int32)
    for (x, y), valid in zip(pts, valid_mask.tolist()):
        if (not valid) or x < 0 or y < 0 or x >= w or y >= h:
            continue
        cv2.circle(image, (int(x), int(y)), int(radius), color, -1, cv2.LINE_AA)


def draw_state_chip(image: np.ndarray, state_name: str, pos: Tuple[int, int]) -> None:
    x, y = pos
    color = STATE_COLORS_BGR.get(state_name, (220, 220, 220))
    cv2.rectangle(image, (x, y), (x + 210, y + 34), color, -1)
    cv2.putText(
        image,
        state_name,
        (x + 10, y + 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )


def draw_timeline(
    image: np.ndarray,
    state_names: Sequence[str],
    current_index: int,
    bar_rect: Tuple[int, int, int, int],
) -> None:
    # 底部时间轴按整段序列状态着色，并用白线标出当前帧。
    x0, y0, width, height = bar_rect
    n = max(1, len(state_names))
    for idx, state_name in enumerate(state_names):
        left = x0 + int(round(idx * width / n))
        right = x0 + int(round((idx + 1) * width / n))
        color = STATE_COLORS_BGR.get(state_name, (180, 180, 180))
        cv2.rectangle(image, (left, y0), (max(left + 1, right), y0 + height), color, -1)
    cur_x = x0 + int(round((current_index + 0.5) * width / n))
    cv2.line(image, (cur_x, y0 - 4), (cur_x, y0 + height + 4), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.rectangle(image, (x0, y0), (x0 + width, y0 + height), (255, 255, 255), 1)


def draw_state_progress_panel(
    image: np.ndarray,
    state_ranges: Dict[str, List[int]],
    current_index: int,
    total_frames: int,
    origin: Tuple[int, int],
    width: int,
) -> None:
    # 参考旧版状态条风格，但这里进一步把四种状态显式划分成四段，
    # 并在每一段下方标注对应状态名，方便直接理解颜色语义。
    x0, y0 = origin
    title_h = 24
    bar_h = 22
    label_h = 26
    panel_h = title_h + bar_h + label_h + 18

    overlay = image.copy()
    cv2.rectangle(overlay, (x0 - 8, y0 - 20), (x0 - 8 + width, y0 - 20 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.42, image, 0.58, 0.0, dst=image)
    cv2.putText(image, "State Progress", (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    bar_y = y0 + 10
    usable_w = max(120, width - 4)
    seg_x = x0
    for idx, state_name in enumerate(STATE_ORDER):
        start_end = state_ranges.get(state_name, [-1, -1])
        start, end = int(start_end[0]), int(start_end[1])
        if start < 0 or end < start or total_frames <= 0:
            seg_w = int(round(usable_w / len(STATE_ORDER)))
        else:
            seg_w = int(round(((end - start + 1) / max(total_frames, 1)) * usable_w))
        if idx == len(STATE_ORDER) - 1:
            next_x = x0 + usable_w
        else:
            next_x = min(x0 + usable_w, seg_x + max(seg_w, 1))
        color = STATE_COLORS_BGR.get(state_name, (180, 180, 180))
        cv2.rectangle(image, (seg_x, bar_y), (max(seg_x + 1, next_x), bar_y + bar_h), color, -1)
        label = state_name
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)[0]
        label_x = int((seg_x + next_x - label_size[0]) / 2)
        label_x = max(x0, min(label_x, x0 + usable_w - label_size[0]))
        cv2.putText(
            image,
            label,
            (label_x, bar_y + bar_h + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            2,
            cv2.LINE_AA,
        )
        seg_x = next_x

    if total_frames > 0:
        cur_x = x0 + int(round((current_index + 0.5) * usable_w / total_frames))
        cv2.line(image, (cur_x, bar_y - 4), (cur_x, bar_y + bar_h + 4), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.rectangle(image, (x0, bar_y), (x0 + usable_w, bar_y + bar_h), (255, 255, 255), 1)


def draw_info_box(image: np.ndarray, lines: Sequence[str], origin: Tuple[int, int] = (16, 20)) -> None:
    x0, y0 = origin
    box_w = 520
    box_h = 14 + 28 * len(lines)
    overlay = image.copy()
    cv2.rectangle(overlay, (x0 - 8, y0 - 18), (x0 - 8 + box_w, y0 - 18 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.42, image, 0.58, 0.0, dst=image)
    for idx, line in enumerate(lines):
        y = y0 + idx * 26
        cv2.putText(image, line, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)


def wrap_text(text: str, max_chars: int = 44) -> List[str]:
    # 简单按单词做自动换行，避免状态描述过长时超出显示框。
    text = str(text or "").strip()
    if not text:
        return []
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_text_panel(
    image: np.ndarray,
    title: str,
    text_lines: Sequence[str],
    origin: Tuple[int, int],
    width: int = 620,
    line_height: int = 24,
) -> None:
    # 独立的文本描述框，用于显示当前状态对应的自然语言说明。
    x0, y0 = origin
    lines = [str(title)] + [str(v) for v in text_lines]
    box_h = 18 + line_height * len(lines)
    overlay = image.copy()
    cv2.rectangle(overlay, (x0 - 8, y0 - 20), (x0 - 8 + width, y0 - 20 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.48, image, 0.52, 0.0, dst=image)
    cv2.putText(image, lines[0], (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    for idx, line in enumerate(lines[1:], start=1):
        y = y0 + idx * line_height
        cv2.putText(image, line, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)


def make_mosaic(
    images: Sequence[np.ndarray],
    titles: Sequence[str],
    n_cols: int = 3,
    cell_size: Tuple[int, int] = (640, 360),
    bg_color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    # 将多个可视化面板统一拼成网格，便于一屏查看固定相机、
    # 动态相机、接触点和状态信息。
    if not images:
        w_cell, h_cell = cell_size
        return np.zeros((h_cell, w_cell, 3), dtype=np.uint8)
    w_cell, h_cell = cell_size
    cells: List[np.ndarray] = []
    for image, title in zip(images, titles):
        canvas = np.full((h_cell, w_cell, 3), bg_color, dtype=np.uint8)
        if image is not None:
            canvas = cv2.resize(image, (w_cell, h_cell))
        cv2.putText(canvas, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cells.append(canvas)
    n_cols = max(1, min(n_cols, len(cells)))
    n_rows = int(math.ceil(len(cells) / float(n_cols)))
    rows: List[np.ndarray] = []
    idx = 0
    for _ in range(n_rows):
        row: List[np.ndarray] = []
        for _ in range(n_cols):
            if idx < len(cells):
                row.append(cells[idx])
            else:
                row.append(np.full((h_cell, w_cell, 3), bg_color, dtype=np.uint8))
            idx += 1
        rows.append(np.concatenate(row, axis=1))
    return np.concatenate(rows, axis=0)


def project_points(X_cam: np.ndarray, K: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    # 标准针孔投影：相机坐标 -> 像素坐标，同时返回 Z>0 的有效掩码。
    Z = X_cam[:, 2:3]
    valid = Z[:, 0] > eps
    X = X_cam[:, 0:1]
    Y = X_cam[:, 1:2]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = fx * (X / (Z + eps)) + cx
    v = fy * (Y / (Z + eps)) + cy
    return np.concatenate([u, v], axis=1), valid


def apply_T(T: np.ndarray, X: np.ndarray) -> np.ndarray:
    # 对点云/顶点批量应用 4x4 刚体变换。
    X = np.asarray(X, dtype=np.float64)
    ones = np.ones((X.shape[0], 1), dtype=np.float64)
    Xh = np.concatenate([X, ones], axis=1)
    Yh = (T @ Xh.T).T
    return Yh[:, :3]


def camera_pose_from_extri(R_wc: np.ndarray, T_wc: np.ndarray) -> np.ndarray:
    # 旧版 verification_ui.py 使用的是同一套转换：
    # OpenCV 相机坐标先转到 camera pose，再补一个 OpenGL 约定变换给 pyrender。
    R_cw = R_wc.T
    t_cw = -R_cw @ T_wc
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R_cw.astype(np.float32)
    pose[:3, 3] = t_cw.reshape(3).astype(np.float32)
    cv_to_gl = np.eye(4, dtype=np.float32)
    cv_to_gl[1, 1] = -1.0
    cv_to_gl[2, 2] = -1.0
    return pose @ cv_to_gl


def render_mesh_rgba(
    verts_world: np.ndarray,
    faces: np.ndarray,
    K: np.ndarray,
    img_wh: Tuple[int, int],
    cam_pose: np.ndarray,
    base_color: Tuple[float, float, float, float],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    # 用 pyrender 做离屏渲染，返回 RGBA 和深度图。
    # 深度图后面会用于多层渲染结果的前后遮挡融合。
    if pyrender is None or trimesh is None:
        return None
    w, h = img_wh
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[0.2, 0.2, 0.2])
    mesh = trimesh.Trimesh(vertices=verts_world, faces=faces, process=False)
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=base_color,
        metallicFactor=0.2,
        roughnessFactor=0.6,
    )
    scene.add(pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True))
    camera = pyrender.IntrinsicsCamera(
        fx=float(K[0, 0]),
        fy=float(K[1, 1]),
        cx=float(K[0, 2]),
        cy=float(K[1, 2]),
    )
    scene.add(camera, pose=cam_pose)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    light2 = pyrender.DirectionalLight(color=np.ones(3), intensity=1.0)
    scene.add(light, pose=cam_pose)
    scene.add(light2, pose=cam_pose)
    renderer = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
    color, depth = renderer.render(
        scene,
        flags=pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SKIP_CULL_FACES,
    )
    renderer.delete()
    return color, depth


def alpha_blend_bgr(
    img_bgr: np.ndarray,
    render_rgba: np.ndarray,
    depth: np.ndarray,
    alpha: float = 0.9,
) -> np.ndarray:
    out = img_bgr.astype(np.float32)
    render_rgb = render_rgba[:, :, :3].astype(np.float32)
    valid = depth > 0
    if not np.any(valid):
        return img_bgr
    a = float(np.clip(alpha, 0.0, 1.0))
    for c in range(3):
        out[:, :, c] = np.where(valid, out[:, :, c] * (1.0 - a) + render_rgb[:, :, c] * a, out[:, :, c])
    return np.clip(out, 0, 255).astype(np.uint8)


def compose_render_layers(
    img_bgr: np.ndarray,
    layers: Sequence[Tuple[np.ndarray, np.ndarray]],
    alpha: float = 0.9,
) -> np.ndarray:
    # 当手和物体都需要渲染时，按深度决定每个像素显示哪一层。
    if not layers:
        return img_bgr
    if len(layers) == 1:
        rgba, depth = layers[0]
        return alpha_blend_bgr(img_bgr, rgba, depth, alpha=alpha)
    rgba_a, depth_a = layers[0]
    rgba_b, depth_b = layers[1]
    mask_a = depth_a > 0
    mask_b = depth_b > 0
    depth_a_inf = np.where(mask_a, depth_a, np.inf)
    depth_b_inf = np.where(mask_b, depth_b, np.inf)
    choose_a = depth_a_inf < depth_b_inf
    combined_rgba = np.zeros_like(rgba_a)
    combined_depth = np.zeros_like(depth_a)
    sel_a = choose_a & mask_a
    sel_b = (~choose_a) & mask_b
    combined_rgba[sel_a] = rgba_a[sel_a]
    combined_rgba[sel_b] = rgba_b[sel_b]
    combined_depth = np.minimum(depth_a_inf, depth_b_inf)
    combined_depth[~(sel_a | sel_b)] = 0
    return alpha_blend_bgr(img_bgr, combined_rgba, combined_depth, alpha=alpha)


# scene3d 动态视角：参考图风格（浅色手网格 + 紫色调物体），非固定相机投影色。
BREAKOUT_HAND_COLOR_RGBA = (0.95, 0.95, 0.92, 1.0)
BREAKOUT_OBJECT_COLOR_RGBA = (0.52, 0.22, 0.78, 1.0)


def build_camera_pose_look_at(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray = None,
) -> np.ndarray:
    """构建 look-at 相机 pose（pyrender：相机局部 -Z 为视线方向）。"""
    if up is None:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    eye = np.asarray(eye, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    up = np.asarray(up, dtype=np.float64).reshape(3)

    forward = target - eye
    dist = np.linalg.norm(forward)
    if dist < 1e-8:
        forward = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        forward = forward / dist

    right = np.cross(forward, up)
    rn = np.linalg.norm(right)
    if rn < 1e-8:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64) if abs(forward[1]) < 0.99 else np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        right = right / rn

    cam_up = np.cross(right, forward)
    cam_up = cam_up / max(np.linalg.norm(cam_up), 1e-8)

    R = np.column_stack([right, cam_up, -forward])
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.astype(np.float32)
    pose[:3, 3] = eye.astype(np.float32)
    return pose


def _scene_center_and_radius(
    hand_world: Optional[np.ndarray],
    object_world: Optional[np.ndarray],
) -> Tuple[np.ndarray, float]:
    parts: List[np.ndarray] = []
    if hand_world is not None and len(hand_world) > 0:
        parts.append(hand_world)
    if object_world is not None and len(object_world) > 0:
        parts.append(object_world)
    if not parts:
        return np.zeros(3, dtype=np.float64), 0.1
    all_pts = np.vstack(parts)
    center = all_pts.mean(axis=0)
    r = float(np.max(np.linalg.norm(all_pts - center, axis=1)))
    return center, max(r, 0.05)


def _virtual_K_from_scene(
    img_wh: Tuple[int, int],
    eye: np.ndarray,
    target: np.ndarray,
    scene_radius: float,
) -> np.ndarray:
    w, h = img_wh
    cx, cy = w / 2.0, h / 2.0
    cam_dist = float(np.linalg.norm(target - eye))
    sr = max(scene_radius, 0.01)
    f = 0.42 * min(w, h) * cam_dist / sr
    return np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def render_breakout_scene_with_image_bg(
    rgb_bgr: np.ndarray,
    hand_world: Optional[np.ndarray],
    object_world: Optional[np.ndarray],
    hand_faces: Optional[np.ndarray],
    mesh_faces: np.ndarray,
    img_wh: Tuple[int, int],
    az_deg: float,
    el_deg: float,
    cam_dist_scale: float,
) -> Optional[np.ndarray]:
    """
    白底 3D 场景：RGB 贴在相机前方更远处的平面上作为背景，手/物体网格在近处，非 2D 投影融合。
    返回 BGR uint8；失败时返回 None。
    """
    if pyrender is None or trimesh is None:
        return None
    w, h = img_wh
    if w < 2 or h < 2:
        return None

    center, scene_radius = _scene_center_and_radius(hand_world, object_world)
    if cam_dist_scale <= 0:
        cam_dist = max(0.4, scene_radius * 3.5)
    else:
        cam_dist = max(0.25, scene_radius * cam_dist_scale)

    az = np.radians(az_deg)
    el = np.radians(el_deg)
    offset = np.array(
        [
            cam_dist * np.cos(el) * np.sin(az),
            cam_dist * np.sin(el),
            cam_dist * np.cos(el) * np.cos(az),
        ],
        dtype=np.float64,
    )
    eye = center + offset
    target = center.copy()

    K = _virtual_K_from_scene((w, h), eye, target, scene_radius)
    cam_pose = build_camera_pose_look_at(eye, target, up=np.array([0.0, 1.0, 0.0]))

    fwd = center - eye
    fn = np.linalg.norm(fwd)
    if fn < 1e-8:
        return None
    fwd = fwd / fn
    plane_center = center + fwd * (scene_radius * 1.35 + 0.06)
    D = float(np.linalg.norm(plane_center - eye))
    fx, fy = float(K[0, 0]), float(K[1, 1])
    quad_half_w = (w / 2.0) * D / max(fx, 1e-6)
    quad_half_h = (h / 2.0) * D / max(fy, 1e-6)

    right = np.cross(fwd, np.array([0.0, 1.0, 0.0], dtype=np.float64))
    rn = np.linalg.norm(right)
    if rn < 1e-8:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        right = right / rn
    up_plane = np.cross(right, fwd)
    up_plane = up_plane / max(np.linalg.norm(up_plane), 1e-8)

    q00 = plane_center - right * quad_half_w - up_plane * quad_half_h
    q10 = plane_center + right * quad_half_w - up_plane * quad_half_h
    q11 = plane_center + right * quad_half_w + up_plane * quad_half_h
    q01 = plane_center - right * quad_half_w + up_plane * quad_half_h
    verts_q = np.asarray([q00, q10, q11, q01], dtype=np.float64)
    faces_q = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    tex = cv2.resize(rgb_bgr, (w, h), interpolation=cv2.INTER_AREA)
    tex_rgb = cv2.cvtColor(tex, cv2.COLOR_BGR2RGB)
    uvs = np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]], dtype=np.float64)

    scene = pyrender.Scene(
        ambient_light=[0.55, 0.55, 0.55],
        bg_color=[1.0, 1.0, 1.0, 1.0],
    )

    try:
        bg_mesh = trimesh.Trimesh(
            vertices=verts_q,
            faces=faces_q,
            visual=trimesh.visual.TextureVisuals(uv=uvs, image=tex_rgb),
            process=False,
        )
        scene.add(pyrender.Mesh.from_trimesh(bg_mesh, smooth=False))
    except Exception:
        bg_mesh = trimesh.Trimesh(vertices=verts_q, faces=faces_q, process=False)
        mat_bg = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=(0.85, 0.85, 0.88, 1.0),
            metallicFactor=0.05,
            roughnessFactor=0.9,
        )
        scene.add(pyrender.Mesh.from_trimesh(bg_mesh, material=mat_bg, smooth=True))

    mat_hand = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=BREAKOUT_HAND_COLOR_RGBA,
        metallicFactor=0.15,
        roughnessFactor=0.55,
    )
    mat_obj = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=BREAKOUT_OBJECT_COLOR_RGBA,
        metallicFactor=0.2,
        roughnessFactor=0.45,
    )

    if object_world is not None and mesh_faces is not None and len(mesh_faces) > 0:
        om = trimesh.Trimesh(vertices=object_world, faces=mesh_faces, process=False)
        scene.add(pyrender.Mesh.from_trimesh(om, material=mat_obj, smooth=True))
    if hand_world is not None and hand_faces is not None and len(hand_faces) > 0:
        hm = trimesh.Trimesh(vertices=hand_world, faces=hand_faces, process=False)
        scene.add(pyrender.Mesh.from_trimesh(hm, material=mat_hand, smooth=True))

    camera = pyrender.IntrinsicsCamera(
        fx=float(K[0, 0]),
        fy=float(K[1, 1]),
        cx=float(K[0, 2]),
        cy=float(K[1, 2]),
    )
    scene.add(camera, pose=cam_pose)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.2), pose=cam_pose)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=0.9), pose=cam_pose)

    renderer = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
    try:
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.SKIP_CULL_FACES)
    finally:
        renderer.delete()
    return cv2.cvtColor(color, cv2.COLOR_RGB2BGR)


def draw_projected_vertices(
    img_bgr: np.ndarray,
    verts_cam: np.ndarray,
    K: np.ndarray,
    stride: int,
    color: Tuple[int, int, int],
) -> None:
    uv, valid = project_points(verts_cam, K)
    draw_points(img_bgr, uv[:: max(1, stride)], valid[:: max(1, stride)], color=color, radius=1)


def load_mesh_template(mesh_path: str) -> Tuple[np.ndarray, np.ndarray]:
    # 优先使用 trimesh / open3d 读取 mesh；
    # 两者都不可用时，再退化到最简 OBJ 解析。
    if trimesh is not None:
        mesh = trimesh.load(mesh_path, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        return verts, faces
    if o3d is not None:
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.triangles, dtype=np.int32)
        return verts, faces
    verts: List[List[float]] = []
    faces: List[List[int]] = []
    with open(mesh_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                idxs = []
                for token in line.strip().split()[1:]:
                    idxs.append(int(token.split("/")[0]) - 1)
                if len(idxs) >= 3:
                    for i in range(1, len(idxs) - 1):
                        faces.append([idxs[0], idxs[i], idxs[i + 1]])
    if not verts or not faces:
        raise RuntimeError(f"failed to load mesh: {mesh_path}")
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int32)


class PublicSequenceVisualizer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        # 序列级元数据、标定和各类标签都在初始化阶段加载好，
        # 后续逐帧渲染时只做索引和计算，不重复读文件。
        self.paths = sequence_paths(args.dataset_root, args.subject_id, args.sequence_id)
        self.sensor_meta = load_json(os.path.join(self.paths.sensor_dir, "meta.json"))
        self.label_meta = load_json(os.path.join(self.paths.label_dir, "meta.json"))
        self.fixed_cameras = load_json(os.path.join(self.paths.calib_dir, "fixed_cameras.json"))
        self.wrist_cameras = load_json(os.path.join(self.paths.calib_dir, "wrist_cameras.json"))
        self.frame_index = load_frame_index_csv(os.path.join(self.paths.label_dir, "frame_index.csv"))
        self.state_language = self._load_state_language()

        self.mano_npz = np.load(os.path.join(self.paths.ann_dir, "mano_world.npz"), allow_pickle=True)
        self.object_npz = np.load(os.path.join(self.paths.ann_dir, "object_pose_world.npz"), allow_pickle=True)
        self.contact_npz = np.load(os.path.join(self.paths.ann_dir, "hand_object_contact.npz"), allow_pickle=True)
        self.state_npz = np.load(os.path.join(self.paths.ann_dir, "interaction_state.npz"), allow_pickle=True)

        self.frame_ids = [str(v) for v in self.mano_npz["frame_ids"].tolist()]
        if args.max_frames and args.max_frames > 0:
            self.frame_ids = self.frame_ids[: args.max_frames]
        if not self.frame_ids:
            raise RuntimeError("no frame_ids found in mano_world.npz")
        self.frame_to_index = {fid: idx for idx, fid in enumerate(self.frame_ids)}

        self.fixed_cam = self._select_fixed_cam(args.fixed_cam)
        self.dynamic_cams = [str(cam) for cam in args.dynamic_cams]

        self.mano_side = args.mano_side or str(self._scalar_from_npz(self.mano_npz["mano_side"], "left"))
        self.mesh_path = os.path.join(self.paths.dataset_root, str(self._scalar_from_npz(self.object_npz["mesh_ref"], "")))
        self.mesh_verts, self.mesh_faces = load_mesh_template(self.mesh_path)
        self.pose_model, self.flat_model = self._build_mano_models()
        self.hand_faces = None if self.pose_model is None else np.asarray(self.pose_model.faces, dtype=np.int32)
        # flat 面板只画稀疏线框，因此预先从三角面片提取边集合。
        self.flat_edges = self._build_mesh_edges(self.hand_faces) if self.hand_faces is not None else []

    @staticmethod
    def _scalar_from_npz(value, default):
        arr = np.asarray(value)
        if arr.ndim == 0:
            return arr.item()
        if arr.size == 1:
            return arr.reshape(-1)[0].item()
        return default

    @staticmethod
    def _build_mesh_edges(faces: Optional[np.ndarray]) -> List[Tuple[int, int]]:
        if faces is None:
            return []
        edges = set()
        for i0, i1, i2 in np.asarray(faces, dtype=np.int32):
            pairs = ((i0, i1), (i1, i2), (i2, i0))
            for a, b in pairs:
                a_i, b_i = int(a), int(b)
                if a_i > b_i:
                    a_i, b_i = b_i, a_i
                edges.add((a_i, b_i))
        return sorted(edges)

    def _load_state_language(self) -> dict:
        path = os.path.join(self.paths.label_dir, "language", "state_descriptions.json")
        if not os.path.isfile(path):
            return {}
        try:
            return load_json(path)
        except Exception:
            return {}

    def _state_ranges(self, state_names: Sequence[str]) -> Dict[str, List[int]]:
        # 优先使用导出时写入的状态范围；若缺失则从当前序列状态列表回退计算。
        raw_ranges = self.state_language.get("state_ranges", {})
        out: Dict[str, List[int]] = {}
        for state_name in STATE_ORDER:
            value = raw_ranges.get(state_name)
            if isinstance(value, (list, tuple)) and len(value) >= 2:
                out[state_name] = [int(value[0]), int(value[1])]
                continue
            indices = [idx for idx, name in enumerate(state_names) if str(name) == state_name]
            out[state_name] = [indices[0], indices[-1]] if indices else [-1, -1]
        return out

    def _build_mano_models(self):
        if torch is None or smplx is None:
            return None, None
        # 这里故意与 verification_ui.py 保持一致：
        # pca=48, flat_hand_mean=False, use_pca=True, num_betas=10。
        # 否则同一份 MANO 参数在可视化里可能出现姿态变形。
        pose_model = smplx.create(
            self.args.mano_model_dir,
            model_type="mano",
            is_rhand=(self.mano_side == "right"),
            flat_hand_mean=False,
            use_pca=True,
            num_betas=10,
            num_pca_comps=48,
        )
        flat_model = smplx.create(
            self.args.mano_model_dir,
            model_type="mano",
            is_rhand=(self.mano_side == "right"),
            # flat 手用于接触点展示，因此这里显式启用 flat hand mean。
            flat_hand_mean=True,
            use_pca=True,
            num_betas=10,
            num_pca_comps=48,
        )
        pose_model.eval()
        flat_model.eval()
        return pose_model, flat_model

    def _select_fixed_cam(self, requested_cam: str) -> str:
        """选择固定相机。优先使用 K 和 T_camera_world 都完整的相机。"""
        requested_cam = str(requested_cam)
        complete = []  # K 和 T 都有的相机
        partial = []   # 只有 T 的相机（K 缺失时用于 fallback）
        for cam, info in self.fixed_cameras.items():
            has_K = info.get("K") is not None
            has_T = info.get("T_camera_world") is not None
            if has_K and has_T:
                complete.append(cam)
            elif has_T:
                partial.append(cam)
        # 优先返回用户指定且标定完整的相机
        if requested_cam in complete:
            return requested_cam
        if complete:
            return sorted(complete)[0]
        # 无完整标定时，若用户指定的相机有 T，则用其（K 将用 fallback）
        if requested_cam in partial:
            return requested_cam
        if partial:
            return sorted(partial)[0]
        raise RuntimeError(
            "no fixed camera with extrinsics (T_camera_world) found. "
            "Export with --intri_yml to include intrinsics, or ensure extri.yml is provided."
        )

    def _get_frame_row(self, frame_id: str) -> dict:
        return self.frame_index.get(frame_id, {})

    def _availability(self, frame_id: str, prefix: str, cam: str) -> bool:
        row = self._get_frame_row(frame_id)
        value = str(row.get(f"{prefix}_{cam}_available", "")).strip().lower()
        return value in ("1", "true", "yes")

    def _sensor_path(self, modality: str, cam: str, frame_id: str) -> Optional[str]:
        root = os.path.join(self.paths.sensor_dir, modality, cam)
        return frame_path(root, frame_id)

    def _fixed_camera_K(self, cam: str) -> Optional[np.ndarray]:
        info = self.fixed_cameras.get(cam, {})
        K = info.get("K")
        if K is None:
            return None
        return np.asarray(K, dtype=np.float64).reshape(3, 3)

    def _fixed_camera_Tcw(self, cam: str) -> Optional[np.ndarray]:
        info = self.fixed_cameras.get(cam, {})
        T = info.get("T_camera_world")
        if T is None:
            return None
        return np.asarray(T, dtype=np.float64).reshape(4, 4)

    def _contact_indices(self, frame_idx: int) -> np.ndarray:
        # 公开版接触标签采用 CSR 风格稀疏存储：
        # offsets 指定每帧在一维 packed 顶点数组中的切片范围。
        offsets = np.asarray(self.contact_npz["hand_vertex_offsets"], dtype=np.int64)
        packed = np.asarray(self.contact_npz["hand_vertex_indices"], dtype=np.int32)
        start = int(offsets[frame_idx])
        end = int(offsets[frame_idx + 1])
        return packed[start:end]

    def _hand_valid(self, frame_idx: int) -> bool:
        valid_mask = np.asarray(self.mano_npz["valid_mask"]).astype(bool)
        return bool(valid_mask[frame_idx])

    def _object_valid(self, frame_idx: int) -> bool:
        valid_mask = np.asarray(self.object_npz["valid_mask"]).astype(bool)
        return bool(valid_mask[frame_idx])

    def _state_valid(self, frame_idx: int) -> bool:
        valid_mask = np.asarray(self.state_npz["valid_mask"]).astype(bool)
        return bool(valid_mask[frame_idx])

    def _reconstruct_hand_world(self, frame_idx: int) -> Optional[np.ndarray]:
        if self.pose_model is None or torch is None:
            return None
        if not self._hand_valid(frame_idx):
            return None
        # 输入字段名虽然换成了公开版命名，
        # 但传给 MANO 的方式严格对齐旧版 verification_ui.py。
        rh = np.asarray(self.mano_npz["global_orient"][frame_idx], dtype=np.float32).reshape(1, 3)
        poses = np.asarray(self.mano_npz["hand_pose"][frame_idx], dtype=np.float32).reshape(1, -1)
        th = np.asarray(self.mano_npz["transl"][frame_idx], dtype=np.float32).reshape(1, 3)
        betas = np.asarray(self.mano_npz["betas"][frame_idx], dtype=np.float32).reshape(1, 10)
        with torch.no_grad():
            out = self.pose_model(
                global_orient=torch.from_numpy(rh),
                hand_pose=torch.from_numpy(poses),
                betas=torch.from_numpy(betas),
                transl=torch.from_numpy(th),
            )
        return out.vertices[0].detach().cpu().numpy()

    def _object_world_vertices(self, frame_idx: int) -> Optional[np.ndarray]:
        if not self._object_valid(frame_idx):
            return None
        # object_pose_world.npz 中保存的是 T_world_object，
        # 因此先把模板 mesh 从物体系变到世界系。
        T_world_object = np.asarray(self.object_npz["T_world_object"][frame_idx], dtype=np.float64).reshape(4, 4)
        return apply_T(T_world_object, self.mesh_verts)

    def _render_object_mapping(
        self,
        rgb: np.ndarray,
        frame_idx: int,
        K: np.ndarray,
        T_camera_world: np.ndarray,
    ) -> np.ndarray:
        out = rgb.copy()
        object_world = self._object_world_vertices(frame_idx)
        if object_world is None:
            cv2.putText(out, "No object pose", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            return out
        # 固定相机下优先直接渲染物体 mesh；
        # 如果环境里没有 pyrender，就退化为顶点投影结果。
        cam_pose = camera_pose_from_extri(T_camera_world[:3, :3], T_camera_world[:3, 3:4])
        layer = render_mesh_rgba(
            verts_world=object_world,
            faces=self.mesh_faces,
            K=K,
            img_wh=(rgb.shape[1], rgb.shape[0]),
            cam_pose=cam_pose,
            base_color=(0.2, 0.8, 0.7, 1.0),
        )
        if layer is not None:
            return compose_render_layers(out, [layer], alpha=0.85)
        object_cam = apply_T(T_camera_world, object_world)
        draw_projected_vertices(out, object_cam, K, stride=5, color=(0, 255, 255))
        return out

    def _render_hand_mapping(
        self,
        rgb: np.ndarray,
        frame_idx: int,
        K: np.ndarray,
        T_camera_world: np.ndarray,
    ) -> np.ndarray:
        out = rgb.copy()
        hand_world = self._reconstruct_hand_world(frame_idx)
        if hand_world is None:
            cv2.putText(out, "No MANO", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            return out
        # 手部映射结果与物体类似：优先渲染 mesh，失败时退化到顶点投影。
        cam_pose = camera_pose_from_extri(T_camera_world[:3, :3], T_camera_world[:3, 3:4])
        layer = None
        if self.hand_faces is not None:
            layer = render_mesh_rgba(
                verts_world=hand_world,
                faces=self.hand_faces,
                K=K,
                img_wh=(rgb.shape[1], rgb.shape[0]),
                cam_pose=cam_pose,
                base_color=(0.85, 0.6, 0.4, 1.0),
            )
        if layer is not None:
            out = compose_render_layers(out, [layer], alpha=0.85)
        else:
            hand_cam = apply_T(T_camera_world, hand_world)
            draw_projected_vertices(out, hand_cam, K, stride=2, color=(0, 180, 255))
        contact_ids = self._contact_indices(frame_idx)
        if contact_ids.size > 0:
            # 接触点仍然是 MANO 顶点索引，因此先把全部顶点投影到 2D，
            # 再只高亮当前帧被标为接触的那一部分顶点。
            hand_cam = apply_T(T_camera_world, hand_world)
            uv, valid = project_points(hand_cam, K)
            mask = np.zeros((len(hand_world),), dtype=bool)
            mask[contact_ids[(contact_ids >= 0) & (contact_ids < len(hand_world))]] = True
            draw_points(out, uv[mask], valid[mask], color=(0, 0, 255), radius=3)
        return out

    def _render_dynamic_view(
        self,
        frame_id: str,
        cam: str,
        fallback_shape: Tuple[int, int],
        frame_idx: int,
    ) -> np.ndarray:
        # 动态相机 02/08 通常没有公开 world 外参：mask 模式为 RGB+hand_mask；scene3d 为白底+远景 RGB 平面+前景网格。
        rgb_path = self._sensor_path("rgb", cam, frame_id)
        rgb = read_rgb_image(rgb_path, fallback_shape=fallback_shape)
        if str(cam) in ("02", "08"):
            rgb = cv2.flip(cv2.flip(rgb, 0), 1)
        h, w = rgb.shape[:2]
        img_wh = (w, h)

        if self.args.dynamic_view_mode == "scene3d":
            az = float(self.args.scene_az_deg) + float(self.args.scene_orbit_deg_per_frame) * float(frame_idx)
            scene_bgr = render_breakout_scene_with_image_bg(
                rgb,
                self._reconstruct_hand_world(frame_idx),
                self._object_world_vertices(frame_idx),
                self.hand_faces,
                self.mesh_faces,
                img_wh,
                az_deg=az,
                el_deg=float(self.args.scene_el_deg),
                cam_dist_scale=float(self.args.scene_cam_dist_scale),
            )
            if scene_bgr is not None:
                cv2.putText(
                    scene_bgr,
                    f"cam{cam} scene3d (no extri; virtual view)",
                    (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (40, 40, 40),
                    2,
                    cv2.LINE_AA,
                )
                return scene_bgr
            canvas = np.full((h, w, 3), 255, dtype=np.uint8)
            cv2.putText(
                canvas,
                "scene3d needs pyrender+trimesh+MANO",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (60, 60, 60),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"cam{cam} fallback",
                (20, 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (80, 80, 80),
                2,
                cv2.LINE_AA,
            )
            small = cv2.resize(rgb, (min(w, 640), int(h * min(1.0, 640 / max(w, 1)))))
            y0 = 100
            x0 = 12
            if y0 + small.shape[0] < h and x0 + small.shape[1] < w:
                canvas[y0 : y0 + small.shape[0], x0 : x0 + small.shape[1]] = small
            return canvas

        mask_path = self._sensor_path("hand_mask", cam, frame_id)
        if not mask_path:
            canvas = rgb.copy()
            cv2.putText(canvas, f"cam{cam} mask missing", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            return canvas
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            canvas = rgb.copy()
            cv2.putText(canvas, f"cam{cam} mask load failed", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            return canvas
        if str(cam) in ("02", "08"):
            mask = cv2.flip(cv2.flip(mask, 0), 1)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask_bool = mask > 0
        overlay = rgb.copy()
        overlay[mask_bool] = (0.35 * overlay[mask_bool] + 0.65 * np.array([0, 255, 0], dtype=np.float32)).astype(np.uint8)
        alpha = 0.35
        out = cv2.addWeighted(overlay, 1.0 - alpha, rgb, alpha, 0.0)
        contours, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (0, 255, 255), 2)
        return out

    def _render_flat_contact_panel(self, frame_idx: int, panel_size: Tuple[int, int]) -> np.ndarray:
        h, w = panel_size
        canvas = np.full((h, w, 3), 18, dtype=np.uint8)
        if self.flat_model is None or torch is None or self.hand_faces is None:
            cv2.putText(canvas, "SMPL-X/MANO unavailable", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            return canvas
        # 这里不显示真实姿态，而是构造一个 flat MANO，
        # 专门用来稳定地查看接触顶点在手模板上的分布。
        betas = np.asarray(self.mano_npz["betas"][frame_idx], dtype=np.float32).reshape(1, 10)
        zeros3 = np.zeros((1, 3), dtype=np.float32)
        zeros_pose = np.zeros((1, int(np.asarray(self.mano_npz["hand_pose"]).shape[1])), dtype=np.float32)
        with torch.no_grad():
            out = self.flat_model(
                global_orient=torch.from_numpy(zeros3),
                hand_pose=torch.from_numpy(zeros_pose),
                transl=torch.from_numpy(zeros3),
                betas=torch.from_numpy(betas),
            )
        verts = out.vertices[0].detach().cpu().numpy().astype(np.float64)
        verts -= verts.mean(axis=0, keepdims=True)
        # 给 flat MANO 一个固定观察角度，让掌面和手指分布更容易看清。
        rx = np.deg2rad(-65.0)
        ry = np.deg2rad(20.0)
        R_x = np.array(
            [[1.0, 0.0, 0.0], [0.0, math.cos(rx), -math.sin(rx)], [0.0, math.sin(rx), math.cos(rx)]],
            dtype=np.float64,
        )
        R_y = np.array(
            [[math.cos(ry), 0.0, math.sin(ry)], [0.0, 1.0, 0.0], [-math.sin(ry), 0.0, math.cos(ry)]],
            dtype=np.float64,
        )
        verts_view = (R_y @ (R_x @ verts.T)).T
        xy = verts_view[:, :2]
        span = np.ptp(xy, axis=0)
        scale = 0.72 * min(w, h) / max(float(max(span[0], span[1], 1e-6)), 1e-6)
        uv = np.zeros((len(verts_view), 2), dtype=np.float64)
        uv[:, 0] = xy[:, 0] * scale + w * 0.5
        uv[:, 1] = -xy[:, 1] * scale + h * 0.58
        uv_i = np.round(uv).astype(np.int32)
        for edge_idx, (a, b) in enumerate(self.flat_edges):
            if edge_idx % 3 != 0:
                continue
            p0 = tuple(uv_i[a])
            p1 = tuple(uv_i[b])
            cv2.line(canvas, p0, p1, (90, 90, 90), 1, cv2.LINE_AA)
        contact_ids = self._contact_indices(frame_idx)
        valid_ids = contact_ids[(contact_ids >= 0) & (contact_ids < len(uv_i))]
        for vid in valid_ids.tolist():
            x, y = uv_i[int(vid)]
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(canvas, (int(x), int(y)), 4, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"contact vertices: {len(valid_ids)}",
            (18, h - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return canvas

    def compose_frame(self, frame_idx: int) -> np.ndarray:
        # 每帧最终输出一个大拼图：
        # 固定相机 RGB / depth / object mapping / hand mapping
        # + 动态相机 RGB+mask
        # + flat MANO 接触点面板
        frame_id = self.frame_ids[frame_idx]
        fixed_rgb_path = self._sensor_path("rgb", self.fixed_cam, frame_id)
        fixed_rgb = read_rgb_image(fixed_rgb_path)
        fixed_depth_path = self._sensor_path("depth", self.fixed_cam, frame_id)
        fixed_depth = read_depth_vis(fixed_depth_path, fallback_shape=fixed_rgb.shape[:2])

        K = self._fixed_camera_K(self.fixed_cam)
        T_camera_world = self._fixed_camera_Tcw(self.fixed_cam)
        if T_camera_world is None:
            raise RuntimeError(f"fixed camera calibration incomplete (missing T_camera_world): {self.fixed_cam}")
        # 当 K 缺失时，用图像尺寸构造 fallback 内参，便于验证 pipeline 是否正常
        if K is None:
            h, w = fixed_rgb.shape[:2]
            f = max(w, h) * 1.2  # 常见近似焦距
            K = np.array(
                [[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]],
                dtype=np.float64,
            )
            if not getattr(self, "_fallback_k_warned", False):
                import sys
                print(
                    f"[WARN] fixed_cam {self.fixed_cam}: K is null in calibration, using fallback from image size {w}x{h}. "
                    "Re-export with --intri_yml for accurate projection.",
                    file=sys.stderr,
                )
                self._fallback_k_warned = True

        object_map = self._render_object_mapping(fixed_rgb, frame_idx, K, T_camera_world)
        hand_map = self._render_hand_mapping(fixed_rgb, frame_idx, K, T_camera_world)

        images: List[np.ndarray] = [
            fixed_rgb,
            fixed_depth,
            object_map,
            hand_map,
        ]
        titles: List[str] = [
            f"Fixed Cam {self.fixed_cam} RGB",
            f"Fixed Cam {self.fixed_cam} Depth",
            "Object Coordinate Mapping",
            "Hand Coordinate Mapping",
        ]

        dyn_title_suffix = "RGB+Mask" if self.args.dynamic_view_mode == "mask" else "Scene3D bg+mesh"
        for cam in self.dynamic_cams:
            images.append(self._render_dynamic_view(frame_id, cam, fixed_rgb.shape[:2], frame_idx))
            titles.append(f"Dynamic Cam {cam} {dyn_title_suffix}")

        images.append(self._render_flat_contact_panel(frame_idx, panel_size=fixed_rgb.shape[:2]))
        titles.append("Flat MANO Contact")

        mosaic = make_mosaic(
            images,
            titles,
            n_cols=3,
            cell_size=(int(self.args.cell_width), int(self.args.cell_height)),
        )

        state_names = [str(v) for v in self.state_npz["state_name"][: len(self.frame_ids)].tolist()]
        state_name = state_names[frame_idx] if frame_idx < len(state_names) and self._state_valid(frame_idx) else "Unknown"
        description_map = self.state_language.get("description_text", {})
        state_description = str(description_map.get(state_name, "")).strip()
        state_ranges = self._state_ranges(state_names)
        contact_count = int(np.asarray(self.contact_npz["contact_vertex_count"])[frame_idx])
        min_dist = float(np.asarray(self.contact_npz["min_contact_distance_m"])[frame_idx])
        dist_text = "nan"
        if np.isfinite(min_dist):
            dist_text = f"{min_dist * 1000.0:.2f} mm"
        info_lines = [
            f"subject: {self.paths.subject_id}  sequence: {self.paths.sequence_id}",
            f"frame: {frame_id} ({frame_idx + 1}/{len(self.frame_ids)})  state: {state_name}",
            f"fixed_cam: {self.fixed_cam}  dynamic_cams: {','.join(self.dynamic_cams)}",
            f"contact vertices: {contact_count}  min contact distance: {dist_text}",
        ]
        draw_info_box(mosaic, info_lines, origin=(16, 24))
        draw_state_chip(mosaic, state_name, (16, 140))

        timeline_panel_width = max(360, mosaic.shape[1] - 32)
        timeline_origin = (16, mosaic.shape[0] - 92)
        draw_state_progress_panel(
            mosaic,
            state_ranges=state_ranges,
            current_index=frame_idx,
            total_frames=len(self.frame_ids),
            origin=timeline_origin,
            width=timeline_panel_width,
        )

        if state_description:
            desc_lines = wrap_text(state_description, max_chars=40)
            desc_width = min(560, max(360, mosaic.shape[1] // 3))
            desc_box_h = 18 + 24 * (1 + len(desc_lines))
            desc_x = mosaic.shape[1] - desc_width - 16
            desc_y = timeline_origin[1] - desc_box_h - 18
            draw_text_panel(
                mosaic,
                title="State Description",
                text_lines=desc_lines,
                origin=(desc_x, desc_y),
                width=desc_width,
            )
        return mosaic

    def run(self) -> None:
        # 支持三种运行模式：
        # 1) 仅预览；2) 仅导出视频/图片；3) 预览同时导出。
        save_frames_dir = ensure_dir(self.args.save_frames_dir) if self.args.save_frames_dir else ""
        writer = None
        video_path = ""
        if self.args.save_video:
            video_path = self.args.save_video
            if not video_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                ensure_dir(video_path)
                video_path = os.path.join(video_path, f"{self.paths.sequence_id}.mp4")

        if self.args.preview:
            cv2.namedWindow("PublicDatasetVis", cv2.WINDOW_NORMAL)

        delay_ms = max(1, int(round(1000.0 / max(self.args.fps, 1e-6))))
        frame_indices = list(range(0, len(self.frame_ids), max(1, self.args.frame_step)))
        for local_idx, frame_idx in enumerate(frame_indices):
            vis = self.compose_frame(frame_idx)
            if writer is None and video_path:
                ensure_dir(os.path.dirname(video_path) or ".")
                writer = cv2.VideoWriter(
                    video_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    float(self.args.fps),
                    (vis.shape[1], vis.shape[0]),
                )
            if writer is not None:
                writer.write(vis)
            if save_frames_dir:
                out_path = os.path.join(save_frames_dir, f"{self.frame_ids[frame_idx]}.jpg")
                cv2.imwrite(out_path, vis)
            if self.args.preview:
                cv2.imshow("PublicDatasetVis", vis)
                key = cv2.waitKey(delay_ms) & 0xFF
                if key == 27:
                    break
            if (not self.args.quiet) and (local_idx % 20 == 0 or local_idx == len(frame_indices) - 1):
                print(f"[VIS] frame {local_idx + 1}/{len(frame_indices)} -> {self.frame_ids[frame_idx]}")

        if writer is not None:
            writer.release()
            if not self.args.quiet:
                print(f"[SAVE] video -> {video_path}")
        if self.args.preview:
            cv2.destroyAllWindows()


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    vis = PublicSequenceVisualizer(args)
    vis.run()


if __name__ == "__main__":
    main()
