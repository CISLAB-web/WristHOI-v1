"""
Wrist cameras 02/08 only: far field as left/right RGB panels (optional gap + slight inward fan),
near field as a single 3D hand-object viewport. Virtual camera behind the wrist; optional --flip_dorsum.

Usage (from repo root; --mano_model_dir defaults to <repo>/models/manov1.2):
  python -m wrist_hoi.viz.scene3d_0208 \\
    --dataset_root /path/to/dataset \\
    --subject_id p002 \\
    --sequence_id p002__banana_g2__T01 \\
    --save_video ./out_scene3d.mp4
"""

from __future__ import annotations

import sys
from pathlib import Path

# 允许直接运行 `python wrist_hoi/viz/scene3d_0208.py ...`：将仓库根目录加入 sys.path
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

DEFAULT_MANO_MODEL_DIR = str(_repo_root / "models" / "manov1.2")

import argparse
import os
from typing import Optional, Tuple

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

from wrist_hoi.viz.public_dataset import (
    BREAKOUT_HAND_COLOR_RGBA,
    apply_T,
    build_camera_pose_look_at,
    compose_render_layers,
    frame_path,
    load_mesh_template,
    read_rgb_image,
    render_mesh_rgba,
    sequence_paths,
)

# 物体网格：暗紫色（glTF/PBR baseColor，线性近似）
SCENE3D_OBJECT_DARK_PURPLE_RGBA = (0.20, 0.16, 0.46, 1.0)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _scalar_from_npz(value, default):
    arr = np.asarray(value)
    if arr.ndim == 0:
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    return default


def axis_angle_to_matrix(rot_vec: np.ndarray) -> np.ndarray:
    """轴角 (3,) -> 旋转矩阵。"""
    rot_vec = np.asarray(rot_vec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(rot_vec))
    if theta < 1e-8:
        return np.eye(3, dtype=np.float64)
    k = rot_vec / theta
    x, y, z = k
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def rotate_vector_around_axis(v: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues：向量 v 绕单位轴 axis 旋转 angle_rad（右手系）。"""
    v = np.asarray(v, dtype=np.float64).reshape(3)
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    an = np.linalg.norm(axis)
    if an < 1e-8:
        return v
    k = axis / an
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return v * c + np.cross(k, v) * s + k * np.dot(k, v) * (1.0 - c)


def wrist_z_axis_world(
    global_orient: np.ndarray,
    joints_world: np.ndarray,
    fwd: np.ndarray,
) -> np.ndarray:
    """
    MANO 手腕局部 +Z 轴在世界系中的方向（由 global_orient 轴角得到），
    与指尖方向近似垂直，用于在掌面内「左右」摆动指尖朝向。
    若与 fwd 近乎共线则改用腕-拇指方向估计掌面法向。
    """
    R = axis_angle_to_matrix(global_orient)
    # MANO 手腕局部：+Z 为手掌法向一侧（与掌面内手指摆动轴一致）
    ez = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    wz = R @ ez
    n = np.linalg.norm(wz)
    if n < 1e-8:
        wz = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        wz = wz / n
    fd = fwd / max(np.linalg.norm(fwd), 1e-8)
    if abs(np.dot(fd, wz)) < 0.96:
        return wz
    wrist = np.asarray(joints_world[0], dtype=np.float64)
    thumb = joints_world[17] if joints_world.shape[0] > 17 else joints_world[min(4, joints_world.shape[0] - 1)]
    side = thumb - wrist
    side = side - np.dot(side, fd) * fd
    sn = np.linalg.norm(side)
    if sn < 1e-6:
        side = np.cross(np.array([0.0, 1.0, 0.0]), fd)
        sn = np.linalg.norm(side)
    if sn < 1e-6:
        return wz
    side = side / sn
    pn = np.cross(fd, side)
    pn = pn / max(np.linalg.norm(pn), 1e-8)
    return pn


def finger_forward_from_mano_joints(joints_world: np.ndarray) -> np.ndarray:
    """
    MANO 21 关节：0 腕；5–8 为中指链，8 为中指指尖（与 smplx MANO 一致）。
    返回从手腕指向指尖的单位向量。
    """
    j = np.asarray(joints_world, dtype=np.float64)
    wrist = j[0]
    if j.shape[0] > 8:
        tip = j[8]
    elif j.shape[0] > 5:
        tip = j[5]
    else:
        tip = wrist + np.array([0.0, 0.0, 0.12], dtype=np.float64)
    v = tip - wrist
    n = np.linalg.norm(v)
    if n < 1e-7:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return v / n


def dorsum_up_from_joints(joints_world: np.ndarray, fwd: np.ndarray, is_right_hand: bool) -> np.ndarray:
    """
    手背朝上的世界方向单位向量：在垂直于指尖 fwd 的平面内，由腕→拇指与 fwd 叉乘得到掌面法向，
    使屏幕「上」为手背、下为手心（与左右手性修正）。
    """
    wrist = np.asarray(joints_world[0], dtype=np.float64)
    fd = np.asarray(fwd, dtype=np.float64).reshape(3)
    fd = fd / max(np.linalg.norm(fd), 1e-8)
    # MANO 21 关节：拇指链末端一般为 20（指尖），否则退到 17
    if joints_world.shape[0] > 20:
        thumb_pt = joints_world[20]
    elif joints_world.shape[0] > 17:
        thumb_pt = joints_world[17]
    else:
        thumb_pt = joints_world[min(4, joints_world.shape[0] - 1)]
    side = np.asarray(thumb_pt - wrist, dtype=np.float64)
    side = side - np.dot(side, fd) * fd
    sn = np.linalg.norm(side)
    if sn < 1e-6:
        side = np.cross(np.array([0.0, 1.0, 0.0]), fd)
        sn = np.linalg.norm(side)
    if sn < 1e-6:
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    side = side / sn
    dorsum = np.cross(fd, side)
    dn = np.linalg.norm(dorsum)
    if dn < 1e-8:
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    dorsum = dorsum / dn
    if not is_right_hand:
        dorsum = -dorsum
    return dorsum


def scene_radius_hand_object(
    hand_world: Optional[np.ndarray],
    object_world: Optional[np.ndarray],
    anchor: np.ndarray,
) -> float:
    parts = []
    if hand_world is not None and len(hand_world):
        parts.append(hand_world)
    if object_world is not None and len(object_world):
        parts.append(object_world)
    if not parts:
        return 0.08
    pts = np.vstack(parts)
    r = float(np.max(np.linalg.norm(pts - anchor.reshape(1, 3), axis=1)))
    return max(r, 0.04)


def wrist_based_camera(
    joints_world: np.ndarray,
    hand_world: np.ndarray,
    object_world: Optional[np.ndarray],
    img_wh: Tuple[int, int],
    cam_dist_scale: float,
    view_yaw_deg: float,
    view_pitch_deg: float,
    view_wrist_z_deg: float,
    global_orient: np.ndarray,
    camera_up_lift: float,
    is_right_hand: bool,
    flip_dorsum: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    相机固定在手腕「上后方」：沿 -手指方向退开，再沿手背方向抬高；
    视线与指尖 fwd 一致；look_at 的 up 用手背方向，使渲染图「上=手背、下=手心」。
    view_* 参数在几何指尖方向 fwd0 上做可选微调。
    """
    wrist = np.asarray(joints_world[0], dtype=np.float64)
    fwd0 = finger_forward_from_mano_joints(joints_world)

    if abs(view_yaw_deg) > 1e-6 or abs(view_pitch_deg) > 1e-6:
        yaw = np.radians(view_yaw_deg)
        pitch = np.radians(view_pitch_deg)
        Ry = np.array(
            [[np.cos(yaw), 0.0, np.sin(yaw)], [0.0, 1.0, 0.0], [-np.sin(yaw), 0.0, np.cos(yaw)]],
            dtype=np.float64,
        )
        Rx = np.array(
            [[1.0, 0.0, 0.0], [0.0, np.cos(pitch), -np.sin(pitch)], [0.0, np.sin(pitch), np.cos(pitch)]],
            dtype=np.float64,
        )
        fwd = Ry @ (Rx @ fwd0.reshape(3))
        fwd = fwd / max(np.linalg.norm(fwd), 1e-8)
    else:
        fwd = fwd0

    if abs(view_wrist_z_deg) > 1e-6:
        go = np.asarray(global_orient, dtype=np.float64).reshape(3)
        axis_z = wrist_z_axis_world(go, joints_world, fwd)
        fwd = rotate_vector_around_axis(fwd, axis_z, np.radians(view_wrist_z_deg))
        fwd = fwd / max(np.linalg.norm(fwd), 1e-8)

    if object_world is not None and len(object_world):
        center = 0.55 * hand_world.mean(axis=0) + 0.45 * object_world.mean(axis=0)
    else:
        center = hand_world.mean(axis=0)

    center = center.astype(np.float64)
    r = scene_radius_hand_object(hand_world, object_world, center)
    back_dist = max(0.12, r * max(cam_dist_scale, 0.5))

    dorsum = dorsum_up_from_joints(joints_world, fwd, is_right_hand)
    # 与 fwd 正交化，避免数值漂移
    dorsum = dorsum - np.dot(dorsum, fwd) * fwd
    dn = np.linalg.norm(dorsum)
    if dn < 1e-8:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if abs(np.dot(fwd, world_up)) > 0.92:
            world_up = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        u_side = np.cross(world_up, fwd)
        u_side = u_side / max(np.linalg.norm(u_side), 1e-8)
        dorsum = np.cross(fwd, u_side)
        dorsum = dorsum / max(np.linalg.norm(dorsum), 1e-8)
    else:
        dorsum = dorsum / dn

    if flip_dorsum:
        dorsum = -dorsum

    lift = float(camera_up_lift) * max(r, 0.04)
    # 手腕上后方：沿 -fwd 后退，再沿手背方向抬高（与画面「上」一致）
    eye = wrist - fwd * back_dist + dorsum * lift

    # 视线与手指方向一致：注视点在指尖射线上，取投影使对准场景中心前方
    s = float(np.dot(center - eye, fwd))
    if s < 0.08 * max(r, 0.05):
        s = max(0.15 * max(r, 0.05), float(np.linalg.norm(center - wrist)))
    target = eye + fwd * s

    w, h = img_wh
    cx, cy = w / 2.0, h / 2.0
    cam_d = float(np.linalg.norm(target - eye))
    f = 0.38 * min(w, h) * cam_d / max(r, 0.02)
    K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    cam_pose = build_camera_pose_look_at(eye, target, up=dorsum)
    return K, cam_pose


def render_hand_object_large(
    hand_world: np.ndarray,
    object_world: Optional[np.ndarray],
    hand_faces: np.ndarray,
    mesh_faces: np.ndarray,
    joints_world: np.ndarray,
    img_wh: Tuple[int, int],
    cam_dist_scale: float,
    view_yaw_deg: float,
    view_pitch_deg: float,
    view_wrist_z_deg: float,
    global_orient: np.ndarray,
    camera_up_lift: float,
    is_right_hand: bool,
    flip_dorsum: bool,
    bg_bgr: np.ndarray,
    blend_alpha: float,
) -> Optional[np.ndarray]:
    """
    在 bg_bgr（与视口同尺寸的 02+08 远景底图）上按深度合成手+物体网格，形成远近重合的透视效果。
    """
    if pyrender is None or trimesh is None:
        return None
    w, h = img_wh
    if bg_bgr.shape[0] != h or bg_bgr.shape[1] != w:
        bg_bgr = cv2.resize(bg_bgr, (w, h), interpolation=cv2.INTER_AREA)
    K, cam_pose = wrist_based_camera(
        joints_world,
        hand_world,
        object_world,
        img_wh,
        cam_dist_scale,
        view_yaw_deg,
        view_pitch_deg,
        view_wrist_z_deg,
        global_orient,
        camera_up_lift,
        is_right_hand,
        flip_dorsum,
    )
    layers = []
    if object_world is not None and mesh_faces is not None and len(mesh_faces):
        layer = render_mesh_rgba(
            object_world,
            mesh_faces,
            K,
            img_wh,
            cam_pose,
            base_color=SCENE3D_OBJECT_DARK_PURPLE_RGBA,
        )
        if layer is not None:
            layers.append(layer)
    layer_h = render_mesh_rgba(
        hand_world,
        hand_faces,
        K,
        img_wh,
        cam_pose,
        base_color=BREAKOUT_HAND_COLOR_RGBA,
    )
    if layer_h is not None:
        layers.append(layer_h)
    if not layers:
        return None
    return compose_render_layers(bg_bgr, layers, alpha=float(blend_alpha))


def fuse_cam_strip(
    rgb02: np.ndarray,
    rgb08: np.ndarray,
    target_height: int,
    cam02: str,
    cam08: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """将 02/08 缩放到同一高度后分别标注、横向拼接；返回 (fused, r2, r8) 供扇形变形使用。"""
    if rgb02.shape[0] > 0 and rgb02.shape[1] > 0:
        rgb02 = cv2.flip(cv2.flip(rgb02, 0), 1)
    if rgb08.shape[0] > 0 and rgb08.shape[1] > 0:
        rgb08 = cv2.flip(cv2.flip(rgb08, 0), 1)
    h2, w2 = rgb02.shape[:2]
    h8, w8 = rgb08.shape[:2]
    th = max(32, int(target_height))
    s2 = th / max(h2, 1)
    s8 = th / max(h8, 1)
    r2 = cv2.resize(rgb02, (int(w2 * s2), th), interpolation=cv2.INTER_AREA)
    r8 = cv2.resize(rgb08, (int(w8 * s8), th), interpolation=cv2.INTER_AREA)
    cv2.putText(
        r2,
        f"cam {cam02}",
        (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        r8,
        f"cam {cam08}",
        (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    fused = np.concatenate([r2, r8], axis=1)
    return fused, r2, r8


def add_canvas_border(img_bgr: np.ndarray, pad: int) -> np.ndarray:
    if pad <= 0:
        return img_bgr
    h, w = img_bgr.shape[:2]
    out = np.full((h + 2 * pad, w + 2 * pad, 3), 255, dtype=np.uint8)
    out[pad : pad + h, pad : pad + w] = img_bgr
    return out


def draw_convergence_guides(
    canvas: np.ndarray,
    x0: int,
    y0: int,
    nw: int,
    nh: int,
    alpha: float,
) -> None:
    """极淡视锥暗示线：底边两角至画面上方灭点，不抢 3D 主体。"""
    if alpha <= 1e-6 or nw < 8 or nh < 8:
        return
    rh, rw = canvas.shape[:2]
    overlay = np.zeros_like(canvas)
    vp = (rw // 2, max(8, int(rh * 0.38)))
    bl = (int(x0), int(y0 + nh - 1))
    br = (int(x0 + nw - 1), int(y0 + nh - 1))
    col = (238, 238, 238)
    cv2.line(overlay, bl, vp, col, 1, cv2.LINE_AA)
    cv2.line(overlay, br, vp, col, 1, cv2.LINE_AA)
    canvas[:] = cv2.addWeighted(canvas, 1.0, overlay, float(alpha), 0.0)


def embed_fused_rgb_on_white(
    fused: np.ndarray,
    out_wh: Tuple[int, int],
    relative_scale: float,
    guide_lines: bool = False,
    guide_alpha: float = 0.14,
) -> np.ndarray:
    """
    将 02+08 融合图缩小后居中贴在白底上，营造「更远、更小」的 2D 远景；relative_scale 为相对画幅的缩放系数。
    relative_scale=1 时仍铺满整幅（与旧行为一致）。
    """
    rw, rh = int(out_wh[0]), int(out_wh[1])
    canvas = np.full((rh, rw, 3), 255, dtype=np.uint8)
    if relative_scale >= 1.0 - 1e-6:
        canvas = cv2.resize(fused, (rw, rh), interpolation=cv2.INTER_AREA)
        if guide_lines:
            draw_convergence_guides(canvas, 0, 0, rw, rh, guide_alpha)
        return canvas
    rs = float(np.clip(relative_scale, 0.05, 0.999))
    max_dim = rs * float(min(rw, rh))
    fh, fw = fused.shape[:2]
    s = max_dim / max(float(fh), float(fw), 1e-6)
    nw = max(1, int(round(fw * s)))
    nh = max(1, int(round(fh * s)))
    nw = min(nw, rw)
    nh = min(nh, rh)
    fused_small = cv2.resize(fused, (nw, nh), interpolation=cv2.INTER_AREA)
    x0 = (rw - nw) // 2
    y0 = (rh - nh) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = fused_small
    if guide_lines:
        draw_convergence_guides(canvas, x0, y0, nw, nh, guide_alpha)
    return canvas


def stitch_panels_with_gap(
    r2: np.ndarray,
    r8: np.ndarray,
    gap_px: int,
    bg: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """左右面板之间插入留白条，向外分开，避免底边拼成突兀实心三角。"""
    h = max(r2.shape[0], r8.shape[0])
    gap_px = max(0, int(gap_px))
    if gap_px <= 0:
        return np.concatenate([r2, r8], axis=1)
    strip = np.full((h, gap_px, 3), bg, dtype=np.uint8)
    return np.concatenate([r2, strip, r8], axis=1)


def warp_fan_panels(
    r2: np.ndarray,
    r8: np.ndarray,
    fan_inset_frac: float,
    panel_gap_frac: float,
) -> np.ndarray:
    """
    左/右面板底边内侧轻微内收形成浅 V；中间插入留白条使两面向外分开，
    透视未填充区用浅灰与纸面一致，减轻中间实心白三角的突兀感。
    """
    wL, hL = r2.shape[1], r2.shape[0]
    wR, hR = r8.shape[1], r8.shape[0]
    gap_px = max(0, int(round(float(panel_gap_frac) * float(wL + wR))))
    gap_px = min(gap_px, (wL + wR) // 3)
    # 与纸面留白接近的浅灰，弱化透视产生的未采样三角
    border = (252, 252, 252)

    if fan_inset_frac <= 1e-6:
        return stitch_panels_with_gap(r2, r8, gap_px, bg=border)

    # 有面板间隙时略减弱内倾，扇形更自然
    scale = 0.62 if gap_px > 0 else 1.0
    inset = int(round(float(fan_inset_frac) * scale * float(min(wL, wR))))
    inset = max(1, min(inset, wL // 5, wR // 5, wL - 2, wR - 2))

    src_l = np.float32([[0, 0], [wL, 0], [wL, hL], [0, hL]])
    dst_l = np.float32([[0, 0], [wL, 0], [wL - inset, hL], [0, hL]])
    ml = cv2.getPerspectiveTransform(src_l, dst_l)
    out_l = cv2.warpPerspective(
        r2,
        ml,
        (wL, hL),
        flags=cv2.INTER_AREA,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )

    src_r = np.float32([[0, 0], [wR, 0], [wR, hR], [0, hR]])
    dst_r = np.float32([[0, 0], [wR, 0], [wR, hR], [inset, hR]])
    mr = cv2.getPerspectiveTransform(src_r, dst_r)
    out_r = cv2.warpPerspective(
        r8,
        mr,
        (wR, hR),
        flags=cv2.INTER_AREA,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )
    return stitch_panels_with_gap(out_l, out_r, gap_px, bg=border)


def embed_fan_rgb_on_white(
    r2: np.ndarray,
    r8: np.ndarray,
    out_wh: Tuple[int, int],
    relative_scale: float,
    fan_inset_frac: float,
    panel_gap_frac: float,
    guide_lines: bool,
    guide_alpha: float,
) -> np.ndarray:
    """扇形 + 面板间隙后的双面板再经 distant_scale 缩小居中贴白底，可选极淡引导线。"""
    fused_fan = warp_fan_panels(r2, r8, fan_inset_frac, panel_gap_frac)
    return embed_fused_rgb_on_white(
        fused_fan,
        out_wh,
        relative_scale,
        guide_lines=guide_lines,
        guide_alpha=guide_alpha,
    )


class Scene3d0208Visualizer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.paths = sequence_paths(args.dataset_root, args.subject_id, args.sequence_id)
        self.mano_npz = np.load(os.path.join(self.paths.ann_dir, "mano_world.npz"), allow_pickle=True)
        self.object_npz = np.load(os.path.join(self.paths.ann_dir, "object_pose_world.npz"), allow_pickle=True)
        self.frame_ids = [str(v) for v in self.mano_npz["frame_ids"].tolist()]
        if args.max_frames and args.max_frames > 0:
            self.frame_ids = self.frame_ids[: args.max_frames]
        if not self.frame_ids:
            raise RuntimeError("mano_world.npz 中无 frame_ids")

        self.mano_side = args.mano_side or str(_scalar_from_npz(self.mano_npz["mano_side"], "left"))
        mesh_ref = str(_scalar_from_npz(self.object_npz["mesh_ref"], ""))
        self.mesh_path = os.path.join(self.paths.dataset_root, mesh_ref)
        self.mesh_verts, self.mesh_faces = load_mesh_template(self.mesh_path)
        self.pose_model = self._build_mano()
        self.hand_faces = None if self.pose_model is None else np.asarray(self.pose_model.faces, dtype=np.int32)

    def _build_mano(self):
        if torch is None or smplx is None:
            return None
        m = smplx.create(
            self.args.mano_model_dir,
            model_type="mano",
            is_rhand=(self.mano_side == "right"),
            flat_hand_mean=False,
            use_pca=True,
            num_betas=10,
            num_pca_comps=48,
        )
        m.eval()
        return m

    def _sensor_path(self, modality: str, cam: str, frame_id: str) -> Optional[str]:
        root = os.path.join(self.paths.sensor_dir, modality, cam)
        return frame_path(root, frame_id)

    def _hand_valid(self, frame_idx: int) -> bool:
        return bool(np.asarray(self.mano_npz["valid_mask"]).astype(bool)[frame_idx])

    def _object_valid(self, frame_idx: int) -> bool:
        return bool(np.asarray(self.object_npz["valid_mask"]).astype(bool)[frame_idx])

    def _reconstruct_hand_and_joints(self, frame_idx: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self.pose_model is None or torch is None or not self._hand_valid(frame_idx):
            return None, None
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
        verts = out.vertices[0].detach().cpu().numpy()
        joints = out.joints[0].detach().cpu().numpy()
        return verts, joints

    def _object_world_vertices(self, frame_idx: int) -> Optional[np.ndarray]:
        if not self._object_valid(frame_idx):
            return None
        T_world_object = np.asarray(self.object_npz["T_world_object"][frame_idx], dtype=np.float64).reshape(4, 4)
        return apply_T(T_world_object, self.mesh_verts)

    def _compose_scene3d_frame_bgr(self, frame_idx: int) -> np.ndarray:
        """单帧 02/08 + 手腕视角 3D 合成（BGR），供本脚本与子类（如带状态条/接触热图）复用。"""
        frame_id = self.frame_ids[frame_idx]
        cam02 = str(self.args.cam02)
        cam08 = str(self.args.cam08)
        p02 = self._sensor_path("rgb", cam02, frame_id)
        p08 = self._sensor_path("rgb", cam08, frame_id)
        rgb02 = read_rgb_image(p02)
        rgb08 = read_rgb_image(p08)

        fused, r2, r8 = fuse_cam_strip(rgb02, rgb08, self.args.fused_strip_height, cam02, cam08)

        hand_world, joints = self._reconstruct_hand_and_joints(frame_idx)
        obj_world = self._object_world_vertices(frame_idx)
        global_orient = np.asarray(self.mano_npz["global_orient"][frame_idx], dtype=np.float64).reshape(3)

        rw = int(self.args.render_width)
        rh = int(self.args.render_height)
        img_wh = (rw, rh)
        g_alpha = float(self.args.rgb_guide_alpha)
        g_on = g_alpha > 1e-6
        gap_frac = float(self.args.rgb_panel_gap)
        if float(self.args.rgb_fan_fold) > 1e-6:
            fused_bg = embed_fan_rgb_on_white(
                r2,
                r8,
                img_wh,
                float(self.args.rgb_distant_scale),
                float(self.args.rgb_fan_fold),
                gap_frac,
                g_on,
                g_alpha,
            )
        else:
            gap_px = max(0, int(round(gap_frac * float(r2.shape[1] + r8.shape[1]))))
            fused_gap = stitch_panels_with_gap(r2, r8, gap_px, bg=(252, 252, 252))
            fused_bg = embed_fused_rgb_on_white(
                fused_gap,
                img_wh,
                float(self.args.rgb_distant_scale),
                guide_lines=g_on,
                guide_alpha=g_alpha,
            )

        if hand_world is None or joints is None or self.hand_faces is None:
            canvas = add_canvas_border(fused_bg.copy(), int(self.args.canvas_pad))
            cv2.putText(
                canvas,
                "MANO unavailable or invalid frame",
                (24, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (40, 40, 40),
                2,
                cv2.LINE_AA,
            )
            return canvas

        render_bgr = render_hand_object_large(
            hand_world,
            obj_world,
            self.hand_faces,
            self.mesh_faces,
            joints,
            img_wh,
            cam_dist_scale=float(self.args.cam_dist_scale),
            view_yaw_deg=float(self.args.view_yaw_deg),
            view_pitch_deg=float(self.args.view_pitch_deg),
            view_wrist_z_deg=float(self.args.view_wrist_z_deg),
            global_orient=global_orient,
            camera_up_lift=float(self.args.camera_up_lift),
            is_right_hand=(self.mano_side == "right"),
            flip_dorsum=bool(getattr(self.args, "flip_dorsum", False)),
            bg_bgr=fused_bg,
            blend_alpha=float(self.args.mesh_blend_alpha),
        )
        if render_bgr is None:
            canvas = add_canvas_border(fused_bg.copy(), int(self.args.canvas_pad))
            cv2.putText(
                canvas,
                "pyrender/trimesh failed",
                (24, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (40, 40, 40),
                2,
                cv2.LINE_AA,
            )
            return canvas

        out = add_canvas_border(render_bgr, int(self.args.canvas_pad))
        if getattr(self.args, "sequence_footer", True):
            cv2.putText(
                out,
                f"{self.paths.sequence_id}  frame {frame_id}",
                (12, out.shape[0] - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (100, 100, 100),
                1,
                cv2.LINE_AA,
            )
        return out

    def compose_frame(self, frame_idx: int) -> np.ndarray:
        return self._compose_scene3d_frame_bgr(frame_idx)

    def run(self) -> None:
        writer = None
        video_path = ""
        if self.args.save_video:
            video_path = self.args.save_video
            if not video_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                ensure_dir(video_path)
                video_path = os.path.join(video_path, f"{self.paths.sequence_id}_scene3d_0208.mp4")

        save_dir = ensure_dir(self.args.save_frames_dir) if self.args.save_frames_dir else ""

        delay_ms = max(1, int(round(1000.0 / max(self.args.fps, 1e-6))))
        if self.args.preview:
            cv2.namedWindow("Scene3D 02+08", cv2.WINDOW_NORMAL)

        indices = list(range(0, len(self.frame_ids), max(1, self.args.frame_step)))
        for k, frame_idx in enumerate(indices):
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
            if save_dir:
                cv2.imwrite(os.path.join(save_dir, f"{self.frame_ids[frame_idx]}.jpg"), vis)
            if self.args.preview:
                cv2.imshow("Scene3D 02+08", vis)
                if (cv2.waitKey(delay_ms) & 0xFF) == 27:
                    break
            if not self.args.quiet and (k % 20 == 0 or k == len(indices) - 1):
                print(f"[scene3d] {k + 1}/{len(indices)} {self.frame_ids[frame_idx]}")

        if writer is not None:
            writer.release()
            if not self.args.quiet:
                print(f"[SAVE] {video_path}")
        if self.args.preview:
            cv2.destroyAllWindows()


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="02/08 融合 + 大尺寸手腕视角 3D 可视化")
    p.add_argument("--dataset_root", type=str, required=True)
    p.add_argument("--subject_id", type=str, required=True)
    p.add_argument("--sequence_id", type=str, required=True)
    p.add_argument(
        "--mano_model_dir",
        type=str,
        default=DEFAULT_MANO_MODEL_DIR,
    )
    p.add_argument("--cam02", type=str, default="02", help="腕部动态相机 id（左半幅）")
    p.add_argument("--cam08", type=str, default="08", help="腕部动态相机 id（右半幅）")
    p.add_argument("--mano_side", type=str, default="left", choices=["", "left", "right"])
    p.add_argument("--frame_step", type=int, default=1)
    p.add_argument("--max_frames", type=int, default=0)
    p.add_argument("--fps", type=float, default=20.0)
    p.add_argument("--save_video", type=str, default="")
    p.add_argument("--save_frames_dir", type=str, default="")
    p.add_argument("--preview", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument(
        "--fused_strip_height",
        type=int,
        default=360,
        help="02+08 先按此高度横向拼接再标字，再经 rgb_distant_scale 缩小居中贴白底作为远景",
    )
    p.add_argument("--render_width", type=int, default=1280, help="输出宽度（与 3D 视口、底图一致）")
    p.add_argument("--render_height", type=int, default=880, help="输出高度（与 3D 视口、底图一致）")
    p.add_argument("--canvas_pad", type=int, default=12, help="合成图外圈白边（像素）")
    p.add_argument(
        "--mesh_blend_alpha",
        type=float,
        default=1.0,
        help="3D 网格与底图 RGB 的混合强度（0~1），越大网格越实",
    )
    p.add_argument(
        "--rgb_distant_scale",
        type=float,
        default=0.74,
        help=(
            "2D 远景（02+08 融合图）相对画幅的缩放：按 min(宽,高)×该系数作为融合图最大边长；"
            "略大则底图更满、少被前景 3D 挡住；=1 铺满整幅"
        ),
    )
    p.add_argument(
        "--rgb_fan_fold",
        type=float,
        default=0.32,
        help=(
            "双视角扇形内倾强度（相对单侧板宽）：底边内侧轻微内收形成浅 V；"
            "0 表示平面仅保留面板间隙"
        ),
    )
    p.add_argument(
        "--rgb_panel_gap",
        type=float,
        default=0.138,
        help="左右 RGB 面板之间留白宽度占「两板宽度之和」的比例，向外分开、中间为留白非实心三角",
    )
    p.add_argument(
        "--rgb_guide_alpha",
        type=float,
        default=0.055,
        help="底图视锥暗示线不透明度，宜小；0 关闭",
    )
    p.add_argument(
        "--cam_dist_scale",
        type=float,
        default=1.25,
        help="相机距离 ≈ 场景半径×该系数；略小则手物体在画面中更大",
    )
    p.add_argument(
        "--view_yaw_deg",
        type=float,
        default=25.0,
        help="在指尖指向上绕世界 Y 轴微调（度）",
    )
    p.add_argument(
        "--view_pitch_deg",
        type=float,
        default=0.0,
        help="绕世界 X 轴微调（度）",
    )
    p.add_argument(
        "--view_wrist_z_deg",
        type=float,
        default=0.0,
        help=(
            "绕 MANO 手腕局部 Z 轴（由 global_orient 将局部 +Z 变到世界系）旋转指尖方向，"
            "在掌面内左右摆动（度）；与指尖近似垂直，独立于世界 X/Y 微调"
        ),
    )
    p.add_argument(
        "--camera_up_lift",
        type=float,
        default=0.15,
        help="相机相对手腕沿「上抬」方向的偏移强度（×场景半径），越大越偏手腕上后方俯视感",
    )
    p.add_argument(
        "--flip_dorsum",
        action="store_true",
        help="若渲染图中仍出现手心在上、手背在下，可加此开关把手背/手心上下对调",
    )
    p.add_argument(
        "--sequence_footer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="主画面底部是否叠加序列名与帧号（scene3d_text 默认关闭）",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()
    if torch is None or smplx is None:
        print("需要 torch 与 smplx", file=sys.stderr)
        sys.exit(1)
    if pyrender is None or trimesh is None:
        print("需要 pyrender 与 trimesh", file=sys.stderr)
        sys.exit(1)
    vis = Scene3d0208Visualizer(args)
    vis.run()


if __name__ == "__main__":
    main()