"""
Extends scene3d_0208 (02/08 hand-object 3D) with:
- Bottom state timeline (centered); top prompt area from labels/.../language/state_descriptions.json
  (description_text when present, else templates with {object}, {grasp_type_text}, etc.).
- Right contact heatmap with letterboxing so MANO is not stretched on non-square viewports.
- Main view sequence footer off by default (set_defaults(sequence_footer=False)).

Usage (from repo root; --mano_model_dir defaults to <repo>/models/manov1.2):
  python -m wrist_hoi.viz.scene3d_text
  or
  python wrist_hoi/viz/scene3d_text.py \\
    --dataset_root /path/to/dataset \\
    --subject_id p002 \\
    --sequence_id p002__banana_g2__T01 \\
    --save_video ./out_scene3d_text.mp4
"""

from __future__ import annotations

import sys
from pathlib import Path

# 允许直接运行 `python wrist_hoi/viz/scene3d_text.py ...`：将仓库根目录加入 sys.path
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import argparse
import os
import re
import shutil
import subprocess
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

from wrist_hoi.viz.contact_heatmap_render import (
    MANO_NUM_VERTICES,
    _render_hand_heatmap_pyrender,
)

from wrist_hoi.viz.public_dataset import (
    STATE_COLORS_BGR,
    load_json,
    wrap_text,
    STATE_ORDER,
)
from wrist_hoi.viz.scene3d_0208 import (
    Scene3d0208Visualizer,
    build_argparser as scene3d_build_argparser,
    ensure_dir,
)


def _ffmpeg_bin() -> Optional[str]:
    return shutil.which("ffmpeg")


def _ffmpeg_first_h264_encoder() -> Optional[str]:
    """
    从 `ffmpeg -encoders` 里选第一个可用的 H.264 编码器。
    无 libx264 时可能仍有 h264_nvenc / h264_qsv / h264_v4l2m2m 等。
    """
    bin_path = _ffmpeg_bin()
    if not bin_path:
        return None
    try:
        r = subprocess.run(
            [bin_path, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        text = (r.stdout or "") + (r.stderr or "")
    except Exception:
        return None
    # 顺序：通用 CPU → NVIDIA → Intel → AMD → Linux V4L2 → Windows MF
    for name in (
        "libx264",
        "h264_nvenc",
        "h264_qsv",
        "h264_amf",
        "h264_v4l2m2m",
        "h264_mf",
    ):
        if re.search(rf"\b{re.escape(name)}\b", text):
            return name
    return None


def _pad_bgr_to_even(frame: np.ndarray) -> np.ndarray:
    """libx264 + yuv420p 需要偶数宽高；在右下补一行/列，与可视化白底一致。"""
    h, w = frame.shape[:2]
    if h % 2 == 0 and w % 2 == 0:
        return frame
    eh, ew = h + (h % 2), w + (w % 2)
    out = np.full((eh, ew, 3), 255, dtype=np.uint8)
    out[:h, :w] = frame
    return out


def _trim_white_top(
    img: np.ndarray,
    *,
    max_trim: int,
    white_thresh: float = 4.0,
    min_h: int = 160,
) -> np.ndarray:
    """裁掉主画面顶部连续近白行，减少 fused RGB 上方留白。"""
    if max_trim <= 0 or img.size == 0 or img.ndim < 2:
        return img
    h, w = img.shape[:2]
    trim = 0
    for y in range(min(h, max_trim)):
        row = img[y]
        diff = float(np.mean(np.abs(row.astype(np.float32) - 255.0)))
        if diff > white_thresh:
            break
        trim = y + 1
    if trim <= 0:
        return img
    if h - trim < min_h:
        return img
    return img[trim:].copy()


class _FfmpegH264Writer:
    """经 ffmpeg 写入 H.264（yuv420p + faststart），浏览器 <video> 可播。"""

    def __init__(
        self,
        path: str,
        fps: float,
        size_hw: Tuple[int, int],
        *,
        encoder: str,
    ) -> None:
        self._h, self._w = int(size_hw[0]), int(size_hw[1])
        fps_f = max(1e-6, float(fps))
        self._encoder = str(encoder).strip()
        if not self._encoder:
            raise ValueError("encoder 不能为空")
        # 绝对路径：相对路径在部分环境下会导致 ffmpeg 无法创建输出文件
        self._out_path = os.path.abspath(path)
        self._stderr_buf = bytearray()
        # rawvideo 使用 rgb24：与 OpenCV BGR 在 write 里转换，兼容性优于 bgr24
        cmd = [
            _ffmpeg_bin() or "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{self._w}x{self._h}",
            "-framerate",
            str(fps_f),
            "-i",
            "-",
            "-an",
            "-c:v",
            self._encoder,
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            # 部分精简版 ffmpeg 不认 -crf / -preset，用码率模式兼容性更好
            "-b:v",
            "8M",
            self._out_path,
        ]
        self._p = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

        def _drain_stderr() -> None:
            if self._p.stderr:
                while True:
                    chunk = self._p.stderr.read(4096)
                    if not chunk:
                        break
                    self._stderr_buf.extend(chunk)

        self._stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
        self._stderr_thread.start()

    def _stderr_text(self) -> str:
        return self._stderr_buf.decode("utf-8", errors="replace").strip()

    def _raise_ffmpeg_failed(self, prefix: str) -> None:
        try:
            if self._p.stdin:
                self._p.stdin.close()
        except BrokenPipeError:
            pass
        except OSError:
            pass
        self._p.wait()
        self._stderr_thread.join(timeout=60.0)
        err = self._stderr_text()
        raise RuntimeError(
            f"{prefix}ffmpeg 退出码 {self._p.returncode}（编码器 {self._encoder}）。"
            + (f"\n--- stderr ---\n{err}" if err else "")
            + "\n请确认：ffmpeg 含 H.264 编码器；输出目录存在且可写；磁盘空间充足。"
        )

    def write(self, frame: np.ndarray) -> None:
        if self._p.poll() is not None:
            self._raise_ffmpeg_failed("ffmpeg 在写入帧前已退出：")
        if frame.shape[0] != self._h or frame.shape[1] != self._w:
            raise ValueError(
                f"frame size {frame.shape[:2]} != expected ({self._h}, {self._w})"
            )
        bgr = np.ascontiguousarray(frame, dtype=np.uint8)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        try:
            self._p.stdin.write(rgb.tobytes())
        except BrokenPipeError:
            self._raise_ffmpeg_failed("写入视频帧时管道断开（ffmpeg 可能已崩溃）：")

    def release(self) -> None:
        if self._p.stdin:
            self._p.stdin.close()
        self._p.wait()
        self._stderr_thread.join(timeout=60.0)
        if self._p.returncode != 0:
            err = self._stderr_text()
            raise RuntimeError(
                f"ffmpeg 编码结束但失败（退出码 {self._p.returncode}，编码器 {self._encoder}）。"
                + (f"\n--- stderr ---\n{err}" if err else "")
                + "\n请确认 ffmpeg 可用且含 H.264 编码器。"
            )


def get_flat_mano_template(
    mano_model_dir: str,
    *,
    is_rhand: bool,
    mano_num_pca: int = 45,
    mano_flat_hand_mean: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """MANO flat 手模板顶点/三角面；与 contact_heatmap._get_flat_mano_verts_faces 一致，但可指定左右手。"""
    if torch is None or smplx is None:
        raise RuntimeError("需要 torch 和 smplx")
    model = smplx.create(
        mano_model_dir,
        model_type="mano",
        is_rhand=is_rhand,
        flat_hand_mean=mano_flat_hand_mean,
        use_pca=True,
        num_betas=10,
        num_pca_comps=mano_num_pca,
    )
    model.eval()
    hand_pose_dim = model.num_pca_comps if model.use_pca else 3 * model.NUM_HAND_JOINTS
    with torch.no_grad():
        out = model(
            global_orient=torch.zeros(1, 3),
            hand_pose=torch.zeros(1, hand_pose_dim),
            transl=torch.zeros(1, 3),
            betas=torch.zeros(1, 10),
        )
    verts = out.vertices[0].detach().cpu().numpy().astype(np.float64)
    faces = np.asarray(model.faces, dtype=np.int32)
    verts = verts - verts.mean(axis=0, keepdims=True)
    return verts, faces


def load_state_language(label_dir: str) -> dict:
    path = os.path.join(label_dir, "language", "state_descriptions.json")
    if not os.path.isfile(path):
        return {}
    try:
        return load_json(path)
    except Exception:
        return {}


def letterbox_bgr(
    img: np.ndarray,
    out_w: int,
    out_h: int,
    bg: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """等比例缩放后居中贴入目标矩形，避免非正方形视口把 3D 手拉扁/拉长。"""
    ih, iw = img.shape[:2]
    if iw < 1 or ih < 1:
        return np.full((out_h, out_w, 3), bg, dtype=np.uint8)
    scale = min(out_w / float(iw), out_h / float(ih))
    nw = max(1, int(round(iw * scale)))
    nh = max(1, int(round(ih * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.full((out_h, out_w, 3), bg, dtype=np.uint8)
    x0 = (out_w - nw) // 2
    y0 = (out_h - nh) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = resized
    return out


def _dict_get_ci(block: dict, key: str) -> str:
    """字典按状态名取值，兼容大小写/首尾空格不一致。"""
    if not isinstance(block, dict) or not key:
        return ""
    if key in block:
        v = block[key]
        return str(v).strip() if v is not None else ""
    kln = key.strip().lower()
    for k, v in block.items():
        if str(k).strip().lower() == kln:
            return str(v).strip() if v is not None else ""
    return ""


def _format_template_placeholders(tpl: str, meta: dict) -> str:
    """
    填充 templates 中的占位符（与公开数据 state_descriptions.json 一致）：
    {object}、{grasp_type_text}、{grasp_type}、{sequence_id}、{subject_id} 等。
    meta 为整份 JSON 根对象（含 object_id、grasp_type_text…）。
    """
    if not tpl or not isinstance(tpl, str):
        return ""
    obj = str(meta.get("object_id") or meta.get("object", "object"))
    try:
        return tpl.format(
            object=obj,
            object_id=str(meta.get("object_id", obj)),
            grasp_type_text=str(meta.get("grasp_type_text", "")),
            grasp_type=str(meta.get("grasp_type", "")),
            sequence_id=str(meta.get("sequence_id", "")),
            subject_id=str(meta.get("subject_id", "")),
            language=str(meta.get("language", "")),
        )
    except (KeyError, IndexError, ValueError):
        return tpl


def _prompt_from_ordered_list(block: Sequence[Any], state_name: str) -> str:
    """
    部分数据集中 prompt_text / description_text 为与 STATE_ORDER 等长的列表，
    下标 0..3 依次对应 Approach, Contact-Start, In-Contact, Release。
    """
    if not isinstance(block, (list, tuple)) or state_name not in STATE_ORDER:
        return ""
    idx = STATE_ORDER.index(state_name)
    if idx >= len(block):
        return ""
    v = block[idx]
    if v is None:
        return ""
    return str(v).strip()


def state_prompt_text(state_language: dict, state_name: str) -> str:
    """
    从 labels/.../language/state_descriptions.json 取当前状态对应文案。

    优先级（与可视化需求一致：优先展示已填好物体名的 description_text）：
    1) description_text：与 state_ranges 分段一一对应的完整句子
    2) prompt / prompt_text（若有）
    3) templates：占位符填充
    4) states[state].*
    """
    if not state_language:
        return ""
    sn = str(state_name).strip()
    if not sn:
        return ""

    dt = state_language.get("description_text")
    if isinstance(dt, dict):
        t = _dict_get_ci(dt, sn)
        if t:
            return t
    if isinstance(dt, (list, tuple)):
        t = _prompt_from_ordered_list(dt, sn)
        if t:
            return t

    for key in ("prompt", "prompt_text", "PromptText", "promptText"):
        block = state_language.get(key)
        if isinstance(block, dict):
            t = _dict_get_ci(block, sn)
            if t:
                return t
        if isinstance(block, (list, tuple)):
            t = _prompt_from_ordered_list(block, sn)
            if t:
                return t

    tmpl = state_language.get("templates")
    if isinstance(tmpl, dict):
        raw = _dict_get_ci(tmpl, sn)
        if raw:
            filled = _format_template_placeholders(raw, state_language)
            if filled:
                return filled.strip()
    if isinstance(tmpl, (list, tuple)):
        raw = _prompt_from_ordered_list(tmpl, sn)
        if raw:
            filled = _format_template_placeholders(raw, state_language)
            if filled:
                return filled.strip()

    states = state_language.get("states")
    if isinstance(states, dict):
        node = states.get(sn)
        if node is None:
            node = _dict_get_node_ci(states, sn)
        if isinstance(node, dict):
            for k in ("prompt", "prompt_text", "description", "text"):
                if k in node:
                    t = str(node[k]).strip()
                    if t:
                        return t
    return ""


def _dict_get_node_ci(states: dict, key: str) -> Any:
    if key in states:
        return states[key]
    kln = key.strip().lower()
    for k, v in states.items():
        if str(k).strip().lower() == kln:
            return v
    return None


def _put_text_center_x(
    image: np.ndarray,
    text: str,
    y_baseline: int,
    font_scale: float,
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> None:
    h, w = image.shape[:2]
    tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][0]
    x = max(0, (w - tw) // 2)
    cv2.putText(
        image,
        text,
        (x, y_baseline),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_state_progress_panel_scene3d(
    image: np.ndarray,
    state_ranges: Dict[str, List[int]],
    current_index: int,
    total_frames: int,
    origin: Tuple[int, int],
    width: int,
) -> None:
    """
    无深色半透明底；四状态名在彩色条上方、较大字号；进度为白条。
    """
    x0, y0 = int(origin[0]), int(origin[1])
    usable_w = max(120, int(width) - 4)
    bar_h = 24
    label_font = 0.82
    label_thick = 2

    seg_x = x0
    segments_info: List[Tuple[str, int, int]] = []
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
        segments_info.append((state_name, seg_x, next_x))
        seg_x = next_x

    label_baseline = y0 + 24
    for state_name, sx, nx in segments_info:
        color = STATE_COLORS_BGR.get(state_name, (180, 180, 180))
        text = state_name
        tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, label_font, label_thick)[0][0]
        tx = int((sx + nx - tw) / 2)
        tx = max(0, min(tx, image.shape[1] - tw - 1))
        cv2.putText(
            image,
            text,
            (tx, label_baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            label_font,
            color,
            label_thick,
            cv2.LINE_AA,
        )

    bar_y = y0 + 36
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
        seg_x = next_x

    if total_frames > 0:
        cur_x = x0 + int(round((current_index + 0.5) * usable_w / total_frames))
        cv2.line(
            image,
            (cur_x, bar_y - 3),
            (cur_x, bar_y + bar_h + 3),
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.rectangle(image, (x0, bar_y), (x0 + usable_w, bar_y + bar_h), (40, 40, 45), 1)


def draw_prompt_state_colored(
    image: np.ndarray,
    title: str,
    text_lines: Sequence[str],
    origin: Tuple[int, int],
    state_color_bgr: Tuple[int, int, int],
    *,
    line_height: int = 34,
    title_scale: float = 0.92,
    body_scale: float = 0.84,
) -> None:
    """无半透明底框；标题与正文使用当前状态色（与状态条一致），字体偏大。"""
    x0, y0 = int(origin[0]), int(origin[1])
    thick_t = 2
    thick_b = 2
    cv2.putText(
        image,
        str(title),
        (x0, y0),
        cv2.FONT_HERSHEY_SIMPLEX,
        title_scale,
        state_color_bgr,
        thick_t,
        cv2.LINE_AA,
    )
    for i, line in enumerate(text_lines):
        y = y0 + (i + 1) * line_height
        cv2.putText(
            image,
            str(line),
            (x0, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            body_scale,
            state_color_bgr,
            thick_b,
            cv2.LINE_AA,
        )


class Scene3dText0208Visualizer(Scene3d0208Visualizer):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.state_language = load_state_language(self.paths.label_dir)
        sp = os.path.join(self.paths.ann_dir, "interaction_state.npz")
        self.state_npz = np.load(sp, allow_pickle=True) if os.path.isfile(sp) else None
        cp = os.path.join(self.paths.ann_dir, "hand_object_contact.npz")
        self.contact_npz = np.load(cp, allow_pickle=True) if os.path.isfile(cp) else None
        self._hm_verts: Optional[np.ndarray] = None
        self._hm_faces: Optional[np.ndarray] = None

    def _ensure_heatmap_mesh(self) -> bool:
        if self._hm_verts is not None and self._hm_faces is not None:
            return True
        if torch is None or smplx is None:
            return False
        try:
            self._hm_verts, self._hm_faces = get_flat_mano_template(
                self.args.mano_model_dir,
                is_rhand=(self.mano_side == "right"),
                mano_num_pca=int(getattr(self.args, "mano_num_pca_heatmap", 45)),
                mano_flat_hand_mean=True,
            )
        except Exception:
            return False
        return True

    def _contact_indices(self, frame_idx: int) -> np.ndarray:
        if self.contact_npz is None:
            return np.zeros(0, dtype=np.int32)
        offsets = np.asarray(self.contact_npz["hand_vertex_offsets"], dtype=np.int64)
        packed = np.asarray(self.contact_npz["hand_vertex_indices"], dtype=np.int32)
        start = int(offsets[frame_idx])
        end = int(offsets[frame_idx + 1])
        return packed[start:end]

    def _state_valid(self, frame_idx: int) -> bool:
        if self.state_npz is None:
            return False
        vm = np.asarray(self.state_npz["valid_mask"]).astype(bool)
        return bool(vm[frame_idx]) if frame_idx < len(vm) else False

    def _state_ranges(self, state_names: Sequence[str]) -> Dict[str, List[int]]:
        raw_ranges = self.state_language.get("state_ranges", {})
        out: Dict[str, List[int]] = {}
        for state_name in STATE_ORDER:
            value = raw_ranges.get(state_name)
            if value is None and isinstance(raw_ranges, dict):
                value = _dict_get_node_ci(raw_ranges, state_name)
            if isinstance(value, (list, tuple)) and len(value) >= 2:
                out[state_name] = [int(value[0]), int(value[1])]
                continue
            indices = [idx for idx, name in enumerate(state_names) if str(name) == state_name]
            out[state_name] = [indices[0], indices[-1]] if indices else [-1, -1]
        return out

    def _resolve_state_name(self, frame_idx: int, state_names: List[str]) -> str:
        """
        当前帧逻辑状态名。

        与 state_descriptions.json 中 description_text / state_ranges 对齐：若 JSON 里
        有 state_ranges，则优先按帧下标落在哪一段确定状态（与文案键一致），避免
        interaction_state.npz 里 state_name 与 JSON 不一致时读不到 description_text。
        """
        sr = self.state_language.get("state_ranges", {})
        if isinstance(sr, dict):
            for st in STATE_ORDER:
                r = sr.get(st)
                if r is None:
                    r = _dict_get_node_ci(sr, st)
                if isinstance(r, (list, tuple)) and len(r) >= 2:
                    s, e = int(r[0]), int(r[1])
                    if s >= 0 and e >= s and s <= frame_idx <= e:
                        return st
        if self.state_npz is not None and frame_idx < len(state_names):
            if self._state_valid(frame_idx):
                sn = str(state_names[frame_idx]).strip()
                if sn and sn.lower() != "unknown":
                    return sn
        cr = self._state_ranges(state_names)
        for st in STATE_ORDER:
            r = cr.get(st, [-1, -1])
            if isinstance(r, (list, tuple)) and len(r) >= 2:
                s, e = int(r[0]), int(r[1])
                if s >= 0 and e >= s and s <= frame_idx <= e:
                    return st
        if frame_idx < len(state_names):
            sn = str(state_names[frame_idx]).strip()
            if sn:
                return sn
        return "Unknown"

    def _frame_contact_vertex_field(self, frame_idx: int) -> np.ndarray:
        """当前帧接触顶点：接触为 1，否则 0，长度 778。"""
        vals = np.zeros(MANO_NUM_VERTICES, dtype=np.float64)
        if self.contact_npz is None:
            return vals
        idx = self._contact_indices(frame_idx)
        valid = (idx >= 0) & (idx < MANO_NUM_VERTICES)
        for vid in idx[valid]:
            vals[int(vid)] = 1.0
        return vals

    def _draw_contact_panel_captions(self, panel: np.ndarray) -> None:
        """热图列底部：左右手说明 + Contact Annotation（居中）。"""
        h = panel.shape[0]
        hand_line = "Right Hand" if self.mano_side == "right" else "Left Hand"
        _put_text_center_x(panel, hand_line, h - 32, 0.62, (45, 45, 55), 2)
        _put_text_center_x(panel, "Contact Annotation", h - 8, 0.52, (95, 95, 105), 1)

    def _render_contact_heatmap_bgr(self, frame_idx: int, out_hw: Tuple[int, int]) -> np.ndarray:
        """与 contact_heatmap 相同视角；上方为等比例热图，下方为标注区。"""
        h_out, w_out = int(out_hw[0]), int(out_hw[1])
        caption_h = 54
        content_h = max(40, h_out - caption_h)
        placeholder = np.full((h_out, w_out, 3), 255, dtype=np.uint8)

        def _fail(msg: str) -> np.ndarray:
            cv2.putText(
                placeholder,
                msg,
                (12, min(36, content_h // 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (120, 120, 130),
                1,
                cv2.LINE_AA,
            )
            self._draw_contact_panel_captions(placeholder)
            return placeholder

        if pyrender is None or trimesh is None:
            return _fail("pyrender unavailable")
        if not self._ensure_heatmap_mesh() or self._hm_verts is None or self._hm_faces is None:
            return _fail("MANO template failed")

        vertex_values = self._frame_contact_vertex_field(frame_idx)
        sz = int(getattr(self.args, "contact_heatmap_size", 480))
        view_angles = tuple(
            float(x) for x in getattr(self.args, "contact_view_angles", (-90.0, 0.0, 90.0))
        )
        try:
            rgba = _render_hand_heatmap_pyrender(
                self._hm_verts,
                self._hm_faces,
                vertex_values,
                view_angles=view_angles,
                img_size=(sz, sz),
                vmin=0.0,
                vmax=1.0,
            )
        except Exception:
            return _fail("heatmap render failed")

        bgr = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
        bgr = letterbox_bgr(bgr, w_out, content_h, bg=(255, 255, 255))
        placeholder[:content_h, :] = bgr
        self._draw_contact_panel_captions(placeholder)
        return placeholder

    def compose_frame(self, frame_idx: int) -> np.ndarray:
        main = self._compose_scene3d_frame_bgr(frame_idx)
        max_trim = int(getattr(self.args, "crop_top_white_max", 280))
        main = _trim_white_top(
            main,
            max_trim=max_trim,
            white_thresh=float(getattr(self.args, "crop_top_white_thresh", 4.0)),
            min_h=int(getattr(self.args, "crop_top_min_height", 160)),
        )
        mh, mw = main.shape[:2]
        gap = int(getattr(self.args, "contact_panel_gap", 10))
        cw = int(getattr(self.args, "contact_panel_width", 400))
        heat = self._render_contact_heatmap_bgr(frame_idx, (mh, cw))
        gap_strip = np.full((mh, gap, 3), 255, dtype=np.uint8)
        row = np.concatenate([main, gap_strip, heat], axis=1)
        total_w = row.shape[1]

        state_names = []
        if self.state_npz is not None:
            sn = np.asarray(self.state_npz["state_name"]).tolist()
            state_names = [str(sn[i]) if i < len(sn) else "Unknown" for i in range(len(self.frame_ids))]
        else:
            state_names = ["Unknown"] * len(self.frame_ids)

        state_name = self._resolve_state_name(frame_idx, state_names)

        prompt_raw = state_prompt_text(self.state_language, state_name)
        max_chars = int(getattr(self.args, "prompt_wrap_chars", 78))
        desc_lines = wrap_text(prompt_raw, max_chars=max_chars) if prompt_raw else []

        prompt_line_h = int(getattr(self.args, "prompt_line_height", 34))
        pad_top = 10
        title_baseline_offset = 26
        gap_prompt_bar = 12
        bottom_panel_h = 76
        if desc_lines:
            # 与 draw_prompt_state_colored 的 y_title_baseline = mh + pad_top + title_baseline_offset 一致
            prompt_block_h = (
                pad_top + title_baseline_offset + len(desc_lines) * prompt_line_h + 14
            )
            footer_h = prompt_block_h + gap_prompt_bar + bottom_panel_h
        else:
            footer_h = max(96, bottom_panel_h + 20)

        total_h = mh + footer_h
        canvas = np.full((total_h, total_w, 3), 255, dtype=np.uint8)
        canvas[:mh, :total_w] = row

        cv2.line(canvas, (0, mh), (total_w, mh), (235, 238, 242), 1, cv2.LINE_AA)

        state_ranges = self._state_ranges(state_names)
        margin_x = 32
        timeline_w = max(360, total_w - 2 * margin_x)
        timeline_x0 = (total_w - timeline_w) // 2
        timeline_y0 = total_h - bottom_panel_h

        if desc_lines:
            state_col = STATE_COLORS_BGR.get(state_name, (95, 95, 100))
            y_title_baseline = mh + pad_top + title_baseline_offset
            draw_prompt_state_colored(
                canvas,
                "Prompt",
                desc_lines,
                (margin_x, y_title_baseline),
                state_col,
                line_height=prompt_line_h,
                title_scale=float(getattr(self.args, "prompt_title_scale", 0.92)),
                body_scale=float(getattr(self.args, "prompt_body_scale", 0.84)),
            )

        draw_state_progress_panel_scene3d(
            canvas,
            state_ranges=state_ranges,
            current_index=frame_idx,
            total_frames=len(self.frame_ids),
            origin=(timeline_x0, timeline_y0),
            width=timeline_w,
        )
        return canvas

    def run(self) -> None:
        writer = None
        video_path = ""
        if self.args.save_video:
            video_path = self.args.save_video
            if not video_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                ensure_dir(video_path)
                video_path = os.path.join(video_path, f"{self.paths.sequence_id}_scene3d_0208_text.mp4")

        save_dir = ensure_dir(self.args.save_frames_dir) if self.args.save_frames_dir else ""
        delay_ms = max(1, int(round(1000.0 / max(self.args.fps, 1e-6))))
        if self.args.preview:
            cv2.namedWindow("Scene3D 02+08 + state + contact", cv2.WINDOW_NORMAL)

        indices = list(range(0, len(self.frame_ids), max(1, self.args.frame_step)))
        ffmpeg_path = _ffmpeg_bin()
        h264_enc = _ffmpeg_first_h264_encoder() if ffmpeg_path else None
        use_ffmpeg_h264 = bool(
            video_path and video_path.lower().endswith(".mp4") and h264_enc is not None
        )
        if not self.args.quiet and use_ffmpeg_h264 and h264_enc:
            print(f"[scene3d_text] ffmpeg H.264 编码器: {h264_enc}")
        for k, frame_idx in enumerate(indices):
            vis = self.compose_frame(frame_idx)
            if writer is None and video_path:
                ensure_dir(os.path.dirname(video_path) or ".")
                if use_ffmpeg_h264:
                    pad = _pad_bgr_to_even(vis)
                    assert h264_enc is not None
                    writer = _FfmpegH264Writer(
                        video_path,
                        float(self.args.fps),
                        (pad.shape[0], pad.shape[1]),
                        encoder=h264_enc,
                    )
                else:
                    if video_path.lower().endswith(".mp4") and not self.args.quiet:
                        if ffmpeg_path is None:
                            print(
                                "[WARN] 未找到 ffmpeg，使用 OpenCV mp4v；"
                                "浏览器可能无法播放，请安装含 H.264 编码器的 ffmpeg。",
                                file=sys.stderr,
                            )
                        else:
                            print(
                                "[WARN] 当前 ffmpeg 未检测到任何 H.264 编码器（如 libx264、h264_nvenc）；"
                                "使用 OpenCV mp4v；浏览器可能无法播放。"
                                "可安装带 libx264 的 ffmpeg，或启用 NVIDIA 驱动后使用含 nvenc 的构建。",
                                file=sys.stderr,
                            )
                    writer = cv2.VideoWriter(
                        video_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        float(self.args.fps),
                        (vis.shape[1], vis.shape[0]),
                    )
            if writer is not None:
                if use_ffmpeg_h264:
                    writer.write(_pad_bgr_to_even(vis))
                else:
                    writer.write(vis)
            if save_dir:
                cv2.imwrite(os.path.join(save_dir, f"{self.frame_ids[frame_idx]}.jpg"), vis)
            if self.args.preview:
                cv2.imshow("Scene3D 02+08 + state + contact", vis)
                if (cv2.waitKey(delay_ms) & 0xFF) == 27:
                    break
            if not self.args.quiet and (k % 20 == 0 or k == len(indices) - 1):
                print(f"[scene3d_text] {k + 1}/{len(indices)} {self.frame_ids[frame_idx]}")

        if writer is not None:
            writer.release()
            if not self.args.quiet:
                print(f"[SAVE] {video_path}")
        if self.args.preview:
            cv2.destroyAllWindows()


def build_argparser() -> argparse.ArgumentParser:
    p = scene3d_build_argparser()
    p.set_defaults(sequence_footer=False)
    p.description = "02/08 手物 3D + 状态条/说明 + 右侧接触热图（contact_heatmap 视角）"
    p.add_argument(
        "--contact_panel_width",
        type=int,
        default=400,
        help="右侧接触热图列宽度（像素），高度与左侧主画面一致",
    )
    p.add_argument(
        "--contact_panel_gap",
        type=int,
        default=10,
        help="主画面与热图之间的留白宽度",
    )
    p.add_argument(
        "--contact_heatmap_size",
        type=int,
        default=480,
        help="pyrender 内部渲染分辨率（正方形），再缩放到面板大小",
    )
    p.add_argument(
        "--mano_num_pca_heatmap",
        type=int,
        default=45,
        help="热图 MANO flat 模板 PCA 维数（与 contact_heatmap 默认一致）",
    )
    p.add_argument(
        "--contact_view_angles",
        type=float,
        nargs=3,
        default=[-90.0, 0.0, 90.0],
        metavar=("RX", "RY", "RZ"),
        help="与 contact_heatmap.py 中 pyrender 热图一致：顶点旋转角 (度)，默认 -90 0 90",
    )
    p.add_argument(
        "--crop_top_white_max",
        type=int,
        default=280,
        help="从主画面（左侧 3D+RGB）顶部裁掉连续白边的最大像素数，0 表示不裁",
    )
    p.add_argument(
        "--crop_top_white_thresh",
        type=float,
        default=4.0,
        help="判定为白边的行平均 |像素-255| 阈值，越大越保守",
    )
    p.add_argument(
        "--crop_top_min_height",
        type=int,
        default=160,
        help="裁顶后主画面至少保留高度，避免裁过头",
    )
    p.add_argument(
        "--prompt_line_height",
        type=int,
        default=34,
        help="Prompt 标题与各行描述之间的基线间距（像素）",
    )
    p.add_argument(
        "--prompt_wrap_chars",
        type=int,
        default=78,
        help="Prompt 英文自动换行每行最大字符数",
    )
    p.add_argument(
        "--prompt_title_scale",
        type=float,
        default=0.92,
        help="Prompt 标题字号（OpenCV putText scale）",
    )
    p.add_argument(
        "--prompt_body_scale",
        type=float,
        default=0.84,
        help="Prompt 正文每行字号",
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
    vis = Scene3dText0208Visualizer(args)
    vis.run()


if __name__ == "__main__":
    main()
