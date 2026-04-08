"""
与 ``wrist_hoi.viz.public_dataset``（原 ``ECCV_data_verfication/visualize_public_dataset.py``）
一致的多相机公开数据加载方案：固定机位 + 手腕机位下 RGB / depth / hand_mask 路径解析，
以及 ``mano_world`` 时间轴与标定 JSON。

可视化拼图仍请使用 ``python -m wrist_hoi.viz.public_dataset``；本模块供训练/脚本按帧枚举所有相机文件。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from wrist_hoi.viz.public_dataset import (
    SequencePaths,
    frame_path,
    load_frame_index_csv,
    load_json,
    sequence_paths,
)

# 与公开 README / ECCV 校验脚本一致：9 路相机编号
STANDARD_CAMERA_IDS: Sequence[str] = tuple(f"{i:02d}" for i in range(1, 10))


@dataclass
class MultiviewFramePaths:
    """单帧下各相机、各模态的文件路径（不存在则为 None）。"""

    frame_id: str
    frame_idx: int
    rgb_paths: Dict[str, Optional[str]] = field(default_factory=dict)
    depth_paths: Dict[str, Optional[str]] = field(default_factory=dict)
    hand_mask_paths: Dict[str, Optional[str]] = field(default_factory=dict)


def _select_fixed_cam(fixed_cameras: dict, requested_cam: str) -> str:
    """与 ``PublicSequenceVisualizer._select_fixed_cam`` 相同逻辑。"""
    requested_cam = str(requested_cam)
    complete: List[str] = []
    partial: List[str] = []
    for cam, info in fixed_cameras.items():
        has_k = info.get("K") is not None
        has_t = info.get("T_camera_world") is not None
        if has_k and has_t:
            complete.append(cam)
        elif has_t:
            partial.append(cam)
    if requested_cam in complete:
        return requested_cam
    if complete:
        return sorted(complete)[0]
    if requested_cam in partial:
        return requested_cam
    if partial:
        return sorted(partial)[0]
    raise RuntimeError(
        "no fixed camera with extrinsics (T_camera_world) found. "
        "Export with --intri_yml to include intrinsics, or ensure extri.yml is provided."
    )


class PublicMultiviewLoader:
    """
    公开格式序列的多相机加载器（与全相机可视化脚本同源路径规则）。

    - ``sensor_data/<seq>/rgb|depth|hand_mask/<cam_id>/<frame_id>.png``
    - ``labels/<seq>/calibration/{fixed_cameras,wrist_cameras}.json``
    - ``labels/<seq>/frame_index.csv`` 中的 ``rgb_XX_available`` 等（若存在）
    - ``annotations/mano_world.npz`` 提供 ``frame_ids`` 主时间轴

    大文件可使用内存映射，避免整包读入 RAM。
    """

    def __init__(
        self,
        dataset_root: str,
        subject_id: str,
        sequence_id: str,
        *,
        max_frames: int = 0,
        mmap_annotations: bool = True,
    ) -> None:
        self.paths: SequencePaths = sequence_paths(dataset_root, subject_id, sequence_id)
        calib_dir = self.paths.calib_dir
        self.fixed_cameras: dict = load_json(os.path.join(calib_dir, "fixed_cameras.json"))
        self.wrist_cameras: dict = load_json(os.path.join(calib_dir, "wrist_cameras.json"))
        self.frame_index: Dict[str, dict] = load_frame_index_csv(
            os.path.join(self.paths.label_dir, "frame_index.csv")
        )

        ann_dir = self.paths.ann_dir
        mano_path = os.path.join(ann_dir, "mano_world.npz")
        if mmap_annotations:
            self.mano_npz = np.load(mano_path, allow_pickle=True, mmap_mode="r")
        else:
            self.mano_npz = np.load(mano_path, allow_pickle=True)

        raw_ids = self.mano_npz["frame_ids"]
        self.frame_ids: List[str] = [str(v) for v in raw_ids.tolist()]
        if max_frames and max_frames > 0:
            self.frame_ids = self.frame_ids[: int(max_frames)]
        if not self.frame_ids:
            raise RuntimeError("no frame_ids found in mano_world.npz")

        self._sensor_dir = self.paths.sensor_dir
        self._cameras_with_rgb: Optional[List[str]] = None

    def list_cameras_under_rgb(self) -> List[str]:
        """``sensor_data/<seq>/rgb/`` 下实际存在的相机子目录（排序）。"""
        if self._cameras_with_rgb is not None:
            return list(self._cameras_with_rgb)
        rgb_root = os.path.join(self._sensor_dir, "rgb")
        if not os.path.isdir(rgb_root):
            self._cameras_with_rgb = []
            return []
        names = [d for d in os.listdir(rgb_root) if os.path.isdir(os.path.join(rgb_root, d))]
        self._cameras_with_rgb = sorted(names)
        return list(self._cameras_with_rgb)

    def select_fixed_cam(self, requested_cam: str = "09") -> str:
        return _select_fixed_cam(self.fixed_cameras, requested_cam)

    def fixed_camera_K(self, cam: str) -> Optional[np.ndarray]:
        info = self.fixed_cameras.get(cam, {})
        k = info.get("K")
        if k is None:
            return None
        return np.asarray(k, dtype=np.float64).reshape(3, 3)

    def fixed_camera_T_world(self, cam: str) -> Optional[np.ndarray]:
        info = self.fixed_cameras.get(cam, {})
        t = info.get("T_camera_world")
        if t is None:
            return None
        return np.asarray(t, dtype=np.float64).reshape(4, 4)

    def rgb_available(self, frame_id: str, cam: str) -> bool:
        row = self.frame_index.get(frame_id, {})
        value = str(row.get(f"rgb_{cam}_available", "")).strip().lower()
        if value:
            return value in ("1", "true", "yes")
        return True

    def _sensor_path(self, modality: str, cam: str, frame_id: str) -> Optional[str]:
        root = os.path.join(self._sensor_dir, modality, cam)
        return frame_path(root, frame_id)

    def build_multiview_paths(
        self,
        frame_idx: int,
        *,
        cameras: Optional[Sequence[str]] = None,
        include_depth: bool = True,
        include_hand_mask: bool = True,
    ) -> MultiviewFramePaths:
        """
        解析单帧在所有（或指定）相机下的文件路径。

        ``cameras`` 默认 ``STANDARD_CAMERA_IDS``（01–09）；也可传入 ``list_cameras_under_rgb()`` 的子集。
        """
        if frame_idx < 0 or frame_idx >= len(self.frame_ids):
            raise IndexError(f"frame_idx {frame_idx} out of range [0, {len(self.frame_ids)})")
        frame_id = self.frame_ids[frame_idx]
        cams = list(cameras) if cameras is not None else list(STANDARD_CAMERA_IDS)

        rgb_paths: Dict[str, Optional[str]] = {}
        depth_paths: Dict[str, Optional[str]] = {}
        hand_mask_paths: Dict[str, Optional[str]] = {}

        for cam in cams:
            if self.rgb_available(frame_id, cam):
                rgb_paths[cam] = self._sensor_path("rgb", cam, frame_id)
            else:
                rgb_paths[cam] = None
            if include_depth:
                depth_paths[cam] = self._sensor_path("depth", cam, frame_id)
            if include_hand_mask:
                hand_mask_paths[cam] = self._sensor_path("hand_mask", cam, frame_id)

        return MultiviewFramePaths(
            frame_id=frame_id,
            frame_idx=frame_idx,
            rgb_paths=rgb_paths,
            depth_paths=depth_paths,
            hand_mask_paths=hand_mask_paths,
        )
