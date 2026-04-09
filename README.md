<div align="center">

# WristHOI: A Dual Wrist-mounted Dataset for Contact-aware Hand-Object Interaction

[//]: # (### ACM MM 2026)

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://cislab-web.github.io/WristHOI/)
[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://your-paper-link)
[![Dataset](https://img.shields.io/badge/Dataset-Release-green)](https://github.com/CISLAB-web/WristHOI-v1)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

**Xin Chu**, **Junghoon Sung**, **Changho Kim**, **Seongwhan Cho**  

**Eunjee Choi**, **Jongha Lee**, **Younggeun Choi**<sup>*</sup>  

Dankook University, Yongin, Korea  

<sup>*</sup> Corresponding author: younggch@dankook.ac.kr

</div>

---

<p align="center">
  <img src="/wrist_hoi/viz/demo.gif" width="60%">
</p>

## Overview

**WristHOI** is a public dataset for contact-aware hand–object interaction understanding. It provides synchronized multi-view RGB-D observations together with unified world-coordinate annotations for hands and objects. In addition to geometric annotations, the dataset also includes interaction states and optional language descriptions for temporal interaction analysis.

This repository releases the public-format dataset organization, annotations, and metadata for research in:

- Hand–Object Interaction Understanding
- 3D Hand Pose / Mesh Estimation
- 6D Object Pose Estimation
- Hand–Object Contact Analysis
- Temporal Interaction State Recognition
- Vision-Language Grounded Interaction Understanding

---

## Directory Structure

The public layout uses a single **`dataset_root/`** directory that contains `assets/`, `metadata/`, and `subjects/`. When working **inside this repository**, place that tree under **`data/`** and pass **`--dataset_root ./data`** (or an absolute path to `.../WristHOI-v1/data`). Subject archives (**demo**, **p001**–**p010**) must be extracted **under `data/subjects/`** so each subject becomes `data/subjects/<subject_id>/` (not directly at the repository root).

```text
dataset_root/                 # e.g. ./data in this repo
├── assets/
│   └── objects/
│       └── <object_class>/         # e.g. banana, cup, RectangularBox
│           ├── mesh.obj            # Object mesh
│           └── meta.json           # Object metadata
├── metadata/
│   ├── sequence_index.csv          # Sequence-level index (recommended entry point)
│   ├── dataset_info.json
│   ├── label_schema.json
│   ├── cameras.json
│   ├── object_class_map.json       # Maps object classes to stable IDs
│   ├── subjects.json
│   ├── train_list.csv
│   ├── test_list.csv
│   └── object_keypoints_cache/     # Cached FPS keypoints per object mesh hash (*.npz)
└── subjects/
    └── <subject_id>/               # e.g. p001, p002, ...
        ├── meta.json
        ├── sensor_data/
        │   └── <sequence_id>/      # e.g. p002__banana_g2__T01
        │       ├── meta.json
        │       ├── frame_index.csv
        │       ├── rgb/
        │       │   └── <camera_id>/    # 01 ... 09 — PNG frames (000000.png, ...)
        │       ├── depth/
        │       │   └── <camera_id>/    # Same camera IDs as rgb/
        │       └── hand_mask/
        │           └── <camera_id>/    # Wrist cameras only (e.g. 02, 08) — PNG masks
        └── labels/
            └── <sequence_id>/
                ├── meta.json
                ├── frame_index.csv
                ├── calibration/
                │   ├── fixed_cameras.json
                │   └── wrist_cameras.json
                ├── annotations/
                │   ├── mano_world.npz
                │   ├── object_pose_world.npz
                │   ├── interaction_state.npz
                │   └── hand_object_contact.npz
                └── language/
                    └── state_descriptions.json
```

### Dataset download

**1. Base metadata and object meshes (download first).** The following archive contains shared **`metadata/`** (e.g. `sequence_index.csv`, schemas, `object_keypoints_cache`, train/test splits) and **`assets/objects/`** (per-class object folders with meshes such as `mesh.obj`). Extract it under **`data/`** so you obtain `data/metadata/` and `data/assets/`. Replace the placeholder URL when the release link is ready.

| Package | Approx. size | Download |
|---------|--------------|----------|
| **base** — `metadata/` + `assets/objects/` | *4.8MB*      | [Download](https://drive.google.com/file/d/11hbm3Ymqtovy-9JAl6ARhBVOLJkvlzD8/view?usp=sharing) |

**2. Subject archives.** Download the packages below, then **extract them into this repository’s `data/subjects/` directory** (repository root → `data/subjects/`), so you obtain paths such as `data/subjects/p001/`, `data/subjects/p002/`.

| # | Package                   | Approx. size | Download |
|---|---------------------------|--------------|----------|
| 1 | **demo** — Dataset review | *820MB*      | [Download](https://drive.google.com/file/d/1mzeN2I9bR26cxDbRx6ZAzMHru5Ojd5fi/view?usp=sharing) |
| 2 | **p001** (`p001.tar.gz`)  | 94 GB        | [Download](https://drive.google.com/file/d/1FB9k-7KviDifYujMdsTMCIwjFW7eSNhp/view?usp=sharing) |
| 3 | **p002** (`p002.tar.gz`)  | 93 GB        | [Download](https://drive.google.com/file/d/1cErP3WYZl6nHQlnLPpQ4CzWBm-x33Pl-/view?usp=sharing) |
| 4 | **p003** (`p003.tar.gz`)  | 100 GB       | [Download](https://drive.google.com/file/d/1MI-rCgL62WXcEZse2sY7isrcacUUMAMa/view?usp=sharing) |
| 5 | **p004** (`p004.tar.gz`)  | 81 GB        | [Download](https://drive.google.com/file/d/1-lSrg-OmI-UbjBwPDZB7L9Db3_1tTulJ/view?usp=sharing) |
| 6 | **p005** (`p005.tar.gz`)  | 88 GB        | [Download](https://drive.google.com/file/d/1-R6Sr8Lhz0d1TMD2VaHaZ887DwH6z9nf/view?usp=sharing) |
| 7 | **p006** (`p006.tar.gz`)  | 84 GB        | [Download](https://drive.google.com/file/d/1_yhKJ2fIT14cFowUnsBsbfFisGksGMTw/view?usp=sharing) |
| 8 | **p007** (`p007.tar.gz`)  | 101 GB       | [Download](https://drive.google.com/file/d/1zQzynAEorFvXtARRuGqOXqLEDZp1tB81/view?usp=sharing) |
| 9 | **p008** (`p008.tar.gz`)  | 95 GB        | [Download](https://drive.google.com/file/d/1BfUC9iM7qK2FGMMkoCJdJgIft9FWdfnM/view?usp=sharing) |
| 10 | **p009** (`p009.tar.gz`)  | 96 GB        | [Download](https://drive.google.com/file/d/1Rstp3JI7ukXAzcuonttqBLmzHwHkwNEC/view?usp=sharing) |
| 11 | **p010** (`p010.tar.gz`)  | 78 GB        | [Download](https://drive.google.com/file/d/15NtcsRom1bhY_2mU4spfcCuF18tQl75c/view?usp=sharing) |

---

## MANO model (before visualization)

The 3D hand mesh pipeline expects **MANO v1.2** on disk. Download it from the official page (registration may be required):

**[https://mano.is.tue.mpg.de/download.php](https://mano.is.tue.mpg.de/download.php)**

After unpacking, place the model directory under this repository as:

```text
models/
└── manov1.2/
    └── mano /
```

(Use the layout from the MANO release, typically including `MANO_RIGHT.pkl` / `MANO_LEFT.pkl` and related assets.)

By default, visualization scripts set `--mano_model_dir` to `<repository_root>/models/manov1.2`. Pass `--mano_model_dir /path/to/manov1.2` if you install MANO elsewhere.

---

## Visualization demo

This repository includes a **wrist cameras 02/08 hand–object 3D scene** visualization with an optional **interaction state timeline**, **language prompts** from `labels/<sequence_id>/language/state_descriptions.json`, and a **MANO contact heatmap** panel (same view as the internal `contact_heatmap` tool). Code lives under `wrist_hoi.viz`.

### Environment setup (conda)

Create and activate a Conda environment named **`wristhoi`**, then install Python dependencies from `requirements.txt`:

```bash
conda create -n wristhoi python=3.9 -y
conda activate wristhoi
cd /path/to/WristHOI-v1
pip install -r requirements.txt
python -m pip install --no-build-isolation "chumpy==0.70"
```

### Usage

From the repository root, point **`--dataset_root`** at **`./data`** after you have unpacked subjects under **`data/subjects/`** (and metadata/assets under **`data/`** as in [Directory Structure](#directory-structure)). If MANO is at the default location, you can omit `--mano_model_dir`:

```bash
python wrist_hoi/viz/scene3d_text.py \
  --dataset_root ./data \
  --subject_id p003 \
  --sequence_id p003__cup_g5__T01 \
  --mano_model_dir ./mano_dir \
  --save_video ./p003_cup_g5_t01.mp4
```

Other flags match the original scripts (e.g. `--preview`, `--frame_step`, `--quiet`). Full list:

```bash
python -m wrist_hoi.viz.scene3d_text --help
```

### Multi-camera mosaic (`public_dataset`)

<p align="center">
  <img src="/wrist_hoi/viz/demo_fix.gif" width="60%">
</p>

For a **single-window overview of the public layout**, use `wrist_hoi.viz.public_dataset`. It builds a tiled mosaic per frame:

- **Fixed camera** (default `--fixed_cam 09`, auto-fallback if extrinsics require it): RGB, depth (pseudo-color), object mesh overlay in that camera, and hand mesh overlay (pyrender; uses `calibration/fixed_cameras.json` intrinsics / extrinsics when available).
- **Wrist / “dynamic” cameras** (`--dynamic_cams`, default `02` and `08`): RGB fused with `hand_mask` (`--dynamic_view_mode mask`) or a synthetic 3D-style panel (`--dynamic_view_mode scene3d`).
- **Flat MANO contact** panel (sparse hand edges + contact vertices from `hand_object_contact.npz`).
- **State** chip, **timeline** from `language/state_descriptions.json` when present, and optional **state description** text.

Requirements are the same as above (PyTorch, **smplx**, **pyrender**, **trimesh**, OpenCV; MANO v1.2 path). Example:

```bash
python -m wrist_hoi.viz.public_dataset \
  --dataset_root ./data \
  --subject_id p003 \
  --sequence_id p003__cup_g5__T01 \
  --preview \
  --save_video ./mosaic_p003_cup_g5_t01.mp4
```

Use `python -m wrist_hoi.viz.public_dataset --help` for `--frame_step`, `--cell_width` / `--cell_height`, and `scene3d` wrist options.

### Programmatic multi-camera paths (`PublicMultiviewLoader`)

If you only need **per-frame file paths** for cameras `01`–`09` (RGB, depth, hand mask) and calibration handles—without opening a window or running MANO rendering—use `wrist_hoi.dataset.PublicMultiviewLoader`. It follows the same directory rules as the mosaic script (`sensor_data/.../rgb|depth|hand_mask/<camera_id>/`, `mano_world.npz` frame list, `frame_index.csv` availability). Example:

```python
from wrist_hoi.dataset import PublicMultiviewLoader

loader = PublicMultiviewLoader("./data", "p003", "p003__cup_g5__T01", max_frames=0)
bundle = loader.build_multiview_paths(0)
print(bundle.rgb_paths.get("09"), bundle.rgb_paths.get("02"))
```



## Acknowledgements
This research was supported by the MSIT (Ministry of Science and ICT), Korea, under the ITRC (Information Technology Research Center) support program (IITP-2025-RS-2024-00437102) and the Global Research Support Program in the Digital Field program (IITP-2025-RS-2024-00418641) supervised by the IITP (Institute for Information \& Communications Technology Planning \& Evaluation).


## Citation

If you find this dataset useful for your research, please consider citing:

```bibtex
@misc{wristhoi2026,
  title        = {WristHOI: A Dual Wrist-mounted Dataset for Contact-aware Hand-Object Interaction},
  author       = {Xin Chu, Junghoon Sung, Changho Kim, Seongwhan Cho, Eunjee Choi, JongHa Lee, Younggeun Choi},
  year         = {2026},
  howpublished = {Under review for ACM Multimedia 2026}
}