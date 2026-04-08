<div align="center">

# WristHOI: A Dual Wrist-mounted Dataset for Contact-aware Hand-Object Interaction

[//]: # (### ACM MM 2026)

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://cislab-web.github.io/WristHOI/)
[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://your-paper-link)
[![Dataset](https://img.shields.io/badge/Dataset-Release-green)](https://github.com/CISLAB-web/WristHOI-v1)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

**Author 1**<sup>1</sup>, **Author 2**<sup>1</sup>, **Author 3**<sup>2</sup>, **Author 4**<sup>1*</sup>  

<sup>1</sup> Affiliation One &nbsp;&nbsp; <sup>2</sup> Affiliation Two  

</div>

---

<p align="center">
  <img src="/wrist_hoi/viz/demo.gif" width="96%">
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

Download the archives below, then **extract them into this repository’s `data/subjects/` directory** (repository root → `data/subjects/`), so you obtain paths such as `data/subjects/p001/`, `data/subjects/p002/`. 

Replace the placeholder URLs below with the real release links when available.

| # | Package | Approx. size | Download |
|---|---------|--------------|----------|
| 1 | **demo** — small subset for testing the visualization pipeline | *(small)* | [Download]() |
| 2 | **p001** (`p001.tar.gz`) | 94 GB | [Download](https://drive.google.com/file/d/1FB9k-7KviDifYujMdsTMCIwjFW7eSNhp/view?usp=sharing) |
| 3 | **p002** (`p002.tar.gz`) | 93 GB | [Download](https://drive.google.com/file/d/1cErP3WYZl6nHQlnLPpQ4CzWBm-x33Pl-/view?usp=sharing) |
| 4 | **p003** (`p003.tar.gz`) | 100 GB | [Download](https://drive.google.com/file/d/1MI-rCgL62WXcEZse2sY7isrcacUUMAMa/view?usp=sharing) |
| 5 | **p004** (`p004.tar.gz`) | 81 GB | [Download]() |
| 6 | **p005** (`p005.tar.gz`) | 88 GB | [Download]() |
| 7 | **p006** (`p006.tar.gz`) | 84 GB | [Download]() |
| 8 | **p007** (`p007.tar.gz`) | 101 GB | [Download]() |
| 9 | **p008** (`p008.tar.gz`) | 95 GB | [Download]() |
| 10 | **p009** (`p009.tar.gz`) | 96 GB | [Download]() |
| 11 | **p010** (`p010.tar.gz`) | 78 GB | [Download]() |

---

## MANO model (before visualization)

The 3D hand mesh pipeline expects **MANO v1.2** on disk. Download it from the official page (registration may be required):

**[https://mano.is.tue.mpg.de/download.php](https://mano.is.tue.mpg.de/download.php)**

After unpacking, place the model directory under this repository as:

```text
models/
└── manov1.2/
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
```

A **GPU** is recommended for visualization. For **CUDA-enabled PyTorch**, use the official instructions at [PyTorch Get Started](https://pytorch.org/get-started/locally/) and install into the **`wristhoi`** environment (conda or pip wheel). If you install PyTorch via Conda, run `pip install -r requirements.txt` afterward and keep a single `torch` install (avoid mixing conflicting conda and pip builds).

**MANO v1.2** must be placed under `models/manov1.2` as described above (or pass `--mano_model_dir`). For **browser-playable H.264 MP4** export, install **ffmpeg** with an H.264 encoder (**libx264**, **h264_nvenc**, etc.), e.g. `conda install -c conda-forge ffmpeg` while `wristhoi` is active, or use your system package manager.

### Usage

From the repository root, point **`--dataset_root`** at **`./data`** after you have unpacked subjects under **`data/subjects/`** (and metadata/assets under **`data/`** as in [Directory Structure](#directory-structure)). If MANO is at the default location, you can omit `--mano_model_dir`:

```bash
python -m wrist_hoi.viz.scene3d_text \
  --dataset_root ./data \
  --subject_id p002 \
  --sequence_id p002__banana_g2__T01 \
  --save_video ./out_scene3d_text.mp4
```

```bash
python wrist_hoi/viz/scene3d_text.py \
  --dataset_root ./data \
  --subject_id p002 \
  --sequence_id p002__banana_g2__T01 \
  --save_video ./out_scene3d_text.mp4
```

Other flags match the original scripts (e.g. `--preview`, `--frame_step`, `--quiet`). Full list:

```bash
python -m wrist_hoi.viz.scene3d_text --help
```
