<div align="center">

# WristHOI: A Dual Wrist-mounted Dataset for Contact-aware Hand-Object Interaction

### ACM MM 2026

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://your-project-page-link)
[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://your-paper-link)
[![Dataset](https://img.shields.io/badge/Dataset-Release-green)](https://your-dataset-link)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

**Author 1**<sup>1</sup>, **Author 2**<sup>1</sup>, **Author 3**<sup>2</sup>, **Author 4**<sup>1*</sup>  

<sup>1</sup> Affiliation One &nbsp;&nbsp; <sup>2</sup> Affiliation Two  

</div>

---

<p align="center">
  <img src="assets/teaser.png" width="96%">
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

数据集根目录（下文记为 `dataset_root/`）建议采用如下布局：

```text
dataset_root/
├── assets/
│   └── objects/                    # 物体 mesh 等资源（若发布）
├── metadata/
│   ├── sequence_index.csv          # 序列级索引（推荐）
│   ├── dataset_info.json
│   ├── label_schema.json
│   └── cameras.json
└── subjects/
    └── <subject_id>/
        ├── meta.json
        ├── sensor_data/
        │   └── <sequence_id>/
        │       ├── meta.json
        │       ├── frame_index.csv
        │       ├── rgb/              # 各相机 RGB
        │       ├── depth/            # 各相机深度（若有）
        │       └── hand_mask/        # 腕部相机手部分割（若有）
        └── labels/
            └── <sequence_id>/
                ├── meta.json
                ├── frame_index.csv
                ├── calibration/
                ├── annotations/
                └── language/
