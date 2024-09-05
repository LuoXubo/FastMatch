# FastMatch

How fast can image matching be?

## Table of Contents

- [FastMatch](#fastmatch)
  - [Table of Contents](#table-of-contents)
  - [1. Introduction](#1-introduction)
  - [2. Installation](#2-installation)
    - [2.1. Clone the repository](#21-clone-the-repository)
    - [2.2. Install dependencies](#22-install-dependencies)
  - [3. Usage](#3-usage)
  - [4. Results](#4-results)
    - [4.1 Time Comparison](#41-time-comparison)
    - [4.2 Video Matching](#42-video-matching)
  - [5. Thanks To](#5-thanks-to)
  - [6. License](#6-license)

## 1. Introduction

This repository aims to combine the fastest feature extraction and matching methods, i.e. [XFeat](https://github.com/verlab/accelerated_features) and [LightGlue](https://github.com/cvg/LightGlue), and evaluate their performance on laptop and service devices.

## 2. Installation

### 2.1. Clone the repository

```bash
git clone https://github.com/LuoXubo/FastMatch.git
```

### 2.2. Install dependencies

```bash
conda create -n fastmatch python=3.8 -y
conda activate fastmatch
pip install -r requirements.txt
```

## 3. Usage

See `demo.ipynb` for a quick start.

## 4. Results

### 4.1 Time Comparison

- Test device: Ubuntu 20.04, RTX 3090 Ti
- Test image size: 800x600

| Method                   | Feature Extraction | Matching | Total Time |
| ------------------------ | ------------------ | -------- | ---------- |
| SP + LG                  | 0.060s             | 0.020s   | 0.080s     |
| Efficient LoFTR          | -                  | -        | 0.063s     |
| XFeat + mnn (sparse)     | 0.016s             | 0.0007s  | 0.0167s    |
| XFeat + mnn (semi-dense) | 0.170s             | 0.007    | 0.177s     |
| XFeat + LightGlue        | -                  | -        | -          |

TODO: XFeat + LightGlue

### 4.2 Video Matching

- Test device: Ubuntu 20.04, RTX 3090 Ti
- Test video size: 1280x720, 30fps, shot by iPhone 13
- Test video length: 12s

XFeat + mnn

![video](assets/xfeat+mnn.gif)

SuperPoint + LightGlue

![video](assets/sp+lg.gif)

Efficient LoFTR

![video](assets/eloftr.gif)

**Usage:**

```bash
python video_matching.py --ref=assets/groot/groot.jpg --tgt=assets/groot/groot.mp4 --method=sp+lg --save_path=assets/groot/groot_sp+lg.mp4
```

## 5. Thanks To

- [XFeat](https://github.com/verlab/accelerated_features)
- [LightGlue](https://github.com/cvg/LightGlue)

## 6. License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
