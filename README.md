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

Test image: 800x600

| Method                   | Feature Extraction | Matching | Total Time |
| ------------------------ | ------------------ | -------- | ---------- |
| SP + LG                  | 0.06s              | 0.02s    | 0.08s      |
| XFeat + mnn (sparse)     | 0.016s             | 0.0007s  | 0.0167s    |
| XFeat + mnn (semi-dense) | 0.17s              | 0.007    | 0.177s     |

## 5. Thanks To

- [XFeat](https://github.com/verlab/accelerated_features)
- [LightGlue](https://github.com/cvg/LightGlue)

## 6. License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
