# FastMatch

How fast can image matching be?

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Thanks To](#thanks-to)
- [License](#license)

## Introduction

This repository aims to combine the fastest feature extraction and matching methods, i.e. [XFeat](https://github.com/verlab/accelerated_features) and [LightGlue](https://github.com/cvg/LightGlue), and evaluate their performance on laptop and service devices.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/LuoXubo/FastMatch.git
```

### 2. Install dependencies

```bash
conda create -n fastmatch python=3.8 -y
conda activate fastmatch
pip install -r requirements.txt
```

## Usage

See `demo.ipynb` for a quick start.

## Results

## Thanks To

- [XFeat](https://github.com/verlab/accelerated_features]
- [LightGlue](https://github.com/cvg/LightGlue]

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
