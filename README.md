<h2 align="center">
  <b>RiEMann: Near Real-Time SE(3)-Equivariant Robot Manipulation without Point Cloud Segmentation</b>

<!-- <div align="center">
    <a href="" target="_blank">
    <img src="https://img.shields.io/badge/Paper-arXiv-green" alt="Paper arXiv"></a>
    <a href="" target="_blank">
    <img src="https://img.shields.io/badge/Page-RiEMann-blue" alt="Project Page"/></a>
</div> -->
</h2>

This is the official repository of **RiEMann: Near Real-Time SE(3)-Equivariant Robot Manipulation without Point Cloud Segmentation**.

For more information, please visit our [**project page**](https://github.com/HeegerGao/RiEMann).

## Overview

## Installation

Please follow the steps below to perform the installation:

### 1. Create virtual environment
```bash
conda create -n equi python==3.8
conda activate equi
```

### 2. Install Special Dependencies
Install [torch-sparse](https://github.com/rusty1s/pytorch_sparse), [torch-scatter](https://github.com/rusty1s/pytorch_scatter), [torch-cluster](https://github.com/rusty1s/pytorch_cluster), and [dgl](https://www.dgl.ai/pages/start.html) according to their official installation guidance.

### 3. Install General Dependiencies

```pip install -r requirements.txt```

## Run

### Data

Please put your data in the `data/exp_name`.

## Training

`python scripts/training/train_seg.py`

`python scripts/training/train_mani.py`

## Evaluation

`python scripts/testing/infer.py`



