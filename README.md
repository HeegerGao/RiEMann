<h2 align="center">
  <b>RiEMann: Near Real-Time SE(3)-Equivariant Robot Manipulation without Point Cloud Segmentation</b>

<div align="center">
    <a href="https://arxiv.org/abs/2403.19460" target="_blank">
    <img src="https://img.shields.io/badge/Paper-arXiv-green" alt="Paper ArXiv"></a>
    <a href="https://riemann-web.github.io/" target="_blank">
    <img src="https://img.shields.io/badge/Page-RiEMann-blue" alt="Project Page"/></a>
</div>
</h2>

This is the official code repository of **RiEMann: Near Real-Time SE(3)-Equivariant Robot Manipulation without Point Cloud Segmentation**.

<!-- For more information, please visit our [**project page**](). -->

## Overview

RiEMann is an SE(3)-equivariant robot manipulation algorithm that can generalize to novel SE(3) object poses with only 5 to 10 demonstrations.

![image](imgs/web_teaser.gif)


## Installation

### 1. Create virtual environment
```bash
conda create -n equi python==3.8
conda activate equi
```

Not sure if other python versions are OK.

### 2. Installation

1. Install [PyTorch](https://pytorch.org/). RiEMann is tested on CUDA version 11.7 and PyTorch version 2.0.1. Not sure if other versions are OK.

2. Install [torch-sparse](https://github.com/rusty1s/pytorch_sparse), [torch-scatter](https://github.com/rusty1s/pytorch_scatter), [torch-cluster](https://github.com/rusty1s/pytorch_cluster), and [dgl](https://www.dgl.ai/pages/start.html) according to their official installation guidance. We recommend to use the following commands to install:
```
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.1+${CUDA}.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.0.1+${CUDA}.html
pip install dgl -f https://data.dgl.ai/wheels/cu117/repo.html
pip install dglgo -f https://data.dgl.ai/wheels-test/repo.html
```

3. ```pip install -r requirements.txt```

## Data Preparation

Please put your data in the `data/{your_exp_name}`. We provide the demonstrations for the mug-pick experiment at `data/mug/pick`, both for training and testing.

The demonstration file is a .npz file and is in the following data structure:
```
{
  "xyz": np.array([traj_num, video_len, point_num, 3]),
  "rgb": np.array([traj_num, video_len, point_num, 3]),
  "seg_center": np.array([traj_num, video_len, 3]), 
  "axes": np.array([traj_num, video_len, 9])
}
```
where the *seg_center* is the target position, and *axes* the target rotation, representated in the unit orthonormal basis of 3 axes v1, v2, v3 of the rotation matrix: [v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z].

## Training

As stated in our paper, there is an SE(3)-invariant network $\phi$ that extracts the saliency map, and an SE(3)-equivariant network $\psi$ that leverages the saliency map to predict the target pose. **We must first train $\phi$, then train $\psi$.**

1. `python scripts/training/train_seg.py`

2. `python scripts/training/train_mani.py`

After these training, you will get a `seg_net.pth` and a `mani_net.pth` under `experiments/{your_exp_name}`.

Different hyperparameters in the config file leads to different performance, training speed, and memory cost. Have a try!

## Evaluation

Run `python scripts/testing/infer.py`. You can select the testing demonstrations in the input arguments. After this you will get a `pred_pose.npz` that records the predicted target pose.

We provide different scripts for result and feature visualization in `scripts/testing`.

## Citing
```
@inproceedings{gao2024riemann,
   title={RiEMann: Near Real-Time SE(3)-Equivariant Robot Manipulation without Point Cloud Segmentation},
   author={Gao, Chongkai and Xue, Zhengrong and Deng, Shuying and Liang, Tianhai and Yang, Siqi and Zhu, Yuke},
   booktitle={arXiv preprint arXiv:2403.19460},
   year={2024}
}
```