# Multi-view Depth Estimation with Uncertainty Constraints for Virtual-Real Occlusion                                                                                                                              

## Overview

This repository contains the implementation of the multi-view depth estimation model proposed in the paper *"Multi-view Depth Estimation with Uncertainty Constraints for Virtual-Real Occlusion"* . The model addresses occlusion handling in Augmented Reality (AR) applications by improving depth estimation accuracy, particularly at object contours, using uncertainty constraints.

### Key Features

- **Bayesian Convolutional Uncertainty Estimation (BCUE)**: Enhances depth contour accuracy by modeling uncertainty in complex scenes.
- **Attention Spatial Convolution Fusion (ASCF)**: Combines multi-head self-attention and spatial-channel reconstruction convolution to improve feature extraction.
- **Depth Edge Padding (DEP)**: An optional post-processing module to refine depth map edges for better contour alignment.
- **Datasets**: Evaluated on ScanNetV2 and 7Scenes, achieving superior performance in depth estimation and virtual-real occlusion tasks.

## Installation

### Prerequisites

- Python 3.8+
- Conda (Anaconda or Miniconda)
- CUDA-enabled GPU (e.g., NVIDIA RTX3090)
- Optional: Segment Anything Model (SAM) for DEP module.

### Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/sdfsfwe/Virtual-Real-Occlusion.git
   cd Virtual-Real-Occlusion
   ```

2. **Create and activate the Conda environment**:

   The required dependencies are specified in the `env.yaml` file. To set up the environment:

   ```bash
   conda env create -f env.yaml
   conda activate multiview-depth
   ```

   This will install all necessary packages, including PyTorch, torchvision, NumPy, OpenCV, and others specified in `env.yaml`.

3. **Verify installation**:

   Ensure the environment is set up correctly by running:

   ```bash
   python -c "import torch; print(torch.__version__)"
   ```

4. **Download and preprocess datasets** (ScanNetV2, 7Scenes):

   Follow instructions in `data/README.md` to set up the datasets.

## Training

By default models and tensorboard event files are saved to `~/tmp/tensorboard/<model_name>`.
This can be changed with the `--log_dir` flag.

We train with a batch_size of 16 with 16-bit precision on two RTX3090 on the default ScanNetv2 split.

Example command to train with two GPUs:

```shell
CUDA_VISIBLE_DEVICES=0,1 python train.py --name HERO_MODEL \
            --log_dir logs \
            --config_file configs/models/hero_model.yaml \
            --data_config configs/data/scannet_default_train.yaml \
            --gpus 2 \
            --batch_size 16;
```

The code supports any number of GPUs for training.
You can specify which GPUs to use with the `CUDA_VISIBLE_DEVICES` environment.

**Different dataset**

You can train on a custom MVS dataset by writing a new dataloader class which inherits from `GenericMVSDataset` at `datasets/generic_mvs_dataset.py`. See the `ScannetDataset` class in `datasets/scannet_dataset.py` or indeed any other class in `datasets` for an example.
