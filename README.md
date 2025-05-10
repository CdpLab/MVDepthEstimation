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
