# Semantic Segmentation using FCCNs (Fully Complex-valued Convolutional Networks)

## Overview

This project explores the application of Fully Complex-valued Convolutional Networks (FCCNs) for semantic segmentation tasks. Leveraging the ability of complex numbers to encapsulate both magnitude and phase information, this work aims to address the gap in complex neural network research, particularly for semantic segmentation.

The core idea involves:
1.  **Input Complexification:** Transforming real-valued input images (RGB) into a complex-valued representation using the iHSV color model methodology. This involves converting RGB to HSV and then deriving three orthogonal complex planes ($I_h^*, I_s^*, I_v^*$).
2.  **FCCN Model:** Utilizing a network architecture composed entirely of complex-valued layers (convolutional layers, activations, batch normalization) with complex weights and biases. This ensures phase information is preserved throughout the network.
3.  **Semantic Segmentation:** Applying the FCCN model to perform per-pixel classification on the complexified input data.
4.  **Evaluation:** Assessing the FCCN model's performance using mIoU and Cross Entropy Loss metrics and comparing it against real-valued counterparts.

## Contributors

* Srijan Anand, IIT Kanpur
* Ayushi Mehta, IIT Kanpur
* Prof. Koteswar Rao Jerripothula (Mentor), IIT Kanpur

## Methodology

### iHSV Color Complexification

Real-valued RGB images are first converted to the HSV color space. The iHSV methodology is then applied to derive three complex-valued channels based on fixing H, S, and V respectively:
* Fixed Hue ($H=I_h$): $I_h^*(p) = I_v(p) + i I_s(p)$ (V as real, S as imaginary)
* Fixed Saturation ($S=I_s$): $I_s^*(p) = I_s(p)I_h(p) + i I_v(p)$ (Arc length $S \times H$ as real, V as imaginary)
* Fixed Value ($V=I_v$): $I_v^*(p) = I_s(p)\cos(I_h(p)) + i I_s(p)\sin(I_h(p))$ ($S\cos H$ as real, $S\sin H$ as imaginary)
These three complex channels form the complex-valued input $I = I_{re} + i I_{im}$ for the network. This conversion is one-to-one and invertible.

### FCCN Architecture

* **Encoder:** A Complex-valued ResNet50 model, pre-trained on ImageNet (classification layers removed), serves as the encoder.
* **Decoder:** Uses complex-valued transposed convolutions to upscale the feature maps from the encoder back to the original input size, producing output maps corresponding to the number of classes.
* **Complex Layers:** The network utilizes custom complex-valued layers including convolutions (`ComplexConv2d`), transposed convolutions (`ComplexTranspose2d`), batch normalization (`ComplexNaiveBatchNorm2d`, `ComplexBatchNorm2d`), activations (`CReLU`), and pooling (`ComplexMaxPool2d`).
* **Skip Connections:** Implemented to fuse features from different stages (coarse, high-level information with fine, low-level information) using element-wise addition, enhancing segmentation accuracy.
* **Output:** The final complex-valued output is converted to real values by taking the absolute magnitude (`abs()`) for loss calculation and prediction.

## Implementation


### Key Files & Logic

* `complexnn.py`: Core implementations of complex-valued convolutional layers, batch normalization, pooling, and linear layers. Follows the standard approach of applying real-valued operations to real and imaginary parts separately based on complex multiplication rules.
* `complex_activations.py`: Defines complex activation functions like CReLU, which applies ReLU independently to real and imaginary parts.
* `networks.py`: Contains the implementation of the Complex ResNet architecture (using `BasicBlock` or `Bottleneck`defined with complex layers) and potentially other complex architectures like AlexNet and VGG.
* `train*.py` scripts: Handle the data loading (e.g., `DroneDataset` or `VocDataset`), preprocessing (including RGB to complex conversion using `rgb_to_hsv_mine`, `ToHSV`, `ToComplex`), training loop (`fit` function), loss calculation (CrossEntropyLoss), optimization (AdamW or SGD), and evaluation (mIoU, pixel accuracy).
* `utils.py`: Provides helper functions, notably the `rgb_to_hsv_mine`, `ToHSV`, and `ToComplex` functions essential for the input complexification pipeline.

## Dataset

The primary dataset used for training and evaluation is the **Semantic Drone Dataset** provided by the Institute of Computer Graphics and Vision at TU Graz.
* Content: Urban scenes from a nadir view (5-30m altitude).
* Resolution: 6000x4000 pixels.
* Size: 400 training images, 200 test images.
* Classes: 23.

A version trained on the PASCAL VOC 2012 dataset (`train_voc.py`) is also included.

## Training Details

* **Loss Function:** Cross Entropy Loss.
* **Optimizer:** AdamW or SGD.
* **Scheduler:** OneCycleLR or ReduceLROnPlateau.
* **Epochs:** 80 epochs reported in the paper, other scripts might use different numbers (e.g., 15, 70, 100, 50).
* **Learning Rate:** Max LR 1e-3.
* **Framework:** PyTorch, using `torch.complex64` for complex operations.

## Evaluation & Results

* **Metrics:** Mean Intersection over Union (mIoU) and Cross Entropy Loss. Pixel accuracy is also used in some training scripts.
* **Performance (Semantic Drone Dataset):**
    * Training Loss: ~0.609
    * Training mIoU: ~0.511
* **Comparison:** The ResNet50-FCCN model achieved an mIoU of 0.511, outperforming real-valued VGG-UNet (0.286 mIoU) and UNet-MobileNet on the same dataset based on reported Kaggle scores.

## References

* FCCNs: Fully Complex-valued Convolutional Networks using Complex-valued Color Model and Loss Function Saurabh Yadav, Koteswar Rao Jerripothula.
* Fully Convolutional Networks for Semantic Segmentation - Jonathan Long, Evan Shelhamer, Trevor Darrell.
* Semantic Drone Dataset - Institute of Computer Graphics & Vision at TU Graz.
* Kaggle (for comparison model scores).
