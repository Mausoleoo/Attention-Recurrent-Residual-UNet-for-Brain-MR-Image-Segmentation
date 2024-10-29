# Attention Recurrent Residual U-Net for Brain MR Image Segmentation
This repository contains an implementation of an Attention Recurrent Residual UNet (AR2-UNet) for semantic segmentation of brain MR images, specifically targeting the FLAIR abnormality regions.

--------------------------------------------------------------------------------------------------------------------------------
# Architecture Overview

![image](https://github.com/user-attachments/assets/aa60dece-c2ef-4df5-af34-1efcb9bc7d14)
Image taken from [1]

The architecture builds upon the U-Net model by incorporating:

Recurrent Residual Blocks: To capture temporal dependencies between convolutions.

Attention Mechanisms: To enhance feature selection and focus on the relevant regions during upsampling.

Deconvolutional Layers: For upsampling during the decoder phase.

--------------------------------------------------------------------------------------------------------------------------------

# Results and Visualization

### Model Metrics 
The model was trained on the provided dataset, achieving the following performance metrics:

* IoU: 0.373
* Dice Coefficient: 0.513
* Pixel Accuracy: 99.33%
* Precision: 0.704
* Recall: 0.437

### Visualization

![image](https://github.com/user-attachments/assets/4a5f2532-ec2e-489d-af05-e5efb5cf0aea)


--------------------------------------------------------------------------------------------------------------------------------

# Dataset
 
This dataset includes brain MR images and manual FLAIR abnormality segmentation masks. The images correspond to 110 patients in the The Cancer Genome Atlas (TCGA) lower-grade glioma collection.

link: https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation

--------------------------------------------------------------------------------------------------------------------------------

# Required Libraries
The following libraries were used:

torch==2.4.1

pillow==10.2.0

matplotlib==3.8.3

--------------------------------------------------------------------------------------------------------------------------------

# Code details

The code can be applied to both 1-dimensional (grayscale) and 3-dimensional (RGB) images. Additionally, it supports training with more than one class by adjusting the model parameters (input_channels for the number of input channels, and num_classes for the number of output classes).

model = AttentionR2UNet(input_channels=3, num_classes=1)

**Note**: If using more than one class, the loss function should be changed to CrossEntropyLoss instead of BCELoss.

--------------------------------------------------------------------------------------------------------------------------------

# Discussion 

Although the model was trained on images from various slices of the brain, it performed well in segmenting the FLAIR abnormalities. With more refined data preprocessing, the model's performance could likely be improved, potentially achieving even higher accuracy. The input images were normalized to values between 0 and 1, ensuring consistent data scaling for better convergence. While the model was trained for a total of 30 epochs, it reached satisfactory results between 10 and 15 epochs, indicating that fewer training iterations might be sufficient for effective segmentation.

The model was trained on Google Colab using an A100 GPU.

--------------------------------------------------------------------------------------------------------------------------------

# Resources

[1] Katsamenis, I., Protopapadakis, E., Bakalos, N., Doulamis, A., Doulamis, N., & Voulodimos, A. (2023). A few-shot attention recurrent residual U-Net for crack segmentation. arXiv. https://arxiv.org/abs/2303.01582
