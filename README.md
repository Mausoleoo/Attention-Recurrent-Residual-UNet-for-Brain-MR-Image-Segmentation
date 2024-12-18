# Attention Recurrent Residual U-Net for Brain MR Image Segmentation

This repository contains an implementation of an Attention Recurrent Residual UNet (AR2-UNet) for semantic segmentation of brain MR images, specifically targeting the FLAIR abnormality regions. Identifying FLAIR abnormalities in brain MR images is critical in detecting and analyzing various neurological conditions, including tumors, strokes, and multiple sclerosis. A model like AR2-UNet, which is tailored to accurately capture these abnormal regions, **can significantly assist radiologists in making faster, more accurate diagnoses**. This contributes to more effective patient management, potentially allowing for earlier interventions and better patient outcomes.

What is FLAIR?

FLAIR, or Fluid-Attenuated Inversion Recovery, is an MRI technique that suppresses signals from cerebrospinal fluid, making it easier to see abnormalities in brain tissue. This helps highlight lesions and damage, useful for diagnosing conditions like multiple sclerosis, strokes, and tumors.

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

These results indicate that:

* Pixel Accuracy (99.33%) is notably high, suggesting that the model correctly labels a significant number of pixels.
  
* IoU (0.373) and Dice Coefficient (0.513) are moderate, showing some room for improvement. These metrics suggest that the model captures FLAIR abnormalities reasonably but may miss some finer details.
  
* Precision (0.704) and Recall (0.437) show a trade-off: the model is better at avoiding false positives than it is at capturing all relevant abnormal areas.

The model won’t display a mask if there is no damage in the brain.

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

# Future Work

To improve the model’s performance, consider:

* Dataset Augmentation: Increase the diversity of training images through augmentations (rotations, intensity variations) to improve recall.
* Refinement of Attention Mechanism: Experiment with different types of attention to enhance model focus on FLAIR abnormalities.
* Ensemble Models: Combine outputs from several architectures, such as standard U-Net and AR2-UNet, to boost performance.

--------------------------------------------------------------------------------------------------------------------------------

# Discussion 

Although the model was trained on images from various slices of the brain, it performed well in segmenting the FLAIR abnormalities. With more refined data preprocessing, the model's performance could likely be improved, potentially achieving even higher accuracy. The input images were normalized to values between 0 and 1, ensuring consistent data scaling for better convergence. While the model was trained for a total of 30 epochs, it reached satisfactory results between 10 and 15 epochs, indicating that fewer training iterations might be sufficient for effective segmentation.

The model was trained on Google Colab using an A100 GPU.


--------------------------------------------------------------------------------------------------------------------------------

# Resources

[1] Katsamenis, I., Protopapadakis, E., Bakalos, N., Doulamis, A., Doulamis, N., & Voulodimos, A. (2023). A few-shot attention recurrent residual U-Net for crack segmentation. arXiv. https://arxiv.org/abs/2303.01582
