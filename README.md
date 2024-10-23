# Attention Recurrent Residual U-Net for Brain MR Image Segmentation
This repository contains an implementation of an Attention Recurrent Residual UNet (AR2-UNet) for semantic segmentation of brain MR images, specifically targeting the FLAIR abnormality regions.

--------------------------------------------------------------------------------------------------------------------------------
# Architecture Overview

![image](https://github.com/user-attachments/assets/aa60dece-c2ef-4df5-af34-1efcb9bc7d14)

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

The dataset used for this project comes from:

* Mateusz Buda, Ashirbani Saha, Maciej A. Mazurowski: "Association of genomic subtypes of lower-grade gliomas with shape features automatically extracted by a deep learning algorithm." Computers in Biology and Medicine, 2019.
  
* Maciej A. Mazurowski et al.: "Radiogenomics of lower-grade glioma: algorithmically-assessed tumor shape is associated with tumor genomic subtypes and patient outcomes in a multi-institutional study with The Cancer Genome Atlas data." Journal of Neuro-Oncology, 2017.
  
This dataset includes brain MR images and manual FLAIR abnormality segmentation masks. The images correspond to 110 patients in the The Cancer Genome Atlas (TCGA) lower-grade glioma collection.

link: https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation

--------------------------------------------------------------------------------------------------------------------------------

# Required Libraries
The following libraries were used:

torch==2.4.1
pillow==10.2.0
matplotlib==3.8.3




