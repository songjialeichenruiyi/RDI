# RDI: A adversarial robustness evaluation metric for deep neural networks based on model statistical features
## Overview
RDI is a adversarial robustness evaluation metric for deep neural networks based on model statistical features. This README provides an explanation of the functionality of the different codes in the repository and how to set up and run the method.
## Requirements
- **Python**: 3.9.0
- **Libraries**:
  - PyTorch 1.13.0
  - Torchvision 0.14.0
## Installation
- **Python Setup**: Ensure that you have the correct version of Python installed. If not, download and install it from [python390](https://www.python.org/downloads/release/python-390/)
- **Library Installation**: <br>
```bash
pip install torch==1.13.0 torchvision==0.14.0
```

## model_dataset.py
Code for training different models with natural samples.
## model_dataset_adv.py
Use PGD attack method to generate dversarial samples under different models and different data sets.
## model_dataset_adv_train.py
Code for adversarial training using adversarial examples.
## PGD_attack.py RFGSM.py Square_Attack.py
Codes for three attack methods.
## RDI.py
The code of the RDI calculation method needs to pass in the feature vectors, and the feature vectors extraction is placed in the code of each model training.
