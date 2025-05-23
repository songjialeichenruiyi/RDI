# RDI: An adversarial robustness evaluation metric for deep neural networks based on sample clustering features
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
## Dataset
### MNIST
Download and prepare the MNIST dataset:
```python
import torchvision
import torchvision.transforms as transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
```
### FashionMNIST
Download and prepare the FashionMNIST dataset:
```python
import torchvision
import torchvision.transforms as transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
```

### CIFAR10
Download and prepare the CIFAR10 dataset:
```python
import torchvision
import torchvision.transforms as transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.2435, 0.2616))
])
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
```

### CIFAR100
Download and prepare the CIFAR100 dataset:
```python
import torchvision
import torchvision.transforms as transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
])
train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
```

### Tiny-ImageNet
Download the ImageNet dataset from the following link: [Tiny-ImageNet](http://cs231n.stanford.edu/tiny-imagenet-200.zip)

### SpeechCommand
The dataset needs to be loaded and preprocessed. Detailed procedures can be found in the data processing section of Speech_XX.py.


## Explanation of the functionality of the different codes
### model_dataset.py
The codes for training and testing different models using natural samples is provided, such as **AlexNet_Cifar10.py, AlexNet.py, and Speech_CNN.py**. In the naming convention, a filename like AlexNet.py without a dataset name indicates that the model is trained on the MNIST dataset. Files prefixed with Speech, such as Speech_CNN.py, correspond to models using the SpeechCommand speech recognition dataset. The same naming scheme is followed throughout the remaining code.
### model_dataset_adv.py
Use PGD attack method to generate dversarial samples under different models and different datasets, such as **AlexNet_Cifar10_adv.py, AlexNet_adv.py, etc.**
### model_dataset_adv_train.py
Codes for adversarial training using adversarial examples, such as **AlexNet_Cifar10_adv_train.py, AlexNet_adv_train.py, etc.**
### PGD_attack.py, RFGSM.py, Square_Attack.py, CW_attack.py, SirenAttack.py
Codes for five attack methods. SirenAttack is an attack method that targets language recognition models.
### RDI.py
The code of the RDI calculation method needs to pass in the feature vectors, and the feature vectors extraction is placed in the codes of each model training (**model_dataset.py and model_dataset_adv_train.py**).

## Usage
After preparing the dataset and the corresponding model (the model structure needs to be the same as the one in the codes), modify the paths for the model and dataset in **model_dataset.py and model_dataset_adv_train.py**, then run the codes directly to obtain the model's RDI value. By adjusting the comments in both types of codes, you can also perform model training.
