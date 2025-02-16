import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchattacks  # Introducing the torchattacks library
import os
from torchvision.transforms import ToPILImage
import numpy as np


import matplotlib.pyplot as plt


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Test model performance using PGD attack
def test_with_pgd_attack(model, test_loader, epsilon=0.3, alpha=0.01, iters=40):
    model.eval()
    attacks = torchattacks.PGD(model, epsilon, alpha, iters)  # Initialize PGD attacks

    correct = 0
    total = 0

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        adv_images = attacks(images, labels)  # Generating Adversarial Samples
        outputs = model(adv_images)
        
        _, predicted = torch.max(outputs, 1)
        print(predicted)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f'Accuracy under PGD attack: {accuracy:.2f}%')


    accuracy = 100 * correct / total
    print(f'Accuracy under PGD attack: {accuracy:.2f}%')

# Testing Model Performance with PGD Attack
def save_pgd_attack(model, test_loader, epsilon=0.3, alpha=0.01, iters=40):
    attack = torchattacks.PGD(model, epsilon, alpha, iters)
    
    # Generate adversarial samples and save
    adv_data = []
    labels = []
    model.eval()
    size = 0
    for images, targets in test_loader:
        images, targets = images.to(device), targets.to(device)
        adv_images = attack(images, targets)
        adv_data.append(adv_images.cpu().numpy())
        labels.append(targets.cpu().numpy())
        size = size + 1
        print(str(size) + '/938')

    # 保存对抗样本和标签
    adv_data = np.concatenate(adv_data, axis=0)
    labels = np.concatenate(labels, axis=0)
    return adv_data, labels





