import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchattacks 
import os
from torchvision.transforms import ToPILImage

import matplotlib.pyplot as plt


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Test the performance of the model using RFGSM attack
def test_with_RFGSM_attack(model, test_loader, epsilon=0.3, alpha=0.01, iters = 40):
    model.eval()
    attacks = torchattacks.RFGSM(model, epsilon, alpha, iters)  # Initializing RFGSM Attacks

    correct = 0
    total = 0

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        adv_images = attacks(images, labels)
        outputs = model(adv_images)
        
        _, predicted = torch.max(outputs, 1)
        print(predicted)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f'Accuracy under RFGSM attack: {accuracy:.2f}%')


    accuracy = 100 * correct / total
    print(f'Accuracy under RFGSM attack: {accuracy:.2f}%')





