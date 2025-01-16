import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchattacks  # 引入 torchattacks 库
import os
from torchvision.transforms import ToPILImage
import numpy as np


import matplotlib.pyplot as plt


# 检查是否有可用的GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 使用PGD攻击测试模型的性能
def test_with_pgd_attack(model, test_loader, epsilon=0.3, alpha=0.01, iters=40):
    model.eval()
    attacks = torchattacks.PGD(model, epsilon, alpha, iters)  # 初始化PGD攻击

    correct = 0
    total = 0

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        adv_images = attacks(images, labels)  # 生成对抗样本
        outputs = model(adv_images)
        
        _, predicted = torch.max(outputs, 1)
        print(predicted)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f'Accuracy under PGD attack: {accuracy:.2f}%')


    accuracy = 100 * correct / total
    print(f'Accuracy under PGD attack: {accuracy:.2f}%')

# 使用PGD攻击测试模型的性能
def save_pgd_attack(model, test_loader, epsilon=0.3, alpha=0.01, iters=40):
    attack = torchattacks.PGD(model, epsilon, alpha, iters)
    
    # 4. 生成对抗样本并保存
    adv_data = []
    labels = []
    model.eval()
    size = 0
    for images, targets in test_loader:
        images, targets = images.to(device), targets.to(device)
        adv_images = attack(images, targets)
        adv_data.append(adv_images.cpu().numpy())  # 保存对抗样本
        labels.append(targets.cpu().numpy())  # 
        size = size + 1
        print(str(size) + '/938')

    # 保存对抗样本和标签
    adv_data = np.concatenate(adv_data, axis=0)
    labels = np.concatenate(labels, axis=0)
    return adv_data, labels





