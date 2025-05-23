import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchattacks  # 引入 torchattacks 库
import os

import matplotlib.pyplot as plt

# 检查是否有可用的GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 使用CW攻击测试模型的性能
def test_with_cw_attack(model, test_loader, c=1, lr=0.01, iters=50):
    model.eval()
    attacks = torchattacks.CW(model, c=c, steps=iters, lr=lr)  # 初始化CW攻击

    correct = 0
    total = 0

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        adv_images = attacks(images, labels)  # 生成对抗样本
        outputs = model(adv_images)
        
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f'Accuracy under CW attack: {accuracy:.2f}%')

    accuracy = 100 * correct / total
    print(f'Final Accuracy under CW attack: {accuracy:.2f}%')
