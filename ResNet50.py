import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms, models
import RDI
import PGD_attack
import PGD_attack2
import os
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
import torchattacks
import RFGSM
import Square_Attack
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load ResNet50 model and modify structure to fit MNIST dataset (28x28 inputs, 10 class outputs)
class ResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet50, self).__init__()
        self.resnet = models.resnet50(weights=None)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False) 
        self.resnet.maxpool = nn.Identity()
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

# Data preprocessing: conversion to Tensor and normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)


model = ResNet50(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")


def test(model, test_loader):
    model.eval()
    all_outputs_before_softmax = []
    all_predictions = []
    correct = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)  
            all_outputs_before_softmax.append(outputs.cpu().numpy())  
            pred = outputs.argmax(dim=1, keepdim=True)
            all_predictions.append(pred.cpu().numpy())
            correct += pred.eq(labels.view_as(pred)).sum().item()

    accuracy = correct / len(test_loader.dataset)
    print(f'\nAccuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * accuracy:.2f}%)\n')

    return all_outputs_before_softmax, all_predictions


# train model
# num_epochs = 5
# train(model, train_loader, criterion, optimizer, num_epochs)


# save model
model_save_path = './models/AdvResNet50_mnist.pth'
# torch.save(model.state_dict(), model_save_path)
# print(f'Model saved to {model_save_path}')

loaded_model = ResNet50(num_classes=10).to(device)
loaded_model.load_state_dict(torch.load(model_save_path))
loaded_model.eval() 

# Test the model and get the output and predictions before the softmax layer
print("Evaluating on test data...")
outputs, predictions = test(loaded_model, test_loader)


class_num = 10
feature_vector = [[] for i in range(class_num)]

for i in range(100):
    for j in range (len(outputs[i])):
        feature_vector[predictions[i][j][0]].append(outputs[i][j])


# RDI
RDI = RDI.features(feature_vector)
print("RDI =", RDI)

# Testing model performance under PGD attack
print("Evaluating under PGD attack...")
PGD_attack.test_with_pgd_attack(loaded_model, test_loader)


# Testing model performance under RFGSM attacks
# print("Evaluating under RFGSM attack...")
# RFGSM.test_with_RFGSM_attack(loaded_model, test_loader)

# Testing model performance under Square attacks
# print("Evaluating under Square attack...")
# Square_Attack.test_with_square_attack(loaded_model, test_loader)
