import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader
import RDI
import PGD_attack
import PGD_attack2
import RFGSM
import Square_Attack
import CW_attack


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Customize the DenseNet161 model class
class DenseNet161_MNIST(nn.Module):
    def __init__(self, num_classes=10):
        super(DenseNet161_MNIST, self).__init__()
        self.model = models.densenet161(weights = None)
        
        self.model.features.conv0 = nn.Conv2d(
            1, 
            96, 
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        self.model.features.pool0 = nn.Identity()
        
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
    
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=200, shuffle=False)


DenseNet161 = DenseNet161_MNIST(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(DenseNet161.parameters(), lr=0.001)


def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

def test_model(model, test_loader):
    model.eval()
    all_outputs_before_softmax = []
    all_predictions = []
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            all_outputs_before_softmax.append(outputs.cpu().numpy())  # 保存softmax层之前的输出
            pred = outputs.argmax(dim=1, keepdim=True)
            all_predictions.append(pred.cpu().numpy())  # 保存所有
            correct += pred.eq(labels.view_as(pred)).sum().item()

    accuracy = correct / len(test_loader.dataset)
    print(f'\nAccuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * accuracy:.2f}%)\n')

    return all_outputs_before_softmax, all_predictions



# num_epochs = 2
# train_model(DenseNet161, train_loader, criterion, optimizer, num_epochs)


model_save_path = './models/FashionMNIST/AdvDenseNet161_FashionMNIST.pth'
# torch.save(DenseNet161.state_dict(), model_save_path)
# print(f'Model saved to {model_save_path}')

loaded_model = DenseNet161_MNIST(num_classes=10).to(device)
loaded_model.load_state_dict(torch.load(model_save_path))
loaded_model.eval()

print("Evaluating on test data...")
outputs, predictions = test_model(loaded_model, test_loader)

class_num = 10
feature_vector = [[] for i in range(class_num)]

for i in range(50):
    for j in range (len(outputs[i])):
        feature_vector[predictions[i][j][0]].append(outputs[i][j])


# RDI
RDI = RDI.features(feature_vector)
print("RDI = ", RDI)

# Testing model performance under PGD attack
print("Evaluating under PGD attack...")
PGD_attack.test_with_pgd_attack(loaded_model, test_loader)

# Testing model performance under RFGSM attack
# print("Evaluating under RFGSM attack...")
# RFGSM.test_with_RFGSM_attack(loaded_model, test_loader)

# Testing model performance under Square attack
# print("Evaluating under Square attack...")
# Square_Attack.test_with_square_attack(loaded_model, test_loader)

# Testing model performance under CW attack
# print("Evaluating under CW attack...")
# CW_attack.test_with_cw_attack(loaded_model, test_loader, 1, 0.01, 50)
