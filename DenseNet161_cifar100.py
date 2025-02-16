import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import RDI
import PGD_attack
import RFGSM
import Square_Attack
import time


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

class ModifiedDenseNet161(nn.Module):
    def __init__(self, num_classes=100): 
        super(ModifiedDenseNet161, self).__init__()
        self.model = models.densenet161(pretrained=True) 
        self.model.features.conv0 = nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.features.pool0 = nn.Identity()
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes) 

    def forward(self, x):
        return self.model(x)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)) 
])

train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)


model = ModifiedDenseNet161(num_classes=100).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0002, weight_decay=5e-4)


def train(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad() 
            outputs = model(images) 
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
            loss = criterion(outputs, labels) 
            loss.backward() 
            optimizer.step() 
            
            running_loss += loss.item()

        accuracy = correct / len(train_loader.dataset)
        print(f'Accuracy: {correct}/{len(train_loader.dataset)} '
              f'({100. * accuracy:.2f}%)')
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")

def test(model, test_loader):
    model.eval() 
    correct = 0
    all_outputs_before_softmax = []
    all_predictions = []
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

    return all_outputs_before_softmax, all_predictions, accuracy



# num_epochs = 2
# train(model, train_loader, criterion, optimizer, num_epochs)

model_save_path = 'boundary_robustness/models/cifar100/DenseNet161_cifar100_3.pth'
# torch.save(model.state_dict(), model_save_path)
# print(f'Model saved to {model_save_path}')


loaded_model = ModifiedDenseNet161(num_classes=100).to(device)
loaded_model.load_state_dict(torch.load(model_save_path))
loaded_model.eval()

print("Testing loaded model...")
outputs, predictions, acc = test(loaded_model, test_loader)


class_num = 100
feature_vector = [[] for i in range(class_num)]

for i in range(100):
    for j in range(len(outputs[i])):
        feature_vector[predictions[i][j][0]].append(outputs[i][j])

# RDI
RDI = myMethod.features(feature_vector)
print("RDI =", RDI)

# Testing model performance under PGD attack
# print("Evaluating under PGD attack...")
# PGD_attack.test_with_pgd_attack(loaded_model, test_loader, 0.025, 0.001, 10)

# Testing model performance under RFGSM attack
# print("Evaluating under RFGSM attack...")
# RFGSM.test_with_RFGSM_attack(loaded_model, test_loader,  0.025, 0.001, 10)

# Testing model performance under Square attack
print("Evaluating under Square attack...")
Square_Attack.test_with_square_attack(loaded_model, test_loader, n_queries=200, eps=0.025)
