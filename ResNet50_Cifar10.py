import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import RDI
import PGD_attack
import RFGSM
import Square_Attack

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

class ModifiedResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super(ModifiedResNet50, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.2435, 0.2616))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

model = ModifiedResNet50(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum = 0.9, weight_decay=1e-4)

def train(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        correct = 0
        running_loss = 0.0
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

# train model
# num_epochs = 25
# train(model, train_loader, criterion, optimizer, num_epochs)

# save model
model_save_path = './models/cifar10/AdvResNet50_cifar10.pth'
# torch.save(model.state_dict(), model_save_path)
# print(f'Model saved to {model_save_path}')

loaded_model = ModifiedResNet50().to(device)
loaded_model.load_state_dict(torch.load(model_save_path))
loaded_model.eval()

print("Testing loaded model...")
outputs, predictions, acc = test(loaded_model, test_loader)

class_num = 10
feature_vector = [[] for i in range(class_num)]

for i in range(100):
    for j in range (len(outputs[i])):
        feature_vector[predictions[i][j][0]].append(outputs[i][j])


# RDI
RDI = RDI.features(feature_vector)
print("RDI = ", RDI)

# Testing the performance of a model using PGD Attack
print("Evaluating under PGD attack...")
PGD_attack.test_with_pgd_attack(loaded_model, test_loader, 0.1, 0.0025, 10)

# Testing the performance of a model using RFGSM Attack
# print("Evaluating under RFGSM attack...")
# RFGSM.test_with_RFGSM_attack(loaded_model, test_loader, 0.1, 0.0025, 10)

# Testing the performance of a model using Square Attack
# print("Evaluating under Square attack...")
# Square_Attack.test_with_square_attack(loaded_model, test_loader, n_queries=200, eps=0.1)
