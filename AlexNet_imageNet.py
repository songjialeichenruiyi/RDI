import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
import RDI
import PGD_attack
import RFGSM
import Square_Attack
import os
from PIL import Image
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

class ModifiedAlexNet(nn.Module):
    def __init__(self, num_classes=200):
        super(ModifiedAlexNet, self).__init__()
        self.model = models.alexnet(pretrained=True)
        # self.model.features[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        # self.model.features[2] = nn.Identity()
        self.model.classifier[6] = nn.Linear(4096, num_classes) 

    def forward(self, x):
        return self.model(x)

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10), 
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))
])


dataset_path = "./boundary_robustness/data/tiny-imagenet-200"
train_dir = os.path.join(dataset_path, 'train')
val_dir = os.path.join(dataset_path, 'val')

# Custom validation dataset class
class TinyImageNetValDataset(Dataset):
    def __init__(self, val_dir, transform=None):
        self.val_dir = val_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Read the val_annotations.txt file
        annotations_file = os.path.join(val_dir, 'val_annotations.txt')
        with open(annotations_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                image_name = parts[0]
                label = parts[1]
                self.image_paths.append(os.path.join(val_dir, 'images', image_name))
                self.labels.append(label)
        
        # Get category name to index mapping
        label_names = sorted(list(set(self.labels)))
        self.label_to_idx = {label: idx for idx, label in enumerate(label_names)}
        
        # Converting labels to indexes
        self.labels = [self.label_to_idx[label] for label in self.labels]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return image, label


# Load training dataset
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)

# Load val dataset
test_dataset = TinyImageNetValDataset(val_dir, transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)


model = ModifiedAlexNet(num_classes=200).to(device)
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0002, weight_decay=1e-4)
optimizer = optim.SGD(model.parameters(), lr=0.004, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

# training function
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

        scheduler.step()

# testing function
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

# train models
# num_epochs = 10
# train(model, train_loader, criterion, optimizer, num_epochs)

model_save_path = '/models/imageNet/AlexNet_imageNet_3.pth'
# torch.save(model.state_dict(), model_save_path)
# print(f'Model saved to {model_save_path}')

loaded_model = ModifiedAlexNet(num_classes=200).to(device)
loaded_model.load_state_dict(torch.load(model_save_path))
loaded_model.eval()

print("Testing loaded model...")
outputs, predictions, acc = test(loaded_model, test_loader)

class_num = 200
feature_vector = [[] for i in range(class_num)]

for i in range(100):
    for j in range(len(outputs[i])):
        feature_vector[predictions[i][j][0]].append(outputs[i][j])

# RDI
RDI = RDI.features(feature_vector)
print("RDI =", RDI)

# Testing model performance under PGD attack
print("Evaluating under PGD attack...")
PGD_attack.test_with_pgd_attack(loaded_model, test_loader, 0.01, 0.001, 10)

# Testing model performance under RFGSM attack
# print("Evaluating under RFGSM attack...")
# RFGSM.test_with_RFGSM_attack(loaded_model, test_loader, 0.01, 0.001, 10)

# Testing model performance under Square attack attack
# print("Evaluating under Square attack...")
# Square_Attack.test_with_square_attack(loaded_model, test_loader, n_queries=200, eps=0.01)
