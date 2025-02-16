import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models
import RDI


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

class ResNet101(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet101, self).__init__()
        self.resnet = models.resnet101(weights=None)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)
    
    
model = ResNet101().to(device)
model_save_path = './models/FashionMNIST/Res101_FashionMNIST.pth'
model.load_state_dict(torch.load(model_save_path))
model.eval()

class AdversarialDataset(Dataset):
    def __init__(self, adv_data_file, label_file, transform=None):
        self.adv_data = np.load(adv_data_file)
        self.labels = np.load(label_file) 
        self.transform = transform

    def __len__(self):
        return len(self.adv_data)

    def __getitem__(self, idx):
        image = torch.tensor(self.adv_data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

transform = transforms.Compose([
    transforms.RandomCrop(28, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.Normalize((0.1307,), (0.3081,))
])

transform2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

trainset = AdversarialDataset(adv_data_file="./attackData/FashionMNIST/ResNet101_datas.npy", label_file="./attackData/FashionMNIST/ResNet101_labels.npy", transform=transform)
train_loader = DataLoader(trainset, batch_size=64, shuffle=True)

test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform2)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.000000001, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5) 

def adv_train(model, train_loader, criterion, optimizer, num_epochs):
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
        scheduler.step()

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

    return all_outputs_before_softmax, all_predictions


# num_epochs = 5
# adv_train(model, train_loader, criterion, optimizer, num_epochs)

# save model
model_save_path2 = './models/FashionMNIST/AdvResNet101_FashionMNIST.pth'
# torch.save(model.state_dict(), model_save_path2)
# print(f'Model saved to {model_save_path2}')


loaded_model = ResNet101().to(device)
loaded_model.load_state_dict(torch.load(model_save_path2))
loaded_model.eval()

print("Testing loaded model...")
outputs, predictions = test(loaded_model, test_loader)


class_num = 10
feature_vector = [[] for i in range(class_num)]

for i in range(10):
    for j in range (len(outputs[i])):
        feature_vector[predictions[i][j][0]].append(outputs[i][j])

# RDI
RDI = RDI.features(feature_vector)
print("RDI = ", RDI)
