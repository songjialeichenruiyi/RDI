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

class ModifiedMobileNetV2(nn.Module):
    def __init__(self, num_classes=10):
        super(ModifiedMobileNetV2, self).__init__()
        self.model = models.mobilenet_v2(weights=None)
        self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)

    def forward(self, x):
        return self.model(x)
    
model = ModifiedMobileNetV2().to(device)
model_save_path = './models/MobileNetV2_mnist3.pth'
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
    transforms.Normalize((0.1307,), (0.3081,))
])

transform2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

trainset = AdversarialDataset(adv_data_file="./attackData/MNIST/MobileNetV2_datas.npy", label_file="./attackData/MNIST/MobileNetV2_labels.npy", transform=transform)
train_loader = DataLoader(trainset, batch_size=64, shuffle=True)

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform2)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# training settings
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
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


# num_epochs = 4
# adv_train(model, train_loader, criterion, optimizer, num_epochs)

model_save_path2 = './models/AdvMobileNetV2_mnist.pth'
# torch.save(model.state_dict(), model_save_path2)
# print(f'Model saved to {model_save_path2}')

loaded_model = ModifiedMobileNetV2().to(device)
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
