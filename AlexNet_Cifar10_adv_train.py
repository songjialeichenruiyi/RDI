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

class ModifiedAlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ModifiedAlexNet, self).__init__()
        self.model = models.alexnet(weights = None)
        self.model.features[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.model.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.model(x)
    

model = ModifiedAlexNet().to(device)
model_save_path = './models/cifar10/AlexNet_cifar10_2.pth'

model.load_state_dict(torch.load(model_save_path))
model.eval()

#Custom dataset class for loading adversarial samples
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
    # transforms.RandomCrop(32, padding=4),  
    # transforms.RandomHorizontalFlip(),    
    # transforms.RandomRotation(15),         
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.2435, 0.2616))
])

transform3 = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  
    transforms.RandomHorizontalFlip(),    
    transforms.RandomRotation(15),        
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.2435, 0.2616))
])

transform2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.2435, 0.2616))
])


trainset = AdversarialDataset(adv_data_file="./attackData/Cifar10/AlexNet_datas.npy", label_file="./attackData/Cifar10/AlexNet_labels.npy", transform=transform)
adv_train_loader = DataLoader(trainset, batch_size=64, shuffle=True)

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform3)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform2)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


def mixed_train_loader(original_loader, adv_loader):
    """
    Take all data from original_loader and adv_loader to form a mixed batch
    Args:
        original_loader: Original Data Loader
        adv_loader: Adversarial Data loader
    Yields:
        mixed_images:
        mixed_labels:
    """
    for (orig_images, orig_labels), (adv_images, adv_labels) in zip(original_loader, adv_loader):
       
        min_batch_size = min(orig_images.size(0), adv_images.size(0))

       
        orig_images_all = orig_images[:min_batch_size]
        orig_labels_all = orig_labels[:min_batch_size]

       
        adv_images_all = adv_images[:min_batch_size]
        adv_labels_all = adv_labels[:min_batch_size]

      
        mixed_images = torch.cat([orig_images_all, adv_images_all], dim=0)
        mixed_labels = torch.cat([orig_labels_all, adv_labels_all], dim=0)

        yield mixed_images, mixed_labels



# Training settings
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)  # 每2个epoch学习率衰减为原来的一半

# training function
def adv_train(model, train_loader, adv_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for mixed_images, mixed_labels in mixed_train_loader(train_loader, adv_loader):
            mixed_images, mixed_labels = mixed_images.to(device), mixed_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(mixed_images)
            loss = criterion(outputs, mixed_labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / (len(train_loader) + len(adv_train_loader))}")
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

    return all_outputs_before_softmax, all_predictions

# num_epochs = 4
# adv_train(model, train_loader, adv_train_loader, criterion, optimizer, num_epochs)


model_save_path2 = './models/cifar10/AdvAlexNet_cifar10.pth'
# torch.save(model.state_dict(), model_save_path2)
# print(f'Model saved to {model_save_path2}')


loaded_model = ModifiedAlexNet().to(device)
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
