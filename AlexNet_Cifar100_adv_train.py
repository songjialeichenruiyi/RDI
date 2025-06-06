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

# Load the AlexNet model in torchvision.models and modify the input layer to fit the CIFAR-100 dataset
class ModifiedAlexNet(nn.Module):
    def __init__(self, num_classes=100):
        super(ModifiedAlexNet, self).__init__()
        self.model = models.alexnet(weights = None)
        # self.model.features[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        # self.model.features[2] = nn.Identity()
        self.model.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.model(x)
    
    
model = ModifiedAlexNet().to(device)
model_save_path = './models/cifar100/AlexNet_cifar100.pth'
model.load_state_dict(torch.load(model_save_path))
model.eval()

# Custom dataset class for loading adversarial samples
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
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), 
    transforms.Resize((224, 224)),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
])

transform3 = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.RandomCrop(224, padding=4), 
    transforms.RandomHorizontalFlip(),     
    transforms.RandomRotation(15),        
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)) 
])

transform2 = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
])

#  Create adversarial datasets
trainset = AdversarialDataset(adv_data_file="./attackData/Cifar100/AlexNet_datas.npy", label_file="./attackData/Cifar100/AlexNet_labels.npy", transform=transform)
adv_train_loader = DataLoader(trainset, batch_size=64, shuffle=True)

train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform3)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform2)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)


# Hybrid Data Loader Generator Functions
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
        # Ensure that both sides take the same amount of data
        min_batch_size = min(orig_images.size(0), adv_images.size(0))

        # Take all from the original sample
        orig_images_all = orig_images[: min_batch_size]
        orig_labels_all = orig_labels[: min_batch_size]

        # Take all from the adversarial sample
        adv_images_all = adv_images[: min_batch_size]
        adv_labels_all = adv_labels[: min_batch_size]

        # Splicing into mixed batches
        mixed_images = torch.cat([orig_images_all, adv_images_all], dim=0)
        mixed_labels = torch.cat([orig_labels_all, adv_labels_all], dim=0)

        yield mixed_images, mixed_labels


criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.8, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5) 

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

# num_epochs = 5
# adv_train(model, train_loader, adv_train_loader, criterion, optimizer, num_epochs)

model_save_path2 = 'boundary_robustness/models/cifar100/AdvAlexNet_cifar100.pth'
# torch.save(model.state_dict(), model_save_path2)
# print(f'Model saved to {model_save_path2}')


loaded_model = ModifiedAlexNet().to(device)
loaded_model.load_state_dict(torch.load(model_save_path2))
loaded_model.eval() 

# Test loaded models
print("Testing loaded model...")
outputs, predictions = test(loaded_model, test_loader)


class_num = 100
feature_vector = [[] for i in range(class_num)]

for i in range(100):
    for j in range (len(outputs[i])):
        feature_vector[predictions[i][j][0]].append(outputs[i][j])
# RDI
RDI = RDI.features(feature_vector)
print("RDI = ", RDI)
