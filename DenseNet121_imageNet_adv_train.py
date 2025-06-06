import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
import RDI
import PGD_attack
import Square_Attack
import RFGSM
import numpy as np
import os
from PIL import Image


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

class ModifiedDenseNet121(nn.Module):
    def __init__(self, num_classes=200):
        super(ModifiedDenseNet121, self).__init__()
        self.model = models.densenet121(weights = None)
        self.model.features.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.features.pool0 = nn.Identity()
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
    
    
model = ModifiedDenseNet121()
model_save_path = './models/imageNet/DenseNet121_imageNet.pth'
state_dict1 = torch.load(model_save_path)
# Check for `module.` prefix and remove, required for multi-GPU post-training tests
new_state_dict = {}
for key, value in state_dict1.items():
    new_key = key.replace('module.', '')  # Remove the 'module.' prefix
    new_state_dict[new_key] = value

model.load_state_dict(new_state_dict)


# Using the DataParallel Wrapper Model
if torch.cuda.is_available():
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    model = model.cuda()


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
    transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))  
])


transform3 = transforms.Compose([
    # transforms.RandomCrop(64, padding=4), 
    transforms.RandomHorizontalFlip(),  
    transforms.RandomRotation(15),         
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), 
    transforms.ToTensor(),
    transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)) 
])

transform2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))  
])


trainset = AdversarialDataset(adv_data_file="./attackData/ImageNet/DenseNet121_datas.npy", label_file="./attackData/ImageNet/DenseNet121_labels.npy", transform=transform)
adv_train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)


dataset_path = "./data/tiny-imagenet-200"
train_dir = os.path.join(dataset_path, 'train')
val_dir = os.path.join(dataset_path, 'val')

class TinyImageNetValDataset(Dataset):
    def __init__(self, val_dir, transform=None):
        self.val_dir = val_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        annotations_file = os.path.join(val_dir, 'val_annotations.txt')
        with open(annotations_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                image_name = parts[0]
                label = parts[1]
                self.image_paths.append(os.path.join(val_dir, 'images', image_name))
                self.labels.append(label)
        
        label_names = sorted(list(set(self.labels)))
        self.label_to_idx = {label: idx for idx, label in enumerate(label_names)}
        
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


train_dataset = datasets.ImageFolder(train_dir, transform=transform3)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)

test_dataset = TinyImageNetValDataset(val_dir, transform=transform2)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)


def mixed_train_loader(original_loader, adv_loader):
  
    for (orig_images, orig_labels), (adv_images, adv_labels) in zip(original_loader, adv_loader):
        min_batch_size = min(orig_images.size(0), adv_images.size(0))
        
        orig_images_all = orig_images[: min_batch_size]
        orig_labels_all = orig_labels[: min_batch_size]

        adv_images_all = adv_images[: min_batch_size]
        adv_labels_all = adv_labels[: min_batch_size]

        mixed_images = torch.cat([orig_images_all, adv_images_all], dim=0)
        mixed_labels = torch.cat([orig_labels_all, adv_labels_all], dim=0)

        yield mixed_images, mixed_labels



# training settings
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0000001, weight_decay=5e-4)
optimizer = optim.SGD(model.parameters(), lr=0.004, momentum=0.5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)


def adv_train(model, train_loader, adv_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for mixed_images, mixed_labels in mixed_train_loader(train_loader, adv_loader):
            mixed_images, mixed_labels = mixed_images.cuda(), mixed_labels.cuda()
            
            optimizer.zero_grad()
            outputs = model(mixed_images) 
            loss = criterion(outputs, mixed_labels)
            loss.backward()
            optimizer.step() 
            
            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / (len(train_loader) + len(adv_train_loader))}")
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

# num_epochs = 6
# adv_train(model, train_loader, adv_train_loader, criterion, optimizer, num_epochs)

model_save_path2 = './models/imageNet/AdvDenseNet121_imageNet.pth'
# torch.save(model.state_dict(), model_save_path2)
# print(f'Model saved to {model_save_path2}')


loaded_model = ModifiedDenseNet121().to(device)

state_dict = torch.load(model_save_path2)
new_state_dict = {}
for key, value in state_dict.items():
    new_key = key.replace('module.', '')
    new_state_dict[new_key] = value
loaded_model.load_state_dict(new_state_dict)

loaded_model.eval() 


print("Testing loaded model...")
outputs, predictions = test(loaded_model, test_loader)


class_num = 200
feature_vector = [[] for i in range(class_num)]

for i in range(100):
    for j in range (len(outputs[i])):
        feature_vector[predictions[i][j][0]].append(outputs[i][j])


# RDI
RDI = RDI.features(feature_vector)
print("RDI = ", RDI)
