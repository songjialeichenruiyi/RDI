import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from torchvision.models import vit_b_16, ViT_B_16_Weights
import RDI
import PGD_attack
import Square_Attack
import RFGSM
import numpy as np
import os
from PIL import Image


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

# Use the ViT model from torchvision.models and modify the output layer to fit the Tiny-ImageNet dataset
class ModifiedViT(nn.Module):
    def __init__(self, num_classes=200):  # Tiny-ImageNet has 200 categories.
        super(ModifiedViT, self).__init__()
        self.model = vit_b_16(weights=None)
        self.model.heads.head = nn.Linear(self.model.hidden_dim, num_classes)

    def forward(self, x):
        return self.model(x)
    
    

# Load the trained model
model = ModifiedViT()
model_save_path = './models/imageNet/VIT_imageNet.pth'
state_dict1 = torch.load(model_save_path)
# Check for `module.` prefix and remove, required for multi-GPU post-training tests
new_state_dict = {}
for key, value in state_dict1.items():
    new_key = key.replace('module.', '')
    new_state_dict[new_key] = value

# Load modified state_dict
model.load_state_dict(new_state_dict)

# Using the DataParallel Wrapper Model
if torch.cuda.is_available():
    model = nn.DataParallel(model, device_ids=[1, 3])
    # model = model.cuda()
    model = model.to('cuda:0')


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

# Data loading
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)) 
])

transform3 = transforms.Compose([
    # transforms.RandomCrop(64, padding=4), 
    # transforms.RandomHorizontalFlip(),     
    # transforms.RandomRotation(15),       
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), 
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))
])

transform2 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))
])

# Creating Adversarial Datasets
trainset = AdversarialDataset(adv_data_file="./attackData/ImageNet/VIT_datas.npy", label_file="./attackData/ImageNet/VIT_labels.npy", transform=transform)
adv_train_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)


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


# Load training dataset
train_dataset = datasets.ImageFolder(train_dir, transform=transform3)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

# Load validation dataset
test_dataset = TinyImageNetValDataset(val_dir, transform=transform2)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)


# Hybrid Data Loader Generator Functions
def mixed_train_loader(original_loader, adv_loader):
    for (orig_images, orig_labels), (adv_images, adv_labels) in zip(original_loader, adv_loader):
        min_batch_size = min(orig_images.size(0), adv_images.size(0)) // 2

        orig_images_all = orig_images[: min_batch_size]
        orig_labels_all = orig_labels[: min_batch_size]

        adv_images_all = adv_images[: min_batch_size]
        adv_labels_all = adv_labels[: min_batch_size]

        mixed_images = torch.cat([orig_images_all, adv_images_all], dim=0)
        mixed_labels = torch.cat([orig_labels_all, adv_labels_all], dim=0)

        yield mixed_images, mixed_labels

# training settings
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

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

    #  Return the output and predictions before the softmax layer
    return all_outputs_before_softmax, all_predictions

# num_epochs = 3
# adv_train(model, train_loader, adv_train_loader, criterion, optimizer, num_epochs)

# save model
model_save_path2 = './models/imageNet/AdvVIT_imageNet.pth'
# torch.save(model.state_dict(), model_save_path2)
# print(f'Model saved to {model_save_path2}')


loaded_model = ModifiedViT().to(device)

state_dict = torch.load(model_save_path2)
# Check for `module.` prefix and remove, required for multi-GPU post-training tests
new_state_dict = {}
for key, value in state_dict.items():
    new_key = key.replace('module.', '') 
    new_state_dict[new_key] = value
loaded_model.load_state_dict(new_state_dict)

loaded_model.eval()  # 切换到评估模式

# Test loaded models
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
