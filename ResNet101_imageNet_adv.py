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

class ModifiedResNet101(nn.Module):
    def __init__(self, num_classes=200): 
        super(ModifiedResNet101, self).__init__()
        self.model = models.resnet101(pretrained=True)  
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)



transform = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(10),      
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))
])

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


train_dataset = datasets.ImageFolder(train_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

# test_dataset = TinyImageNetValDataset(val_dir, transform=transform_test)
# test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)

model = ModifiedResNet101().to(device)
model_save_path = './models/imageNet/ResNet101_imageNet.pth'
model.load_state_dict(torch.load(model_save_path))
model.eval()

adv_data, labels = PGD_attack.save_pgd_attack(model=model, test_loader=train_loader, epsilon=0.005, alpha=0.00025, iters=5)

np.save('./attackData/ImageNet/ResNet101_datas.npy', adv_data) 
np.save('./attackData/ImageNet/ResNet101_labels.npy', labels)
