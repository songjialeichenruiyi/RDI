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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

class ModifiedViT(nn.Module):
    def __init__(self, num_classes=200):
        super(ModifiedViT, self).__init__()
        self.model = vit_b_16(weights=None)
        self.model.heads.head = nn.Linear(self.model.hidden_dim, num_classes)

    def forward(self, x):
        return self.model(x)

    

transform = transforms.Compose([
    transforms.Resize((224, 224)),  
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
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

# test_dataset = TinyImageNetValDataset(val_dir, transform=transform_test)
# test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)

model = ModifiedViT().to(device)
model_save_path = './models/imageNet/VIT_imageNet_2.pth'
model.load_state_dict(torch.load(model_save_path))
model.eval()

adv_data, labels = PGD_attack.save_pgd_attack(model=model, test_loader=train_loader, epsilon=0.005, alpha=0.00025, iters=5)

np.save('./attackData/ImageNet/VIT_datas.npy', adv_data) 
np.save('./attackData/ImageNet/VIT_labels.npy', labels) 
