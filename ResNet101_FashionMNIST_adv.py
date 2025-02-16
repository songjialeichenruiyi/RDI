import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import RDI
import PGD_attack
import Square_Attack
import RFGSM
import numpy as np

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
    

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

model = ResNet101().to(device)
model_save_path = './models/FashionMNIST/Res101_FashionMNIST.pth'
model.load_state_dict(torch.load(model_save_path))
model.eval() 

adv_data, labels = PGD_attack.save_pgd_attack(model=model, test_loader=train_loader)

np.save('./attackData/FashionMNIST/ResNet101_datas.npy', adv_data)
np.save('./attackData/FashionMNIST/ResNet101_labels.npy', labels) 
