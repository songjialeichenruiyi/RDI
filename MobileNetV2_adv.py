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

class ModifiedMobileNetV2(nn.Module):
    def __init__(self, num_classes=10):
        super(ModifiedMobileNetV2, self).__init__()
        self.model = models.mobilenet_v2(weights=None)
        self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)

    def forward(self, x):
        return self.model(x)
    

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

model = ModifiedMobileNetV2().to(device)
model_save_path = './models/MobileNetV2_mnist3.pth'
model.load_state_dict(torch.load(model_save_path))
model.eval()

adv_data, labels = PGD_attack.save_pgd_attack(model=model, test_loader=train_loader)

np.save('./attackData/MNIST/MobileNetV2_datas.npy', adv_data) 
np.save('./attackData/MNIST/MobileNetV2_labels.npy', labels) 
