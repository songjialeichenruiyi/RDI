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


class ModifiedDenseNet161(nn.Module):
    def __init__(self, num_classes=10):
        super(ModifiedDenseNet161, self).__init__()
        self.model = models.densenet161(pretrained=True)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes) 

    def forward(self, x):
        return self.model(x)
    

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.2435, 0.2616))
])


train_dataset = datasets.CIFAR10(root='./boundary_robustness/data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./boundary_robustness/data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

model = ModifiedDenseNet161().to(device)
model_save_path = './models/cifar10/DenseNet161_cifar10.pth'
model.load_state_dict(torch.load(model_save_path))
model.eval() 

adv_data, labels = PGD_attack.save_pgd_attack(model=model, test_loader=train_loader, epsilon=0.01, alpha=0.001, iters=10)

np.save('./attackData/Cifar10/DenseNet161_datas.npy', adv_data)
np.save('./attackData/Cifar10/DenseNet161_labels.npy', labels)
