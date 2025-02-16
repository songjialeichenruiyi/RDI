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

class DenseNet121_MNIST(nn.Module):
    def __init__(self, num_classes=10):
        super(DenseNet121_MNIST, self).__init__()
        self.model = models.densenet121(weights = None)
        
        self.model.features.conv0 = nn.Conv2d(
            1, 
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        self.model.features.pool0 = nn.Identity()
        
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_classes)

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

model = DenseNet121_MNIST().to(device)
model_save_path = './models/DenseNet121_mnist.pth'
model.load_state_dict(torch.load(model_save_path))
model.eval()

adv_data, labels = PGD_attack.save_pgd_attack(model=model, test_loader=test_loader)

np.save('./attackData/MNIST/DenseNet121_datas2.npy', adv_data)  # save adversarial samples
np.save('./attackData/MNIST/DenseNet121_labels2.npy', labels)   # save labels
