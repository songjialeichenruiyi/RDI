import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
# import ROBY
import RDI
import PGD_attack
import Square_Attack
import RFGSM
import numpy as np

# 检查是否有可用的GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())


# 使用torch库里自带的AlexNet模型，适应28x28的输入
class ModifiedAlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ModifiedAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  # 修改输入通道数
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出尺寸：14x14x64
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出尺寸：7x7x192
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 输出尺寸：3x3x256
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 3 * 3, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    

# 数据预处理：转换为Tensor并进行归一化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载训练集和测试集
train_dataset = datasets.FashionMNIST(root='./boundary_robustness/data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./boundary_robustness/data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 创建新的LeNet模型实例
model = ModifiedAlexNet().to(device)
model_save_path = 'boundary_robustness/models/FashionMNIST/AlexNet_fashionMNIST_2.pth'
# 加载模型参数
model.load_state_dict(torch.load(model_save_path))
model.eval()  # 切换到评估模式

adv_data, labels = PGD_attack.save_pgd_attack(model=model, test_loader=train_loader)

np.save('boundary_robustness/attackData/FashionMNIST/AlexNet_datas.npy', adv_data)  # 保存对抗样本
np.save('boundary_robustness/attackData/FashionMNIST/AlexNet_labels.npy', labels)   # 保存标签
