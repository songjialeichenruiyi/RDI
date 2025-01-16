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


# 加载ResNet50模型并修改结构以适应MNIST数据集（28x28输入，10类输出）
class ResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet50, self).__init__()
        # 使用 torchvision.models 中的预训练 ResNet50
        self.resnet = models.resnet50(weights=None)  # 不加载预训练权重
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)  # 修改第一层
        self.resnet.maxpool = nn.Identity()  # 删除 MaxPool 层，以适应28x28输入
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)  # 修改最后全连接层

    def forward(self, x):
        return self.resnet(x)
    

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
model = ResNet50().to(device)
model_save_path = 'boundary_robustness/models/FashionMNIST/ResNet50_FashionMNIST_3.pth'
# 加载模型参数
model.load_state_dict(torch.load(model_save_path))
model.eval()  # 切换到评估模式

adv_data, labels = PGD_attack.save_pgd_attack(model=model, test_loader=train_loader)

np.save('boundary_robustness/attackData/FashionMNIST/ResNet50_datas.npy', adv_data)  # 保存对抗样本
np.save('boundary_robustness/attackData/FashionMNIST/ResNet50_labels.npy', labels)   # 保存标签
