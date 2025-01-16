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


# 自定义DenseNet121模型类
class DenseNet161_MNIST(nn.Module):
    def __init__(self, num_classes=10):
        super(DenseNet161_MNIST, self).__init__()
        # 加载预训练的DenseNet161模型
        self.model = models.densenet161(weights = None)
        
        # 修改第一层卷积层，适应1通道的MNIST数据
        self.model.features.conv0 = nn.Conv2d(
            1,  # 输入通道改为1 (灰度图像)
            96,  # 输出通道保持不变
            kernel_size=7,  # 核大小与原模型保持一致
            stride=2,
            padding=3,
            bias=False
        )

        # 移除第一个最大池化层，防止特征图缩小过快
        self.model.features.pool0 = nn.Identity()
        
        # 替换分类层为10类输出
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

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
model = DenseNet161_MNIST().to(device)
model_save_path = 'boundary_robustness/models/FashionMNIST/DenseNet161_FashionMNIST.pth'
# 加载模型参数
model.load_state_dict(torch.load(model_save_path))
model.eval()  # 切换到评估模式

adv_data, labels = PGD_attack.save_pgd_attack(model=model, test_loader=train_loader)

np.save('boundary_robustness/attackData/FashionMNIST/DenseNet161_datas.npy', adv_data)  # 保存对抗样本
np.save('boundary_robustness/attackData/FashionMNIST/DenseNet161_labels.npy', labels)   # 保存标签
