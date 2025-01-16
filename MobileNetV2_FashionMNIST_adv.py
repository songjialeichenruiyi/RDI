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


# 定义MobileNetV2类，修改输入通道数为1，适应MNIST数据集
class ModifiedMobileNetV2(nn.Module):
    def __init__(self, num_classes=10):
        super(ModifiedMobileNetV2, self).__init__()
        self.model = models.mobilenet_v2(weights=None)
        # 修改第一层卷积层输入通道数为1，因为MNIST是灰度图像
        self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.model.features[0][1] = nn.Identity()  # 移除第一层的最大池化层
        # 修改最后一层分类器，输出类别数为10（MNIST共有10类）
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)

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
model = ModifiedMobileNetV2().to(device)
model_save_path = 'boundary_robustness/models/FashionMNIST/MobileNetV2_FashionMNIST_3.pth'
# 加载模型参数
model.load_state_dict(torch.load(model_save_path))
model.eval()  # 切换到评估模式

adv_data, labels = PGD_attack.save_pgd_attack(model=model, test_loader=train_loader)

np.save('boundary_robustness/attackData/FashionMNIST/MobileNetV2_datas.npy', adv_data)  # 保存对抗样本
np.save('boundary_robustness/attackData/FashionMNIST/MobileNetV2_labels.npy', labels)   # 保存标签
