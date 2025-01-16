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

# 使用 torchvision.models 中的 MobileNetV2 模型，并修改输出层以适应 CIFAR-100 数据集
class ModifiedMobileNetV2(nn.Module):
    def __init__(self, num_classes=100):  # CIFAR-100 类别数为 100
        super(ModifiedMobileNetV2, self).__init__()
        self.model = models.mobilenet_v2(pretrained=True)  # 使用预训练的 MobileNetV2
        # 修改第一层卷积来适应 CIFAR-100 小尺寸输入
        self.model.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        # 修改池化层，因为 CIFAR-100 输入尺寸较小，移除最大池化层以保留更多信息
        self.model.features[0][1] = nn.Identity()  # 移除第一层的最大池化层
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)  # 修改输出层

    def forward(self, x):
        return self.model(x)

    

# 数据预处理：转换为Tensor并进行归一化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))  # CIFAR-100 的均值和标准差
])


# 加载训练集和测试集
train_dataset = datasets.CIFAR100(root='./boundary_robustness/data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR100(root='./boundary_robustness/data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 创建新的MobileNet模型实例
model = ModifiedMobileNetV2().to(device)
model_save_path = 'boundary_robustness/models/cifar100/MobileNetV2_cifar100_2.pth'
# 加载模型参数
model.load_state_dict(torch.load(model_save_path))
model.eval()  # 切换到评估模式

adv_data, labels = PGD_attack.save_pgd_attack(model=model, test_loader=train_loader, epsilon=0.01, alpha=0.001, iters=10)

np.save('boundary_robustness/attackData/Cifar100/MobileNetV2_datas.npy', adv_data)  # 保存对抗样本
np.save('boundary_robustness/attackData/Cifar100/MobileNetV2_labels.npy', labels)   # 保存标签
