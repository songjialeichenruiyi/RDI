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


# 加载torchvision.models中的DenseNet161模型，并修改输出层以适应CIFAR-10数据集
class ModifiedDenseNet161(nn.Module):
    def __init__(self, num_classes=10):
        super(ModifiedDenseNet161, self).__init__()
        self.model = models.densenet161(pretrained=True)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)  # 修改输出层以适应CIFAR-10的类别数量

    def forward(self, x):
        return self.model(x)
    

# 数据预处理：转换为Tensor并进行归一化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.2435, 0.2616))
])


# 加载训练集和测试集
train_dataset = datasets.CIFAR10(root='./boundary_robustness/data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./boundary_robustness/data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 创建新的MobileNet模型实例
model = ModifiedDenseNet161().to(device)
model_save_path = 'boundary_robustness/models/cifar10/DenseNet161_cifar10_2.pth'
# 加载模型参数
model.load_state_dict(torch.load(model_save_path))
model.eval()  # 切换到评估模式

adv_data, labels = PGD_attack.save_pgd_attack(model=model, test_loader=train_loader, epsilon=0.01, alpha=0.001, iters=10)

np.save('boundary_robustness/attackData/Cifar10/DenseNet161_datas.npy', adv_data)  # 保存对抗样本
np.save('boundary_robustness/attackData/Cifar10/DenseNet161_labels.npy', labels)   # 保存标签
