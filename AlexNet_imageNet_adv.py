import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
# import ROBY
import RDI
import PGD_attack
import Square_Attack
import RFGSM
import numpy as np
import os
from PIL import Image

# 检查是否有可用的GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

# 加载 torchvision.models 中的 AlexNet 模型，并修改输入层以适应TinyImageNet数据集
class ModifiedAlexNet(nn.Module):
    def __init__(self, num_classes=200):  # 将类别数量改为 200
        super(ModifiedAlexNet, self).__init__()
        self.model = models.alexnet(weights = None)
        # 修改第一层卷积来适应小尺寸输入
        # self.model.features[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        # 修改最大池化层，减少池化窗口的大小和步幅
        # self.model.features[2] = nn.Identity()
        self.model.classifier[6] = nn.Linear(4096, num_classes)  # 修改输出层以适应Tiny-imageNet200的类别数量

    def forward(self, x):
        return self.model(x)

    

# 数据预处理：转换为 Tensor 并进行归一化
transform = transforms.Compose([
    # transforms.RandomHorizontalFlip(),  # 随机水平翻转
    # transforms.RandomRotation(10),      # 随机旋转角度
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # 色彩抖动
    # 改变图像大小
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))  # Tiny-imageNet200的均值和标准差
])

dataset_path = "./boundary_robustness/data/tiny-imagenet-200"
train_dir = os.path.join(dataset_path, 'train')
val_dir = os.path.join(dataset_path, 'val')

# 自定义验证集数据集类
class TinyImageNetValDataset(Dataset):
    def __init__(self, val_dir, transform=None):
        self.val_dir = val_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # 读取 val_annotations.txt 文件
        annotations_file = os.path.join(val_dir, 'val_annotations.txt')
        with open(annotations_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                image_name = parts[0]
                label = parts[1]
                self.image_paths.append(os.path.join(val_dir, 'images', image_name))
                self.labels.append(label)
        
        # 获取类别名称到索引的映射
        label_names = sorted(list(set(self.labels)))
        self.label_to_idx = {label: idx for idx, label in enumerate(label_names)}
        
        # 将标签转换为索引
        self.labels = [self.label_to_idx[label] for label in self.labels]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return image, label


# 加载训练集
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)

# 加载验证集
# test_dataset = TinyImageNetValDataset(val_dir, transform=transform_test)
# test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)

# 创建新的MobileNet模型实例
model = ModifiedAlexNet().to(device)
model_save_path = 'boundary_robustness/models/imageNet/AlexNet_imageNet_3.pth'
# 加载模型参数
model.load_state_dict(torch.load(model_save_path))
model.eval()  # 切换到评估模式

adv_data, labels = PGD_attack.save_pgd_attack(model=model, test_loader=train_loader, epsilon=0.005, alpha=0.00025, iters=5)

np.save('boundary_robustness/attackData/ImageNet/AlexNet_datas.npy', adv_data)  # 保存对抗样本
np.save('boundary_robustness/attackData/ImageNet/AlexNet_labels.npy', labels)   # 保存标签
