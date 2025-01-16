import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models
# import ROBY
import RDI

# 检查是否有可用的GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

# 加载 torchvision.models 中的 AlexNet 模型，并修改输入层以适应 CIFAR-100 数据集
class ModifiedAlexNet(nn.Module):
    def __init__(self, num_classes=100):  # 将类别数量改为 100
        super(ModifiedAlexNet, self).__init__()
        self.model = models.alexnet(weights = None)
        # 修改第一层卷积来适应 CIFAR-100 小尺寸输入
        # self.model.features[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        # 修改最大池化层，减少池化窗口的大小和步幅
        # self.model.features[2] = nn.Identity()
        self.model.classifier[6] = nn.Linear(4096, num_classes)  # 修改输出层以适应 CIFAR-100 的类别数量

    def forward(self, x):
        return self.model(x)
    
    
# 1. 加载训练好的模型
model = ModifiedAlexNet().to(device)
model_save_path = 'boundary_robustness/models/cifar100/AlexNet_cifar100.pth'
# 加载模型参数
model.load_state_dict(torch.load(model_save_path))
model.eval()  # 切换到评估模式

# 2. 自定义数据集类，用于加载对抗样本
class AdversarialDataset(Dataset):
    def __init__(self, adv_data_file, label_file, transform=None):
        self.adv_data = np.load(adv_data_file)  # 加载对抗样本
        self.labels = np.load(label_file)       # 加载标签
        self.transform = transform

    def __len__(self):
        return len(self.adv_data)

    def __getitem__(self, idx):
        image = torch.tensor(self.adv_data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# 3. 数据加载部分
transform = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),  # 随机裁剪到 32x32，并在边界填充 4 像素
    # transforms.RandomHorizontalFlip(),     # 随机水平翻转
    # transforms.RandomRotation(15),         # 随机旋转 [-15, 15] 度
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 调整亮度、对比度、饱和度和色调
    transforms.Resize((224, 224)),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))  # CIFAR-100 的均值和标准差
])

# 3. 数据加载部分
transform3 = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.RandomCrop(224, padding=4),  # 随机裁剪到 32x32，并在边界填充 4 像素
    transforms.RandomHorizontalFlip(),     # 随机水平翻转
    transforms.RandomRotation(15),         # 随机旋转 [-15, 15] 度
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 调整亮度、对比度、饱和度和色调
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))  # CIFAR-100 的均值和标准差
])

transform2 = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))  # CIFAR-100 的均值和标准差
])

# 创建对抗数据集
trainset = AdversarialDataset(adv_data_file="boundary_robustness/attackData/Cifar100/AlexNet_datas.npy", label_file="boundary_robustness/attackData/Cifar100/AlexNet_labels.npy", transform=transform)
adv_train_loader = DataLoader(trainset, batch_size=64, shuffle=True)

train_dataset = datasets.CIFAR100(root='./boundary_robustness/data', train=True, download=True, transform=transform3)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = datasets.CIFAR100(root='./boundary_robustness/data', train=False, download=True, transform=transform2)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)


# 混合数据加载器的生成函数
def mixed_train_loader(original_loader, adv_loader):
    """
    从 original_loader 和 adv_loader 取全部数据组成混合批次。
    Args:
        original_loader: 原始数据加载器
        adv_loader: 对抗样本数据加载器
    Yields:
        mixed_images: 混合的图像张量
        mixed_labels: 混合的标签张量
    """
    for (orig_images, orig_labels), (adv_images, adv_labels) in zip(original_loader, adv_loader):
        # 确保两边取数据量一致
        min_batch_size = min(orig_images.size(0), adv_images.size(0))

        # 从原始样本中取全部
        orig_images_all = orig_images[: min_batch_size]
        orig_labels_all = orig_labels[: min_batch_size]

        # 从对抗样本中取全部
        adv_images_all = adv_images[: min_batch_size]
        adv_labels_all = adv_labels[: min_batch_size]

        # 拼接成混合批次
        mixed_images = torch.cat([orig_images_all, adv_images_all], dim=0)
        mixed_labels = torch.cat([orig_labels_all, adv_labels_all], dim=0)

        yield mixed_images, mixed_labels



# 4. 训练设置
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.8, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)  # 每2个epoch学习率衰减为原来的一半

# 训练函数
def adv_train(model, train_loader, adv_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for mixed_images, mixed_labels in mixed_train_loader(train_loader, adv_loader):
            mixed_images, mixed_labels = mixed_images.to(device), mixed_labels.to(device)
            
            optimizer.zero_grad()  # 清零梯度
            outputs = model(mixed_images)  # 前向传播
            loss = criterion(outputs, mixed_labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重
            
            running_loss += loss.item()

        # 打印每个 epoch 的损失
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / (len(train_loader) + len(adv_train_loader))}")
        scheduler.step()

# 测试函数
def test(model, test_loader):
    model.eval()  # 设置模型为评估模式
    correct = 0
    all_outputs_before_softmax = []
    all_predictions = []

    with torch.no_grad():  # 禁止计算梯度
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)  # 获得softmax层之前的输出
            all_outputs_before_softmax.append(outputs.cpu().numpy())  # 保存softmax层之前的输出
            pred = outputs.argmax(dim=1, keepdim=True)
            all_predictions.append(pred.cpu().numpy())  # 保存所有预测结果
            correct += pred.eq(labels.view_as(pred)).sum().item()

    accuracy = correct / len(test_loader.dataset)
    print(f'\nAccuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * accuracy:.2f}%)\n')

    # 返回softmax层之前的输出和预测结果
    return all_outputs_before_softmax, all_predictions

num_epochs = 5
adv_train(model, train_loader, adv_train_loader, criterion, optimizer, num_epochs)

# 保存模型
model_save_path2 = 'boundary_robustness/models/cifar100/AdvAlexNet_cifar100.pth'
torch.save(model.state_dict(), model_save_path2)
print(f'Model saved to {model_save_path2}')


# 创建新的AlexNet模型实例
loaded_model = ModifiedAlexNet().to(device)
# 加载模型参数
loaded_model.load_state_dict(torch.load(model_save_path2))
loaded_model.eval()  # 切换到评估模式

# 测试加载的模型
print("Testing loaded model...")
outputs, predictions = test(loaded_model, test_loader)


class_num = 100
# 倒数第二层的输出
feature_vector = [[] for i in range(class_num)]

for i in range(100):
    for j in range (len(outputs[i])):
        feature_vector[predictions[i][j][0]].append(outputs[i][j])


# FSA, FSD, FSC = ROBY.feature_sta(feature_vector)
# print("FSA={}, FSD={}, FSC={}".format(FSA,FSD,FSC))
RDI = RDI.features(feature_vector)
print("RDI = ", RDI)
