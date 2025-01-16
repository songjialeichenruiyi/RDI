import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms, models
# import ROBY
import RDI
import PGD_attack
import PGD_attack2
import os
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
import torchattacks
import RFGSM
import Square_Attack
# 检查是否有可用的GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    transforms.Normalize((0.1307,), (0.3081,))  # 均值和标准差
])

# 加载MNIST数据集
train_dataset = datasets.FashionMNIST(root='./boundary_robustness/data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./boundary_robustness/data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=200, shuffle=False)

# 定义模型、损失函数和优化器
model = ResNet50(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 训练函数
def train(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")

# 测试函数，返回softmax之前的输出和预测结果
def test(model, test_loader):
    model.eval()
    all_outputs_before_softmax = []
    all_predictions = []
    correct = 0

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


# 训练模型
# num_epochs = 3
# train(model, train_loader, criterion, optimizer, num_epochs)


# 保存模型
model_save_path = 'boundary_robustness/models/FashionMNIST/AdvResNet50_FashionMNIST_3.pth'
# torch.save(model.state_dict(), model_save_path)
# print(f'Model saved to {model_save_path}')

# 创建新的LeNet模型实例
loaded_model = ResNet50(num_classes=10).to(device)
# 加载模型参数
loaded_model.load_state_dict(torch.load(model_save_path))
loaded_model.eval()  # 切换到评估模式

# 测试模型，并获取softmax层之前的输出和预测结果
print("Evaluating on test data...")
outputs, predictions = test(loaded_model, test_loader)

class_num = 10
# 倒数第二层的输出
feature_vector = [[] for i in range(class_num)]

for i in range(50):
    for j in range (len(outputs[i])):
        feature_vector[predictions[i][j][0]].append(outputs[i][j])


# FSA, FSD, FSC = ROBY.feature_sta(feature_vector)
# print("FSA={}, FSD={}, FSC={}".format(FSA,FSD,FSC))
RDI = RDI.features(feature_vector)
print("RDI = ", RDI)

# 测试模型在PGD攻击下的性能
print("Evaluating under PGD attack...")
PGD_attack.test_with_pgd_attack(loaded_model, test_loader)

# 测试模型在RFGSM攻击下的性能
# print("Evaluating under RFGSM attack...")
# RFGSM.test_with_RFGSM_attack(loaded_model, test_loader)

# 测试模型在Square_Attack攻击下的性能
# print("Evaluating under Square attack...")
# Square_Attack.test_with_square_attack(loaded_model, test_loader)
