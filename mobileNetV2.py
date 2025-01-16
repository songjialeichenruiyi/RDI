import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader
# import ROBY
import RDI
import PGD_attack
import PGD_attack2
import RFGSM
import Square_Attack
import time


# 设置设备为GPU（如果可用）
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 定义MobileNetV2类，修改输入通道数为1，适应MNIST数据集
class ModifiedMobileNetV2(nn.Module):
    def __init__(self, num_classes=10):
        super(ModifiedMobileNetV2, self).__init__()
        self.model = models.mobilenet_v2(weights=None)
        # 修改第一层卷积层输入通道数为1，因为MNIST是灰度图像
        self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        # 修改最后一层分类器，输出类别数为10（MNIST共有10类）
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)

    def forward(self, x):
        return self.model(x)
    
# 数据预处理：将图像转换为Tensor并进行标准化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # 对灰度图像进行归一化
])

# 加载MNIST数据集
train_dataset = torchvision.datasets.MNIST(root='./boundary_robustness/data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./boundary_robustness/data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

# 加载模型
MobileNetV2 = ModifiedMobileNetV2(num_classes=10).to(device)
# 设置损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(MobileNetV2.parameters(), lr=0.0006)


def train_model(model, train_loader, criterion, optimizer, num_epochs):
    # 训练模型
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

def test_model(model, test_loader):
    # 测试模型
    model.eval()
    all_outputs_before_softmax = []
    all_predictions = []
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            all_outputs_before_softmax.append(outputs.cpu().numpy())  # 保存softmax层之前的输出
            pred = outputs.argmax(dim=1, keepdim=True)
            all_predictions.append(pred.cpu().numpy())  # 保存所有
            correct += pred.eq(labels.view_as(pred)).sum().item()

    accuracy = correct / len(test_loader.dataset)
    print(f'\nAccuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * accuracy:.2f}%)\n')

    # 返回softmax层之前的输出和预测结果
    return all_outputs_before_softmax, all_predictions

# 测试函数
def test2(model, test_loader):
    all_outputs_before_softmax = []
    all_predictions = []

    with torch.no_grad():  # 禁止计算梯度
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)  # 获得softmax层之前的输出
            all_outputs_before_softmax.append(outputs.cpu().numpy())  # 保存softmax层之前的输出
            pred = outputs.argmax(dim=1, keepdim=True)
            all_predictions.append(pred.cpu().numpy())  # 保存所有预测结果

    # 返回softmax层之前的输出和预测结果
    return all_outputs_before_softmax, all_predictions

# 训练模型
# num_epochs = 8
# train_model(MobileNetV2, train_loader, criterion, optimizer, num_epochs)

# 保存模型
model_save_path = 'boundary_robustness/models/AdvMobileNetV2_mnist3.pth'
# torch.save(MobileNetV2.state_dict(), model_save_path)
# print(f'Model saved to {model_save_path}')

# 创建新的LeNet模型实例
loaded_model = ModifiedMobileNetV2(num_classes=10).to(device)
# 加载模型参数
loaded_model.load_state_dict(torch.load(model_save_path))
loaded_model.eval()  # 切换到评估模式

# 测试模型，并获取softmax层之前的输出和预测结果
# print("Evaluating on test data...")
# outputs, predictions = test_model(loaded_model, test_loader)

# 测试模型，并获取softmax层之前的输出和预测结果
print("Evaluating on test data...")
start_time = time.time()
outputs, predictions = test2(loaded_model, test_loader)

class_num = 10
# 倒数第二层的输出
feature_vector = [[] for i in range(class_num)]

for i in range(100):
    for j in range (len(outputs[i])):
        feature_vector[predictions[i][j][0]].append(outputs[i][j])


# 测量FSC的运行时间
# FSC = ROBY.feature_sta(feature_vector)
# end_time = time.time()
# print("FSC = {}".format(FSC))
# print("FSC运行时间: {:.6f}秒".format(end_time - start_time))

# 测量RDI的运行时间
RDI = RDI.features(feature_vector)
end_time = time.time()
print("RDI =", RDI)
print("RDI运行时间: {:.6f}秒".format(end_time - start_time))

# 测试模型在PGD攻击下的性能
print("Evaluating under PGD attack...")
PGD_attack.test_with_pgd_attack(loaded_model, test_loader)

# 测试模型在RFGSM攻击下的性能
# print("Evaluating under RFGSM attack...")
# RFGSM.test_with_RFGSM_attack(loaded_model, test_loader)

# 测试模型在Square_Attack攻击下的性能
# print("Evaluating under Square attack...")
# Square_Attack.test_with_square_attack(loaded_model, test_loader)
