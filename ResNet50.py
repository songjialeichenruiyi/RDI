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
import time
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
train_dataset = datasets.MNIST(root='./boundary_robustness/data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./boundary_robustness/data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)

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

# 定义加载对抗样本的函数
def load_adversarial_examples(directory):
    adv_images_list = []
    adv_labels_list = []
    
    # 遍历目录中的每个对抗样本文件
    for filename in os.listdir(directory):
        if filename.endswith('.pt'):
            filepath = os.path.join(directory, filename)
            images, labels = torch.load(filepath)
            adv_images_list.append(images)
            adv_labels_list.append(labels)
    
    # 将所有批次的对抗样本合并
    adv_images = torch.cat(adv_images_list, dim=0)
    adv_labels = torch.cat(adv_labels_list, dim=0)
    
    # 创建一个 TensorDataset
    adv_dataset = TensorDataset(adv_images, adv_labels)
    
    return adv_dataset


# 定义对抗训练函数
def adversarial_training(model, adv_loader, optimizer, criterion, epochs=5):
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        # 混合干净样本和对抗样本进行训练
        for adv_images, adv_labels in adv_loader:
            # 将图像和标签放到 GPU
            adv_images, adv_labels = adv_images.to(device), adv_labels.to(device)
            optimizer.zero_grad()
            
            # 对抗样本前向传播
            outputs_adv = model(adv_images)
            loss = criterion(outputs_adv, adv_labels)

            # 反向传播并优化
            loss.backward()
            optimizer.step()
            
            # 统计损失和准确率
            running_loss += loss.item()
            _, predicted_adv = outputs_adv.max(1)
            total += adv_labels.size(0)
            correct += predicted_adv.eq(adv_labels).sum().item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(adv_loader):.4f}, Accuracy: {100. * correct / total:.2f}%")


# 训练模型
# num_epochs = 1
# train(model, train_loader, criterion, optimizer, num_epochs)
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

# 保存模型
model_save_path = 'boundary_robustness/models/AdvResNet50_mnist2.pth'
# torch.save(model.state_dict(), model_save_path)
# print(f'Model saved to {model_save_path}')

# 创建新的LeNet模型实例
loaded_model = ResNet50(num_classes=10).to(device)
# 加载模型参数
loaded_model.load_state_dict(torch.load(model_save_path))
loaded_model.eval()  # 切换到评估模式

# 测试模型，并获取softmax层之前的输出和预测结果
# print("Evaluating on test data...")
# outputs, predictions = test(loaded_model, test_loader)

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
