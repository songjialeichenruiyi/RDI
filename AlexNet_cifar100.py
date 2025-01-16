import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
# import ROBY
import RDI
import PGD_attack
import RFGSM
import Square_Attack
import time

# 检查是否有可用的 GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

# 加载 torchvision.models 中的 AlexNet 模型，并修改输入层以适应 CIFAR-100 数据集
class ModifiedAlexNet(nn.Module):
    def __init__(self, num_classes=100):  # 将类别数量改为 100
        super(ModifiedAlexNet, self).__init__()
        self.model = models.alexnet(pretrained=True)
        # 修改第一层卷积来适应 CIFAR-100 小尺寸输入
        # self.model.features[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        # 修改最大池化层，减少池化窗口的大小和步幅
        # self.model.features[2] = nn.Identity()
        self.model.classifier[6] = nn.Linear(4096, num_classes)  # 修改输出层以适应 CIFAR-100 的类别数量

    def forward(self, x):
        return self.model(x)

# 数据预处理：转换为 Tensor 并进行归一化
transform = transforms.Compose([
    # 改变图像大小
    transforms.Resize((224, 224)),  
    # transforms.RandomCrop(32, padding=4),  # 随机裁剪到 32x32，并在边界填充 4 像素
    transforms.RandomHorizontalFlip(),     # 随机水平翻转
    transforms.RandomRotation(15),         # 随机旋转 [-15, 15] 度
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))  # CIFAR-100 的均值和标准差
])

# 数据预处理：转换为 Tensor 并进行归一化
transform2 = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))  # CIFAR-100 的均值和标准差
])

# 加载训练集和测试集
train_dataset = datasets.CIFAR100(root='./boundary_robustness/data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR100(root='./boundary_robustness/data', train=False, download=True, transform=transform2)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)

# 定义模型、损失函数和优化器
model = ModifiedAlexNet(num_classes=100).to(device)
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0004, weight_decay=1e-4)
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.8, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)  # 每2个epoch学习率衰减为原来的一半

# 训练函数
def train(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
        
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()  # 清零梯度
            outputs = model(images)  # 前向传播
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重
            
            running_loss += loss.item()

        accuracy = correct / len(train_loader.dataset)
        print(f'Accuracy: {correct}/{len(train_loader.dataset)} '
              f'({100. * accuracy:.2f}%)')
        # 打印每个 epoch 的损失
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")
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
            outputs = model(images)
            all_outputs_before_softmax.append(outputs.cpu().numpy())  # 保存 softmax 层之前的输出
            pred = outputs.argmax(dim=1, keepdim=True)
            all_predictions.append(pred.cpu().numpy())  # 保存所有预测结果
            correct += pred.eq(labels.view_as(pred)).sum().item()

    accuracy = correct / len(test_loader.dataset)
    print(f'\nAccuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * accuracy:.2f}%)\n')

    # 返回 softmax 层之前的输出和预测结果
    return all_outputs_before_softmax, all_predictions, accuracy


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
# num_epochs = 6
# train(model, train_loader, criterion, optimizer, num_epochs)

# 保存模型
model_save_path = 'boundary_robustness/models/cifar100/AlexNet_cifar100.pth'
# torch.save(model.state_dict(), model_save_path)
# print(f'Model saved to {model_save_path}')

# 加载模型并测试
loaded_model = ModifiedAlexNet(num_classes=100).to(device)
loaded_model.load_state_dict(torch.load(model_save_path))
loaded_model.eval()  # 切换到评估模式

# print("Testing loaded model...")
# outputs, predictions, acc = test(loaded_model, test_loader)

print("Testing loaded model...")
start_time = time.time()
outputs, predictions = test2(loaded_model, test_loader)

class_num = 100
# 倒数第二层的输出
feature_vector = [[] for i in range(class_num)]

for i in range(100):
    for j in range(len(outputs[i])):
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

# 测试模型在 PGD 攻击下的性能
print("Evaluating under PGD attack...")
PGD_attack.test_with_pgd_attack(loaded_model, test_loader, 0.025, 0.001, 10)

# 测试模型在RFGSM攻击下的性能
# print("Evaluating under RFGSM attack...")
# RFGSM.test_with_RFGSM_attack(loaded_model, test_loader, 0.025, 0.001, 10)

# 测试模型在Square_Attack攻击下的性能
# print("Evaluating under Square attack...")
# Square_Attack.test_with_square_attack(loaded_model, test_loader, n_queries=200, eps=0.025)