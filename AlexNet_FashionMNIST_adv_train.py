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

# 使用torch库里自带的AlexNet模型，适应28x28的输入
class ModifiedAlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ModifiedAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  # 修改输入通道数
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出尺寸：14x14x64
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出尺寸：7x7x192
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 输出尺寸：3x3x256
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 3 * 3, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    
# 1. 加载训练好的模型
model = ModifiedAlexNet().to(device)
model_save_path = 'boundary_robustness/models/FashionMNIST/AlexNet_fashionMNIST_2.pth'
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
    # transforms.RandomCrop(28, padding=4),
    # transforms.RandomHorizontalFlip(),
    transforms.Normalize((0.1307,), (0.3081,))
])

transform2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 创建对抗数据集
trainset = AdversarialDataset(adv_data_file="boundary_robustness/attackData/FashionMNIST/AlexNet_datas.npy", label_file="boundary_robustness/attackData/FashionMNIST/AlexNet_labels.npy", transform=transform)
train_loader = DataLoader(trainset, batch_size=64, shuffle=True)

test_dataset = datasets.FashionMNIST(root='./boundary_robustness/data', train=False, download=True, transform=transform2)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 4. 训练设置
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)  # 每2个epoch学习率衰减为原来的一半

# 训练函数
def adv_train(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()  # 清零梯度
            outputs = model(images)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重
            
            running_loss += loss.item()

        # 打印每个epoch的损失
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


num_epochs = 4
adv_train(model, train_loader, criterion, optimizer, num_epochs)

# 保存模型
model_save_path2 = 'boundary_robustness/models/FashionMNIST/AdvAlexNet_FashionMNIST_2.pth'
torch.save(model.state_dict(), model_save_path2)
print(f'Model saved to {model_save_path2}')

# 创建新的MobileNetV2模型实例
loaded_model = ModifiedAlexNet().to(device)
# 加载模型参数
loaded_model.load_state_dict(torch.load(model_save_path2))
loaded_model.eval()  # 切换到评估模式

# 测试加载的模型
print("Testing loaded model...")
outputs, predictions = test(loaded_model, test_loader)


class_num = 10
# 倒数第二层的输出
feature_vector = [[] for i in range(class_num)]

for i in range(10):
    for j in range (len(outputs[i])):
        feature_vector[predictions[i][j][0]].append(outputs[i][j])


# FSA, FSD, FSC = ROBY.feature_sta(feature_vector)
# print("FSA={}, FSD={}, FSC={}".format(FSA,FSD,FSC))
RDI = RDI.features(feature_vector)
print("RDI = ", RDI)
