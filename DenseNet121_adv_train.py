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

# 自定义DenseNet121模型类
class DenseNet121_MNIST(nn.Module):
    def __init__(self, num_classes=10):
        super(DenseNet121_MNIST, self).__init__()
        # 加载预训练的DenseNet121模型
        self.model = models.densenet121(weights = None)
        
        # 修改第一层卷积层，适应1通道的MNIST数据
        self.model.features.conv0 = nn.Conv2d(
            1,  # 输入通道改为1 (灰度图像)
            64,  # 输出通道保持不变
            kernel_size=7,  # 核大小与原模型保持一致
            stride=2,
            padding=3,
            bias=False
        )

        # 移除第一个最大池化层，防止特征图缩小过快
        self.model.features.pool0 = nn.Identity()
        
        # 替换分类层为10类输出
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
    
# 1. 加载训练好的模型
model = DenseNet121_MNIST().to(device)
model_save_path = 'boundary_robustness/models/DenseNet121_mnist.pth'
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
    transforms.RandomCrop(28, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize((0.1307,), (0.3081,))
])

transform2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 创建对抗数据集
trainset = AdversarialDataset(adv_data_file="boundary_robustness/attackData/MNIST/DenseNet121_datas.npy", label_file="boundary_robustness/attackData/MNIST/DenseNet121_labels.npy", transform=transform)
train_loader = DataLoader(trainset, batch_size=64, shuffle=True)

test_dataset = datasets.MNIST(root='./boundary_robustness/data', train=False, download=True, transform=transform2)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 4. 训练设置
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0000002, weight_decay=5e-4)
# optimizer = optim.SGD(model.parameters(), lr=0.00001, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)  # 每5个epoch学习率衰减为原来的一半

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


num_epochs = 1
adv_train(model, train_loader, criterion, optimizer, num_epochs)

# 保存模型
model_save_path2 = 'boundary_robustness/models/AdvDenseNet121_mnist.pth'
torch.save(model.state_dict(), model_save_path2)
print(f'Model saved to {model_save_path2}')

# 创建新的LeNet模型实例
loaded_model = DenseNet121_MNIST().to(device)
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
