import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from torchvision.models import vit_b_16, ViT_B_16_Weights
import os
from PIL import Image
# import ROBY
import RDI
import PGD_attack
import torch.nn.functional as F
import RFGSM
import Square_Attack
import time

# 检查是否有可用的 GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

# 使用 torchvision.models 中的 ViT 模型，并修改输出层以适应 Tiny-ImageNet 数据集
class ModifiedViT(nn.Module):
    def __init__(self, num_classes=200):  # Tiny-ImageNet 类别数为 200
        super(ModifiedViT, self).__init__()
        # 加载预训练的 ViT 模型
        self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        # 修改分类器，以适应 200 个类别
        self.model.heads.head = nn.Linear(self.model.hidden_dim, num_classes)

    def forward(self, x):
        return self.model(x)

# 数据预处理：转换为 Tensor 并进行归一化
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))  # Tiny-ImageNet200 的均值和标准差
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))  # Tiny-ImageNet200 的均值和标准差
])


# 自定义验证集数据集类
dataset_path = "./boundary_robustness/data/tiny-imagenet-200"
train_dir = os.path.join(dataset_path, 'train')
val_dir = os.path.join(dataset_path, 'val')

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
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)

# 加载验证集
test_dataset = TinyImageNetValDataset(val_dir, transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False, num_workers=4)

# 定义模型、损失函数和优化器
model = ModifiedViT(num_classes=200).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.004, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)  # 每 3 个 epoch 学习率衰减为原来的一半

# 训练函数
def train(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
        
    for epoch in range(num_epochs):
        model.train()
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
# num_epochs = 1
# train(model, train_loader, criterion, optimizer, num_epochs)

# 保存模型
model_save_path = 'boundary_robustness/models/imageNet/VIT_imageNet_2.pth'
# torch.save(model.state_dict(), model_save_path)
# print(f'Model saved to {model_save_path}')


# 创建新的MobileNetV2模型实例
loaded_model = ModifiedViT().to(device)

state_dict = torch.load(model_save_path)
# 检查是否有 `module.` 前缀，并移除，多GPU训练后测试需要
new_state_dict = {}
for key, value in state_dict.items():
    new_key = key.replace('module.', '')  # 删除 'module.' 前缀
    new_state_dict[new_key] = value
# 加载修改后的 state_dict
loaded_model.load_state_dict(new_state_dict)

# 加载模型参数
# loaded_model.load_state_dict(torch.load(model_save_path2))
loaded_model.eval()  # 切换到评估模式

# print("Testing loaded model...")
# outputs, predictions, acc = test(loaded_model, test_loader)


print("Testing loaded model...")
start_time = time.time()
outputs, predictions = test2(loaded_model, test_loader)


class_num = 200
# 倒数第二层的输出
feature_vector = [[] for i in range(class_num)]

for i in range(200):
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
PGD_attack.test_with_pgd_attack(loaded_model, test_loader, 0.01, 0.001, 10)

# 测试模型在RFGSM攻击下的性能
# print("Evaluating under RFGSM attack...")
# RFGSM.test_with_RFGSM_attack(loaded_model, test_loader, 0.01, 0.001, 10)

# 测试模型在Square_Attack攻击下的性能
# print("Evaluating under Square attack...")
# Square_Attack.test_with_square_attack(loaded_model, test_loader, n_queries=200, eps=0.01)