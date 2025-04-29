import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio.transforms import MelSpectrogram, Resample
import os
import ROBY
import RDI
import SirenAttack

# 检查是否有可用的GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
print(f"CUDA可用: {torch.cuda.is_available()}")

# 定义数据集类，这里使用SpeechCommands数据集
class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__("./boundary_robustness/data", download=True)
        
        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]
        
        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]

# 只选择10个类别
labels = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']

# 标签映射
label_map = {label: i for i, label in enumerate(labels)}

# 创建音频预处理转换器
melspec = MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    hop_length=512,
    n_mels=64
)
amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

# 预处理音频并提取特征 - 在CPU上完成
def extract_features(waveform):
    """提取梅尔频谱图特征"""
    # 确保音频长度一致 (调整到1秒)
    if waveform.shape[1] < 16000:
        # 填充音频
        padding = torch.zeros(1, 16000 - waveform.shape[1])
        waveform = torch.cat([waveform, padding], dim=1)
    else:
        # 截断音频
        waveform = waveform[:, :16000]
    
    # 提取特征
    with torch.no_grad():
        # 转换为梅尔频谱图
        spec = melspec(waveform)
        # 转换为分贝单位
        spec_db = amplitude_to_db(spec)
        # 添加通道维度
        spec_db = spec_db.unsqueeze(0)  # 形状: [1, n_mels, time]
    
    return spec_db

# 自定义数据收集函数 - 预先提取特征，避免在GPU上进行特征提取
def collate_fn(batch):
    features = []
    labels = []
    for waveform, sample_rate, label, speaker_id, utterance_number in batch:
        # 只处理目标类别
        if label in label_map:
            # 在CPU上提取特征
            feature = extract_features(waveform)
            features.append(feature)
            labels.append(label_map[label])
    
    # 如果批次为空，返回空张量
    if len(features) == 0:
        return torch.empty(0), torch.empty(0, dtype=torch.long)
    
    # 转换为tensor
    features = torch.cat(features, dim=0)  # 形状: [batch_size, 1, n_mels, time]
    labels = torch.tensor(labels, dtype=torch.long)
    return features, labels

# 加载数据集
train_dataset = SubsetSC("training")
test_dataset = SubsetSC("testing")
val_dataset = SubsetSC("validation")

# 批次大小
batch_size = 64

# 加载训练集和测试集
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    pin_memory=True if torch.cuda.is_available() else False,
    num_workers=4  # 使用多个工作进程加速数据加载
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    pin_memory=True if torch.cuda.is_available() else False,
    num_workers=4
)


# 选项2: 语音专用CNN - 基于DeepSpeech和其他语音识别模型的改进架构
class SpeechCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SpeechCNN, self).__init__()
        
        # 第一个卷积块 - 学习局部特征
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 第二个卷积块 - 学习更复杂的特征
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 第三个卷积块 - 更深的特征
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 第四个卷积块 - 更深的特征
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 使用自适应池化以适应不同输入大小
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 全连接分类器
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # 应用卷积块
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        
        # 自适应池化
        x = self.adaptive_pool(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 分类器
        x = self.classifier(x)
        return x



model = SpeechCNN(num_classes=len(labels)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

# 学习率调度器 - 帮助模型更好地收敛
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# 训练函数
def train(model, train_loader, criterion, optimizer, scheduler, num_epochs):
    best_accuracy = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        correct = 0
        total = 0
        running_loss = 0.0
        
        for features, labels in train_loader:
            # 跳过空批次
            if features.size(0) == 0:
                continue
                
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()  # 清零梯度
            outputs = model(features)  # 前向传播
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
            total += labels.size(0)
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重
            
            running_loss += loss.item()
        
        # 更新学习率
        scheduler.step()
        
        if total > 0:  # 避免除零错误
            accuracy = correct / total
            print(f'训练准确率: {correct}/{total} ({100. * accuracy:.2f}%)')
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")
            

# 测试函数
def test(model, test_loader):
    model.eval()  # 设置模型为评估模式
    correct = 0
    total = 0
    all_outputs_before_softmax = []
    all_predictions = []
    
    with torch.no_grad():  # 禁止计算梯度
        for features, labels in test_loader:
            # 跳过空批次
            if features.size(0) == 0:
                continue
                
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            all_outputs_before_softmax.append(outputs.cpu().numpy())  # 保存softmax层之前的输出
            pred = outputs.argmax(dim=1, keepdim=True)
            all_predictions.append(pred.cpu().numpy())  # 保存所有预测结果
            correct += pred.eq(labels.view_as(pred)).sum().item()
            total += labels.size(0)
    
    accuracy = 0
    if total > 0:  # 避免除零错误
        accuracy = correct / total
        print(f'\n测试准确率: {correct}/{total} ({100. * accuracy:.2f}%)\n')
    
    # 返回softmax层之前的输出和预测结果
    return all_outputs_before_softmax, all_predictions, accuracy

# 训练模型
# num_epochs = 5
# train(model, train_loader, criterion, optimizer, scheduler, num_epochs)

# 保存最终模型
model_save_path = './boundary_robustness/models/speech/speech_cnn_10class.pth'
# torch.save(model.state_dict(), model_save_path)
# print(f'最终模型已保存到 {model_save_path}')

# 加载模型并测试
print("加载最佳模型...")
best_model = SpeechCNN(num_classes=len(labels)).to(device)  # 确保与训练的模型架构一致
best_model.load_state_dict(torch.load(model_save_path))
best_model.eval()

print("测试最佳模型...")
outputs, predictions, acc = test(best_model, test_loader)

class_num = len(labels)
# 倒数第二层的输出
feature_vector = [[] for i in range(class_num)]

for i in range(len(outputs)):
    for j in range(len(outputs[i])):
        pred_class = predictions[i][j][0]
        feature_vector[pred_class].append(outputs[i][j])

FSC = ROBY.feature_sta(feature_vector)
print("FSC = {}".format(FSC))

RDI = RDI.features(feature_vector)
print("RDI = ", RDI)

print("Evaluating under PGD attack...")
SirenAttack.evaluate_siren_attack(best_model, test_loader, epsilon=0.5, alpha=0.05, steps=50, freq_range=(20, 8000))
