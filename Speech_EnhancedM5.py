import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio.transforms import MelSpectrogram, Resample
import os
import RDI
import PGD_attack
import RFGSM
import Square_Attack
import gauss
import CW_attack
import SirenAttack

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the dataset class, here using the SpeechCommands dataset
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

# Only 10 categories will be selected
labels = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']

label_map = {label: i for i, label in enumerate(labels)}

# Create an audio pre-processing converter
melspec = MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    hop_length=512,
    n_mels=64
)
amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

# Preprocess audio and extract features
def extract_features(waveform):
    """Extract Mel spectrogram features"""
    if waveform.shape[1] < 16000:
        padding = torch.zeros(1, 16000 - waveform.shape[1])
        waveform = torch.cat([waveform, padding], dim=1)
    else:
        waveform = waveform[:, :16000]
    
    with torch.no_grad():
        spec = melspec(waveform)
        spec_db = amplitude_to_db(spec)
        spec_db = spec_db.unsqueeze(0)
    
    return spec_db

def collate_fn(batch):
    features = []
    labels = []
    for waveform, sample_rate, label, speaker_id, utterance_number in batch:
        if label in label_map:
            feature = extract_features(waveform)
            features.append(feature)
            labels.append(label_map[label])
    
    if len(features) == 0:
        return torch.empty(0), torch.empty(0, dtype=torch.long)
    
    features = torch.cat(features, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return features, labels

train_dataset = SubsetSC("training")
test_dataset = SubsetSC("testing")
val_dataset = SubsetSC("validation")

batch_size = 64

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    pin_memory=True if torch.cuda.is_available() else False,
    num_workers=4
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    pin_memory=True if torch.cuda.is_available() else False,
    num_workers=4
)

class EnhancedM5(nn.Module):
    def __init__(self, n_input=1, n_output=10, stride=16, n_channel=64):
        super().__init__()
        self.conv1 = nn.Conv2d(n_input, n_channel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(n_channel)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(n_channel, n_channel, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(n_channel)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(n_channel, 2 * n_channel, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(2 * n_channel)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(2 * n_channel, 2 * n_channel, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(2 * n_channel)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2 * n_channel, 2 * n_channel // 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(2 * n_channel // 8, 2 * n_channel, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size=1)
        
        self.fc1 = nn.Linear(2 * n_channel, n_output)
        
    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = nn.functional.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = nn.functional.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = nn.functional.relu(self.bn4(x))
        x = self.pool4(x)
        
        attn = self.channel_attention(x)
        x = x * attn
        
        x = self.adaptive_pool(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        return x

model = EnhancedM5(n_input=1, n_output=len(labels)).to(device)

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
optimizer = optim.Adam(model.parameters(), lr=0.001,weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# train model
def train(model, train_loader, criterion, optimizer, scheduler, num_epochs):
    best_accuracy = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        correct = 0
        total = 0
        running_loss = 0.0
        
        for features, labels in train_loader:
        
            if features.size(0) == 0:
                continue
                
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)  
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
            total += labels.size(0)
            loss = criterion(outputs, labels) 
            loss.backward() 
            optimizer.step() 
            
            running_loss += loss.item()
        
        scheduler.step()
        
        if total > 0:  
            accuracy = correct / total
            print(f'train Acc: {correct}/{total} ({100. * accuracy:.2f}%)')
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")
            

# test model
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_outputs_before_softmax = []
    all_predictions = []
    
    with torch.no_grad():  
        for features, labels in test_loader:
            if features.size(0) == 0:
                continue
                
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            all_outputs_before_softmax.append(outputs.cpu().numpy()) 
            pred = outputs.argmax(dim=1, keepdim=True)
            all_predictions.append(pred.cpu().numpy()) 
            correct += pred.eq(labels.view_as(pred)).sum().item()
            total += labels.size(0)
    
    accuracy = 0
    if total > 0:  
        accuracy = correct / total
        print(f'\ntest acc: {correct}/{total} ({100. * accuracy:.2f}%)\n')

    return all_outputs_before_softmax, all_predictions, accuracy

num_epochs = 5
train(model, train_loader, criterion, optimizer, scheduler, num_epochs)

model_save_path = './boundary_robustness/models/speech/speech_enhancedM5_10class.pth'
torch.save(model.state_dict(), model_save_path)
print(f'The final model has been saved to {model_save_path}')

best_model = EnhancedM5().to(device)
best_model.load_state_dict(torch.load(model_save_path))
best_model.eval()

outputs, predictions, acc = test(best_model, test_loader)

class_num = len(labels)
feature_vector = [[] for i in range(class_num)]

for i in range(len(outputs)):
    for j in range(len(outputs[i])):
        pred_class = predictions[i][j][0]
        feature_vector[pred_class].append(outputs[i][j])

RDI = RDI.features(feature_vector)
print("RDI = ", RDI)

print("Evaluating under SirenAttack attack...")
SirenAttack.evaluate_siren_attack(best_model, test_loader, epsilon=0.5, alpha=0.05, steps=50, freq_range=(20, 8000))
