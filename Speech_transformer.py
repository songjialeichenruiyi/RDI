import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio.transforms import MelSpectrogram, Resample
import os
import math
import RDI
import SirenAttack

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

labels = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']

label_map = {label: i for i, label in enumerate(labels)}

melspec = MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    hop_length=512,
    n_mels=64
)
amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

def extract_features(waveform):
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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)

class TransformerAudioNet(nn.Module):
    def __init__(self, n_input=1, n_output=10, d_model=128, nhead=4, 
                 num_layers=4, dim_feedforward=512, dropout=0.1):
        super(TransformerAudioNet, self).__init__()
        
        self.conv1 = nn.Conv2d(n_input, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, d_model, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(d_model)
        
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.fc1 = nn.Linear(d_model, 256)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, n_output)
        
        self._initialize_weights()
        
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))  # [batch_size, d_model, h/4, w/4]
        
        batch_size, d_model, h, w = x.shape
        x = x.permute(0, 2, 3, 1)  # [batch_size, h/4, w/4, d_model]
        x = x.reshape(batch_size, h*w, d_model)  # [batch_size, seq_len, d_model]
        
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)  # [batch_size, seq_len, d_model]
        
        x = x.mean(dim=1)  # [batch_size, d_model]
        
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

model = TransformerAudioNet(n_input=1, n_output=len(labels)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

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
            print(f'train acc: {correct}/{total} ({100. * accuracy:.2f}%)')
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")
            

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

# num_epochs = 5
# train(model, train_loader, criterion, optimizer, scheduler, num_epochs)

model_save_path = './boundary_robustness/models/speech/speech_transformer_10class.pth'
# torch.save(model.state_dict(), model_save_path)

best_model = TransformerAudioNet(n_input=1, n_output=len(labels)).to(device)
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
