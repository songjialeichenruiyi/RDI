import torch
import torch.nn.functional as F

def evaluate_siren_attack(model, test_loader, epsilon=0.3, alpha=0.03, steps=100, freq_range=(20, 8000)):
    """
    SirenAttack
    """
    device = next(model.parameters()).device
    
    clean_correct = 0
    adv_correct = 0
    total = 0
    
    was_training = model.training
    
    for features, labels in test_loader:
        if features.size(0) == 0:
            continue
            
        features, labels = features.to(device), labels.to(device)
        batch_size, channels, mel_bins, time_frames = features.shape
        
        min_freq_idx = max(0, int(freq_range[0] / (8000 / mel_bins)))
        max_freq_idx = min(mel_bins - 1, int(freq_range[1] / (8000 / mel_bins)))
        freq_mask = torch.zeros_like(features)
        freq_mask[:, :, min_freq_idx:max_freq_idx+1, :] = 1.0
        
        model.eval()
        with torch.no_grad():
            outputs = model(features)
            clean_pred = outputs.argmax(dim=1)
            clean_correct += (clean_pred == labels).sum().item()
        
        adv_features = features.clone().detach()
        
        noise = torch.FloatTensor(features.shape).uniform_(-epsilon/3, epsilon/3).to(device)
        noise = noise * freq_mask
        adv_features = torch.clamp(adv_features + noise, -80.0, 0.0)
        
        for i in range(steps):
            adv_features.requires_grad = True
            
            model.train()
            for module in model.modules():
                if isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.BatchNorm1d):
                    module.eval()
                    module.weight.requires_grad = False
                    module.bias.requires_grad = False
            
            outputs = model(adv_features)
            
            onehot = torch.zeros(outputs.shape).to(device)
            onehot.scatter_(1, labels.unsqueeze(1), 1)
            real = (onehot * outputs).sum(dim=1)
            other = ((1. - onehot) * outputs - onehot * 10000.).max(dim=1)[0]
            loss = (other - real + 20).clamp(min=0)
            loss = loss.sum()
            

            model.zero_grad()
            loss.backward()
            
            with torch.no_grad():
                if adv_features.grad is None:
           
                    grad = torch.randn_like(adv_features) * 0.01
                else:
                    grad = adv_features.grad.clone()
                
                grad = grad * freq_mask
                
                adv_features = adv_features.detach() + alpha * torch.sign(grad)
                
                delta = adv_features - features
                delta = torch.clamp(delta, -epsilon, epsilon)
                adv_features = features + delta
                
                adv_features = torch.clamp(adv_features, -80.0, 0.0)
        
        model.eval()
        
        with torch.no_grad():
            adv_outputs = model(adv_features)
            adv_pred = adv_outputs.argmax(dim=1)
            adv_correct += (adv_pred == labels).sum().item()
        
        total += labels.size(0)
    
    if was_training:
        model.train()
    else:
        model.eval()
    
    clean_acc = 100. * clean_correct / total
    adv_acc = 100. * adv_correct / total
    attack_success_rate = 100. - adv_acc
    
    print(f"Normal samples accuracy: {clean_acc:.2f}%")
    print(f"Adversarial samples accuracy: {adv_acc:.2f}%")
    print(f"ASR: {attack_success_rate:.2f}%")
    
    return clean_acc, adv_acc
