import torch
import torch.nn.functional as F

def evaluate_siren_attack(model, test_loader, epsilon=0.3, alpha=0.03, steps=100, freq_range=(20, 8000)):
    """
    SirenAttack
    解决 "cudnn RNN backward can only be called in training mode" 错误
    """
    device = next(model.parameters()).device
    
    clean_correct = 0
    adv_correct = 0
    total = 0
    
    # 保存模型原始状态
    was_training = model.training
    
    for features, labels in test_loader:
        if features.size(0) == 0:
            continue
            
        features, labels = features.to(device), labels.to(device)
        batch_size, channels, mel_bins, time_frames = features.shape
        
        # 计算频率范围掩码
        min_freq_idx = max(0, int(freq_range[0] / (8000 / mel_bins)))
        max_freq_idx = min(mel_bins - 1, int(freq_range[1] / (8000 / mel_bins)))
        freq_mask = torch.zeros_like(features)
        freq_mask[:, :, min_freq_idx:max_freq_idx+1, :] = 1.0
        
        # 常规预测 (评估模式)
        model.eval()
        with torch.no_grad():
            outputs = model(features)
            clean_pred = outputs.argmax(dim=1)
            clean_correct += (clean_pred == labels).sum().item()
        
        # 初始化对抗样本
        adv_features = features.clone().detach()
        
        # 为CRNN模型添加噪声初始化
        noise = torch.FloatTensor(features.shape).uniform_(-epsilon/3, epsilon/3).to(device)
        noise = noise * freq_mask
        adv_features = torch.clamp(adv_features + noise, -80.0, 0.0)
        
        # 迭代添加扰动
        for i in range(steps):
            adv_features.requires_grad = True
            
            # 为RNN反向传播设置训练模式，但冻结BN层状态
            model.train()
            # 冻结所有BN层
            for module in model.modules():
                if isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.BatchNorm1d):
                    module.eval()
                    module.weight.requires_grad = False
                    module.bias.requires_grad = False
            
            # 前向传播
            outputs = model(adv_features)
            
            # 损失计算 - 使用更强的多类别边界损失
            onehot = torch.zeros(outputs.shape).to(device)
            onehot.scatter_(1, labels.unsqueeze(1), 1)
            real = (onehot * outputs).sum(dim=1)
            other = ((1. - onehot) * outputs - onehot * 10000.).max(dim=1)[0]
            loss = (other - real + 20).clamp(min=0)
            loss = loss.sum()
            
            # 反向传播
            model.zero_grad()
            loss.backward()
            
            # 获取并处理梯度
            with torch.no_grad():
                if adv_features.grad is None:
                    # 使用随机梯度
                    grad = torch.randn_like(adv_features) * 0.01
                else:
                    grad = adv_features.grad.clone()
                
                # 应用频率掩码
                grad = grad * freq_mask
                
                # 更新对抗样本 - 使用PGD方法
                adv_features = adv_features.detach() + alpha * torch.sign(grad)
                
                # 限制总扰动
                delta = adv_features - features
                delta = torch.clamp(delta, -epsilon, epsilon)
                adv_features = features + delta
                
                # 确保在有效范围内
                adv_features = torch.clamp(adv_features, -80.0, 0.0)
        
        # 恢复评估模式进行测试
        model.eval()
        
        # 对抗样本的预测
        with torch.no_grad():
            adv_outputs = model(adv_features)
            adv_pred = adv_outputs.argmax(dim=1)
            adv_correct += (adv_pred == labels).sum().item()
        
        total += labels.size(0)
    
    # 恢复模型原始状态
    if was_training:
        model.train()
    else:
        model.eval()
    
    # 计算准确率
    clean_acc = 100. * clean_correct / total
    adv_acc = 100. * adv_correct / total
    attack_success_rate = 100. - adv_acc
    
    print(f"正常样本准确率: {clean_acc:.2f}%")
    print(f"对抗样本准确率: {adv_acc:.2f}%")
    print(f"攻击成功率: {attack_success_rate:.2f}%")
    
    return clean_acc, adv_acc
