import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# 检查是否有可用的GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 定义L∞ PGD攻击类
class LinfPGD:
    def __init__(self, model, epsilon, num_steps, step_size, random_start):
        """Attack parameter initialization. The attack performs k steps of
           size a, while always staying within epsilon from the initial
           point."""
        self.model = model
        self.epsilon = epsilon  # 扰动范围
        self.num_steps = num_steps  # 攻击步数
        self.step_size = step_size  # 每一步步长
        self.random_start = random_start  # 是否进行随机初始化
        self.loss_fn = nn.CrossEntropyLoss()  # 损失函数和tensorflow版相同

    def perturb(self, x_nat, y):
        """
        Given a set of examples (x_nat, y), returns a set of adversarial
        examples within epsilon of x_nat in l_infinity norm.
        """
        x = x_nat.clone().detach().to(torch.device('cuda'))  # 确保x在CUDA上
        y = y.to(torch.device('cuda'))

        if self.random_start:
            # 随机初始化，在 epsilon 范围内加扰动
            x = x + torch.empty_like(x).uniform_(-self.epsilon, self.epsilon)
            x = torch.clamp(x, 0, 1)  # 确保图片像素值依然在 [0, 1] 范围内

        # 进行多步迭代的PGD攻击
        for i in range(self.num_steps):
            # 每次迭代都需要让 x 可计算梯度
            x.requires_grad_()

            # 计算模型输出
            outputs = self.model(x)
            loss = self.loss_fn(outputs, y)

            # 清空之前的梯度，进行反向传播
            self.model.zero_grad()
            loss.backward()

            # 获取梯度信息，并更新x
            grad = x.grad.data.sign()
            x = x + self.step_size * grad

            # 将扰动限制在无穷范数范围内，并确保加上扰动后的像素值合法
            x = torch.clamp(x, x_nat - self.epsilon, x_nat + self.epsilon)
            x = torch.clamp(x, 0, 1)  # 再次确保像素值在 [0, 1] 范围

            # Detach x，防止计算图一直累积
            x = x.detach()

        return x  # 返回对抗样本

    

# 使用PGD攻击测试模型的性能
def test_with_pgd_attack(model, test_loader, epsilon=0.3, alpha=0.01, iters=40):
    attack = LinfPGD(model = model,
                         epsilon = 0.3,
                         num_steps = 40,
                         step_size = 0.01,
                         random_start = True)
    model.eval()
    correct = 0
    total = 0

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        adv_images = attack.perturb(images, labels)  # 生成对抗样本
        outputs = model(adv_images)
        _, predicted = torch.max(outputs, 1)
        # print(predicted)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy under PGD attack: {accuracy:.2f}%')


# 使用PGD攻击测试模型的性能
def save_pgd_attack(model, test_loader, epsilon=0.3, alpha=0.01, iters=40):
    attack = LinfPGD(model = model,
                         epsilon = 0.3,
                         num_steps = 40,
                         step_size = 0.01,
                         random_start = True)
    
    adv_examples = []
    model.eval()
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        adv_images = attack.perturb(images, labels)
        adv_examples.append((adv_images, labels))
    return adv_examples