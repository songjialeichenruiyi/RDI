import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Defining the L∞ PGD attack class
class LinfPGD:
    def __init__(self, model, epsilon, num_steps, step_size, random_start):
        """Attack parameter initialization. The attack performs k steps of
           size a, while always staying within epsilon from the initial
           point."""
        self.model = model
        self.epsilon = epsilon  
        self.num_steps = num_steps  
        self.step_size = step_size 
        self.random_start = random_start  
        self.loss_fn = nn.CrossEntropyLoss() 

    def perturb(self, x_nat, y):
        """
        Given a set of examples (x_nat, y), returns a set of adversarial
        examples within epsilon of x_nat in l_infinity norm.
        """
        x = x_nat.clone().detach().to(torch.device('cuda')) 
        y = y.to(torch.device('cuda'))

        if self.random_start:
            # Random initialization with perturbations in the epsilon range
            x = x + torch.empty_like(x).uniform_(-self.epsilon, self.epsilon)
            x = torch.clamp(x, 0, 1)

        # Perform multi-step iterative PGD attacks
        for i in range(self.num_steps):
            x.requires_grad_()

            outputs = self.model(x)
            loss = self.loss_fn(outputs, y)

            # Empty the previous gradient for backpropagation
            self.model.zero_grad()
            loss.backward()

            # Get the gradient information and update x
            grad = x.grad.data.sign()
            x = x + self.step_size * grad

            # Limit the perturbation to an infinite number of paradigms and make sure that the pixel values after adding the perturbation are legal
            x = torch.clamp(x, x_nat - self.epsilon, x_nat + self.epsilon)
            x = torch.clamp(x, 0, 1)

            # Detach x，Prevents the calculation graph from accumulating all the time
            x = x.detach()

        return x

    

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
        adv_images = attack.perturb(images, labels)
        outputs = model(adv_images)
        _, predicted = torch.max(outputs, 1)
        # print(predicted)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy under PGD attack: {accuracy:.2f}%')


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
