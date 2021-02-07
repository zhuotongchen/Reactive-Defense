import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

from Adversarial_attack import fgsm, Random, pgd, CW_attack, Manifold_attack

def testing(test_loader, model, step_size, eps, attack='None', device=None):    
    model.eval()
    total_step = len(test_loader)
    test_loss = 0
    correct = 0
    total = 0
    for i, (inputs, labels) in enumerate(test_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        if attack == 'None':
            x = inputs
        elif attack == 'fgsm':
            x = fgsm(inputs, labels, eps, model, device)
        elif attack == 'Random':
            x = Random(inputs, labels, eps, model, device)
        elif attack == 'pgd':
            x, _ = pgd(model, inputs, labels, epsilon=eps,
                          num_steps=20, step_size=step_size, rand_init=False, device=device)
        elif attack == 'cw':
            print('Processing CW attack on batch:', i)
            CW = CW_attack(model)
            x = CW.attack(inputs, labels, eps)
        outputs = model(x)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = 100.*correct/total
    # print('Testing accuracy:', accuracy)
    return accuracy

def manifold_attack(test_loader, model, eps, basis, device):
    model.eval()
    total_step = len(test_loader)
    test_loss = 0
    correct = 0
    total = 0
    for i, (inputs, labels) in enumerate(test_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        print('Processing CW attack on batch:', i)
        Man_attack = Manifold_attack(model, basis)
        x = Man_attack.attack(inputs, labels, eps)
        outputs = model(x)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = 100.*correct/total
    print('Testing accuracy:', accuracy)
    return accuracy        
    