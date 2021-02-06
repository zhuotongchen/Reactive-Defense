import torch
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import os
import numpy as np
import random
import argparse

from model import *
from train_models import training
from test_models import testing, manifold_attack

# For reproducibility
torch.manual_seed(999)
np.random.seed(999)
random.seed(999)
torch.cuda.manual_seed_all(999)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic=True


device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='Train neural networks')
parser.add_argument('--data_set', type=str, default='cifar10', help='Can be either cifar10 | cifar100')
parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='l2 regularization')
parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
parser.add_argument('--batch_size', default=128, type=int, help='batch_size')
parser.add_argument('--learning_rate', default=0.1, type=float, help='learning_rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--epsilon', default=0.031, type=float, help='adversarial training epsilon')
parser.add_argument('--step_size', default=0.007, type=float, help='adversarial training step_size')
parser.add_argument('--num_steps', default=10, type=int, help='adversarial training num_steps')
parser.add_argument('--begin_epoch', default=60, type=int, help='begin epoch for GAIRAT')


parser.add_argument('--training_method', default='standard', type=str, help='standard, pgd, GAIRAT',
                    choices=['standard, pgd, GAIRAT'])
parser.add_argument('--number_of_workers', default=0, type=int, help='number_of_workers')

args = parser.parse_args()
workers = args.number_of_workers
batch_size = args.batch_size

if args.data_set == 'cifar10':
    num_classes = 10
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
        ]), download=True),
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
        ])),
        batch_size=128, shuffle=False,
        num_workers=workers, pin_memory=True)
elif args.data_set == 'cifar100':
    num_classes = 100
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
        ]), download=True),
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
        ])),
        batch_size=128, shuffle=False,
        num_workers=workers, pin_memory=True)


print('==> Building model..')
net = resnet20(num_classes=num_classes)
net = net.to(device)

training(train_loader, test_loader, net, args, device)




