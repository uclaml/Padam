import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import os
import argparse

from models import *
from utils import progress_bar
from torch.autograd import Variable
 
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingLR


import json
from copy import deepcopy

 
    
parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--method', '-m', help='optimization method')
parser.add_argument('--net', '-n', help='network archtecture')
parser.add_argument('--partial', default=1/8, type=float, help='partially adaptive parameter p in Padam')
parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
parser.add_argument('--Nepoch', default=200, type=int, help='number of epoch')
parser.add_argument('--beta1', default=0.9, type=float, help='beta1')
parser.add_argument('--beta2', default=0.999, type=float, help='beta2')


args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
train_errs = []
test_errs = []
train_losses = []
test_losses = []

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

 
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/cnn_cifar10_'+args.method)
    model = checkpoint['model']
    start_epoch = checkpoint['epoch']
    train_losses = checkpoint['train_losses']
    test_losses = checkpoint['test_losses']
    train_errs = checkpoint['train_errs']
    test_errs = checkpoint['test_errs']
else:
    print('==> Building model..')


if args.net == 'vggnet':
    from models import vgg
    model = vgg.VGG('VGG16', num_classes = 10)
#     model = models.vgg16_bn(num_classes=10)
elif args.net == 'resnet':
    from models import resnet
    model = resnet.ResNet18(num_classes = 10)
#     model = models.resnet18(num_classes=10)
elif args.net == 'wideresnet':
    from models import wideresnet
    model = wideresnet.WResNet_cifar10(num_classes = 10, depth=16, multiplier=4)
else:
    print ('Network undefined!')



if use_cuda:
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
 
    
criterion = nn.CrossEntropyLoss()

betas = (args.beta1, args.beta2)
import Padam
optimizer = Padam.Padam(model.parameters(), lr=args.lr, partial = args.partial, weight_decay = args.wd, betas = betas)
 

scheduler = MultiStepLR(optimizer, milestones=[100,150], gamma=0.1)


for epoch in range(start_epoch+1, args.Nepoch+1):
    
    scheduler.step() 
    print ('\nEpoch: %d' % epoch, ' Learning rate:', scheduler.get_lr())
    model.train()  # Training

    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        def closure():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            return loss

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step(closure)

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.0/total*(correct), correct, total))
 
    # Compute training error 
    
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.0/total*(correct), correct, total))
    train_errs.append(1 - correct/total)
    train_losses.append(train_loss/(batch_idx+1))

    model.eval() # Testing
 
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().item()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.0/ total *(correct), correct, total))
    test_errs.append(1 - correct/total)
    test_losses.append(test_loss/(batch_idx+1))

    # Save checkpoint
    acc = 100.0/total*(correct)
    if acc > best_acc:
        print('Saving..')
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        state = {
                'model': model,
                'epoch': epoch,
                'train_errs':train_errs,
                'test_errs':test_errs,
                'train_losses':train_losses,
                'test_losses':test_losses
                }
        torch.save(state, './checkpoint/cnn_cifar10_' + args.method)
        best_acc = acc
        
 
     