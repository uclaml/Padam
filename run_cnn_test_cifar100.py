import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

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

def dist(x0, xt):
    norm_dis = 0
    for p0, pt in zip(x0.parameters(), xt.parameters()):
        norm_dis += (p0.data - pt.data).norm(2)**2
    norm_dis = np.sqrt(norm_dis)   
    return norm_dis

 
parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--method', '-m', help='optimization method')
parser.add_argument('--net', '-n', help='network archtecture')
parser.add_argument('--partial', default=1/8, type=float, help='partially adaptive parameter p in Padam')
parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
parser.add_argument('--Nepoch', default=100, type=int, help='number of epoch')
 
args = parser.parse_args()
    

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
train_errs = []
test_errs = []
train_losses = []
test_losses = []
dists = []


print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
])


trainset = torchvision.datasets.CIFAR100(root='./data/CIFAR100', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root='./data/CIFAR100', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

  
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/cnn_cifar100_'+args.method)
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
    model = vgg.VGG('VGG16', num_classes = 100)
elif args.net == 'resnet':
    from models import resnet
    model = resnet.ResNet18(num_classes = 100)
elif args.net == 'wideresnet':
    from models import wideresnet
    model = wideresnet.WResNet_cifar10(num_classes = 100, depth=16, multiplier=4)
else:
    print ('Network undefined!')


if use_cuda:
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
    
x0 = deepcopy(model)

criterion = nn.CrossEntropyLoss()

if args.method == 'sgdm':
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum= 0.9, weight_decay = args.wd)
elif args.method == 'adam':
    import Adam
    optimizer = Adam.Adam(model.parameters(), lr=args.lr, weight_decay = args.wd)
elif args.method == 'amsgrad':
    import Adam
    optimizer = Adam.Adam(model.parameters(), lr=args.lr, amsgrad = True, weight_decay = args.wd)
elif args.method == 'padam':
    import Padam
    optimizer = Padam.Padam(model.parameters(), lr=args.lr, partial = args.partial, weight_decay = args.wd)
else:
    print ('Optimizer undefined!')
    
scheduler = MultiStepLR(optimizer, milestones=[30,60,80], gamma=0.1)

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
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step(closure)

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
 
    # Compute training error 
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    train_errs.append(1 - correct/total)
    train_losses.append(train_loss/(batch_idx+1))

    model.eval() # Testing
 
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    test_errs.append(1 - correct/total)
    test_losses.append(test_loss/(batch_idx+1))

    # Save checkpoint
    acc = 100.*correct/total
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
        torch.save(state, './checkpoint/cnn_cifar100_' + args.method)
        best_acc = acc
         
    d = dist(x0, model)
    dists.append(d)
    print ('Dist: ', d)
 
 
