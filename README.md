# Padam
Partially Adaptive Momentum Estimation (Padam)

Prerequisites: 
* Pytorch
* CUDA

Usage:
run run_cnn_test_cifar10.py for experiments on Cifar10 and run_cnn_test_cifar100.py for experiments on Cifar100

Parameters:
* --lr: (start) learning rate 
* --method: optimization method, e.g., sgdm, adam, amsgrad, padam
* --net: network architecture, e.g. vggnet, resnet, wideresnet
* --partial: partially adaptive parameter for Padam method
* --wd: weight decay
* --Nepoch: number of training epochs
* --resume: whether resume from previous training process

Usage Example:
* Run experiments on Cifar10:
  -  python run_cnn_test_cifar10.py  --lr 0.1 --method "padam" --net "vggnet"  --partial 0.125 --wd 5e-4
* Run experiments on Cifar100:
  -  python run_cnn_test_cifar100.py  --lr 0.1 --method "padam" --net "resnet"  --partial 0.125 --wd 5e-4

Note:
* Adam.py is a copy of the newest offical Pytorch implementation of Adam method, with Amsgrad support. We add it to support earlier version of Pytorch.
