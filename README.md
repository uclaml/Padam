# Padam
This repository contains our pytorch implementation of Partially Adaptive Momentum Estimation method (Padam) in the paper [Closing the Generalization Gap of Adaptive Gradient Methods in Training Deep Neural Networks](https://arxiv.org/abs/1806.06763) (accepted by IJCAI 2020). 

## Prerequisites: 
* Pytorch
* CUDA

## Usage:
Use python to run run_cnn_test_cifar10.py for experiments on [Cifar10](https://www.cs.toronto.edu/~kriz/cifar.html) and run_cnn_test_cifar100.py for experiments on [Cifar100](https://www.cs.toronto.edu/~kriz/cifar.html)

## Command Line Arguments:
* --lr: (start) learning rate 
* --method: optimization method, e.g., "sgdm", "adam", "amsgrad", "padam"
* --net: network architecture, e.g. ["vggnet"](https://arxiv.org/abs/1409.1556), ["resnet"](https://arxiv.org/abs/1512.03385), ["wideresnet"](https://arxiv.org/abs/1605.07146)
* --partial: partially adaptive parameter for Padam method
* --wd: weight decay
* --Nepoch: number of training epochs
* --resume: whether resume from previous training process

## Usage Examples:
* Run experiments on Cifar10:
```bash
  -  python run_cnn_test_cifar10.py  --lr 0.1 --method "padam" --net "vggnet"  --partial 0.125 --wd 5e-4
```
* Run experiments on Cifar100:
```bash
  -  python run_cnn_test_cifar100.py  --lr 0.1 --method "padam" --net "resnet"  --partial 0.125 --wd 5e-4
```
## Citation
Please check our paper for technical details and full results. 

```
@inproceedings{chen2020closing,
  title={Closing the Generalization Gap of Adaptive Gradient Methods in Training Deep Neural Networks},
  author={Chen, Jinghui and Zhou, Dongruo and Tang, Yiqi and Yang, Ziyan and Cao, Yuan and Gu, Quanquan},
  booktitle={International Joint Conferences on Artificial Intelligence},
  year={2020}
}
```
