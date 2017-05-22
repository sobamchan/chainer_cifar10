from chainer import optimizers

from sobamchan.sobamchan_cifar10 import train
from model import CNN, ResCNN, ResNCNN, ExpoCNN, SigCNN, AutoResNCNN
from resnet import ResNet

def main():
    opts = {}

    optimizer = optimizers.AdaGrad()
    opts['optimizer'] = optimizer
    opts['model'] = CNN

    train(opts)

if __name__ == '__main__':
    main()
