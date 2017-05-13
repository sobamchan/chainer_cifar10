from sobamchan.sobamchan_cifar10 import train
from model import CNN, ResCNN, ResNCNN, ExpoCNN, SigCNN, AutoResNCNN
from resnet import ResNet

def main():
    train(AutoResNCNN)

if __name__ == '__main__':
    main()
