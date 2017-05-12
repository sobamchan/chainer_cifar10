from trainer import train
from model import CNN, ResCNN, ResNCNN, ExpoCNN, SigCNN
from resnet import ResNet

def main():
    train(ResNCNN)

if __name__ == '__main__':
    main()
