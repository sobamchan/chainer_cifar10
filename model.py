import numpy as np
import chainer.links as L
import chainer.functions as F
from chainer import Variable

from sobamchan.sobamchan_chainer import Model

class CNN(Model):

    def __init__(self):
        super(CNN, self).__init__(
            conv1=L.Convolution2D(3, 1, ksize=4, stride=1),
            conv2=L.Convolution2D(1, 3, ksize=4, stride=1),
            fc=L.Linear(None, 10)
        )

    def __call__(self, x, t, train=True):
        x = self.fwd(x, train)
        return F.softmax_cross_entropy(x, t), F.accuracy(x, t)

    def fwd(self, x, train):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.fc(x)
        return x

class ExpoCNN(Model):

    def __init__(self):
        super(ExpoCNN, self).__init__(
            conv1=L.Convolution2D(3, 1, ksize=4, stride=1),
            conv2=L.Convolution2D(1, 3, ksize=4, stride=1),
            fc=L.Linear(None, 10)
        )

    def __call__(self, x, t, train=True):
        x = self.fwd(x, train)
        return F.softmax_cross_entropy(x, t), F.accuracy(x, t)

    def fwd(self, x, train):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x**2
        x = self.fc(x)
        return x

class SigCNN(Model):

    def __init__(self):
        super(SigCNN, self).__init__(
            conv1=L.Convolution2D(3, 1, ksize=4, stride=1),
            conv2=L.Convolution2D(1, 3, ksize=4, stride=1),
            fc=L.Linear(None, 10)
        )

    def __call__(self, x, t, train=True):
        x = self.fwd(x, train)
        return F.softmax_cross_entropy(x, t), F.accuracy(x, t)

    def fwd(self, x, train):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.sigmoid(x)
        x = self.fc(x)
        return x

class ResCNN(Model):

    def __init__(self):
        super(ResCNN, self).__init__(
            conv1=L.Convolution2D(3, 1, ksize=4, stride=1),
            conv2=L.Convolution2D(1, 3, ksize=4, stride=1),
            fc=L.Linear(None, 10)
        )

    def __call__(self, x, t, train=True):
        x = self.fwd(x, train)
        return F.softmax_cross_entropy(x, t), F.accuracy(x, t)

    def fwd(self, x, train):
        _, _, x_h, x_w = x.shape
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h_b, h_c, h_h, h_w = h.shape
        if x_h != h_h and x_w != h_w:
            pad = Variable(np.zeros((h_b, h_c, x_h-h_h, h_w)).astype(np.float32), volatile=x.volatile)
            if np.ndarray is not type(h.data):
                pad.to_gpu()
            h = F.concat((h, pad), axis=2)
            pad = Variable(np.zeros((h_b, h_c, x_h, x_w-h_w)).astype(np.float32), volatile=x.volatile)
            if np.ndarray is not type(h.data):
                pad.to_gpu()
            h = F.concat((h, pad), axis=3)
        h += x
        h = self.fc(h)
        return h


class ResNCNN(Model):

    def __init__(self):
        super(ResNCNN, self).__init__(
            conv1=L.Convolution2D(3, 1, ksize=4, stride=1),
            conv2=L.Convolution2D(1, 3, ksize=4, stride=1),
            fc=L.Linear(None, 10)
        )

    def __call__(self, x, t, train=True):
        x = self.fwd(x, train)
        return F.softmax_cross_entropy(x, t), F.accuracy(x, t)

    def fwd(self, x, train):
        n = 2
        _, _, x_h, x_w = x.shape
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h_b, h_c, h_h, h_w = h.shape
        if x_h != h_h and x_w != h_w:
            pad = Variable(np.zeros((h_b, h_c, x_h-h_h, h_w)).astype(np.float32), volatile=x.volatile)
            if np.ndarray is not type(h.data):
                pad.to_gpu()
            h = F.concat((h, pad), axis=2)
            pad = Variable(np.zeros((h_b, h_c, x_h, x_w-h_w)).astype(np.float32), volatile=x.volatile)
            if np.ndarray is not type(h.data):
                pad.to_gpu()
            h = F.concat((h, pad), axis=3)
        for _ in range(n):
            h += x
        h = self.fc(h)
        return h
