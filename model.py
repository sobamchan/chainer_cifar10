import chainer.links as L
import chainer.functions as F

from sobamchan.sobamchan_chainer import Model

class CNN(Model):

    def __init__(self):
        super(CNN, self).__init__(
            conv1=L.Convolution2D(3, 1, ksize=4, stride=1),
            conv2=L.Convolution2D(1, 1, ksize=4, stride=1),
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
