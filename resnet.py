import numpy as np
import chainer
from sobamchan import sobamchan_chainer
import chainer.links as L
import chainer.functions as F
from chainer import Variable
from chainer import flag

class ResBlock(sobamchan_chainer.Model):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__(
            conv1=L.Convolution2D(in_channels, out_channels, ksize=(3,3), stride=1, pad=1),
            conv2=L.Convolution2D(out_channels, out_channels, ksize=(3,3), stride=stride, pad=1),
            bn1=L.BatchNormalization(out_channels),
            bn2=L.BatchNormalization(out_channels),
        )

    def __call__(self, x):
        h = self.fwd(x)
        return h

    def fwd(self, x):
        h = F.relu(self.bn1((self.conv1(x))))
        h = self.bn2((self.conv2(h)))
        _, x_channels, x_h, x_w = x.shape
        h_batch_size, h_channels, h_h, h_w = h.shape
        if x_h != h_h:
            x = F.max_pooling_2d(x, ksize=2, stride=2)
        if x_channels != h_channels:
            pad = Variable(np.zeros((h_batch_size, h_channels - x_channels, h_h, h_w)).astype(np.float32), volatile=x.volatile)
            if np.ndarray is not type(h.data):
                pad.to_gpu()
            x = F.concat((x, pad))
        return F.relu(h + x)

class ResNet(sobamchan_chainer.Model):

    def __init__(self, in_channels=3):
        super(ResNet, self).__init__()
        n = 3
        modules = []

        # first
        modules += [('conv', L.Convolution2D(in_channels, 16, (3,3)))]
        modules += [('bn', L.BatchNormalization(16))]
        # 16
        for i in range(5):
            modules += [('resblock_16_{}'.format(i), ResBlock(16, 16))]
        modules += [('resblock_16_{}'.format(5), ResBlock(16, 32, 2))]
        # 32
        for i in range(5):
            modules += [('resblock_32_{}'.format(i), ResBlock(32, 32))]
        modules += [('resblock_32_{}'.format(5), ResBlock(32, 64, 2))]
        # 64
        for i in range(6):
            modules += [('resblock_64_{}'.format(i), ResBlock(64, 64))]
        # last
        modules += [('fc', L.Linear(None, 10))]
        
        # register
        [ self.add_link(*link) for link in modules ]
        self.modules = modules

    def __call__(self, x, t, train=True):
        y = self.fwd(x)
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

    def fwd(self, x):
        # convs and bns
        # first
        h = self['conv'](x)
        h = self['bn'](h)
        # 16
        for i in range(5):
            h = self['resblock_16_{}'.format(i)](h)
        h = self['resblock_16_{}'.format(5)](h)
        # 32 
        for i in range(5):
            h = self['resblock_32_{}'.format(i)](h)
        h = self['resblock_32_{}'.format(5)](h)
        # 64
        for i in range(5):
            h = self['resblock_64_{}'.format(i)](h)
        h = self['resblock_64_{}'.format(5)](h)
        h = F.average_pooling_2d(h, (2,2), stride=1)
        # fc
        h = self['fc'](h)
        print('done')
        return h
