import numpy as np
import cupy as cp

import chainer
from chainer import reporter
from chainer import functions as F
from chainer import links as L

from lib.loss import WeightedNormalizedAbsoluteError


def zero_plus(x):
    return F.softplus(x) - 0.6931472


class ElementLayerNormalization(chainer.links.LayerNormalization):

    def __call__(self, x):

        shape = x.shape
        h = F.reshape(x, (-1, shape[-1]))
        h = super(ElementLayerNormalization, self).__call__(h)
        h = F.reshape(h, shape)
        return h


class ElementLinear(chainer.links.Linear):

    def __call__(self, x):

        shape = x.shape
        h = F.reshape(x, (-1, shape[-1]))
        h = super(ElementLinear, self).__call__(h)
        shape_after = shape[:-1] + (self.out_size,)
        h = F.reshape(h, shape_after)
        return h


class EdgeUpdate(chainer.Chain):

    def __init__(self, C):
        super(EdgeUpdate, self).__init__()
        with self.init_scope():
            self.W1 = ElementLinear(2 * C, nobias=True)
            self.W2 = ElementLinear(C, nobias=True)
            self.bn = ElementLayerNormalization(C)

    def __call__(self, edge, h):
        num_atom = edge.shape[1]
        h1 = F.tile(F.expand_dims(h, 1), (1, num_atom, 1, 1))
        h2 = F.tile(F.expand_dims(h, 2), (1, 1, num_atom, 1))
        concat = F.concat([h1, h2, edge], axis=3)

        add = zero_plus(self.W2(zero_plus(self.W1(concat))))

        return edge + self.bn(add)


class InteractionNetwork(chainer.Chain):

    def __init__(self, C):
        super(InteractionNetwork, self).__init__()
        with self.init_scope():
            self.W1 = ElementLinear(C, nobias=True)
            self.W2 = ElementLinear(C, nobias=True)
            self.W3 = ElementLinear(C, nobias=True)
            self.W4 = ElementLinear(C, nobias=True)
            self.W5 = ElementLinear(C, nobias=True)
            self.bn = ElementLayerNormalization(C)

    def __call__(self, h, edge):
        mt = zero_plus(self.W3(zero_plus(self.W2(edge))))
        mt = self.W1(h) * F.sum(mt, axis=1)
        h_add = self.W5(zero_plus(self.W4(mt)))

        return h + self.bn(h_add)


class EdgeUpdateNet(chainer.Chain):

    def __init__(self):
        super().__init__()
        self.num_layer = 6
        node_dim = 64
        edge_dim = 64
        self.edge_dim = 64
        self.loss = WeightedNormalizedAbsoluteError()

        with self.init_scope():
            self.gn1 = ElementLinear(node_dim)
            for layer in range(self.num_layer):
                self.add_link('eup{}'.format(layer), EdgeUpdate(edge_dim))
                self.add_link('int{}'.format(layer), InteractionNetwork(node_dim))

            self.gn2 = ElementLinear(8)
            self.n1 = L.Linear(256)
            self.n2 = L.Linear(5)

            self.conv_3d_1 = L.Convolution3D(1, 4, ksize=3, pad=1, stride=2)
            self.conv_3d_2 = L.Convolution3D(4, 8, ksize=3, pad=1)
            self.conv_3d_3 = L.Convolution3D(8, 16, ksize=3, pad=1)
            self.conv_3d_4 = L.Convolution3D(16, 32, ksize=3, pad=1)

            self.bn1 = L.BatchNormalization(4)
            self.bn2 = L.BatchNormalization(8)
            self.bn3 = L.BatchNormalization(16)
            self.bn4 = L.BatchNormalization(32)

    def __call__(self, targets, **kwargs):
        out = self.predict(**kwargs)
        loss = self.loss(out, targets)
        reporter.report({'loss': loss}, self)
        return loss

    def predict(self, loading, fnc, net_type, spatial_map, **kwargs):

        batch_size, channel, x, y, z = spatial_map.shape

        xp = chainer.cuda.get_array_module(spatial_map)
        spatial_map = spatial_map.astype(xp.float32) / 3000.0
        spatial_map[spatial_map > 2.0] = 2.0
        spatial_map[spatial_map < -2.0] = -2.0

        sm = spatial_map.reshape(batch_size * channel, 1, x, y, z)

        sm = F.pad(sm, ((0, 0), (0, 0), (6, 5), (1, 0), (6, 6)), 'constant', constant_values=0)

        sm = F.max_pooling_3d(F.relu(self.bn1(self.conv_3d_1(sm))), ksize=2)  # 64 x 64
        sm = F.max_pooling_3d(F.relu(self.bn2(self.conv_3d_2(sm))), ksize=2)  # 32 x 32
        sm = F.max_pooling_3d(F.relu(self.bn3(self.conv_3d_3(sm))), ksize=2)  # 16 x 16
        sm = F.max_pooling_3d(F.relu(self.bn4(self.conv_3d_4(sm))), ksize=4)  # 8 x 8

        sm = sm.reshape(batch_size, channel, 32)

        h = self.gn1(F.concat([net_type, sm], axis=2))
        e = F.expand_dims(fnc, 3)

        # adj = (xp.abs(fnc) > 1e-5).astype(np.float32)
        # adj_mask = F.tile(F.expand_dims(adj, 3), (1, 1, 1, self.edge_dim))

        for layer in range(self.num_layer):
            e = self['eup{}'.format(layer)](e, h)
            h = self['int{}'.format(layer)](h, e)

        h_mean = F.mean(h, axis=1)
        h_max = F.mean(h, axis=1)
        h = F.concat([loading * 10.0, h_mean, h_max], 1)

        # h = self.gn2(h)
        # h = F.concat([loading * 10.0, h.reshape(batch_size, -1)], 1)

        h = F.relu(self.n1(h))
        out = self.n2(h)
        return out


def main():

    model = EdgeUpdateNet()
    model.to_gpu()

    batch_size = 8
    loading = cp.zeros((batch_size, 26), dtype=cp.float32)
    fnc = cp.zeros((batch_size, 53, 53), dtype=cp.float32)
    net_type = cp.zeros((batch_size, 53, 7), dtype=cp.float32)
    spatial_map = cp.zeros((batch_size, 53, 53, 63, 52), dtype=cp.int16)

    out = model.predict(loading, fnc, net_type, spatial_map)
    print(out.shape)


if __name__ == '__main__':
    main()
