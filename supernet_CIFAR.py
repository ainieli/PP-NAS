import math
import numpy as np

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import RandomNormal, Constant
from hanser.models.layers import Conv2d, Norm, Act, Linear, Pool2d, Identity, GlobalAvgPool

from hanser.models.cifar.res2net.layers import Res2Conv

from operations import OPS
from primitives import get_primitives
from genotypes import Genotype


class MixedOp(Layer):

    def __init__(self, C, stride):
        super().__init__()
        self._ops = []
        for primitive in get_primitives():
            op = OPS[primitive](C, stride)
            self._ops.append(op)

    def call(self, x, weights):
        return sum(weights[i] * op(x) for i, op in enumerate(self._ops))


class PPConv(Layer):

    def __init__(self, channels, splits):
        super().__init__()
        self.splits = splits
        C = channels // splits

        self.ops = []
        for i in range(self.splits):
            op = MixedOp(C, 1)
            self.ops.append(op)

    def call(self, x, alphas, betas):
        states = list(tf.split(x, self.splits, axis=-1))
        offset = 0
        for i in range(self.splits):
            x = sum(alphas[offset + j] * h for j, h in enumerate(states)) / tf.reduce_sum(
                alphas[offset:offset + len(states)])
            x = self.ops[i](x, betas[i])
            offset += len(states)
            states.append(x)

        return tf.concat(states[-self.splits:], axis=-1)


class Bottleneck(Layer):
    expansion = 4

    def __init__(self, in_channels, channels, stride, base_width, splits):
        super().__init__()
        out_channels = channels * self.expansion
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        width = math.floor(out_channels // self.expansion * (base_width / 64)) * splits
        self.conv1 = Conv2d(in_channels, width, kernel_size=1,
                            norm='def', act='def')
        if stride != 1 or in_channels != out_channels:
            layers = []
            if stride != 1:
                layers.append(Pool2d(3, stride=2, type='avg'))      # avd
            layers.append(
                Res2Conv(width, width, kernel_size=3, stride=1, scale=splits,
                         norm='def', act='def', start_block=True))
            self.conv2 = Sequential(layers)
        else:
            self.conv2 = PPConv(width, splits=splits)
        self.conv3 = Conv2d(width, out_channels, kernel_size=1)
        self.bn3 = Norm(out_channels, gamma_init='zeros')

        if stride != 1 or in_channels != out_channels:
            shortcut = []
            if stride != 1:
                shortcut.append(Pool2d(3, 2, type='avg'))
            shortcut.append(
                Conv2d(in_channels, out_channels, kernel_size=1, norm='def'))
            self.shortcut = Sequential(shortcut)
        else:
            self.shortcut = Identity()

        self.act = Act()

    def call(self, x, alphas, betas):
        identity = self.shortcut(x)
        x = self.conv1(x)
        if self.stride != 1 or self.in_channels != self.out_channels:
            x = self.conv2(x)
        else:
            x = self.conv2(x, alphas, betas)
        x = self.conv3(x)
        x = self.bn3(x)
        x = x + identity
        x = self.act(x)
        return x


class Network(Model):

    def __init__(self, depth=110, base_width=24, splits=4, num_classes=10, stages=(64, 64, 128, 256)):
        super().__init__()
        self.stages = stages
        self.splits = splits
        block = Bottleneck
        layers = [(depth - 2) // 9] * 3
        self.num_stages = len(layers)

        self.stem = Conv2d(3, self.stages[0], kernel_size=3, norm='def', act='def')
        self.in_channels = self.stages[0]

        self.layer1 = self._make_layer(
            block, self.stages[1], layers[0], stride=1,
            base_width=base_width, splits=splits)
        self.layer2 = self._make_layer(
            block, self.stages[2], layers[1], stride=2,
            base_width=base_width, splits=splits)
        self.layer3 = self._make_layer(
            block, self.stages[3], layers[2], stride=2,
            base_width=base_width, splits=splits)

        self.avgpool = GlobalAvgPool()
        self.fc = Linear(self.in_channels, num_classes)

        self._initialize_alphas()

        self.fair_loss_weight = self.add_weight(
            name="fair_loss_weight", shape=(),
            dtype=self.dtype, initializer=Constant(1.),
            trainable=False,
        )   # L0-1

        self.edge_loss_weight = self.add_weight(
            name="edge_loss_weight", shape=(),
            dtype=self.dtype, initializer=Constant(1.),
            trainable=False,
        )   # Lconn

        self.l2_loss_weight = self.add_weight(
            name="l2_loss_weight", shape=(),
            dtype=self.dtype, initializer=Constant(5e-4),
            trainable=False,
        )   # L2

    def param_splits(self):
        return slice(None, -2), slice(-2, None)

    def _initialize_alphas(self):
        k = sum(4 + i for i in range(self.splits))
        num_ops = len(get_primitives())

        self.alphas = self.add_weight(
            'alphas', (self.num_stages, k), initializer=Constant(0.), trainable=True,
        )
        self.betas = self.add_weight(
            'betas', (self.num_stages * self.splits, num_ops), initializer=RandomNormal(stddev=1e-3), trainable=True,
        )

    def _make_layer(self, block, channels, blocks, stride, **kwargs):
        layers = [block(self.in_channels, channels, stride=stride, **kwargs)]
        self.in_channels = channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, channels, stride=1, **kwargs))
        return layers

    def call(self, x):
        alphas = tf.nn.sigmoid(self.alphas)
        betas = tf.nn.softmax(self.betas, axis=1)

        x = self.stem(x)
        alphas = tf.cast(alphas, x.dtype)
        betas = tf.cast(betas, x.dtype)

        for l in self.layer1:
            x = l(x, alphas[0], betas[:self.splits])
        for l in self.layer2:
            x = l(x, alphas[1], betas[self.splits:2 * self.splits])
        for l in self.layer3:
            x = l(x, alphas[2], betas[2 * self.splits:])

        x = self.avgpool(x)
        x = self.fc(x)
        return x

    def arch_loss(self):
        probs = tf.nn.sigmoid(self.alphas)
        fair_loss = -tf.square((probs - 0.5))
        fair_loss = tf.reduce_mean(fair_loss)
        fair_loss = fair_loss + 0.25    # non-negative representation without changing the gradient

        k = self.splits
        n = tf.range(k)
        indices = k * n + n * (n - 1) // 2
        indices = n[:, None] + indices[None, :]
        weights = tf.gather(probs, indices, axis=1)
        weight_sum = tf.reduce_sum(weights, axis=2)
        edge_loss = tf.where(
            weight_sum > 1.,
            tf.zeros_like(weight_sum),
            tf.square(weight_sum - 1),
        )
        edge_loss = tf.reduce_mean(edge_loss)

        offset = 0
        shape = [2 * k - 1]
        indices_tmp = []
        mask_tmp = []
        for i in range(k):
            indices_stage = tf.range(k + i)[:, None]

            updates = tf.cast(tf.range(offset, offset + k + i), indices_stage.dtype)
            indices = tf.scatter_nd(indices_stage, updates, shape)
            indices_tmp.append(indices)

            mask_stage = tf.cast(tf.ones(k + i), indices_stage.dtype)
            mask = tf.scatter_nd(indices_stage, mask_stage, shape)
            mask_tmp.append(mask)

            offset += k + i
        indices = tf.stack(indices_tmp, axis=0)
        mask = tf.cast(tf.stack(mask_tmp, axis=0), probs.dtype)

        weights = tf.gather(probs, indices, axis=1)
        weight_sum = tf.reduce_sum(weights * mask[None], axis=2)
        edge_loss_2 = tf.where(
            weight_sum > 1.,
            tf.zeros_like(weight_sum),
            tf.square(weight_sum - 1),
        )
        edge_loss_2 = tf.reduce_mean(edge_loss_2)
        edge_loss = edge_loss + edge_loss_2

        l2_loss = 0.5 * (tf.reduce_mean(tf.square(self.alphas)) + tf.reduce_mean(tf.square(self.betas)))
        return self.fair_loss_weight * fair_loss + self.edge_loss_weight * edge_loss + self.l2_loss_weight * l2_loss

    def genotype(self, threshold=0.9):
        primitives = get_primitives()
        alphas = tf.convert_to_tensor(self.alphas.numpy())
        alphas = tf.nn.sigmoid(alphas).numpy()

        betas = tf.convert_to_tensor(self.betas.numpy())
        betas = tf.nn.softmax(betas, axis=1).numpy()

        normal = []
        for s in range(self.num_stages):
            conns = []
            offset = 0
            for i in range(self.splits):
                a = alphas[s, offset:offset + i + self.splits]
                cs = np.arange(len(a))[a > threshold] + 1
                op = primitives[betas[s * self.splits + i].argmax()]
                conns.append((*cs, op))
                offset += self.splits + i
            normal.append(conns)
        return Genotype(normal=normal)