import tensorflow as tf
from hanser.train.callbacks import Callback


class PrintGenotype(Callback):

    def __init__(self, from_epoch):
        super().__init__()
        self.from_epoch = from_epoch

    def after_epoch(self, state):
        if state['epoch'] + 1 < self.from_epoch:
            return
        print(self.learner.model.genotype())


class PrintArchParams(Callback):

    def __init__(self, from_epoch):
        super().__init__()
        self.from_epoch = from_epoch

    def after_epoch(self, state):
        if state['epoch'] + 1 < self.from_epoch:
            return
        print('Alphas(Sigmoid):')
        print(tf.nn.sigmoid(tf.convert_to_tensor(self.learner.model.alphas.numpy())))
        print('Betas:')
        print(self.learner.model.betas.numpy())