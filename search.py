import os

from toolz import curry

import tensorflow as tf
from tensorflow.keras.metrics import CategoricalAccuracy, Mean, CategoricalCrossentropy

from hanser.distribute import setup_runtime, distribute_datasets
from hanser.datasets.cifar import make_cifar10_dataset, make_cifar100_dataset
from hanser.transform import random_crop, normalize, to_tensor, cutout
from hanser.transform.autoaugment import autoaugment

from hanser.train.optimizers import SGD, AdamW
from hanser.train.lr_schedule import CosineLR
from hanser.losses import CrossEntropy

from learner import PPNASLearner
from supernet_CIFAR import Network
from callbacks import PrintGenotype, PrintArchParams


USE_CIFAR_10 = True
if USE_CIFAR_10:
    N_C = 10
    make_dataset = make_cifar10_dataset
else:
    N_C = 100
    make_dataset = make_cifar100_dataset


@curry
def transform(image, label, training):
    if training:
        image = random_crop(image, (32, 32), (4, 4))
        image = tf.image.random_flip_left_right(image)
        image = autoaugment(image, augmentation_name='CIFAR10')

    image, label = to_tensor(image, label)

    if training:
        image = cutout(image, 16)

    image = normalize(image, [0.491, 0.482, 0.447], [0.247, 0.243, 0.262])
    label = tf.one_hot(label, N_C)

    return image, label


batch_size = 128
eval_batch_size = 2048

ds_train, ds_test, steps_per_epoch, test_steps = make_dataset(
    batch_size, eval_batch_size, transform)
setup_runtime(fp16=True)
ds_train, ds_test = distribute_datasets(ds_train, ds_test)

model = Network(depth=110, base_width=24, splits=4, num_classes=N_C, stages=(64, 64, 128, 256))
model.build((None, 32, 32, 3))
model.summary()

criterion = CrossEntropy()  # Other loss terms are defined in the Network and added in the learner.

base_lr = 0.1
epochs = 300
lr_schedule = CosineLR(base_lr, steps_per_epoch, epochs=epochs, min_lr=0)
optimizer_w = SGD(lr_schedule, momentum=0.9, weight_decay=5e-4, nesterov=True)
optimizer_a = AdamW(1e-3, beta_1=0.9, beta_2=0.999, weight_decay=0)
train_metrics = {
    'loss': Mean(),
    'acc': CategoricalAccuracy(),
}
eval_metrics = {
    'loss': CategoricalCrossentropy(from_logits=True),
    'acc': CategoricalAccuracy(),
}

learner = PPNASLearner(
    model, criterion, optimizer_a, optimizer_w, grad_clip_norm=5.0, add_arch_loss=True,
    train_metrics=train_metrics, eval_metrics=eval_metrics,
    work_dir='.')

learner.fit(ds_train, epochs, ds_test, val_freq=1,
            steps_per_epoch=steps_per_epoch, val_steps=test_steps,
            reuse_train_iterator=True, callbacks=[PrintGenotype(from_epoch=0), PrintArchParams(from_epoch=0)])