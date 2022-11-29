from toolz import curry

import tensorflow as tf
from tensorflow.keras.metrics import CategoricalAccuracy, Mean, CategoricalCrossentropy

from hanser.distribute import setup_runtime, distribute_datasets
from hanser.datasets.cifar import make_cifar10_dataset, make_cifar100_dataset
from hanser.transform import random_crop, normalize, to_tensor
from hanser.train.optimizers import SGD
from hanser.models.cifar.ppnas.resnet_avd_fair import ResNet
from hanser.train.cls import SuperLearner
from hanser.train.lr_schedule import CosineLR
from hanser.losses import CrossEntropy

from genotypes import CIFAR10_Geno, CIFAR100_Geno


USE_CIFAR10 = True
if USE_CIFAR10:
    N_C = 10
    make_dataset = make_cifar10_dataset
    Geno = CIFAR10_Geno
else:
    N_C = 100
    make_dataset = make_cifar100_dataset
    Geno = CIFAR100_Geno


@curry
def transform(image, label, training):
    if training:
        image = random_crop(image, (32, 32), (4, 4))
        image = tf.image.random_flip_left_right(image)

    image, label = to_tensor(image, label)
    image = normalize(image, [0.491, 0.482, 0.447], [0.247, 0.243, 0.262])

    label = tf.one_hot(label, N_C)

    return image, label


batch_size = 128
eval_batch_size = 2048

ds_train, ds_test, steps_per_epoch, test_steps = make_dataset(
    batch_size, eval_batch_size, transform)
setup_runtime(fp16=True)
ds_train, ds_test = distribute_datasets(ds_train, ds_test)

model = ResNet(Geno, depth=110, base_width=26, splits=4, num_classes=N_C, stages=[64, 64, 128, 256])
model.build((None, 32, 32, 3))
model.summary()

criterion = CrossEntropy()

base_lr = 0.1
epochs = 200
lr_schedule = CosineLR(base_lr, steps_per_epoch, epochs=epochs, min_lr=0)
optimizer = SGD(lr_schedule, momentum=0.9, weight_decay=5e-4, nesterov=True)
train_metrics = {
    'loss': Mean(),
    'acc': CategoricalAccuracy(),
}
eval_metrics = {
    'loss': CategoricalCrossentropy(from_logits=True),
    'acc': CategoricalAccuracy(),
}

learner = SuperLearner(
    model, criterion, optimizer,
    train_metrics=train_metrics, eval_metrics=eval_metrics,
    work_dir='.')

learner.fit(ds_train, epochs, ds_test, val_freq=1,
            steps_per_epoch=steps_per_epoch, val_steps=test_steps,
            reuse_train_iterator=True)