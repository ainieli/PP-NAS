import tensorflow as tf
from hanser.train.learner import Learner, cast


class PPNASLearner(Learner):

    def __init__(self, model: Network, criterion, optimizer_arch, optimizer_model,
                 grad_clip_norm=0.0, add_arch_loss=False, **kwargs):
        self.grad_clip_norm = grad_clip_norm
        self.add_arch_loss = add_arch_loss
        super().__init__(model, criterion, (optimizer_arch, optimizer_model), **kwargs)

    def train_batch(self, batch):
        model = self.model
        optimizer_arch, optimizer_model = self.optimizers

        inputs, target = batch
        with tf.GradientTape() as tape:
            inputs = cast(inputs, self.dtype)
            logits = model(inputs, training=True)
            logits = cast(logits, tf.float32)
            per_example_loss = self.criterion(target, logits)
            loss = self.reduce_loss(per_example_loss)
            if self.add_arch_loss:
                arch_loss = model.arch_loss()
                arch_loss = tf.reduce_mean(arch_loss)
                loss = loss + arch_loss

        variables = model.trainable_variables
        grads = tape.gradient(loss, variables)
        model_slice, arch_slice = model.param_splits()
        self.apply_gradients(optimizer_model, grads[model_slice], variables[model_slice], self.grad_clip_norm)
        self.apply_gradients(optimizer_arch, grads[arch_slice], variables[arch_slice])
        self.update_metrics(self.train_metrics, target, logits, per_example_loss)

    def train_batches(self, *batches):
        batch = tuple(tf.concat(xs, axis=0) for xs in zip(*batches))
        return self.train_batch(batch)

    def eval_batch(self, batch):
        inputs, target = batch
        inputs = cast(inputs, self.dtype)
        preds = self.model(inputs, training=False)
        preds = cast(preds, tf.float32)
        self.update_metrics(self.eval_metrics, target, preds)