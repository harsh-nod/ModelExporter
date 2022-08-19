import tensorflow as tf

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()

    def call(self, x, y, logits):
        a = tf.matmul(x, y)
        b = tf.reshape(a, [x.shape[0], -1])
        c = tf.nn.softmax_cross_entropy_with_logits(b, logits)
        return c
