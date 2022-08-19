import tensorflow as tf

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()

    def call(self, x, y):
        return tf.transpose(x) + y
