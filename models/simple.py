import tensorflow as tf

class Model(tf.keras.Model):
    def __init__(self):
        pass

    def predict(self, x):
        self.y = tf.Variable(tf.ones(x.shape), trainable=False)
        return tf.transpose(x) + self.y
