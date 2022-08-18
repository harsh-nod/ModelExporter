import tensorflow as tf
from iree import runtime as ireert
from iree.compiler import tf as tfc
from iree.compiler import compile_str
import os
from arguments import ArgumentParser

def constructModel(input_shapes, input_dtype, model, func):
    class WrappedModel(tf.Module):
        def __init__(self):
            exec(f"from models import {model}")
            self.model = eval(f"{model}()")
            self.func = getattr(self.model, func)
        @tf.signature(input_signature = tf.TensorSpec(shape = input_shape,
                                                      dtype = input_dtype))
        def predict(self, x):
            return self.func(x)
    return WrappedModel

class Exporter:
    def __init__(self):
        self.parser = ArgumentParser()
        self.compile()
    def compile(self):
        module = tfc.compile_module(constructModel(self.input_shape,
                                                   self.input_dtype,
                                                   self.model,
                                                   self.func), \
                                    exported_names = ['predict'], \
                                    import_only = True,
                                    output_mlir_debuginfo = False)
    def translate_to_linalg(self):
        pass

if __name__ == "__main__":
    e = Exporter()
