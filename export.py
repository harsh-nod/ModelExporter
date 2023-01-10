import tensorflow as tf
from iree import runtime as ireert
from iree.compiler import tf as tfc
from iree.compiler import compile_str
import os
from arguments import ArgumentParser

def constructModel(input_shapes, input_dtypes, model):
    input_signature = []
    for shape, dtype in zip(input_shapes, input_dtypes):
        input_signature.append(tf.TensorSpec(shape=shape, dtype=dtype))

    class WrappedModel(tf.Module):
        def __init__(self):
            super(WrappedModel, self).__init__()
            exec(f"from models.{model} import Model")
            self.model = eval(f"Model()")

        @tf.function(input_signature = input_signature)
        def predict(self, *args):
            return self.model.call(*args)
    return WrappedModel

class Exporter:
    def __init__(self):
        self.parser = ArgumentParser()
        input_shapes = self.parser.args.input_shapes.split(',')
        self.input_shapes = [[] for _ in range(len(input_shapes))]
        self.input_dtypes = [None for _ in range(len(input_shapes))]
        for i, shape in enumerate(input_shapes):
            tokens = [x for x in shape.split('x')]
            self.input_shapes[i] = [int(x) for x in tokens[:-1]]
            if tokens[-1] == "f32":
                self.input_dtypes[i] = tf.float32
        self.compile()
        self.translate_to_linalg()
    def compile(self):
        modelType = constructModel(self.input_shapes, self.input_dtypes,
                                   self.parser.args.model)
        self.module = tfc.compile_module(modelType(),
                                    exported_names = ['predict'],
                                    import_only = True,
                                    output_mlir_debuginfo = False,
                                    import_extra_args=['--output-format=mlir-ir'])
        with open('tmp.mlir', 'wt') as output_file:
            output_file.write(self.module.decode("utf-8"))
    def translate_to_linalg(self):
        cmd = [self.parser.args.iree_opt,
               "--iree-mhlo-input-transformation-pipeline", \
               "--iree-mhlo-to-linalg-on-tensors", \
               "--linalg-fuse-elementwise-ops", \
               "--allow-unregistered-dialect", \
               "tmp.mlir", ">", f"{self.parser.args.model}.mlir"]
        print(' '.join(cmd))
        os.system(' '.join(cmd))
        os.remove('tmp.mlir')

if __name__ == "__main__":
    e = Exporter()
