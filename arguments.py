import argparse

class ArgumentParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='MLIR Model Exporter')
        self.define_options()
        self.args = self.parser.parse_args()

    def define_options(self):
        self.parser.add_argument("-iree", type=str, help="Path to iree-opt")
        self.parser.add_argument("-model", type=str, help="Path to TF model to be exported")
        self.parser.add_argument("-func", type=str, help="Name of function in model to be exported")
        self.parser.add_argument("-input_shape", action='append', help="Shape of input tensor")
        self.parser.add_argument("-input_dtype", type=str, help="Dtype of input tensor")

