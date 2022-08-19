import argparse

class ArgumentParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='MLIR Model Exporter')
        self.define_options()
        self.args = self.parser.parse_args()

    def define_options(self):
        self.parser.add_argument("-iree_opt", type=str, help="Path to iree-opt")
        self.parser.add_argument("-model", type=str, help="Path to TF model to be exported")
        self.parser.add_argument("-input_shapes", type=str, help="Shape of input tensor")

