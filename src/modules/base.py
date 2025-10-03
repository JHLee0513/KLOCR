
class Module:
    """Base class for all the modules in the KLOSer pipeline.
    All modules should inherit from this class for consistent flow."""
    def __init__(self, module_args, input_key=None, output_key=None):
        self.module_args = module_args
        self.input_key=input_key
        self.output_key=output_key
    
    def process(self, input_dict):
        raise NotImplementedError()

    