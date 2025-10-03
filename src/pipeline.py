import yaml
import time
from . import modules
import copy

def build_pipeline(config):
    pipeline = {}
    for name, spec in config.items():
        class_name = spec['class']
        mod_cls = None
        for module in [modules]:
            if hasattr(module, class_name):
                mod_cls = getattr(module, class_name)
        assert(mod_cls is not None), f'Could not find {class_name}.'

        generator_args = copy.deepcopy(spec.get('module_args', {}))
        try:
            mod = mod_cls(generator_args, input_key=spec.get('input_key'), output_key=spec.get('output_key'))
        except:
            print(f'Error creating {mod_cls}')
            raise
        pipeline[name] = mod
    return pipeline


def run_pipeline(pipeline, input_dict, verbose=False):
    p_start = time.time()
    for k,v in pipeline.items():
        start=time.time()
        k_out = v.process(input_dict)
        if verbose:
            print(f"Elapsed time to process [{type(v).__name__}] is {time.time()-start:.2f}")
        input_dict.update(k_out)
    if verbose:
        print(f"Total elapsed time to process pipeline is {time.time()-p_start:.2f}")
    return input_dict

class Pipeline:
    """OCR Pipeline
    """
    
    def __init__(
        self,
        cfg,
        verbose=False
    ):
        self.verbose = verbose
        with open(cfg, 'r') as f:
            cfg = yaml.safe_load(f)
            self.pipeline = build_pipeline(cfg)

    def run(
        self,
        input_dict
    ):
        output = run_pipeline(self.pipeline, input_dict, verbose=self.verbose)
        return output

