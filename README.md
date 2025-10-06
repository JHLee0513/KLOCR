<div align="center">
</div>

# KLOCR: Korean Language OCR

Paper: [Exploring OCR-augmented Generation for Bilingual VQA](https://arxiv.org/abs/2510.02543)

## Overview

KLOCR is an open-source Korean & English bilingual OCR model trained on data from publicly available sources. This repository provides a package to run the model as an API service.

## Model weights

Model weights for KLOCR and checkbox detection model can be downloaded here: https://drive.google.com/drive/folders/1pah84yNveLA9SJGGw-CFJpxQ1Ot5ARYx?usp=sharing

Place the downloaded files in the `weights/` directory.

## Get Started

The fastest way to try KLOCR is to run the API with docker (both [docker](https://docs.docker.com/engine/) and [nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) should be installed):
```bash
# within directory
docker compose up --build -d 
```

Once the image is built and running, we can use the api:
```python
import json
import requests
import numpy as np

LOCAL_IP = "xxx"
url = f"{LOCAL_IP}:80/ocr/roi"
payload = {
    "image": np.clip(np.random.normal(255,100, (1000,1000,3)), 0, 255).astype(np.uint8).tolist()
}
headers = {'content-type': 'application/json'}

r = requests.post(url, data=json.dumps(payload), headers=headers)
print(r.json())
```

If you only need to run the KLOCR model itself, the model weights and the [huggingface transformers](https://huggingface.co/docs/transformers/en/index) library are all you need:
```python
# https://huggingface.co/docs/transformers/en/model_doc/trocr
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests

# the KLOCR model is fintuned from "team-lucid/trocr-small-korean"
processor = TrOCRProcessor.from_pretrained("team-lucid/trocr-small-korean")
model = VisionEncoderDecoderModel.from_pretrained("weights/trocr/exp9/066")

image = Image.open("example/0.png").convert("RGB")

pixel_values = processor(image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)

generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)
```

To run KLOCR package and use the pipeline function:
```python
import numpy as np
import kloser
# this may involve downloading model weights for PaddlePaddle
pipeline = kloser.Pipeline("configs/default.yaml")
inp = {
  "image": np.clip(np.random.normal(255,100, (2000,2000,3)), 0, 255).astype(np.uint8)
}
out_dict = pipeline.run(inp)
```

The default pipeline runs PaddleDet + KLOCR and outputs with keys `['image', 'roi', 'text']`.


## Pipeline design
Each processing block follows Module base class for consistency.

```Python
class Module:
    """Base class for all the modules in the KLOCR pipeline.
    All modules should inherit from this class for consistent flow."""
    def __init__(self, module_args, input_key=None, output_key=None):
        self.module_args = module_args
        self.input_key=input_key
        self.output_key=output_key
    
    def process(self, input_dict):
        raise NotImplementedError()
```

## Installation

This package does assume you have a GPU. We tested KLOCR on a RTX 6000 (Turing) and RTX A6000 (Ampere) GPU.

**We suggest running KLOCR pipeline/package with docker since CUDNN installation for PaddlePaddle can become tricky.**

There are two ways to run the KLOCR package:
1. Docker
2. Installation as a python package (in a conda environment)

### Conda/Virtual Env
```bash
conda create -n ocr python==3.10
conda activate ocr
git clone https://github.com/JHLee0513/KLOCR.git
cd KLOCR
pip install .
```

### Docker
Both [docker](https://docs.docker.com/get-docker/) and the [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) should be installed.

We support [Docker-compose](https://docs.docker.com/compose/):
```bash
git clone https://github.com/JHLee0513/KLOCR.git
cd KLOCR
docker compose up --build -d
```

Above command runs KLOCR with default configuration (PaddleDet Text detection + KLOCR text recognition). To separately build and run the docker image, refer to below:

```bash
git clone https://github.com/JHLee0513/KLOCR.git
cd KLOCR
docker build -t kloser .
# Run the docker
docker run --runtime=nvidia --gpus all -it --entrypoint bash -v $(pwd):/workspace/kloser -p 80:80 kloser
# inside docker you can run it via FastAPI with below command
uvicorn src.app:app --host 0.0.0.0 --port 80 --workers 1
```

## Citing KLOCR

If you use this package or the KLOCR model, found it useful, or need to reference it for your project/paper, please consider citation as below:
```
@misc{lee2025exploringocr,
      title={Exploring OCR-augmented Generation for Bilingual VQA}, 
      author={JoonHo Lee and Sunho Park},
      year={2025},
      eprint={2510.02543},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.02543}, 
}

@software{kloser2025github,
  author = {JoonHo Lee, Sunho Park},
  title = {{KLOCR}:Korean Language OCR},
  url = {http://github.com/JHLee0513/KLOCR},
  version = {0.0.1},
  year = {2025},
}
```

## FAQ

### PaddleOCR CuDNN Error

CuDNN is not included in the driver or cuda toolkit installation. Consider downloading [cudnn](https://developer.nvidia.com/cudnn-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local) separately to your environment as follows. Note that the correct one for your setup should be installed. Because of this, we suggest running on docker as this minimizes setup-related issues.
