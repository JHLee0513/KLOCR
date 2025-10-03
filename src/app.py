import io
import numpy as np
from typing import Union, List, Optional
from fastapi import FastAPI, status
from pydantic import BaseModel, ConfigDict, Field
from PIL import Image
from . import pipeline
import base64

class Item(BaseModel):
    image: Optional[List] = None
    image_base64: Optional[str] = None
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )
    
    def model_post_init(self, __context):
        if self.image is None and self.image_base64 is None:
            raise ValueError("Either 'image' or 'image_base64' must be provided")


app = FastAPI()
pipe = pipeline.Pipeline("configs/default.yaml", verbose=True)


class HealthCheck(BaseModel):
    """Response model to validate and return when performing a health check."""

    status: str = "OK"

def decode_image_from_base64(base64_string):
    img_bytes = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_bytes))
    img_array = np.array(img)
    return img_array


def _pipeline_req(input_dict):
    pipeline_out = pipe.run(input_dict)
    if "image" in pipeline_out:
        pipeline_out.pop("image")
    if "image_base64" in pipeline_out:
        pipeline_out.pop("image_base64")
    updated_out = {}
    for k, v in pipeline_out.items():
        if k not in ['roi', 'text', 'checkbox_detections']:
            continue
        if k in ['roi']:
            if not isinstance(v, list):
                v = v.tolist()
            updated_out[k] = v 
        else:
            updated_out[k] = v
    return updated_out


@app.get(
    "/health",
    tags=["healthcheck"],
    summary="Perform a Health Check",
    response_description="Return HTTP Status Code 200 (OK)",
    status_code=status.HTTP_200_OK,
    response_model=HealthCheck,
)
def get_health() -> HealthCheck:
    """
    ## Perform a Health Check
    Endpoint to perform a healthcheck on. This endpoint can primarily be used Docker
    to ensure a robust container orchestration and management is in place. Other
    services which rely on proper functioning of the API service will not deploy if this
    endpoint returns any other HTTP status code except 200 (OK).
    Returns:
        HealthCheck: Returns a JSON response with the health status
    """
    return HealthCheck(status="OK")

@app.post("/ocr/roi")
def roi(item: Item):
    input_dict = item.dict()
    if input_dict.get('image') is not None:
        input_dict["image"] = np.asarray(input_dict.get('image'), dtype=np.uint8)
    elif input_dict.get('image_base64') is not None:
        input_dict["image"] = decode_image_from_base64(input_dict.get('image_base64'))
    return _pipeline_req(input_dict)


if __name__ == "__main__":

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)