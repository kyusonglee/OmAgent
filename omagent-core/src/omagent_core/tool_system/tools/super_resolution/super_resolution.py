from ...base import BaseModelTool, ArgSchema
from ....utils.registry import registry
import requests
from PIL import Image
from io import BytesIO
import base64
import torch
from diffusers import LDMSuperResolutionPipeline

ARGSCHEMA = {
    "image": {
        "type": "dict",
        "description": "Image to be processed, can be a URL, local path, or base64 encoded image.",
        "required": True,
    }
}

@registry.register_tool()
class SuperResolution(BaseModelTool):
    args_schema: ArgSchema = ArgSchema(**ARGSCHEMA)
    description: str = "Super resolution tool using LDM Super Resolution pipeline from Diffusers."
    model_id: str = "CompVis/ldm-super-resolution-4x-openimages"      

    def _run(
        self,
        image,
        *args,
        **kwargs,
    ) -> dict:
        """
        Run super resolution on the provided image.

        Args:
            image: Image input which can be a URL string, base64 string, or PIL image.
        Returns:
            dict: A dictionary containing the upscaled PIL image.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = LDMSuperResolutionPipeline.from_pretrained(self.model_id)
        self.pipeline = self.pipeline.to(device)

        # Load image based on its type
        if isinstance(image, str):
            # If image is a URL
            if image.startswith('http://') or image.startswith('https://'):
                response = requests.get(image)
                pil_image = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                # Assume it's a base64 string
                image_bytes = base64.b64decode(image)
                pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        elif isinstance(image, Image.Image):
            # If image is a PIL image
            pil_image = image
        else:
            raise ValueError("Unsupported image type. Please use a URL string, base64 string, or PIL image.")

        # Run the super resolution pipeline
        upscaled_image = self.pipeline(pil_image, num_inference_steps=100, eta=1.0).images[0]

        return {"upscaled_image": upscaled_image}
