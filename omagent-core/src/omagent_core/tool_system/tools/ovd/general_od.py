from ....utils.registry import registry
from ...base import BaseModelTool, ArgSchema
import requests
import base64
from ....models.od.schemas import Target
from PIL import Image
import io

ARGSCHEMA = {
    "image": {
        "type": "dict",
        "description": "Image to be processed, it can be a url, local path, or base64 encoded image.",
        "required": True,
    },
    "labels": {
        "type": "string",
        "description": "Labels the object detection tool will use to detect objects in the image, split by comma.",
        "required": True,
    },
    "threshold": {
        "type": "float",
        "description": "Threshold for the object detection tool.",
        "required": False,
        "default": 0.3,
    },
    "nms_threshold": {
        "type": "float",
        "description": "NMS threshold for the object detection tool.",
        "required": False,
        "default": 0.5,
    },  
}

@registry.register_tool()
class GeneralOD(BaseModelTool):
    args_schema: ArgSchema = ArgSchema(**ARGSCHEMA)

    url: str = "http://10.0.0.132:8005/inf_predict"
    description: str = (
        "General object detection tool, which can detect any objects and add visual prompting(bounding box and label) to the image."
        "Tasks like object counting, specific object detection, etc. must use this tool."
    )
    def _run(self, image, labels: str, model_id: str = "OmDetV2T_base_CXT_B_n104.pth", threshold: float = 0.3, nms_threshold: float = 0.5, *args, **kwargs) -> dict:        
        """ example input:
        image: "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg" or base64 string or PIL image
        labels: "person"
        """    
        labels = labels.split(",")
        # Prepare payload for the API
        payload = {
            "model_id": model_id,
            "task": "detect: " + "; ".join(labels),
            "labels": labels,
            "include_classes": labels,
            "threshold": threshold,
            "nms_threshold": nms_threshold
        }
        
        # Include image data in the payload
        if isinstance(image, str):
            # If image is a URL
            if image.startswith('http://') or image.startswith('https://'):
                payload['data'] = [image]
                payload['src_type'] = 'url'
                pil_image = Image.open(requests.get(image, stream=True).raw)
            else:
                # Assume it's a local file path
                with open(image, 'rb') as image_file:
                    base64_encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                payload['data'] = [base64_encoded_image]
                payload['src_type'] = 'base64'
                pil_image = Image.open(image)
        elif isinstance(image, bytes):
            # If image is base64 encoded
            payload['data'] = [image.decode('utf-8')]
            payload['src_type'] = 'base64'
            pil_image = Image.open(base64.b64decode(image))
        elif isinstance(image, Image.Image):
            # If image is a PIL image
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            base64_encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
            payload['data'] = [base64_encoded_image]
            payload['src_type'] = 'base64'
            pil_image = image
        else:
            raise ValueError("Unsupported image type")

        # Send request to the object detection API
        print(payload)
        res = requests.post(self.url, json=payload)
        print(res.json())
        # Handle API response
        if res.status_code != 200:
            raise ValueError(
                f"OVD tool failed to detect objects in the images. {res.text}"
            )
        res = res.json()
        targets = []
        for img in res["objects"]:
            current_img_targets = []
            for bbox in img:
                x0 = bbox["xmin"]
                y0 = bbox["ymin"]
                x1 = bbox["xmax"]
                y1 = bbox["ymax"]
                conf = bbox["conf"]
                label = bbox["label"]
                current_img_targets.append(
                    Target(bbox=[x0, y0, x1, y1], conf=conf, label=label)
                )
            targets.append(current_img_targets)
        annotated_pil_image = self.visual_prompting(pil_image, targets)
        return {"annotated_pil_image": annotated_pil_image, "targets": targets}

