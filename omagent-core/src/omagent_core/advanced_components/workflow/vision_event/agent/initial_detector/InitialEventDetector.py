from pathlib import Path
from omagent_core.engine.worker.base import BaseWorker
from omagent_core.models.llms.base import BaseLLMBackend
from omagent_core.utils.registry import registry
from omagent_core.models.llms.base import BaseLLM
from omagent_core.models.llms.prompt.prompt import PromptTemplate
from pydantic import Field
from typing import List
from omagent_core.utils.logger import logging
import base64

CURRENT_PATH = Path(__file__).parents[0]

@registry.register_worker()
class InitialEventDetector(BaseWorker, BaseLLMBackend):
    """Initial event detector that uses Qwen2-VL to analyze images.
    
    This detector performs the first-pass analysis of the image using Qwen2-VL
    to determine if the specified event is present.
    """
    llm: BaseLLM
    prompts: List[PromptTemplate] = Field(
        default=[
            PromptTemplate.from_file(
                CURRENT_PATH.joinpath("vision_prompt.prompt"), role="user"
            ),
        ]
    )

    def encode_image(self, image_path):
        """Convert image to base64 encoding."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _run(self, image_path: str, event_prompt: str, *args, **kwargs):
        """Perform initial event detection on the image.
        
        Args:
            image_path: Path to the input image
            event_prompt: Description of the event to detect
            
        Returns:
            dict containing:
                - result: Boolean indicating if event was detected
                - confidence: Confidence score of the detection
                - requires_detailed: Whether detailed analysis is needed
        """
        if not image_path or not event_prompt:
            return {'result': False, 'confidence': 0.0, 'requires_detailed': False}

        # Prepare messages for Qwen2-VL
        messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "data": image_path},
            {
                "type": "text",
                "data": (
                    "Explain your reasoning briefly.\n"
                    "You are a professional image detail analysis expert. I will provide you with an image, and based on the input 'event to identify,' you will analyze the image and determine the 'location' where the event occurs.\n\n"
                    "Event to Identify:\n"
                    f"{event_prompt}\n\n"
                    "Identification Requirements:\n"
                    "For the 'event to identify,' analyze the image based on the following location definitions:\n\n"
                    "Top-left, bottom-left, top-right, bottom-right: Divide the image into four equal sections using a vertical and horizontal cut at the center point of the image. "
                    "The resulting sections are labeled top-left, bottom-left, top-right, and bottom-right.\n"
                    "If the event is detected in the image, output the location(s) in the original image (e.g., top-left, bottom-right). Provide an analysis of where the event occurs in the image. "
                    "Format your response as a dictionary:\n\n"
                    '{"Event Location Analysis":"xxx","Locations":[xxx,...,xxx],"Event Detected":"Yes"}\n'
                    "If there is any anomaly, output this format.\n\n"
                    "If the event is not detected in the image, output:\n"
                    '{"Event Location Analysis":"xxx","Event Detected":"No"}\n\n'
                    "Examples:\n"
                    '{"Event Location Analysis":"xxx","Locations":["Top-left","Bottom-right"],"Event Detected":"Yes"}\n'
                    '{"Event Location Analysis":"xxx","Event Detected":"No"}\n'
                    '{"Event Location Analysis":"xxx","Locations":["Bottom-left"],"Event Detected":"Yes"}\n\n'
                    "Important Notes:\n"
                    "1. Follow the output examples strictly and only output a single dictionary, without any additional explanation.\n"
                    "2. Do not output phrases like 'I can’t assist' or 'I’m unable to recognize the image.' Instead, provide suggestions or outputs based on the event.\n"
                    "3. For determining the 'event,' do not consider the context of the scene—if the event occurs anywhere in the image, it is considered 'detected.'\n"
                    "4. The 'Event Location Analysis' should specify the location of the event in the image. For example, in the case of 'hard hat identification,' locate where the head is in the image. "
                    "If the head is in the 'bottom-left' region, return 'bottom-left.'\n"
                    "5. If the event occurs in multiple locations in the image, return a list of positions in the 'Locations' key, such as ['Top-left','Bottom-right']."
                )
            }
        ]
    }
]

        # Get model response
        response = self.llm._call({"messages": messages})
        result = response["responses"][0]
        return result
