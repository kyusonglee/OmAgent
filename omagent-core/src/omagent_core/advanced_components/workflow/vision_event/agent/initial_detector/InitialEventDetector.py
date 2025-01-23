from pathlib import Path
from omagent_core.engine.worker.base import BaseWorker
from omagent_core.models.llms.base import BaseLLMBackend
from omagent_core.utils.registry import registry
from omagent_core.models.llms.qwen2_vl import Qwen2_VL
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
    llm: Qwen2_VL
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
                    {"type": "text", "data": f"""Analyze this image and determine if the following event is present: {event_prompt}
                    If you're very confident (>90%) that the event is NOT present, respond with 'CLEARLY_FALSE'.
                    If you're very confident (>90%) that the event IS present, respond with 'CLEARLY_TRUE'.
                    If you're unsure or need closer inspection, respond with 'NEEDS_ANALYSIS'.
                    Explain your reasoning briefly."""}
                ]
            }
        ]

        # Get model response
        print ("messages:", messages)
        response = self.llm._call({"messages": messages})
        result = response["responses"][0]
        print (result)
        # Process response
        if "CLEARLY_FALSE" in result:
            return {'result': False, 'confidence': 0.95, 'requires_detailed': False}
        elif "CLEARLY_TRUE" in result:
            return {'result': True, 'confidence': 0.95, 'requires_detailed': False}
        else:
            return {'result': None, 'confidence': 0.5, 'requires_detailed': True} 