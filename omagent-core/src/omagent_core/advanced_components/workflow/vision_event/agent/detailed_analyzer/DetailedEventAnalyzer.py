from pathlib import Path
from omagent_core.engine.worker.base import BaseWorker
from omagent_core.utils.registry import registry
from omagent_core.models.llms.qwen2_vl import Qwen2_VL
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch
import cv2
import numpy as np
from PIL import Image
import base64
import os

CURRENT_PATH = Path(__file__).parents[0]

@registry.register_worker()
class DetailedEventAnalyzer(BaseWorker):
    """Detailed event analyzer that uses SAM2 and Qwen2-VL for high-resolution analysis."""
    def __init__(self, **data) -> None:
        super().__init__(**data)
        # Initialize SAM2 predictor
        self.predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large", device="cpu")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #device = "cpu"
        #predictor.to(device)
        self.llm = Qwen2_VL()

    def get_regions_of_interest(self, image_path):
        """Use SAM2 to identify potential regions of interest."""
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Set image for SAM2
        with torch.inference_mode(), torch.autocast(self.device.type, dtype=torch.bfloat16):
            self.predictor.set_image(image)
            
            # Generate automatic masks without prompts
            masks, _, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                multimask_output=True
            )
        
        regions = []
        # Process each mask to get bounding boxes
        for mask in masks:
            # Convert mask to numpy if it's a tensor
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
                
            # Get bounding box coordinates
            y_indices, x_indices = np.where(mask)
            if len(y_indices) == 0 or len(x_indices) == 0:
                continue
                
            # Add padding to the bounding box (10% of dimensions)
            x1, x2 = np.min(x_indices), np.max(x_indices)
            y1, y2 = np.min(y_indices), np.max(y_indices)
            
            width = x2 - x1
            height = y2 - y1
            padding_x = int(width * 0.1)
            padding_y = int(height * 0.1)
            
            # Apply padding while keeping within image bounds
            x1 = max(0, x1 - padding_x)
            y1 = max(0, y1 - padding_y)
            x2 = min(image.shape[1], x2 + padding_x)
            y2 = min(image.shape[0], y2 + padding_y)
            
            regions.append((x1, y1, x2, y2))
            
        return regions, image

    def enhance_region(self, image, bbox, scale=2.0):
        """Crop and enhance resolution of a region."""
        x1, y1, x2, y2 = bbox
        crop = image[y1:y2, x1:x2]
        
        # Increase resolution using super-resolution if available
        height, width = crop.shape[:2]
        new_height, new_width = int(height * scale), int(width * scale)
        
        # Use Lanczos4 for better quality upscaling
        enhanced = cv2.resize(crop, (new_width, new_height), 
                            interpolation=cv2.INTER_LANCZOS4)
        
        return enhanced

    def analyze_region_with_qwen(self, image_path: str, event_prompt: str):
        """Analyze an image region using Qwen2-VL."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "data": image_path},
                    {"type": "text", "data": f"In this specific region of the image, is the following event present: {event_prompt}? "
                                           f"Respond with YES or NO and explain briefly."}
                ]
            }
        ]
        
        response = self.llm._call({"messages": messages})
        return response["responses"][0]

    def _run(self, image_path: str, event_prompt: str, initial_result: dict, 
             confidence: float, *args, **kwargs):
        """Perform detailed event analysis if needed.
        
        Args:
            image_path: Path to the input image
            event_prompt: Description of the event to detect
            initial_result: Result from initial detector
            confidence: Confidence score from initial detector
            
        Returns:
            dict containing:
                - final_result: Boolean indicating if event was detected
                - confidence: Final confidence score
                - analysis_details: Details of the analysis process
        """
        
        # If initial detector was confident, return its result
        if initial_result is not None:        
            return {
                'final_result': initial_result,
                'confidence': confidence,
                'analysis_details': 'Used initial detection result'
            }

        # Get regions of interest using SAM2
        regions, original_image = self.get_regions_of_interest(image_path)
        print ("regions:", regions)
        
        
        if not regions:
            return {
                'final_result': False,
                'confidence': 0.7,
                'analysis_details': 'No significant regions detected for analysis'
            }
        
        # Analyze each region
        for i, region in enumerate(regions):
            enhanced_region = self.enhance_region(original_image, region)
            
            # Save enhanced region temporarily
            temp_path = f"temp_region_{i}.jpg"
            cv2.imwrite(temp_path, cv2.cvtColor(enhanced_region, cv2.COLOR_RGB2BGR))
            
            # Analyze with Qwen2-VL
            result = self.analyze_region_with_qwen(temp_path, event_prompt)
            
            # Clean up temporary file
            os.remove(temp_path)
            
            if "YES" in result:
                return {
                    'final_result': True,
                    'confidence': 0.9,
                    'analysis_details': f'Event detected in region {i+1} at coordinates {region}'
                }

        # If no regions contained the event
        return {
            'final_result': False,
            'confidence': 0.8,
            'analysis_details': f'Analyzed {len(regions)} regions, event not detected in any'
        } 