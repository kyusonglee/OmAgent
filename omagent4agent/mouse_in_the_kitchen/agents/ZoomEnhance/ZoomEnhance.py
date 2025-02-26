
from omagent_core.engine.worker.base import BaseWorker
from omagent_core.tool_system.manager import ToolManager
from omagent_core.utils.registry import registry
from omagent_core.utils.general import read_image, encode_image
from PIL import Image
import logging

@registry.register_worker()
class ZoomEnhance(BaseWorker):
    tool_manager: ToolManager

    def _run(self, *args, **kwargs):
        """
        Enhance and zoom areas likely containing a mouse.
        """
        # Load the image stored in STM by ImageInputHandler
        image = self.stm(self.workflow_instance_id).get("image")

        # Load detection results from ObjectDetector
        detection_results = self.stm(self.workflow_instance_id).get("mouse_detected_objects", [])

        if image and detection_results:
            for detected_obj in detection_results:
                if detected_obj['label'] == 'mouse':
                    xmin, ymin, xmax, ymax = (detected_obj['xmin'], detected_obj['ymin'], 
                                              detected_obj['xmax'], detected_obj['ymax'])

                    # Crop and enhance the detected mouse area
                    mouse_area = image.crop((xmin, ymin, xmax, ymax))
                    
                    # Enhance the cropped image using SuperResolution
                    upscaled_result = self.tool_manager.execute("SuperResolution", {"image": encode_image(mouse_area)})
                    enhanced_mouse_area = upscaled_result["upscaled_image"]

                    # Update the image with enhanced area
                    image.paste(enhanced_mouse_area, (xmin, ymin))

            # Store the enhanced image back in STM
            self.stm(self.workflow_instance_id)["enhanced_image"] = image

        else:
            # If no detection result or image is missing, log an info
            logging.info("No mouse detected or image missing, skipping enhancement step.")
