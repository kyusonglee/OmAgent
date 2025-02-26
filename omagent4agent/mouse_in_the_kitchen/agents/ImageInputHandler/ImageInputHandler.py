
from omagent_core.engine.worker.base import BaseWorker
from omagent_core.utils.general import read_image
from omagent_core.utils.logger import logging
from omagent_core.utils.registry import registry

@registry.register_worker()
class ImageInputHandler(BaseWorker):
    def _run(self, image_path: str, *args, **kwargs):
        """
        Handles the input image path. Supports image URL and local path.
        
        Args:
            image_path (str): The path or URL to the image.
        
        Returns:
            None
        """
        try:
            # Read the image using the provided image path or URL
            image = read_image(image_path)
            
            # If image is read successfully, store it in short term memory
            if image:
                # Cache the image in STM for later use in the workflow
                self.stm(self.workflow_instance_id)["image"] = image
                logging.info(f"Image successfully loaded and stored: {image_path}")
            else:
                logging.error("Failed to read image.")
        
        except ValueError as e:
            logging.error(f"ImageInputHandler encountered error: {e}")
