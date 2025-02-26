
from omagent_core.engine.worker.base import BaseWorker
from omagent_core.tool_system.manager import ToolManager
from omagent_core.utils.registry import registry
from omagent_core.utils.general import encode_image, read_image
from PIL import Image

@registry.register_worker()
class SecondaryDetector(BaseWorker):
    tool_manager: ToolManager

    def _run(self, *args, **kwargs):
        """
        Confirm detection using another method.
        """
        # Load the enhanced image stored in STM by ZoomEnhance
        enhanced_image = self.stm(self.workflow_instance_id).get("enhanced_image")

        if enhanced_image:
            # Use a secondary detection method to confirm mouse detection
            detection_results = self.tool_manager.execute(
                "DetectAll",
                {"image": encode_image(enhanced_image)}
            )

            # Confirm detection of mouse object
            mouse_confirmed = any(obj['label'] == 'mouse' for obj in detection_results["objects"])

            # Store the result of the secondary detection in STM for the SWITCH task
            self.stm(self.workflow_instance_id)["mouse_confirmed"] = mouse_confirmed

            # Return result for SWITCH task
            return {"mouse_detected": mouse_confirmed}
        else:
            # If no enhanced image available, assume detection failed
            return {"mouse_detected": False}
