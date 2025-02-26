
from omagent_core.engine.worker.base import BaseWorker
from omagent_core.tool_system.manager import ToolManager
from omagent_core.utils.registry import registry

@registry.register_worker()
class ObjectDetector(BaseWorker):
    tool_manager: ToolManager

    def _run(self, *args, **kwargs):
        """
        Primary detection for objects, especially mouse.
        """
        # Load the image stored in STM by ImageInputHandler
        image = self.stm(self.workflow_instance_id).get("image")

        if image:
            # Perform primary object detection specifically for mouse
            detection_results = self.tool_manager.execute(
                "GeneralOD",
                {"image": image, "labels": "mouse", "threshold": 0.5, "nms_threshold": 0.5}
            )

            # Check if a mouse was detected
            mouse_detected = any(obj['label'] == 'mouse' for obj in detection_results["objects"])

            # Store detection result in short-term memory for use in SWITCH task
            self.stm(self.workflow_instance_id)["mouse_detected"] = mouse_detected
        else:
            mouse_detected = False

        # Return result for SWITCH task
        return {"mouse_detected": mouse_detected}
