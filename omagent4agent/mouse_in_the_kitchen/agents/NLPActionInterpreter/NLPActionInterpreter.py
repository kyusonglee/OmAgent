
from omagent_core.engine.worker.base import BaseWorker
from omagent_core.utils.registry import registry

@registry.register_worker()
class NLPActionInterpreter(BaseWorker):
    def _run(self, *args, **kwargs):
        """
        Interprets the recognized actions of the detected mouse into a human-readable format.
        """
        # Load the recognized action from STM set by ActionRecognizer
        mouse_action = self.stm(self.workflow_instance_id).get("mouse_action", "no action detected")

        # Interpret the recognized action into a human-readable format
        if mouse_action == "mouse moving":
            human_readable_action = "The mouse is on the move, possibly exploring its environment."
        elif mouse_action == "no action detected":
            human_readable_action = "No distinct movement or actions detected from the mouse."
        else:
            human_readable_action = "The mouse is performing an unspecified action."

        # Store the interpreted action in STM for subsequent tasks or outputs
        self.stm(self.workflow_instance_id)["human_readable_action"] = human_readable_action

        # As this is the last worker, return the result for final output
        return {"interpreted_action": human_readable_action}
