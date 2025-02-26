
from omagent_core.engine.worker.base import BaseWorker
from omagent_core.utils.registry import registry

@registry.register_worker()
class DetectionConfirmation(BaseWorker):
    def _run(self, switchCaseValue: bool, *args, **kwargs) -> dict:
        """
        Confirms detection presence and determines further action based on secondary detection output.

        Args:
            switchCaseValue (bool): Indicates if mouse detection was confirmed.

        Returns:
            dict: Decision case value for SWITCH task.
        """
        # Decision value for SWITCH task based on secondary detection confirmation
        if switchCaseValue:
            decision = "true"  # Proceed to action recognition and interpretation
        else:
            decision = "false"  # Enhance image and perform secondary detection again

        # Return the decision for the SWITCH task
        return {"switch_case_value": decision}
