from omagent_core.engine.workflow.conductor_workflow import ConductorWorkflow
from omagent_core.engine.workflow.task.simple_task import simple_task
from typing import List

class VisionEventWorkflow(ConductorWorkflow):
    """Vision-based event detection workflow that analyzes images for specific events.
    
    This workflow uses a two-stage approach:
    1. Initial detection using vision LLM
    2. Detailed analysis with SAM2 segmentation if needed
    """
    def __init__(self):
        super().__init__(name='vision_event_workflow')
        
    def set_input(self, image_path: str, event_prompt: str):
        """Set input parameters and configure workflow.
        
        Args:
            image_path: Path to the input image
            event_prompt: Description of the event to detect (e.g., "detect if a worker doesn't wear helmet")
        """
        self.image_path = image_path
        self.event_prompt = event_prompt
        self._configure_tasks()
        self._configure_workflow()

    def _configure_tasks(self):
        """Configure the detection tasks with input parameters."""
        self.initial_detector_task = simple_task(
            task_def_name="InitialEventDetector",  
            task_reference_name='initial_detector',
            inputs={
                'image_path': self.image_path, 
                'event_prompt': self.event_prompt
            }
        )
        
        self.detailed_analyzer_task = simple_task(
            task_def_name="DetailedEventAnalyzer",  
            task_reference_name='detailed_analyzer',
            inputs={
                'image_path': self.image_path,
                'event_prompt': self.event_prompt,
                'initial_result': self.initial_detector_task.output('result'),
                'confidence': self.initial_detector_task.output('confidence')
            }
        )

    def _configure_workflow(self):
        """Configure workflow execution flow and output."""
        self >> self.initial_detector_task >> self.detailed_analyzer_task
        self.detection_result = self.detailed_analyzer_task.output('final_result')
        self.confidence_score = self.detailed_analyzer_task.output('confidence')
        self.analysis_details = self.detailed_analyzer_task.output('analysis_details') 