from pathlib import Path

from omagent_core.engine.worker.base import BaseWorker
from omagent_core.utils.general import read_image
from omagent_core.utils.logger import logging
from omagent_core.utils.registry import registry

CURRENT_PATH = Path(__file__).parents[0]


@registry.register_worker()
class TestInput(BaseWorker):

    def _run(self, *args, **kwargs):
        # Read user input through configured input interface
        
        input = self.input.read_input(workflow_instance_id=self.workflow_instance_id, input_prompt="Path to the folder where the agent will be created.")
        content = input['messages'][-1]['content']
        for content_item in content:
            if content_item['type'] == 'text':
                folder_path = content_item['data']
        print ("folder_path",folder_path)

        input = self.input.read_input(workflow_instance_id=self.workflow_instance_id, input_prompt="Give me an example input for the agent.")
        content = input['messages'][-1]['content']
        for content_item in content:
            if content_item['type'] == 'text':
                example_input = content_item['data']

        print ("example_input",example_input)
        self.stm(self.workflow_instance_id)["folder_path"] = folder_path
        self.stm(self.workflow_instance_id)["example_inputs"] = example_input      
        return {"folder_path": folder_path, "example_inputs": example_input}
