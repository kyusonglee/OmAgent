from pathlib import Path

from omagent_core.engine.worker.base import BaseWorker
from omagent_core.utils.general import read_image
from omagent_core.utils.logger import logging
from omagent_core.utils.registry import registry
import glob
import os
import sys
import json
from omagent_core.engine.workflow.conductor_workflow import ConductorWorkflow
from omagent_core.clients.devices.programmatic.client import ProgrammaticClient

CURRENT_PATH = Path(__file__).parents[0]


@registry.register_worker()
class ExecuteAgent(BaseWorker):
    def _run(self, *args, **kwargs):
        logging.init_logger("omagent", "omagent", level="INFO")
        folder = self.stm(self.workflow_instance_id)["folder_path"]
        inputs = self.stm(self.workflow_instance_id)["example_inputs"]
        workflow_path = glob.glob(os.path.join(folder, "*_workflow.json"))[0]
        
        target_folder = os.path.abspath(folder)
        if target_folder not in sys.path:
            sys.path.insert(0, target_folder)
        print(f"Added {target_folder} to sys.path")
        os.environ["OMAGENT_MODE"] = "lite"
        registry.import_module(os.path.join(target_folder, "agents"))

        with open(workflow_path) as f:
            workflow_json = json.load(f)

        workflow = ConductorWorkflow(name=workflow_json["name"])
        workflow.load(workflow_path)
        client = ProgrammaticClient(
            processor=workflow,
            config_path="/".join(workflow_path.split("/")[:-1])+"/configs",
        )
        output = client.start_processor_with_input(inputs)  
        print (output)