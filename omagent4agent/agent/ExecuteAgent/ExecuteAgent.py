
from pathlib import Path

from omagent_core.engine.worker.base import BaseWorker
from omagent_core.utils.general import read_image
from omagent_core.utils.logger import logging
from omagent_core.utils.registry import registry
import glob
import os
import sys
import json
import traceback

CURRENT_PATH = Path(__file__).parents[0]


@registry.register_worker()
class ExecuteAgent(BaseWorker):
    def _run(self, folder_path, example_inputs,*args, **kwargs):
        try:
            mode = os.getenv("OMAGENT_MODE")            
            os.environ["OMAGENT_MODE"] = "lite"            
            from omagent_core.engine.workflow.conductor_workflow import ConductorWorkflow
            from omagent_core.clients.devices.programmatic import ProgrammaticClient
            logging.init_logger("omagent", "omagent", level="INFO")
            
            if type(example_inputs) == str:
                example_inputs = json.loads(example_inputs)

            workflow_path = glob.glob(os.path.join(folder_path, "*_workflow.json"))[0]
            
            target_folder = os.path.abspath(folder_path)
            if target_folder not in sys.path:
                sys.path.insert(0, target_folder)
            print(f"Added {target_folder} to sys.path")
            registry.import_module(os.path.join(target_folder, "agent"))

            with open(workflow_path) as f:
                workflow_json = json.load(f)

            workflow = ConductorWorkflow(name=workflow_json["name"])
            workflow.load(workflow_path)
            client = ProgrammaticClient(
                processor=workflow,
                config_path="/".join(workflow_path.split("/")[:-1])+"/configs",
            )
            self.stm(self.workflow_instance_id)["example_inputs"] = example_inputs

            output = client.start_processor_with_input(example_inputs)  
            os.environ["OMAGENT_MODE"] = mode
            self.stm(self.workflow_instance_id)["error_message"] = None
            self.stm(self.workflow_instance_id)["traceback"] = None
            self.stm(self.workflow_instance_id)["workflow_json"] = workflow_json            
            
            return {"output": output, "error": None, "traceback": None, "has_error": False}
        except Exception as e:
            os.environ["OMAGENT_MODE"] = mode
            logging.error(f"Error while executing agent: {e}")
            self.stm(self.workflow_instance_id)["error_message"] = str(e)
            self.stm(self.workflow_instance_id)["traceback"] = traceback.format_exc()
            self.stm(self.workflow_instance_id)["workflow_json"] = workflow_json

            return {"output": None, "error": str(e), "traceback": traceback.format_exc(), "has_error": True}    
