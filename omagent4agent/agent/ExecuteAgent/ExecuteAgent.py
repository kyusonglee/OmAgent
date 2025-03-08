
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
    def _run(self, *args, **kwargs):
        try:
            mode = os.getenv("OMAGENT_MODE")            
            os.environ["OMAGENT_MODE"] = "lite"            
            from omagent_core.engine.workflow.conductor_workflow import ConductorWorkflow
            from omagent_core.clients.devices.programmatic import ProgrammaticClient
            logging.init_logger("omagent", "omagent", level="INFO")
            folder_path = self.inputs["folder_path"]
            example_inputs = self.inputs["example_inputs"]
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
            output = client.start_processor_with_input(example_inputs)  
            print (output)            
            os.environ["OMAGENT_MODE"] = mode
            return {"output": output, "error": None, "traceback": None}
        except Exception as e:
            os.environ["OMAGENT_MODE"] = mode
            logging.error(f"Error while executing agent: {e}")
            return {"output": None, "error": str(e), "traceback": traceback.format_exc()}
