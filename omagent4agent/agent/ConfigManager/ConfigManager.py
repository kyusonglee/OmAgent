from pathlib import Path

from omagent_core.engine.worker.base import BaseWorker
from omagent_core.utils.general import read_image
from omagent_core.utils.logger import logging
from omagent_core.utils.registry import registry
import os
import json
import shutil
CURRENT_PATH = Path(__file__).parents[0]


@registry.register_worker()
class ConfigManager(BaseWorker):
    def _run(self, *args, **kwargs):
        generated_workers = self.stm(self.workflow_instance_id)["generated_workers"]
        folder_path = self.stm(self.workflow_instance_id)["folder_path"]
        name = generated_workers["name"]
        source = os.path.join(CURRENT_PATH, "configs","llms")
        target = os.path.join(folder_path, "configs","llms" )
        shutil.copytree(source, target)

        source = os.path.join(CURRENT_PATH, "configs","tools")
        target = os.path.join(folder_path, "configs","tools" )
        shutil.copytree(source, target) 
    
        for worker in generated_workers["workers"]:
            config_path = os.path.join(folder_path, "configs","workers", worker["worker_name"].lower()+".yaml")
            with open(config_path, 'w') as f:
                f.write("name:"+worker["worker_name"]+"\n")
                if "llm:" in worker["code"] and "image" in worker["code"]:
                    f.write("llm: ${sub|vlm}\n")
                if "llm:" in worker["code"] and not "image" in worker["code"]:
                    f.write("llm: ${sub|llm_text}\n")
                if "tool_manager:" in worker["code"]:
                    f.write("tool_manager: ${sub|all_tools}\n") 
