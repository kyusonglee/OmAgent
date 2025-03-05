from pathlib import Path
from typing import List

from omagent_core.advanced_components.workflow.dnc.schemas.dnc_structure import \
    TaskTree
from omagent_core.engine.worker.base import BaseWorker
from omagent_core.memories.ltms.ltm import LTM
from omagent_core.models.llms.base import BaseLLMBackend, BaseLLM
from omagent_core.models.llms.prompt import PromptTemplate
from omagent_core.utils.logger import logging
from omagent_core.utils.registry import registry
from openai import Stream
from pydantic import Field
from collections.abc import Iterator
from omagent_core.models.llms.prompt.parser import *    
import os

CURRENT_PATH = root_path = Path(__file__).parents[0]

ROOT_PATH = Path(__file__).parents[1]

@registry.register_worker()
class WorkerManager(BaseLLMBackend, BaseWorker):
    prompts: List[PromptTemplate] = Field(
        default=[
            PromptTemplate.from_file(CURRENT_PATH.joinpath("worker_system.prompt"), role="system"),
            PromptTemplate.from_file(CURRENT_PATH.joinpath("worker_user.prompt"), role="user"),   
        ]
    )
    llm: BaseLLM
    def _run(self, *args, **kwargs):    
        print ("CURRENT_PATH",CURRENT_PATH)
        print ("ROOT_PATH",ROOT_PATH)  
        workflow_json = json.loads(self.stm(self.workflow_instance_id)["workflow_json"])
        folder_path = self.stm(self.workflow_instance_id)["folder_path"]
        workflow_file_name = f"{workflow_json['name']}_workflow.json"
        workflow_path = os.path.join(folder_path, workflow_file_name)
        os.makedirs(folder_path, exist_ok=True)
        with open(workflow_path, 'w') as f:
            json.dump(workflow_json, f, indent=4)

        workflow_path = os.path.join(folder_path, workflow_file_name)
        print ("workflow_path",workflow_path)
        workflow_name = workflow_json["name"]
        workers_section = workflow_json["description"]
        generated_workers = {"workers": [], "name":workflow_name, "workflow_path": workflow_path}
        codes = []
        for worker in workers_section:   
            code = self.simple_infer(workflow_name=worker["Worker_Name"], worker_description=worker["Role"], workflow=workflow_json, previous_codes="\n".join(codes))["choices"][0]["message"].get("content")
            code = self.parse_code(code)
            codes.append("# worker name: "+worker["Worker_Name"]+"\n"+code)
            worker_dir = os.path.join(folder_path, 'agent', worker["Worker_Name"])
            os.makedirs(worker_dir, exist_ok=True)
            worker_file_path = os.path.join(worker_dir, f"{worker['Worker_Name']}.py")
            with open(worker_file_path, 'w') as f:
                f.write(code)
            print ("worker_file_path",worker_file_path)
            print ("code",code)
            generated_workers["workers"].append({"worker_name": worker['Worker_Name'], "worker_file_path": worker_file_path, "code": code })
        
        agents_dir = os.path.join(folder_path, "agent")
        for root, dirs, files in os.walk(agents_dir):
            for d in dirs:
                sub_folder_path = os.path.join(root, d)
                init_file = os.path.join(sub_folder_path, "__init__.py")
                if not os.path.isfile(init_file):
                    with open(init_file, "w") as f:
                        f.write("# Auto-generated __init__.py for agents sub-package\n")
                    print(f"Ensured sub-package: {init_file}")

        self.stm(self.workflow_instance_id)["generated_workers"] = generated_workers
        
    def parse_code(self, code: str):
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        return code