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
from pprint import pprint
CURRENT_PATH = root_path = Path(__file__).parents[0]


@registry.register_worker()
class WorkerVerifier(BaseLLMBackend, BaseWorker):  
    prompts: List[PromptTemplate] = Field(
        default=[
            PromptTemplate.from_file(CURRENT_PATH.joinpath("worker_verifier_system.prompt"), role="system"),
            PromptTemplate.from_file(CURRENT_PATH.joinpath("worker_verifier_user.prompt"), role="user"),   
        ]
    )
    llm: BaseLLM
    output_parser: DictParser = DictParser()
    def _run(self, *args, **kwargs):        
        
        generated_workers = self.stm(self.workflow_instance_id)["generated_workers"]       
        folder_path = self.stm(self.workflow_instance_id)["folder_path"]
        for worker in generated_workers["workers"]:
            code = worker["code"]
            worker_name = worker["worker_name"]
            worker_file_path = worker["worker_file_path"]            
            output = self.simple_infer(code=code)["choices"][0]["message"].get("content")
            print (output)
            output = self.output_parser.parse(output)
            print (output)
            if output["is_correct"]:
                print ("The code is correct and working.", worker_name, worker_file_path)
                continue
            else:
                print ("The code is not correct.", worker_name, worker_file_path)
                corrected_code = output["corrected_code"]
                new_code = output["new_code"]
                error_message = output["error_message"]
                self.stm(self.workflow_instance_id)["workers"][worker_name]["code"] = new_code
                file_path = folder_path.joinpath(worker_file_path)
                print (f"Error: {error_message}")
                print (f"Corrected code: {corrected_code}")
                print (f"New code: {new_code}")
                yn = input("Do you want to save the new code? (y/n)")
                if yn == "y":
                    with open(file_path, "w") as f:
                        f.write(new_code)
                else:
                    print ("The new code is not saved.")


    
