from pathlib import Path
from typing import List
from omagent_core.engine.worker.base import BaseWorker
from omagent_core.models.llms.base import BaseLLMBackend, BaseLLM
from omagent_core.models.llms.prompt import PromptTemplate
from omagent_core.utils.logger import logging
from omagent_core.utils.registry import registry
from openai import Stream
from pydantic import Field
from collections.abc import Iterator
from omagent_core.models.llms.prompt.parser import *    
from pprint import pprint
import os

CURRENT_PATH = root_path = Path(__file__).parents[0]


@registry.register_worker()
class DebugAgent(BaseLLMBackend, BaseWorker):
    prompts: List[PromptTemplate] = Field(
        default=[
            PromptTemplate.from_file(CURRENT_PATH.joinpath("debug_agent_system.prompt"), role="system"),
            PromptTemplate.from_file(CURRENT_PATH.joinpath("debug_agent_user.prompt"), role="user"),   
        ]
    )
    llm: BaseLLM
    def _run(self, *args, **kwargs):  
        error_message = self.stm(self.workflow_instance_id)["error_message"]
        traceback = self.stm(self.workflow_instance_id)["traceback"]
        workflow = self.stm(self.workflow_instance_id)["workflow_json"]
        #code = self.stm(self.workflow_instance_id)["code"]
        example_input = self.stm(self.workflow_instance_id)["example_inputs"]
        print ("example_input: ",example_input)
        folder_path = self.stm(self.workflow_instance_id)["folder_path"]
        name = workflow["name"]
        file_path = os.path.join(folder_path, name, name+".py")
        if os.path.exists(file_path):
            print ("File exists")
        else:
            print ("File does not exist")
        print (error_message, traceback, workflow, example_input)

        """
        new_code = self.simple_infer(traceback=traceback, error_message=error_message, workflow=workflow, code=code, input=example_input)["choices"][0]["message"].get("content")        
        file_path = os.path.join(folder_path, name, name+".py")
        with open(file_path, "w") as f:
            f.write(new_code)

        #return {"new_code": new_code}   
        #return {"has_no_error": True}
        """

