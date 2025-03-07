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
class WorkflowManager(BaseLLMBackend, BaseWorker):
    prompts: List[PromptTemplate] = Field(
        default=[
            PromptTemplate.from_file(CURRENT_PATH.joinpath("workflow_manager_system.prompt"), role="system"),
            PromptTemplate.from_file(CURRENT_PATH.joinpath("workflow_manager_user.prompt"), role="user"),   
        ]
    )
    llm: BaseLLM
    def _run(self, *args, **kwargs):        
        initial_description = self.stm(self.workflow_instance_id)["initial_description"]
        example_input = self.stm(self.workflow_instance_id)["example_input"]
        plan = self.stm(self.workflow_instance_id)["plan"]       
       
        workflow_json = self.simple_infer(content=initial_description, plan=plan, input=example_input)["choices"][0]["message"].get("content")
        #print (type(workflow_json))
        #print (pprint(json.loads(workflow_json)))
        self.callback.info(self.workflow_instance_id, progress="WorkflowManager", message=workflow_json)
        self.stm(self.workflow_instance_id)["workflow_json"] = workflow_json
        return workflow_json

