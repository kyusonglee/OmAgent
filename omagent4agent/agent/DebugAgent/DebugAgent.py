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
        workflow = self.stm(self.workflow_instance_id)["workflow"]
        code = self.stm(self.workflow_instance_id)["code"]
        example_input = self.stm(self.workflow_instance_id)["example_input"]
       
        llm_response = self.simple_infer(error_message=error_message, workflow=workflow, code=code, input=example_input)["choices"][0]["message"].get("content")
        
        
        return llm_response

