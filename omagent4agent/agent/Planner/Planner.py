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


CURRENT_PATH = root_path = Path(__file__).parents[0]


@registry.register_worker()
class Planner(BaseLLMBackend, BaseWorker):
    prompts: List[PromptTemplate] = Field(
        default=[
            PromptTemplate.from_file(CURRENT_PATH.joinpath("planner_system.prompt"), role="system"),
            PromptTemplate.from_file(CURRENT_PATH.joinpath("planner_user.prompt"), role="user"),
        ]
    )
    llm: BaseLLM
    output_parser: StrParser = StrParser()
    def _run(self, initial_description: str, *args, **kwargs):       
        print ("input",initial_description)
        plan_text = self.simple_infer(prompt=initial_description)["choices"][0]["message"].get("content")
        print (plan_text)
        self.stm(self.workflow_instance_id)["plan"] = plan_text
        self.stm(self.workflow_instance_id)["initial_description"] = initial_description

