from typing import List
from pydantic import Field
from omagent_core.models.llms.base import BaseLLMBackend
from omagent_core.models.llms.openai_gpt import OpenaiGPTLLM
from omagent_core.models.llms.prompt import PromptTemplate
from omagent_core.models.llms.prompt.parser import *
from PIL import Image
from omagent_core.utils.container import container
import os 
from omagent_core.memories.stms.stm_sharedMem import SharedMemSTM
from omagent_core.utils.registry import registry
from pathlib import Path
from omagent_core.utils.general import encode_image, read_image

os.environ["OMAGENT_MODE"] = "lite"
# Set current working directory path
CURRENT_PATH = Path(__file__).parents[0]

# Import registered modules
registry.import_module(project_path=CURRENT_PATH.joinpath("agent"))
container.register_stm("SharedMemSTM")

class LLMTest(BaseLLMBackend):
    prompts: List[PromptTemplate] = Field(
        default=[
            PromptTemplate.from_template("You are a helpful assistant.", role="system"),
            PromptTemplate.from_template("describe the {{image}}", role="user"),
        ]
    )
    llm: OpenaiGPTLLM ={
        "name": "OpenaiGPTLLM", 
        "model_id": "gpt-4o", 
        "api_key": os.getenv("custom_openai_key"), 
        "vision": True,
        "response_format": "text",
        "use_default_sys_prompt":False
        }
    output_parser: StrParser = StrParser()

llm_test = LLMTest(workflow_instance_id="temp")

img = read_image(input_source="https://media.githubusercontent.com/media/om-ai-lab/OmAgent/refs/heads/main/docs/images/simpleVQA_webpage.png")

chat_completion_res = llm_test.simple_infer(image=img)["choices"][0]["message"].get("content")
print(llm_test.output_parser.parse(chat_completion_res))