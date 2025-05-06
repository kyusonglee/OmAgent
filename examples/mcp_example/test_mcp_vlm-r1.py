import os
os.environ["OMAGENT_MODE"] = "lite"
from typing import List
from pydantic import Field
from omagent_core.models.llms.base import BaseLLMBackend
from omagent_core.models.llms.openai_gpt import OpenaiGPTLLM
from omagent_core.models.llms.prompt import PromptTemplate
from omagent_core.models.llms.prompt.parser import *
from PIL import Image
from omagent_core.utils.container import container
from omagent_core.memories.stms.stm_sharedMem import SharedMemSTM
from omagent_core.utils.registry import registry
from pathlib import Path
from omagent_core.utils.general import encode_image, read_image
from omagent_core.tool_system.manager import ToolManager
import asyncio
import os

# 设置 CUDA_VISIBLE_DEVICES 环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 只使用第一个 GPU
# Set current working directory path

CURRENT_PATH = Path(__file__).parents[0]

# Import registered modules
registry.import_module(project_path=CURRENT_PATH.joinpath("agent"))
container.register_stm("SharedMemSTM")

class LLMTest(BaseLLMBackend):
    llm: OpenaiGPTLLM ={
        "name": "OpenaiGPTLLM", 
        "model_id": "gpt-4o", 
        "api_key": os.getenv("custom_openai_key"), 
        "endpoint": os.getenv("custom_openai_endpoint"),   
        "vision": False,
        "response_format": "text",
        "use_default_sys_prompt": False,
        }
    tool_manager: ToolManager ={
        "llm": llm, 
        "tools": []
    }

llm_test = LLMTest(workflow_instance_id="temp")

tool_manager = llm_test.tool_manager
#x = tool_manager.execute_task("请描述一下这张图片'/data0/qdl/test/old_women.png'",)   
x = tool_manager.execute_task("你能使用哪些工具？",)    
print(x)

