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
import glob2
import difflib
import sys

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
        example_input = self.stm(self.workflow_instance_id)["example_input"]
        print ("example_input: ",example_input)
        folder_path = self.stm(self.workflow_instance_id)["folder_path"]
        name = workflow["name"]
        file_paths = glob2.glob(os.path.join(folder_path, "agent", "**", "*.py"))
        code = ""
        dict_code = {}
        for file_path in file_paths:
            with open(file_path, "r") as f:
                if "__init__.py" in file_path:
                    continue
                code += "# File_path: "+file_path+"\n"
                temp =f.read()
                code += temp
                code += "\n\n"
                dict_code[file_path] = temp
        if error_message:
            self.callback.info(agent_id=self.workflow_instance_id, progress="DEBUGING", message=error_message)
            new_codes = self.simple_infer(traceback=traceback, error_message=error_message, workflow=workflow, code=code, input=example_input)["choices"][0]["message"].get("content")        
            new_codes = self.parse(new_codes)
            for new_code in new_codes:
                file_path = new_code["file_path"]
                old_code = dict_code[file_path]         
                diff_code = self.diff(old_code, new_code["code"])  
                self.callback.info(agent_id=self.workflow_instance_id, progress="Suggestion: "+file_path, message=diff_code)
                input = self.input.read_input(workflow_instance_id=self.workflow_instance_id, input_prompt="Do you want to fix the error? (yes/no)")
                content = input['messages'][-1]['content']
                for content_item in content:
                    if content_item['type'] == 'text':
                        fix_error = content_item['data']
                if fix_error == "yes":
                    with open(file_path, "w") as f:                                                             
                        f.write(new_code["code"])
                    self.clear_modules(folder_path)
            return {"finished": False}
        else:
            return {"finished": True}
        
        

    def parse(self, code: str):
        return json.loads(code.replace("```json", "").replace("```", ""))

    def diff(self, code_a: str, code_b: str):
        """
        Compares two code snippets and highlights only the differences.
        """
        code_a_lines = code_a.splitlines()
        code_b_lines = code_b.splitlines()
        
        diff_generator = difflib.unified_diff(code_a_lines, code_b_lines, lineterm='')
        
        diff_output = '\n'.join(diff_generator)
        
        return diff_output

    def clear_modules(self, folder: str):        
        agents_dir = os.path.join(folder, "agent")
        for module_name in list(sys.modules.keys()):
            module = sys.modules[module_name]
            if module_name.startswith("agent") or (hasattr(module, '__file__') and module.__file__ and agents_dir in module.__file__):
                del sys.modules[module_name]
                # Extract the class name from the module name
                class_name = module_name.split('.')[-1]
                try:
                    registry.unregister("worker", class_name)
                except KeyError:
                    print(f"Module {class_name} not found in registry, skipping unregistration.")
