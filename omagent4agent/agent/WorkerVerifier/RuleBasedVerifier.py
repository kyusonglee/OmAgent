from pathlib import Path
from omagent_core.engine.worker.base import BaseWorker
from omagent_core.utils.registry import registry
import re
import json 
from typing import List
from omagent_core.engine.worker.base import BaseWorker
from omagent_core.models.llms.base import BaseLLMBackend, BaseLLM
from omagent_core.models.llms.prompt import PromptTemplate
from pydantic import Field
import os 


CURRENT_PATH = root_path = Path(__file__).parents[0]

def get_returns(code):
    # Regex to find the dictionary inside the return statement
    match = re.search(r'return\s*{([^}]*)}', code, re.DOTALL)
    if not match:
        return []
    dict_content = match.group(1)
    # Regex to capture all keys in the dictionary
    keys = re.findall(r'"(\w+)"\s*:', dict_content)
    return keys


@registry.register_worker()
class RuleBasedVerifier(BaseLLMBackend, BaseWorker):  
    prompts: List[PromptTemplate] = Field(
        default=[
            PromptTemplate.from_file(CURRENT_PATH.joinpath("worker_verifier_system.prompt"), role="system"),
            PromptTemplate.from_file(CURRENT_PATH.joinpath("rule_based_verifier_user.prompt"), role="user"),   
        ]
    )
    llm: BaseLLM

    def _run(self, *args, **kwargs):                
        generated_workers = self.stm(self.workflow_instance_id)["generated_workers"]       
        workflow_json = self.stm(self.workflow_instance_id)["workflow_json"]
        results = {}
        for i, worker in enumerate(generated_workers["workers"]):
            code = worker["code"]
            if "kwargs.get(" in code:
                if not worker_name in results:
                    results[worker_name] = []
                results[worker_name].append({"is_correct": False, "error_message": "The worker has kwargs.get. This is not allowed. Please use input parameters instead."})

            worker_name = worker["worker_name"]
            worker_file_path = worker["worker_file_path"]  
            output = self.verify_input_parameters(code, workflow_json, worker_name)            
            if output["is_correct"]:
                print ("The input parameters are correct.", worker_name, worker_file_path)
            else:                
                if not worker_name in results:
                    results[worker_name] = []
                results[worker_name].append(output)
                print ("The input parameters are not correct.", worker_name, worker_file_path)
        
        for i, worker in enumerate(generated_workers["workers"]):
            output = self.test_import(worker["code"])
            if output["is_correct"]:
                print ("The import is correct.", worker_name, worker_file_path)
            else:
                if not worker_name in results:
                    results[worker_name] = []
                results[worker_name].append(output)
                print ("The import is not correct.", worker_name, worker_file_path)
        

        results = self.verify_output_parameters(generated_workers["workers"], workflow_json, results)
        folder = self.stm(self.workflow_instance_id)["folder_path"]
        for w in generated_workers["workers"]:                    
            if w["worker_name"] in results:
                error_messages = " ".join([x["error_message"] for x in results[w["worker_name"]]])
                output = self.simple_infer(code=w["code"], workflow_json=workflow_json, error_messages=error_messages)["choices"][0]["message"].get("content")
                w["code"] = self.parser(output)
                self.callback.info(agent_id=self.workflow_instance_id, progress="RuleBasedVerifier", message=f"The code is corrected. {w['worker_name']}")
                self.callback.info(agent_id=self.workflow_instance_id, progress="RuleBasedVerifier", message=w["code"])

                task_name = w["worker_name"]
                code_path = os.path.join(folder, "agent", task_name, f"{task_name}.py")
                with open(code_path, "w") as f:
                    f.write(w["code"])
          
    def parser(self, output):
        if "```python" in output:
            output = output.split("```python")[1].split("```")[0]
        return output
                
    def verify_input_parameters(self, code, workflow_json, worker_name):   
        class_name = worker_name
        # Corrected regex pattern to capture function parameters
        print (code)
        function_parameters = re.findall(r"def _run\(self, ([^)]*)\)", code)
        if len(function_parameters) > 0:
            function_parameters = function_parameters[0].split(",")
        function_parameters = [x.strip().split(":")[0] for x in function_parameters if not x.strip().startswith("*")]        
        print (function_parameters)
        tasks = {}    
        for task in json.loads(workflow_json)["tasks"]:            
            if task["type"] == "SWITCH":
                for case in task["decisionCases"]:
                    if type(case) == list:
                        for c in case:
                            tasks[c["name"]] = c
                    else:
                        print ("case",case)
                        tasks[case["name"]] = case
            elif task["type"] == "DO_WHILE":
                for c in task["loopOver"]:
                    tasks[c["name"]] = c  
            
            tasks[task["name"]] = task


        task = tasks[class_name]
        if not "inputParameters" in task:
            return {"is_correct": True, "error_message": ""}
        if not task["inputParameters"] == {}:
            input_parameters = task["inputParameters"].keys()
            
            if set(input_parameters) != set(function_parameters):
                print(input_parameters, function_parameters)
                return {"is_correct": False, "error_message": "The input parameters are not correct. Input parameters should contains: "+", ".join(input_parameters)}
            else:
                return {"is_correct": True, "error_message": ""}
        else:
            if len(function_parameters) > 0:
                return {"is_correct": False, "error_message": "The input parameters are not correct. Input parameters are empty. but function parameters have values:" + ", ".join(function_parameters)}
            else:
                return {"is_correct": True, "error_message": ""}
    
    def verify_output_parameters(self, workers, workflow_json, results):    
        match = re.findall(r'\${(.*?)\.output\.(.*?)}', str(workflow_json))
        workers_codes = {}
        workers_names = {}
        print ("match:",match)
        if type(workflow_json) == str:
            workflow_json = json.loads(workflow_json)
        Dic = {}
        for w in workflow_json["tasks"]:     
            if w["type"] == "SWITCH":
                for case in w["decisionCases"]:
                    for c in case:
                        Dic[c["name"]] = c["taskReferenceName"]
                        workers_names[c["taskReferenceName"]] = c["name"]
            elif w["type"] == "DO_WHILE":
                for c in w["loopOver"]:
                    Dic[c["name"]] = c["taskReferenceName"]
                    workers_names[c["taskReferenceName"]] = c["name"]
            else:
                Dic[w["name"]] = w["taskReferenceName"]
                workers_names[w["taskReferenceName"]] = w["name"]

        for w in workers:            
            try:
                workers_codes[Dic[w["worker_name"]]] = w["code"]
            except:
                pass
                #print ("worker_name:",w["worker_name"])
                #print ("Dic:",Dic)

        #print (workers_codes)
        #print (workers_names)
        if len(match) > 0:
            for x in match:
                ref_name = x[0]
                output_param = x[1]
                get_returns = get_returns(workers_codes[ref_name])            
                print (get_returns)
                print (output_param)
                if set(output_param) == set(get_returns):
                    continue
                else:
                    if ref_name in results:
                        results[workers_names[ref_name]] = []
                    results[workers_names[ref_name]].append({"is_correct": False, "error_message": f"The output parameter is not in the return statement."})
              
        return results

    def test_import(self, code):
        import_code_only = []
        for line in code.split("\n"):            
            if "@registry.register_worker()" in line:
                break
            import_code_only.append(line)
        try:            
            exec("\n".join(import_code_only))
        except ImportError as e:
            return {"is_correct": False, "error_message": str(e)}
        return {"is_correct": True, "error_message": ""}
    
