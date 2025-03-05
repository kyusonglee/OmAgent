
from omagent_core.engine.worker.base import BaseWorker
from omagent_core.utils.registry import registry
import json

@registry.register_worker()
class WorkflowVerifier(BaseWorker):    
    def _run(self, *args, **kwargs):          
        # Check for SWITCH task type
        workflow_json = self.stm(self.workflow_instance_id)["workflow_json"]
        try:
            workflow_json = json.loads(workflow_json)
        except:
            print (workflow_json)
            return {"valid_json": False, "error": "Invalid JSON"}

        for task in workflow_json.get('tasks', []):
            if task['type'] == 'SWITCH':
                if not task.get('inputParameters'):
                    print(f"Error: SWITCH task '{task['name']}' has empty inputParameters.")
                    return {"valid_json": False, "error": "SWITCH task has empty inputParameters"}

        # Check for DO_WHILE task type
        for task in workflow_json.get('tasks', []):
            if task['type'] == 'DO_WHILE':
                loop_condition = task.get('loopCondition')
                loop_over = task.get('loopOver', [])
                if not loop_condition or not loop_over:
                    print(f"Error: DO_WHILE task '{task['name']}' is missing loopCondition or loopOver.")
                    return {"valid_json": False, "error": "DO_WHILE task is missing loopCondition or loopOver"}
                # Extract taskReferenceName from loopCondition
                task_ref_name = loop_condition.split('$.')[1].split('[')[0]
                if not any(t['taskReferenceName'] == task_ref_name for t in loop_over):
                    print(f"Error: taskReferenceName '{task_ref_name}' in loopCondition is not in loopOver for task '{task['name']}'.")
                    return {"valid_json": False, "error": "taskReferenceName in loopCondition is not in loopOver"}

        # Check for description inclusion
        description = workflow_json.get('description', [])
        all_worker_names = {task['name'] for task in workflow_json.get('tasks', [])}
        for desc in description:
            if desc['Worker_Name'] not in all_worker_names:
                print(f"Error: Worker '{desc['Worker_Name']}' in description is not in tasks.")
                return {"valid_json": False, "error": "Worker in description is not in tasks"}
        print ("Workflow is valid")
        
        return {"valid_json": True}