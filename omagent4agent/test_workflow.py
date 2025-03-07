from agent.Planner.Planner import Planner
from agent.WorkflowVerifier.WorkflowVerifier import WorkflowVerifier
from agent.WorkflowManager.WorkflowManager import WorkflowManager
from agent.WorkerManager.WorkerManager import WorkerManager
from agent.WorkerVerifier.WorkerVerifier import WorkerVerifier

from omagent_core.engine.workflow.conductor_workflow import ConductorWorkflow
from omagent_core.engine.workflow.task.do_while_task import DoWhileTask
from omagent_core.engine.workflow.task.simple_task import simple_task
from omagent_core.engine.workflow.task.switch_task import SwitchTask
from agent.ConfigManager.ConfigManager import ConfigManager
from agent.ExecuteAgent.ExecuteAgent import ExecuteAgent


class OmAgent4Agent(ConductorWorkflow): 
    def __init__(self):
        super().__init__(name="OmAgent4AgentTester")

    def set_input(self,  folder_path: str, example_inputs: str):        
        self.folder_path = folder_path
        self.example_inputs = example_inputs
        print ("folder_path",self.folder_path)
        print ("example_inputs",self.example_inputs)
        self._configure_tasks()
        self._configure_workflow()

    def _configure_tasks(self):
        """
        self.planner_task = simple_task(
            task_def_name=Planner,
            task_reference_name="planner",
            inputs={"initial_description": self.initial_description},
        )
        """

        self.workflow_manager_task = simple_task(
            task_def_name=WorkflowManager,
            task_reference_name="workflow_manager",
        )
        self.workflow_verifier_task = simple_task(
            task_def_name=WorkflowVerifier,
            task_reference_name="workflow_verifier"
        )

        self.workflow_loop_task = DoWhileTask(
            task_ref_name="workflow_loop",
            tasks=[self.workflow_manager_task, self.workflow_verifier_task],
            termination_condition='if ($.workflow_verifier["valid_json"] == true){false;} else {true;} ',
        )        

        self.worker_manager_task = simple_task(
            task_def_name=WorkerManager,
            task_reference_name="worker_manager"
        )
        self.worker_verifier_task = simple_task(
            task_def_name=WorkerVerifier,
            task_reference_name="worker_verifier"
        )

        self.config_manager_task = simple_task(
            task_def_name=ConfigManager,
            task_reference_name="config_manager"
        )

        self.execute_agent_task = simple_task(
            task_def_name=ExecuteAgent,
            task_reference_name="execute_agent",
            inputs={"folder_path": self.folder_path, "example_inputs": self.example_inputs}
        )
        """
        self.switch_task = SwitchTask(
            task_ref_name="switch_task",
            case_expression=self.conqueror_task.output("switch_case_value"),
        )
        self.switch_task.switch_case("complex", self.divider_task)
        self.switch_task.switch_case("failed", self.rescue_task)
        """


    def _configure_workflow(self):
        # configure workflow execution flow
        #self >> self.planner_task >> self.workflow_manager_task >> self.worker_manager_task >> self.worker_verifier_task
        #self >> self.planner_task >> self.workflow_loop_task >> self.worker_manager_task >> self.config_manager_task 
        self >> self.execute_agent_task 
        #self.dnc_structure = self.task_exit_monitor_task.output("dnc_structure")
        #self.last_output = self.task_exit_monitor_task.output("last_output")
