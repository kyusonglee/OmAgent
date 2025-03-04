from agent.Planner.Planner import Planner
from agent.WorkflowVerifier.WorkflowVerifier import WorkflowVerifier
from agent.WorkflowManager.WorkflowManager import WorkflowManager
from agent.WorkerManager.WorkerManager import WorkerManager

from omagent_core.engine.workflow.conductor_workflow import ConductorWorkflow
from omagent_core.engine.workflow.task.do_while_task import DoWhileTask
from omagent_core.engine.workflow.task.simple_task import simple_task
from omagent_core.engine.workflow.task.switch_task import SwitchTask


class OmAgent4Agent(ConductorWorkflow):
    def __init__(self):
        super().__init__(name="OmAgent4Agent")

    def set_input(self,  initial_description: str):        
        self.initial_description = initial_description
        print ("111initial_description",self.initial_description)
        self._configure_tasks()
        self._configure_workflow()

    def _configure_tasks(self):
        self.planner_task = simple_task(
            task_def_name=Planner,
            task_reference_name="planner",
            inputs={"initial_description": self.initial_description},
        )

        self.workflow_manager_task = simple_task(
            task_def_name=WorkflowManager,
            task_reference_name="workflow_manager",
        )
        self.workflow_verifier_task = simple_task(
            task_def_name=WorkflowVerifier,
            task_reference_name="workflow_verifier"
        )

        self.worker_manager_task = simple_task(
            task_def_name=WorkerManager,
            task_reference_name="worker_manager"
        )
        """
        self.switch_task = SwitchTask(
            task_ref_name="switch_task",
            case_expression=self.conqueror_task.output("switch_case_value"),
        )
        self.switch_task.switch_case("complex", self.divider_task)
        self.switch_task.switch_case("failed", self.rescue_task)
        """

        # DnC loop task for task loop
        self.dncloop_task = DoWhileTask(
            task_ref_name="workflow_loop",
            tasks=[self.workflow_manager_task, self.workflow_verifier_task],
            termination_condition=self.workflow_verifier_task.output("loop_condition"),
        )

    def _configure_workflow(self):
        # configure workflow execution flow
        self >> self.planner_task >> self.workflow_manager_task >> self.worker_manager_task 
        #self.dnc_structure = self.task_exit_monitor_task.output("dnc_structure")
        #self.last_output = self.task_exit_monitor_task.output("last_output")
