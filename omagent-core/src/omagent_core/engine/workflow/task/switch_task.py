from copy import deepcopy
from enum import Enum
from typing import List

from omagent_core.engine.http.models.workflow_task import WorkflowTask
from omagent_core.engine.workflow.task.task import (
    TaskInterface, get_task_interface_list_as_workflow_task_list)
from omagent_core.engine.workflow.task.task_type import TaskType
from typing_extensions import Self
import os

class EvaluatorType(str, Enum):
    JAVASCRIPT = ("javascript",)
    ECMASCRIPT = ("graaljs",)
    VALUE_PARAM = "value-param"

def get_task_interface_list_as_workflow_task_list_for_lite(*tasks: Self) -> List[WorkflowTask]:
    converted_tasks = []
    for task in tasks:        
        #converted_tasks.append(simple_task(task_def_name=task.name, task_reference_name=task.task_reference_name, inputs=task.input_parameters))
        converted_tasks.append(task)
    return converted_tasks

class SwitchTask(TaskInterface):
    def __init__(
        self, task_ref_name: str, case_expression: str, use_javascript: bool = False
    ) -> Self:
        super().__init__(
            task_reference_name=task_ref_name,
            task_type=TaskType.SWITCH,
        )
        self._default_case = None
        self._decision_cases = {}
        self._expression = deepcopy(case_expression)
        self._use_javascript = deepcopy(use_javascript)

    def switch_case(self, case_name: str, tasks: List[TaskInterface]) -> Self:
        if isinstance(tasks, List):
            self._decision_cases[case_name] = deepcopy(tasks)
        else:
            self._decision_cases[case_name] = [deepcopy(tasks)]
        return self

    def default_case(self, tasks: List[TaskInterface]) -> Self:
        if isinstance(tasks, List):
            self._default_case = deepcopy(tasks)
        else:
            self._default_case = [deepcopy(tasks)]
        return self



    def to_workflow_task(self) -> WorkflowTask:
        workflow = super().to_workflow_task()
        if self._use_javascript:
            workflow.evaluator_type = EvaluatorType.ECMASCRIPT
            workflow.expression = self._expression
        else:
            workflow.evaluator_type = EvaluatorType.VALUE_PARAM
            workflow.input_parameters["switchCaseValue"] = self._expression
            workflow.expression = "switchCaseValue"
        workflow.decision_cases = {}
        for case_value, tasks in self._decision_cases.items():
            if os.getenv("OMAGENT_MODE") == "lite":
                workflow.decision_cases[case_value] = (
                    get_task_interface_list_as_workflow_task_list_for_lite(
                        *tasks,
                    )
                )
            else:
                workflow.decision_cases[case_value] = (
                    get_task_interface_list_as_workflow_task_list(
                        *tasks,
                    )
            )
        if self._default_case is None:
            self._default_case = []
        if os.getenv("OMAGENT_MODE") == "lite":
            workflow.default_case = get_task_interface_list_as_workflow_task_list_for_lite(
                *self._default_case
            )
        else:
            workflow.default_case = get_task_interface_list_as_workflow_task_list(
                *self._default_case
            )
        return workflow
