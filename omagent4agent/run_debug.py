import os
os.environ["OMAGENT_MODE"] = "lite"

# Import required modules and components
from omagent_core.utils.container import container
from omagent_core.engine.workflow.conductor_workflow import ConductorWorkflow
from omagent_core.engine.workflow.task.simple_task import simple_task
from pathlib import Path
from omagent_core.utils.registry import registry
from omagent_core.clients.devices.cli import DefaultClient

#from omagent_core.engine.workflow.task.set_variable_task import SetVariableTask
from omagent_core.utils.logger import logging
#from omagent_core.engine.workflow.task.do_while_task import DoWhileTask
from omagent_core.clients.devices.webpage.client import WebpageClient

from agent.InputInterface.TestInput import TestInput
from debug_workflow import OmAgent4Agent

# Initialize logging
logging.init_logger("omagent", "omagent", level="INFO")

# Set current working directory path
CURRENT_PATH = Path(__file__).parents[0]

# Import registered modules
registry.import_module(project_path=CURRENT_PATH.joinpath('agent'))

container.register_stm("SharedMemSTM")
# Load container configuration from YAML file
container.from_config(CURRENT_PATH.joinpath('container.yaml'))



# Initialize Got workflow
workflow = ConductorWorkflow(name='OmAgent4AgentDebug')

# Configure workflow tasks:
client_input_task = simple_task(task_def_name=TestInput, task_reference_name='input_task')

got_workflow = OmAgent4Agent()
got_workflow.set_input(folder_path=client_input_task.output('folder_path'), example_inputs=client_input_task.output('example_inputs'))
workflow >> client_input_task >> got_workflow

# Register workflow
workflow.register(True)

# Initialize and start CLI client with workflow configuration
config_path = CURRENT_PATH.joinpath('configs')
cli_client = DefaultClient(interactor=workflow, config_path=config_path, workers=[TestInput()])
#cli_client = WebpageClient(interactor=workflow, config_path=config_path, workers=[TestInput()])
cli_client.start_interactor()

