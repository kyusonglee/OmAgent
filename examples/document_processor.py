from omagent_core.engine.workflow.conductor_workflow import ConductorWorkflow
from omagent_core.engine.workflow.task.simple_task import simple_task
from omagent_core.engine.worker.base import BaseWorker
from omagent_core.utils.registry import registry
from omagent_core.models.llms.base import BaseLLMBackend
from omagent_core.models.llms.openai_gpt import OpenaiGPTLLM
from omagent_core.models.llms.prompt.prompt import PromptTemplate
from omagent_core.models.llms.prompt.parser import StrParser
from omagent_core.engine.automator.task_handler import TaskHandler
from pathlib import Path
from typing import List, Dict
from pydantic import Field
import yaml
from time import sleep
from omagent_core.utils.compile import compile


# 1. Document Validator Worker
@registry.register_worker()
class DocumentValidator(BaseWorker):
    def _run(self, document_text: str, *args, **kwargs):
        # Validate document length and basic structure
        if not document_text or len(document_text) < 10:
            return {
                'is_valid': False,
                'error': 'Document too short'
            }
        
        return {
            'is_valid': True,
            'processed_text': document_text.strip(),
            'word_count': len(document_text.split())
        }

# 2. Text Analyzer Worker
@registry.register_worker()
class TextAnalyzer(BaseLLMBackend, BaseWorker):
    llm: OpenaiGPTLLM
    output_parser: StrParser
    prompts: List[PromptTemplate] = Field(
        default=[
            PromptTemplate.from_template(
                template="Analyze the following text and identify key themes and sentiment:\n{text}"
            )
        ]
    )

    def _run(self, processed_text: str, *args, **kwargs):
        analysis_result = self.simple_infer(variables={'text': processed_text})
        return {
            'analysis': analysis_result,
            'text_for_summary': processed_text
        }

# 3. Summarizer Worker
@registry.register_worker()
class Summarizer(BaseLLMBackend, BaseWorker):
    llm: OpenaiGPTLLM
    output_parser: StrParser
    prompts: List[PromptTemplate] = Field(
        default=[
            PromptTemplate.from_template(
                template="Provide a concise summary of the following text in 2-3 sentences:\n{text}"
            )
        ]
    )

    def _run(self, text_for_summary: str, analysis: str, *args, **kwargs):
        summary = self.simple_infer(variables={'text': text_for_summary})
        return {
            'summary': summary,
            'analysis': analysis
        }

# Main workflow setup
def setup_document_workflow():
    # Create workflow
    workflow = ConductorWorkflow(name='document_processor')

    # Define tasks
    validator_task = simple_task(
        task_def_name='DocumentValidator',
        task_reference_name='validator',
        inputs={'document_text': workflow.input('document_text')}
    )

    analyzer_task = simple_task(
        task_def_name='TextAnalyzer',
        task_reference_name='analyzer',
        inputs={'processed_text': validator_task.output('processed_text')}
    )

    summarizer_task = simple_task(
        task_def_name='Summarizer',
        task_reference_name='summarizer',
        inputs={
            'text_for_summary': analyzer_task.output('text_for_summary'),
            'analysis': analyzer_task.output('analysis')
        }
    )

    # Define workflow sequence
    workflow >> validator_task >> analyzer_task >> summarizer_task

    return workflow

# Example usage
if __name__ == "__main__":
    # Setup workflow
    workflow = setup_document_workflow()
    
    # Compile workflow
    #compile(workflow, Path('./'), True)

    # Initialize task handler
    worker_config = yaml.load(open('worker.yaml', "r"), Loader=yaml.FullLoader)
    task_handler = TaskHandler(worker_config=worker_config)
    task_handler.start_processes()

    # Example document
    sample_text = """
    Artificial Intelligence has transformed various industries in recent years.
    From healthcare to finance, AI applications are becoming increasingly prevalent.
    However, concerns about ethics and privacy remain important considerations
    that need to be addressed as the technology continues to evolve.
    """

    # Start workflow execution
    workflow_execution_id = workflow.start_workflow_with_input(
        workflow_input={'document_text': sample_text}
    )

    # Monitor execution
    while True:
        status = workflow.get_workflow(workflow_id=workflow_execution_id).status
        print(f"Current status: {status}")
        if status == 'COMPLETED':
            break
        sleep(1)

    # Get results
    result = workflow.get_workflow(workflow_id=workflow_execution_id)
    print("\nWorkflow Results:")
    print(f"Summary: {result.output.get('summary')}")
    print(f"Analysis: {result.output.get('analysis')}")

    # Cleanup
    task_handler.stop_processes() 