<div>
    <h1> <img src="docs/images/logo.png" height=33 align="texttop">OmAgent</h1>
</div>

<p align="center">
  <img src="docs/images/icon.png" width="300"/>
</p>

<p align="center">
  <a href="https://twitter.com/intent/follow?screen_name=OmAI_lab" target="_blank">
    <img alt="X (formerly Twitter) Follow" src="https://img.shields.io/twitter/follow/OmAI_lab">
  </a>
  <a href="https://discord.gg/9JfTJ7bk" target="_blank">
    <img alt="Discord" src="https://img.shields.io/discord/1296666215548321822?style=flat&logo=discord">
  </a>
</p>

<p align="center">
    <a>English</a> | <a href="README_ZH.md">中文</a> | <a href="README_JP.md">日本語</a> | <a href="README_FR.md">Français</a>
</p>

## 🗓️ Updates
* 10/20/2024: We are actively engaged in developing version 2.0.0 🚧 Exciting new features are underway! You are welcome to join us on X and Discord~
* 09/20/2024: Our paper has been accepted by EMNLP 2024. See you in Miami!🏝
* 07/04/2024: The OmAgent open-source project has been unveiled. 🎉
* 06/24/2024: [The OmAgent research paper has been published.](https://arxiv.org/abs/2406.16620)




## 📖 Introduction
OmAgent, an open-source framework from OmAI Research, streamlines the development and management of applications leveraging large language models (LLMs). This powerful framework supports the orchestration of multiple AI agents that collaborate through conversational workflows to tackle complex, multifaceted tasks. With built-in integration for multimodal contexts, OmAgent extends its capabilities to encompass in-depth video understanding and analysis.

Key features of OmAgent include:

- **Multimodal Capabilities**: Supports contexts that include rich, comprehensive video analysis, allowing agents to perform detailed understanding and interpretation of video content alongside text-based data.

- **Seamless Open-Source Integration**: OmAgent integrates with platforms like Conductor OSS, which offers robust flow control constructs for orchestrating complex, multi-step processes, further enhancing its functionality.

- **Tool Integration**: Enables the use of external tools and code execution within workflows, equipping agents to handle a wide array of tasks efficiently.

- **Scalability and Flexibility**: OmAgent provides an intuitive interface for building scalable agents, enabling developers to construct agents tailored to specific roles and highly adaptive to various applications. 

These features make OmAgent an accessible yet powerful framework for building sophisticated AI-driven systems that bridge LLMs, multimodal understanding, and tool-assisted workflows.

<p align="center">
  <img src="docs/images/OmAgent.png" width="700"/>
</p>

For more details, check out our paper **[OmAgent: A Multi-modal Agent Framework for Complex Video Understanding with Task Divide-and-Conquer](https://arxiv.org/abs/2406.16620)**

## 🛠️ How To Install
- python >= 3.10
- Install omagent_core
  ```bash
  cd omagent-core
  pip install -e .
  ```
- Other requirements
  ```bash
  cd ..
  pip install -r requirements.txt
  ```
- Set Up Conductor Server (Docker-Compose) Docker-compose includes conductor-server, Elasticsearch, and Redis.
  ```bash
  cd docker
  docker-compose up -d
  ```

- Optional: Install Milvus VectorDB for Long-Term Memory
  ```shell
  # Download milvus startup script
  curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
  # Start milvus in standalone mode
  bash standalone_embed.sh start
  ```

## 🚀 Quick Start & Examples
### General Processing
1. **Worker Classes**
There are several worker classes defined, each registered using a decorator `@registry.register_worker()`:

  ```python
  @registry.register_worker()
  class SimpleWorker(BaseWorker):
      # Basic worker that returns a dictionary with static values
      def _run(self, *args, **kwargs):        
          return 
      
  @registry.register_worker()
  class Conclude(BaseLLMBackend, BaseWorker):
      # Integrates with an LLM (OpenAI GPT) for inference
      llm: OpenaiGPTLLM 
      output_parser: StrParser
      prompts: List[PromptTemplate] = Field(
          default=[
              PromptTemplate.from_template(template="Hello?")
          ]
      )
      def _run(self,*args, **kwargs):
          chat_complete_res = self.simple_infer()
          return chat_complete_res
  ```
- `BaseWorker` is a base class that provides the fundamental structure for all worker implementations. 
- The `_run()` method is where you implement the actual business logic of your worker. Here are some examples with explanations:
- `BaseLLMBackend` class is designed to handle interactions with Language Learning Models (LLMs). 


2. **Workflow Definition**
  ```python
  workflow = ConductorWorkflow(name='my_exp')
  simple_task = simple_task(task_def_name='SimpleWorker', task_reference_name='ref_name_worker')
  task_conclude = simple_task(task_def_name='Conclude', task_reference_name='ref_name_conclude')
  workflow >> simple_task >> task_conclude
  ```
- Creates a workflow with a single task
- `ConductorWorkflow` class is a core component for defining and managing workflows.

- Uses operator `>>` to define task sequence


3. **Compile Configuration**
The compile function generates configuration files for workflows and workers, saving the output file as `worker.yaml` in the specified output_dir

  ```python
  # Compile workflow
  compile(
      workflow=workflow,
      output_dir=Path('./'),
      register=True
  )
  ```


4. **Loading Worker Configuration**
The system reads a configuration file (`worker.yaml`) that was previously generated by the compile function. 

  ```python
  worker_config = yaml.load(open('worker.yaml', "r"), Loader=yaml.FullLoader)
  ```

5. **Task Handler Setup and start Workflow**
The `TaskHandler` is like a manager that oversees all the workers. It reads the configuration and creates all the necessary worker instances.
  ```python
  task_handler = TaskHandler(worker_config=worker_config)
  #Starting the Workflow
  task_handler.start_processes()

  # Start workflow execution
  workflow_execution_id = workflow.start_workflow_with_input(workflow_input={'my_name': 'OmAgent'})
  ```

6. **Monitoring the Execution**
The while loop acts like a supervisor constantly checking on the workflow's progress. It will exit when the status is `COMPLETED`

  ```python
  while True:
      status = workflow.get_workflow(workflow_id=workflow_execution_id).status
      if status == 'COMPLETED':
          break
  ```

### Example Agents
You can find the full list of examples in the [examples](./examples/) directory.

#### Text based Agent
- [Simple LLM chatbot](./examples/chat.py): A basic agent demonstrating how to build an agent with OmAgent. This agent has a straightforward flow with short-term memory and connects directly to ChatGPT.
- [Document Analysis Agent](./examples/document_analysis.py): A more advanced agent utilizing an LLM, short-term and long-term memory, and custom tools for document analysis.

#### Multi-modal Agent
- [Style Bot](./examples/style_bot.py): Offers advice on fashion-related topics such as outfit ideas, accessory pairing, seasonal trends, color coordination, and wardrobe organization.
- [Video Analysis](./examples/DnC.py): Efficiently stores and retrieves relevant video frames for specific queries, preserving detailed video content. ([read EMNLP paper](https://arxiv.org/abs/2406.16620))


## API Documentation
The API documentation is available [here](https://om-ai-lab.github.io/OmAgentDocs/).

## 🔗 Related works
If you are intrigued by multimodal algorithms, large language models, and agent technologies, we invite you to delve deeper into our research endeavors:  
🔆 [How to Evaluate the Generalization of Detection? A Benchmark for Comprehensive Open-Vocabulary Detection](https://arxiv.org/abs/2308.13177) (AAAI24)   
🏠 [GitHub Repository](https://github.com/om-ai-lab/OVDEval/tree/main)

🔆 [OmDet: Large-scale vision-language multi-dataset pre-training with multimodal detection network](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/cvi2.12268) (IET Computer Vision)  
🏠 [Github Repository](https://github.com/om-ai-lab/OmDet)

## ⭐️ Citation

If you find our repository beneficial, please cite our paper:  
```angular2
@article{zhang2024omagent,
  title={OmAgent: A Multi-modal Agent Framework for Complex Video Understanding with Task Divide-and-Conquer},
  author={Zhang, Lu and Zhao, Tiancheng and Ying, Heting and Ma, Yibo and Lee, Kyusong},
  journal={arXiv preprint arXiv:2406.16620},
  year={2024}
}
```


## Star History
[![Star History Chart](https://api.star-history.com/svg?repos=om-ai-lab/OmAgent&type=Date)](https://star-history.com/#om-ai-lab/OmAgent&Date)
