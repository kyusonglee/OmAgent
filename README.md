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


### Video Understanding Task
#### Environment Preparation
- **```Optional```** OmAgent, uses Milvus Lite as a vector database to store vector data by default. If you wish to use the full Milvus service, you can deploy it [milvus vector database](https://milvus.io/docs/install_standalone-docker.md) using docker. The vector database is used to store video feature vectors and retrieve relevant vectors based on queries to reduce MLLM computation. Not installed docker? Refer to [docker installation guide](https://docs.docker.com/get-docker/).
    ```shell
       # Download milvus startup script
       curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
       # Start milvus in standalone mode
       bash standalone_embed.sh start
    ```
    Fill in the relevant configuration information after the deployment ```workflows/video_understanding/config.yml```  
    
- **```Optional```** Configure the face recognition algorithm. The face recognition algorithm can be called as a tool by the agent, but it is optional. You can disable this feature by modifying the ```workflows/video_understanding/tools/video_tools.json``` configuration file and removing the FaceRecognition section. The default face recognition database is stored in the ```data/face_db``` directory, with different folders corresponding to different individuals.
- **```Optional```** Open Vocabulary Detection (OVD) service, used to enhance OmAgent's ability to recognize various objects. The OVD tools depend on this service, but it is optional. You can disable OVD tools by following these steps. Remove the following from ```workflows/video_understanding/tools/video_tools.json```
    ```json 
       {
            "name": "ObjectDetection",
            "ovd_endpoint": "$<ovd_endpoint::http://host_ip:8000/inf_predict>",
            "model_id": "$<ovd_model_id::OmDet-Turbo_tiny_SWIN_T>"
       }
    ```
  
  If using ovd tools, we use [OmDet](https://github.com/om-ai-lab/OmDet/tree/main) for demonstration.
  1. Install OmDet and its environment according to the [OmDet Installation Guide](https://github.com/om-ai-lab/OmDet/blob/main/install.md).
  2. Install requirements to turn OmDet Inference into API calls
     ```text
      pip install pydantic fastapi uvicorn
     ```
  3. Create a ```wsgi.py``` file to expose OmDet Inference as an API
     ```shell
      cd OmDet && vim wsgi.py
     ```
     Copy the [OmDet Inference API code](docs/ovd_api_doc.md) to wsgi.py
  4. Start OmDet Inference API, the default port is 8000
     ```shell
     python wsgi.py
     ```
- Download some interesting videos

#### Running Preparation
1. Create a configuration file and set some necessary environment variables
   ```shell
   cd workflows/video_understanding && vim config.yaml
   ```
2. Configure the API addresses and API keys for MLLM and tools.

   | Configuration Name       | Usage                                                                                   |
   |--------------------------|-----------------------------------------------------------------------------------------|
   | custom_openai_endpoint   | API address for calling OpenAI GPT or other MLLM, format: ```{custom_openai_endpoint}/chat/completions``` |
   | custom_openai_key        | api_key provided by the respective API provider                                          |
   | bing_api_key             | Bing's api key, used for web search                                                      |
   | ovd_endpoint             | ovd tool API address. If using OmDet, the address should be ```http://host:8000/inf_predict``` |
   | ovd_model_id             | Model ID used by the ovd tool. If using OmDet, the model ID should be ```OmDet-Turbo_tiny_SWIN_T``` |

   
2. Set up ```run.py```
    ```python
    def run_agent(task):
        logging.init_logger("omagent", "omagent", level="INFO")
        registry.import_module(project_root=Path(__file__).parent, custom=["./engine"])
        bot_builder = Builder.from_file("workflows/video_understanding") # Video understanding task workflow configuration directory
        input = BaseWorkflowContext(bot_id="1", task=AgentTask(id=0, task=task))
    
        bot_builder.run_bot(input)
        return input.last_output
    
    
    if __name__ == "__main__":
        run_agent("") # You will be prompted to enter the query in the console
    ```
3. Start OmAgent by running ```python run.py```. Enter the path of the video you want to process, wait a moment, then enter your query, and OmAgent will answer based on the query.

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
