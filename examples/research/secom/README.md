# **SeCom: On Memory Construction and Retrieval for Personalized Conversational Agents**

This repository implements the SeCom memory management system described in the paper ["SeCom: On Memory Construction and Retrieval for Personalized Conversational Agents"](https://arxiv.org/abs/XXXXX) by Zhuoshi Pan, Qianhui Wu, Huiqiang Jiang, et al., using the OmAgent framework.

## **Overview**
SeCom introduces a novel method to construct and retrieve memory in conversational agents, focusing on long-term interactions. By organizing memory at a **segment level** and applying **compression-based denoising**, SeCom balances retrieval accuracy, relevance, and efficiency. This implementation showcases:
- Conversation segmentation into topically coherent segments.
- Compression techniques to remove redundancy from memory units.
- Improved retrieval and response generation for long-term conversational benchmarks.


## **Usage**

### **Running the Workflow**
To run the SeCom workflow from the terminal:
```bash
python run_cli.py
```

### **Example Input and Output**
#### Input:
```plaintext
Conversation: "What’s your favorite movie? I love Star Wars. How about dinner plans?"
```

#### Output:
```json
{
  "segmented_history": [
    {"segment_id": 0, "topic": "Movies", "content": "What’s your favorite movie? I love Star Wars."},
    {"segment_id": 1, "topic": "Dinner Plans", "content": "How about dinner plans?"}
  ],
  "response": "Sure, I love Star Wars too! Let's plan dinner for Italian cuisine."
}
```

---

## **Implementation Details**
The SeCom implementation comprises the following workers:

### **1. InputInterface**
- Handles user input and stores conversation history.
- Provides the input data for segmentation and retrieval tasks.

### **2. SeCom**
- Constructs the memory bank at the **segment level** by dividing long conversations into topically coherent units.
- Applies **compression-based denoising** using models like LLMLingua-2 to improve retrieval performance.

### **3. SimpleQA**
- Integrates the segmented memory to generate accurate and context-aware responses.

### **4. Workflow**
- Orchestrates tasks using the `DoWhileTask` to ensure iterative improvements and updates until a termination condition is met.

---

## **Development and Customization**
You can modify the segmentation model, denoising technique, or retrieval logic to fit your needs. Key files include:
- **`input_interface.py`**: Input handling and storage.
- **`secom.py`**: Core segmentation and compression-based memory construction.
- **`run_cli.py`**: Workflow execution script.

---

## **Citing This Work**
If you use this implementation in your research, please cite the original paper:
```bibtex
@article{secom2024,
  title={SeCom: On Memory Construction and Retrieval for Personalized Conversational Agents},
  author={Pan, Zhuoshi and Wu, Qianhui and Jiang, Huiqiang and others},
  journal={arXiv preprint arXiv:XXXXX},
  year={2024}
}
```

---

## **Contributors**
- Zhuoshi Pan, Qianhui Wu, Huiqiang Jiang, and others.

---

## **License**
This project is licensed under the MIT License. See the LICENSE file for more details.

---

## **Acknowledgments**
This work builds on ideas from retrieval-augmented generation, conversational segmentation, and compression techniques. The LOCOMO and Long-MT-Bench+ datasets were used for benchmarking.

--- 

This README provides an easy-to-understand structure for users to run, understand, and modify the implementation.