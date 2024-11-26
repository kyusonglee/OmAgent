# Introduction
This project implements the Divide-and-Conquer (DnC) text generation method described in the paper "Control Large Language Models via Divide and Conquer" by Li, Bingxuan, et al., in the OmAgent framework. The implementation enables iterative keyword-based text generation by leveraging modular task workers and workflows.

# Overview
The DnCGeneration worker iteratively generates text that satisfies specified constraints (keywords). If any constraints are not met in the initial generation, the process continues by focusing on the unmet constraints and merging the results until all constraints are satisfied or the maximum iterations (k=5) are reached.

# An example
```
python run_cli.py
Input: "Ben Smith, 29-year-old"
Output: 
{
  "final_response": "Ben Smith, a 29-year-old entrepreneur known for his innovative contributions to the tech industry, recently celebrated his birthday with friends in Seattle while making waves with his startup.",
  "record": [
    {
      "rest_set": ["29-year-old"],
      "old_response": "Ben Smith is an entrepreneur.",
      "new_response": "He is a 29-year-old making waves in tech.",
      "merged_response": "Ben Smith, a 29-year-old entrepreneur making waves in tech."
    }
  ],
  "success": true
}
```




