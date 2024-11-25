#  The Implementation of DnC text Generation by Li, Bingxuan, et al. "Control Large Language Models via Divide and Conquer." EMNLP2024.

from omagent_core.engine.worker.base import BaseWorker
from omagent_core.utils.registry import registry
from omagent_core.models.llms.schemas import Message, Content
import string


@registry.register_worker()
class DnCGeneration(BaseWorker):
    """
    Worker for Divide-and-Conquer text generation.
    Iteratively generates text to include specified keywords and merges results.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from nltk.stem import PorterStemmer
        self.ps = PorterStemmer()  # Initialize PorterStemmer once

    def _run(self, concepts, k, model_name, *args, **kwargs):
        response = self._generate_initial_response(concepts)
        rest_set = self._get_rest_concepts(response, concepts)
        record = []
        count = 0

        while count < k and len(rest_set) > 0:
            new_response = self._generate_response(rest_set)
            merged_response = self._merge_responses(response, new_response)
            
            record.append({
                "rest_set": rest_set,
                "old_response": response,
                "new_response": new_response,
                "merged_response": merged_response
            })

            rest_set = self._get_rest_concepts(merged_response, concepts)
            response = merged_response
            count += 1

        return {
            "final_response": response,
            "record": record,
            "success": len(rest_set) == 0
        }

    def _generate_initial_response(self, concepts):
        prompt = self._build_prompt(concepts)
        return self._call_model(prompt)

    def _generate_response(self, rest_set):
        prompt = self._build_prompt(rest_set)
        return self._call_model(prompt)

    def _merge_responses(self, response1, response2):
        prompt = f"Merge the following two sentences into one:\n'{response1}',\n'{response2}'."
        return self._call_model(prompt)

    def _call_model(self, prompt):
        # Initialize chat message with the user prompt
        chat_message = [
            Message(role="user", message_type="text", content=prompt)
        ]
        chat_complete_res = self.llm.generate(records=chat_message)
        response = chat_complete_res["choices"][0]["message"]["content"]

        return response

    def get_rest_concepts(self, ans_con, gt_con):
        
        for punc in string.punctuation:
            ans_con = ans_con.replace(punc, ' ')
        
        # Stem words in ans_con and gt_con
        ans_word_set = set([self.ps.stem(word_.strip().lower()) for word_ in ans_con.split(' ') if word_ != ''])
        gt_word_set = [self.ps.stem(word_.strip(string.punctuation).strip().lower()) for word_ in gt_con]

        # Identify missing concepts
        rest_set = []
        for i in range(len(gt_word_set)):
            if gt_word_set[i] not in ans_word_set:
                rest_set.append(gt_con[i])
        
        return rest_set

    def _build_prompt(self, keywords):        
        return f"Generate a sentence including the following keywords: {', '.join(keywords)}."