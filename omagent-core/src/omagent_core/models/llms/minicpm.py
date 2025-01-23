import os
from datetime import datetime
from typing import Any, Dict, List

from pydantic import Field
from .schemas import Content, Message
from ...utils.registry import registry
from .base import BaseLLM
import torch
import sysconfig
from qwen_vl_utils import process_vision_info

@registry.register_llm()
class MiniCPM(BaseLLM):
    model_name: str = Field(default=os.getenv("MODEL_NAME", "openbmb/MiniCPM-o-2_6"), description="The Hugging Face model name")
    max_tokens: int = Field(default=128, description="The maximum number of tokens for the model")
    temperature: float = Field(default=0.1, description="The sampling temperature for generation")
    device: str = Field(default="cuda" if torch.cuda.is_available() else "cpu", description="The device to run the model on")

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            attn_implementation='sdpa',
            torch_dtype=torch.bfloat16
        ).eval().to(self.device)

    def _call(self, records: List[Message], **kwargs) -> Dict:
        messages = self._prepare_inputs(records)
        response = self.model.chat(
            msgs=messages,
            tokenizer=self.tokenizer
        )
        return {"responses": [response]}

    async def _acall(self, records: List[Message], **kwargs) -> Dict:
        raise NotImplementedError("Async calls are not yet supported for MiniCPM models.")

    def convert_messages(self, messages):
        converted = []
        for message in messages:
            msg_dict = {"role": message["role"]}
            content_list = []
            
            for content in message.get("content", []):
                if content.get("type") == "image":
                    from PIL import Image
                    import io
                    image_data = content.get("data")
                    if isinstance(image_data, bytes):
                        image = Image.open(io.BytesIO(image_data)).convert('RGB')
                    else:
                        image = image_data
                    content_list.append(image)
                elif content.get("type") == "text":
                    content_list.append(content.get("data", ""))
            
            msg_dict["content"] = content_list
            converted.append(msg_dict)
        return converted

    def _prepare_inputs(self, records: List[Message]):
        records = records["messages"]
        messages = self.convert_messages(records)
        return messages 