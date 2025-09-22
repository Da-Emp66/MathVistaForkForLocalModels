from typing import Optional

import torch
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
)

class InternVL3_Model:
    def __init__(self, model_path: str):
        self.model_id = model_path
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            device_map="cuda",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

    def get_response(self, user_prompt: str, decoded_image: Optional[Image.Image] = None):
        messages = [
            {
                "role": "user",
                "content": [
                    { "type": "image", "image": decoded_image },
                    { "type": "text", "text": user_prompt },
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)
        generate_ids = self.model.generate(**inputs, max_new_tokens=600)
        decoded_output = self.processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        return decoded_output
    