from typing import Optional

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

SYS_PROMPT = """You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
The reasoning process MUST BE enclosed within <think> </think> tagsdd. The final answer MUST BE enclosed within <answer> </answer> tags, for example <think>your_thinking_process</think><answer>your_final_answer</answer>. If you use formula, please use LaTeX format."""

QUESTION_TEMPLATE = (
    "{Question}\n"
    "Please think about this question as if you were a human pondering deeply. "
    "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
    "It's encouraged to include self-reflection or verification in the reasoning process. "
    "Provide your detailed reasoning between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags."
)

TYPE_TEMPLATE = {
    "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
    "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
    "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
    "free-form": " Please provide your text answer within the <answer> </answer> tags."
}

class SophiaVL_R1_Model:
    def __init__(self, model_path: str = "bunny127/SophiaVL-R1-7B", max_tokens: int = 4096):
        self.model_id = model_path
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.max_tokens = max_tokens

    def get_response(self, user_prompt: str, decoded_image: Optional[Image.Image] = None):
        max_tokens = self.max_tokens

        image = decoded_image
        if image is not None:
            image_path = None
            if hasattr(image, "name"):
                image_path = image.name # type: ignore
            if image_path is None:
                image_format = (image.format) or "jpg"
                image_path = f"tmp.{str(image_format).lower()}" # type: ignore
                image.save(image_path)
            image_local_path = "file://" + image_path
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=user_prompt) + TYPE_TEMPLATE["free-form"]},
                        {"image": image_local_path},
                    ]
                },
            ]
        else:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=user_prompt) + TYPE_TEMPLATE["free-form"]},
                    ]
                },
            ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if image is not None:
            inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt")
        else:
            inputs = self.processor(text=[text], padding=True, return_tensors="pt")
        inputs = inputs.to('cuda')
        output_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return output_text[0]
    