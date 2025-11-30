import re
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM

from model_adapters import BaseAdapter
from utils.constants import *


class DeepseekVLAdapter(BaseAdapter):
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        max_new_tokens: int = 48,
    ):
        super().__init__(model, tokenizer)
        self.processor = VLChatProcessor.from_pretrained(tokenizer.name_or_path)
        self.max_new_tokens = max_new_tokens

    def generate(
        self,
        query: str,
        image: Image,
        task_type: str,
    ) -> str:
        image = image.convert("RGB")

        # Prepare inputs
        conv = [
            {
                "role": "User",
                "content": "<image_placeholder>" + query,
                "images": [image],
            },
            {"role": "Assistant", "content": ""},
        ]
        prepared = self.processor(
            conversations=conv,
            images=[image],
            force_batchify=True,
        ).to(self.model.device)

        inputs_embeds = self.model.prepare_inputs_embeds(**prepared)
        out_ids = self.model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepared.attention_mask,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=False,
        )

        decoded = self.tokenizer.decode(out_ids[0].tolist(), skip_special_tokens=True)
        outputs = decoded.strip()

        # if task_type == CAPTION_TASK:
        #     m = re.findall(r'<meta name="description" content="(.*)">', outputs)
        #     return m[0] if m else outputs

        # if task_type == ELEMENT_OCR_TASK:
        #     if ":" not in outputs:
        #         return outputs
        #     outputs = outputs.split(":", 1)[1]
        #     return outputs.strip().strip('"').strip("'")

        # if task_type == ACTION_PREDICTION_TASK:
        #     return outputs[:1].upper() if outputs else outputs

        return outputs
