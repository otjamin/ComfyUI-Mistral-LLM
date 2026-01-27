import os
import torch

from pathlib import Path

from transformers import Mistral3ForConditionalGeneration
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.chunk import ImageChunk

import comfy.model_management as mm  # pyright: ignore[reportMissingImports]
import folder_paths  # pyright: ignore[reportMissingImports]
from comfy_extras.nodes_dataset import tensor_to_pil  # pyright: ignore[reportMissingImports]

model_directory = Path(folder_paths.models_dir) / "LLM"
os.makedirs(model_directory, exist_ok=True)

model_list = ["mistralai/Mistral-Small-3.2-24B-Instruct-2506"]


class DownloadAndLoadMistral3Model:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    model_list,
                    {"default": "mistralai/Mistral-Small-3.2-24B-Instruct-2506"},
                ),
            },
        }

    RETURN_TYPES = ("M3MODEL",)
    RETURN_NAMES = ("mistral3_model",)
    FUNCTION = "load_model"
    CATEGORY = "llm"

    def load_model(self, model: str):
        offload_device = mm.unet_offload_device()

        model_name = model.rsplit("/", 1)[-1]
        model_path = Path(model_directory) / model_name

        if not model_path.exists():
            print(f"Downloading Mistral 3 model to: {model_path}")  # noqa: T201
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=model,
                local_dir=model_path,
                local_dir_use_symlinks=False,
                ignore_patterns="consolidated.safetensors",  # avoid duplicate model download
            )

        model = Mistral3ForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16
        ).to(offload_device)

        tokenizer = MistralTokenizer.from_file(model_path / "tekken.json")

        mistral3_model = {
            "model": model,
            "tokenizer": tokenizer,
        }

        return (mistral3_model,)


class MistralLLMNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mistral3_model": ("M3MODEL",),
                "user_prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "image": ("IMAGE",),
                "system_prompt": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output",)
    CATEGORY = "llm"
    FUNCTION = "generate"

    def generate(
        self, mistral3_model, user_prompt: str, image=None, system_prompt: str = ""
    ):
        device = mm.get_torch_device()

        model: Mistral3ForConditionalGeneration = mistral3_model["model"]
        tokenizer: MistralTokenizer = mistral3_model["tokenizer"]

        model.to(device)

        messages = []

        SYSTEM_PROMPT = system_prompt
        if SYSTEM_PROMPT:
            messages.append({"role": "system", "content": SYSTEM_PROMPT})

        if image:
            image = tensor_to_pil(image)
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt,
                        },
                        ImageChunk(
                            image=image,
                        ),
                    ],
                }
            )
        else:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt,
                        }
                    ],
                }
            )

        tokenized = tokenizer.encode_chat_completion(
            ChatCompletionRequest(messages=messages)
        )

        input_ids = torch.tensor([tokenized.tokens]).to(device)
        attention_mask = torch.ones_like(input_ids).to(device)
        if image:
            pixel_values = (
                torch.tensor(tokenized.images[0], dtype=torch.bfloat16)
                .unsqueeze(0)
                .to(device)
            )
            image_sizes = torch.tensor([pixel_values.shape[-2:]]).to(device)

            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                max_new_tokens=1000,
            )[0]
        else:
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1000,
            )[0]

        decoded_output = tokenizer.decode(output[len(tokenized.tokens) :])

        return (decoded_output,)


NODE_CLASS_MAPPINGS = {
    "MistralLLMNode": MistralLLMNode,
    "DownloadAndLoadMistral3Model": DownloadAndLoadMistral3Model,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MistralLLMNode": "Mistral LLM",
    "DownloadAndLoadMistral3Model": "Load Mistral 3 Model",
}
