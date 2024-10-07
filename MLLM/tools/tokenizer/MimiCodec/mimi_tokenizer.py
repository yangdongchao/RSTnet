import os
import sys
# # Add MimiCodec to the system path
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'MimiCodec'))
# sys.path.clear()

from omegaconf import OmegaConf
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_model
from tools.tokenizer.MimiCodec.model.models.MimiCodec import MimiCodec
from tools.tokenizer.abs_tokenizer import AbsTokenizer


class MimiTokenizer(AbsTokenizer):
    def __init__(self, device=torch.device('cpu')):
        super(MimiTokenizer, self).__init__()

        ckpt_path = "Moshi/ckpts/moshiko-pytorch-bf16/tokenizer-e351c8d8-checkpoint125.safetensors"
        
        # GPU is only for offline tokenization
        # So, when distributed training is launched, this should still be on CPU
        self.device = device
        working_dir = os.path.dirname(__file__)
        config_path = os.path.join(working_dir, "mimi_config.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        with open(config_path, "r") as f:
            config = OmegaConf.load(f)
        
        self.model = MimiCodec(**config.generator.config)
        if ckpt_path is None or not os.path.exists(ckpt_path):
            ckpt_path = hf_hub_download("kyutai/moshiko-pytorch-bf16", "tokenizer-e351c8d8-checkpoint125.safetensors")
        load_model(self.model, ckpt_path, strict=False)
        self.model.eval()
        # self.model.set_num_codebooks(8)
        self.model = self.model.to(self.device)

    def tokenize(self, wav):
        wav = wav.to(self.device)
        with torch.no_grad():
            codes = self.model.encode(wav)
        return codes
