"""
this code aims to extract semantic features from pre-trained WavLM model
"""
import torch
from semantic_features.WavLM import WavLM, WavLMConfig
import torchaudio
import torch.nn as nn

class WavLMFeature(nn.Module):
    def __init__(self, ckpt_path, device='cpu'):
        super().__init__()
        checkpoint = torch.load(ckpt_path)
        self.cfg = WavLMConfig(checkpoint['cfg'])
        self.model = WavLM(self.cfg)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        self.model = self.model.to(device)
        self.device = device
        self.freeze()

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def extract(self, x):
        """
        extract the feature from last layer of wavlm
        input: <B, T>
        output: <B, T/320, D>
        """
        if len(x.size()) == 3:
            x = x.squeeze(1) # from (B,1,T) ---> (B, T)
        assert len(x.size()) == 2
        #x = torch.cat([x, torch.zeros(x.shape[0], 320).to(x.device)], dim=1)
        if self.cfg.normalize:
            wav_input_16khz = torch.nn.functional.layer_norm(x.to(self.device) , x.shape)
        rep = self.model.extract_features(wav_input_16khz)[0]
        return rep

# ex = WavLMFeature(ckpt_path='/mnt/moonfs/audio-m2/dcyang/exp_data/Moshi/MimiCodec/WavLM-Large.pt', device='cuda')
# x,sr = torchaudio.load('/mnt/moonfs/audio-m2/dcyang/data/speech_data/sub_1/103_1782_000100.wav')
# print(ex.extract(x).shape)
    