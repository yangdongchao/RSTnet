from transformers import HubertModel, Wav2Vec2Processor, Wav2Vec2FeatureExtractor
import torch
import torch.nn as nn

class HuBertFeature(nn.Module):
    def __init__(self, ckpt_path, device='cpu'):
        super(HuBertFeature, self).__init__()
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(ckpt_path)
        self.model = HubertModel.from_pretrained(ckpt_path)
        self.model.eval()
        self.model = self.model.to(device)
        self.device = device
        self.freeze()

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def extract(self, x):
        """
        Extract features from HuBert model
        Input: <B, T>
        Output: <B, T/320, D>
        """
        if len(x.size()) == 3:
            x = x.squeeze(1) # from (B,1,T) ---> (B, T)
        assert len(x.size()) == 2

        with torch.no_grad():
            outputs = self.model(x)
            last_hidden_state = outputs['last_hidden_state'].to(torch.float32)  # (B, ssl_dim, T)
        return last_hidden_state
