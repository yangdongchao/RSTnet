import os
import sys
sys.path.append('/weka2/home-dongchao/code3/RSTnet_private/MLLM2_11_24')
from omegaconf import OmegaConf
import torch
from tools.tokenizer.abs_tokenizer import AbsTokenizer
import torchaudio
import torch
import librosa
import yaml
from transformers import Wav2Vec2BertModel, SeamlessM4TFeatureExtractor
import safetensors
from tools.tokenizer.SSLTokenizer.utils import RepCodec, CodecEncoder, CodecDecoder, SoundStorm
import accelerate
import soundfile as sf
import math
from einops import rearrange

class SSLTokenizer(AbsTokenizer):
    def __init__(self, device=torch.device('cpu')):
        super(SSLTokenizer, self).__init__()
        self.device = device
        print(self.device)
        # tokenize
        feat_stats_path = '/weka2/home-dongchao/code/tokenizer/checkpoints/tokenizer/mls_wav2vec2bert_stats.pt'
        wav2vec_ckpt = 'facebook/w2v-bert-2.0'
        kmeans_ckpt = '/weka2/home-dongchao/code/tokenizer/checkpoints/tokenizer/86k_steps/model.safetensors'
        feat_stats = torch.load(feat_stats_path, map_location='cpu')
        self.feat_mean = feat_stats['mean']
        self.feat_std = torch.sqrt(feat_stats['var'])
        self.semantic_model = Wav2Vec2BertModel.from_pretrained(wav2vec_ckpt)
        self.semantic_model.eval()
        self.semantic_model.to(self.device)
        self.semantic_processor = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")

        self.kmeans_model = RepCodec()
        self.kmeans_model.eval()
        safetensors.torch.load_model(self.kmeans_model, kmeans_ckpt)
        self.kmeans_model.to(self.device)
        self.sr = 16000
        self.max_length = 2048
        
        # self.acoustic_encoder = CodecEncoder()
        # self.acoustic_decoder = CodecDecoder()

        # acoustic_encoder_path = '/weka2/home-dongchao/code/tokenizer/checkpoints/detokenizer/acoustic/24k/model.safetensors'
        # acoustic_decoder_path = '/weka2/home-dongchao/code/tokenizer/checkpoints/detokenizer/acoustic/24k/model_1.safetensors'
        # accelerate.load_checkpoint_and_dispatch(self.acoustic_encoder, acoustic_encoder_path) 
        # accelerate.load_checkpoint_and_dispatch(self.acoustic_decoder, acoustic_decoder_path) 
        # self.acoustic_encoder.to(self.device)
        # self.acoustic_decoder.to(self.device)

        # self.default_timbre = '/weka2/home-dongchao/code/tokenizer/example/en.wav'
        # self.default_timbre_secs = 5
        # self.default_timbre_speech_16k = librosa.load(self.default_timbre, sr=16000, duration=self.default_timbre_secs)[0]
        # self.default_timbre_speech_16k = torch.tensor(self.default_timbre_speech_16k)[None, ...].to(self.device)
        # self.default_timbre_speech_24k = librosa.load(self.default_timbre, sr=24000, duration=self.default_timbre_secs)[0]
        # self.default_timbre_speech_24k = torch.tensor(self.default_timbre_speech_24k)[None, ...].to(self.device)
        

        # self.default_timbre_semantic = self.tokenize(self.default_timbre_speech_16k).unsqueeze(0).long().to(self.device)
        # self.default_timbre_acoustic = self.tokenize_acoustic(self.default_timbre_speech_24k)
        
        # self.soundstorm_1layer = SoundStorm(
        #     num_quantizer=1,
        #     predict_layer_1=True
        # )
        # self.soundstorm_full = SoundStorm(
        #     num_quantizer=12, 
        #     predict_layer_1=False,
        # )
        # soundstorm_1layer_path = '/weka2/home-dongchao/code/tokenizer/checkpoints/detokenizer/1layer/331ksteps/model.safetensors'
        # soundstorm_full_path = '/weka2/home-dongchao/code/tokenizer/checkpoints/detokenizer/full/519ksteps/model.safetensors'
        # safetensors.torch.load_model(self.soundstorm_1layer, soundstorm_1layer_path)
        # safetensors.torch.load_model(self.soundstorm_full, soundstorm_full_path)
        # self.soundstorm_1layer.eval().to(self.device)
        # self.soundstorm_full.eval().to(self.device)


    @torch.no_grad()
    def encode(self, speech):
        # Input:
        # speech: torch tensor, shape[B, N_speech]
        # Output:
        # semantic token: torch tensor, shape[B, N]
        inputs = self.semantic_processor(speech, sampling_rate=16000, return_tensors="pt")
        input_features = inputs["input_features"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        seg_num = math.ceil(input_features.shape[1] / self.max_length)
        pad_num = seg_num * self.max_length - input_features.shape[1]
        input_features = torch.nn.functional.pad(input_features, (0, 0, 0, pad_num, 0,0), value=0)
        attention_mask = torch.nn.functional.pad(attention_mask, (0, pad_num, 0, 0), value=0)
        input_features = rearrange(input_features, "b (s n) d -> (b s) n d", s =seg_num)
        attention_mask = rearrange(attention_mask, "b (s n) -> (b s) n", s=seg_num)

        feats = self.semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = feats.hidden_states[17]  
        feat = rearrange(feat, "(b s) n d -> b (s n) d", s=seg_num)
        feat = feat[:, :feat.shape[1]-pad_num, :]
        feat = (feat - self.feat_mean.to(feat)) / self.feat_std.to(feat)
        semantic_token, _ = self.kmeans_model.quantize(feat)  
        semantic_token = semantic_token.squeeze(0).detach().cpu().to(torch.int16)
        return semantic_token 

    def find_length(self, x):
        return x.shape[1]

    def tokenize2(self, token):
        if isinstance(token, torch.Tensor):
            return token.to(torch.int64)
        else:
            raise NotImplementedError
    
    def tokenize(self, wav, sample_rate=16000):
        if isinstance(wav, str):
            # if x is the wave path
            wav, sr = torchaudio.load(wav)
            if wav.numel() == 0:
                return None
            if sr != self.sr:
                wav = torchaudio.transforms.Resample(sr, self.sr)(wav)
            return self.encode(wav)
        elif isinstance(wav, torch.Tensor):
            if wav.dim() == 1: # already done offline
                return wav 
            if wav.dim() == 2: # 
                if wav.numel() == 0:
                    return None
                if sample_rate != self.sr:
                    wav = torchaudio.transforms.Resample(sample_rate, self.sr)(wav)
                return self.encode(wav)
        else:
            raise NotImplementedError

    @torch.no_grad()
    def detokenize(self, semantic_token, timbre_speech=None, seg_len=None, diff_steps=None):
        # Input:
        # semantic token: torch tensor, shape[B, N]
        # timbre speech: torch tensor, 16k speech, shape [B, N_speech_16k]
        # Output:
        # speech: torch tensor, shape[B, N_speech_24k]
        if diff_steps is None:
            diff_steps = [40,16,1,1,1,1,1,1,1,1,1,1]
        #print('seg_len:', seg_len)
        tot_len = semantic_token.shape[1]
        if seg_len is None:
            seg_len = tot_len
        last_semantic = None
        last_acoustic = None
        if timbre_speech is None:
            print(f'use default timbre: {self.default_timbre} first {self.default_timbre_secs} secs')
            timbre_semantic = self.default_timbre_semantic
            timbre_acoustic = self.default_timbre_acoustic
        else:
            timbre_semantic = self.tokenize(timbre_speech)
            timbre_speech_24k = torch.tensor(librosa.resample(timbre_speech.cpu().numpy(), orig_sr=16000, target_sr=24000)).to(self.device)
            timbre_acoustic = self.tokenize_acoustic(timbre_speech_24k)
        acoustic_token = list()
        for seg_start in range(0, tot_len, seg_len):
            cur_semantic_seg = semantic_token[:, seg_start: seg_start + seg_len]
            if last_semantic is not None and last_acoustic is not None:      
                cur_semantic = torch.cat([timbre_semantic, last_semantic, cur_semantic_seg], dim=-1)
            else:
                cur_semantic = torch.cat([timbre_semantic, cur_semantic_seg], dim=-1)

            if self.soundstorm_1layer.cond_code_layers == 1:
                cond = self.soundstorm_1layer.cond_emb(cur_semantic)
            else:
                cond = self.soundstorm_1layer.cond_emb[0](cur_semantic[0,:,:])
                for i in range(1, self.soundstorm_1layer.cond_code_layers):
                    cond += self.soundstorm_1layer.cond_emb[i](cur_semantic[i,:,:])
                cond  = cond / math.sqrt(self.soundstorm_1layer.cond_code_layers)

            if last_semantic is not None and last_acoustic is not None:
                prompt = torch.cat([timbre_acoustic, last_acoustic], dim=1)
            else:
                prompt = timbre_acoustic
            predict_1layer = self.soundstorm_1layer.reverse_diffusion(cond=cond, prompt=prompt, temp=1.5, filter_thres=0.98, n_timesteps=[diff_steps[0]], cfg=2.5, rescale_cfg=0.75)

            if self.soundstorm_full.cond_code_layers == 1:
                cond = self.soundstorm_full.cond_emb(cur_semantic)
            else:
                cond = self.soundstorm_full.cond_emb[0](cur_semantic[0,:,:])
                for i in range(1, self.soundstorm_full.cond_code_layers):
                    cond += self.soundstorm_full.cond_emb[i](cur_semantic[i,:,:])
                cond  = cond / math.sqrt(self.soundstorm_full.cond_code_layers)

            cur_acoustic = self.soundstorm_full.reverse_diffusion(cond=cond, prompt=prompt, temp=1.5, filter_thres=0.98, n_timesteps=diff_steps, cfg=2.5, rescale_cfg=0.75, gt_code=predict_1layer)
            acoustic_token.append(cur_acoustic)
            
            last_acoustic = cur_acoustic
            last_semantic = cur_semantic_seg
        acoustic_token = torch.cat(acoustic_token, dim=1)
        return self.detokenize_acoustic(acoustic_token)

    
    @torch.no_grad()
    def tokenize_acoustic(self, speech):
        # Input:
        # speech: torch tensor, shape[B, N_speech_24k]
        # Output:
        # acoustic token: torch tensor, shape[B, N, RVQ]
        vq_emb = self.acoustic_encoder(speech.unsqueeze(1))
        _, vq, _, _, _ = self.acoustic_decoder.quantizer(vq_emb)
        acoustic_token = vq.permute(
            1, 2, 0
        )
        return acoustic_token

    @torch.no_grad()
    def detokenize_acoustic(self, acoustic_token):
        # Input:
        # acoustic token: torch tensor, shape[B, N, RVQ]
        # Output:
        # speech: torch tensor, shape[B, N_speech_24k]
        vq_emb = self.acoustic_decoder.vq2emb(acoustic_token.permute(2,0,1), n_quantizers=12)
        speech = self.acoustic_decoder(vq_emb).squeeze(0)
        return speech

    

if __name__ == '__main__':
    tokenizer = SSLTokenizer(device=torch.device('cuda:0')).cuda()
    test_wav2 = '/weka2/home-dongchao/data/source/p225_001.wav'
    wav, sr = torchaudio.load(test_wav2)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    #wav = wav.cuda()
    inps = [wav, wav, wav]
    codes = tokenizer.encode(inps)
    print('codes ', codes.shape)
    # wav = tokenizer.detokenize(codes.unsqueeze(0).long().cuda())
    # # print('wav ', wav.shape)
    # torchaudio.save('sound1.wav', wav.detach().cpu(), 24000)
    pass
    # assert 1==2
    # wav = tokenizer.detokenize(codes)
    # torchaudio.save('sound1.wav', wav, 24000)
