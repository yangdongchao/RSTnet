import os
import sys
sys.path.append('/weka2/home-dongchao/code3/RSTnet_private/MLLM2_11_24')
import os
import io
import glob
import math
import tarfile
import torch
import torchaudio
import safetensors
from tools.tokenizer.GLM4V.configuration_whisper import WhisperVQConfig
from tools.tokenizer.GLM4V.modeling_whisper import WhisperVQEncoder, WhisperVQForConditionalGeneration
from transformers import WhisperFeatureExtractor, WhisperTokenizerFast
from tools.tokenizer.abs_tokenizer import AbsTokenizer
from tools.tokenizer.GLM4V.flow_inference import AudioDecoder

class SSLTokenizer(AbsTokenizer):
    def __init__(self, device=torch.device('cpu')):
        super(SSLTokenizer, self).__init__()
        self.device = device
        model_path = '/weka2/home-dongchao/data/GLM-4-Voice/glm-4-voice-tokenizer'
        flow_config = '/weka2/home-dongchao/data/GLM-4-Voice/glm-4-voice-decoder/config.yaml'
        flow_checkpoint = '/weka2/home-dongchao/data/GLM-4-Voice/glm-4-voice-decoder/flow.pt'
        hift_checkpoint = '/weka2/home-dongchao/data/GLM-4-Voice/glm-4-voice-decoder/hift.pt'
        self.model = WhisperVQEncoder.from_pretrained(model_path).eval().to(device)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)
        self._resample_buffer: dict[int, torchaudio.transforms.Resample] = {}
        # self.audio_decoder = AudioDecoder(config_path=flow_config, flow_ckpt_path=flow_checkpoint,
        #                               hift_ckpt_path=hift_checkpoint,
        #                              device=device)
    def extract_speech_token(self, utts):
        with torch.no_grad():
            audios, indices = [], []
            for idx, utt in enumerate(utts):
                if isinstance(utt, tuple):
                    audio, sample_rate = utt
                else:
                    audio, sample_rate = torchaudio.load(utt)
                audio = audio.cuda()
                if sample_rate != 16000:
                    if sample_rate not in self._resample_buffer:
                        self._resample_buffer[sample_rate] = torchaudio.transforms.Resample(
                            orig_freq=sample_rate,
                            new_freq=16000
                        ).to('cuda')
                    audio = self._resample_buffer[sample_rate](audio)
                # if audio.shape[0] > 1:
                #     audio = audio[:1]
                audio = audio[0]
                audio = audio.cpu().numpy()
                time_step = 0
                while time_step * 16000 < audio.shape[0]:
                    audio_segment = audio[time_step * 16000: (time_step + 30) * 16000]
                    audios.append(audio_segment)
                    indices.append(idx)
                    time_step += 30
            pooling_kernel_size = self.model.config.pooling_kernel_size or 1
            stride = self.model.conv1.stride[0] * self.model.conv2.stride[0] * pooling_kernel_size * self.feature_extractor.hop_length
            all_speech_tokens = [[] for _ in range(len(utts))]
            batch_size = 128
            print('len(audios) ', len(audios))
            for start in range(0, len(audios), batch_size):
                #print('audios[start: start + batch_size] ', audios[start: start + batch_size][0].shape)
                features = self.feature_extractor(audios[start: start + batch_size], sampling_rate=16000,
                                            return_attention_mask=True, return_tensors="pt", device='cuda',
                                            padding="longest", pad_to_multiple_of=stride)
                features = features.to(device="cuda")
                outputs = self.model(**features)
                speech_tokens = outputs.quantized_token_ids
                #print('speech_tokens ', speech_tokens.shape)
                attention_mask = features.attention_mask[:, ::self.model.conv1.stride[0] * self.model.conv2.stride[0]]
                attention_mask = attention_mask[:, ::self.model.config.pooling_kernel_size]
                #print('attention_mask ', attention_mask[0].bool())
                assert attention_mask.shape == speech_tokens.shape
                for i in range(len(speech_tokens)):
                    idx = indices[start + i]
                    speech_token = speech_tokens[i][attention_mask[i].bool()].tolist()
                    all_speech_tokens[idx].extend(speech_token)
            return all_speech_tokens


    def find_length(self, x):
        return x.shape[1]

    def tokenize2(self, token):
        if isinstance(token, torch.Tensor):
            return token.to(torch.int64)
        else:
            raise NotImplementedError
    
    def tokenize(self, utts):
        # a list of audio
        with torch.no_grad():
            audios, indices = [], []
            for idx, utt in enumerate(utts):
                if isinstance(utt, tuple):
                    audio, sample_rate = utt # make sure the audio is (1, T)
                else:
                    audio, sample_rate = torchaudio.load(utt)
                audio = audio.to(self.device)
                if sample_rate != 16000:
                    if sample_rate not in self._resample_buffer:
                        self._resample_buffer[sample_rate] = torchaudio.transforms.Resample(
                            orig_freq=sample_rate,
                            new_freq=16000
                        ).to(self.device)
                    audio = self._resample_buffer[sample_rate](audio)
                # if audio.shape[0] > 1:
                #     audio = audio[:1]
                audio = audio[0]
                audio = audio.cpu().numpy()
                time_step = 0
                while time_step * 16000 < audio.shape[0]:
                    audio_segment = audio[time_step * 16000: (time_step + 30) * 16000]
                    audios.append(audio_segment)
                    indices.append(idx)
                    time_step += 30
            pooling_kernel_size = self.model.config.pooling_kernel_size or 1
            stride = self.model.conv1.stride[0] * self.model.conv2.stride[0] * pooling_kernel_size * self.feature_extractor.hop_length
            all_speech_tokens = [[] for _ in range(len(utts))]
            batch_size = 128
            # print('len(audios) ', len(audios))
            for start in range(0, len(audios), batch_size):
                #print('audios[start: start + batch_size] ', audios[start: start + batch_size][0].shape)
                features = self.feature_extractor(audios[start: start + batch_size], sampling_rate=16000,
                                            return_attention_mask=True, return_tensors="pt", device=self.device,
                                            padding="longest", pad_to_multiple_of=stride)
                features = features.to(device=self.device)
                outputs = self.model(**features)
                speech_tokens = outputs.quantized_token_ids
                # print('speech_tokens ', speech_tokens.shape)
                attention_mask = features.attention_mask[:, ::self.model.conv1.stride[0] * self.model.conv2.stride[0]]
                attention_mask = attention_mask[:, ::self.model.config.pooling_kernel_size]
                # print('attention_mask ', attention_mask[0].bool())
                assert attention_mask.shape == speech_tokens.shape
                for i in range(len(speech_tokens)):
                    idx = indices[start + i]
                    speech_token = speech_tokens[i][attention_mask[i].bool()].tolist()
                    all_speech_tokens[idx].extend(speech_token)
            return all_speech_tokens

    @torch.no_grad()
    def detokenize(self, semantic_token):
        return self.audio_decoder.offline_inference(semantic_token)

if __name__ == '__main__':
    pass
    # tokenizer = SSLTokenizer(device=torch.device('cuda:0')).cuda()
    # test_wav2 = '/weka2/home-dongchao/data/source/p225_001.wav'
    # test_wav3 = '/weka2/home-dongchao/data/source/p225_002.wav'
    # # wav, sr = torchaudio.load(test_wav2)
    # # # if sr != 16000:
    # # #     wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    # # #wav = wav.cuda()
    # codes = tokenizer.extract_speech_token([test_wav2, test_wav3])
    # for c in codes:
    #     c = torch.tensor(c)
    #     print(c.shape)
    #     wav = tokenizer.detokenize(c.unsqueeze(0).long().cuda())
    #     torchaudio.save('sound2.wav', wav.detach().cpu(), 16000)
    #     print('wav ', wav.shape)
    #     assert 1==2
    # torchaudio.save('sound1.wav', wav.detach().cpu(), 24000)
    #pass
    # assert 1==2
    # wav = tokenizer.detokenize(codes)
    # torchaudio.save('sound1.wav', wav, 24000)
