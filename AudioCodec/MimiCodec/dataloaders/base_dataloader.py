"""
The codec dataloader. It return 24khz audio and 16khz audio for semantic
"""
import random
import torch
from torch.utils.data import Dataset
import torchaudio
from torchaudio.transforms import Resample
import os
import glob
import torchaudio
import torch

class WaveDataset(Dataset):
    def __init__(
        self,
        flist_file,
        segment_size,
        sampling_rate,
        split=True, # whether or not to get a segment of an audio sample to form the batch
        shuffle=False,
        audio_norm_scale: float = 1.0,
    ):
        """
        flist_file: the scp file path
        segment_size: the training segments seconds
        """
        self.file_list = self.get_filelist(flist_file)
        if shuffle:
            random.shuffle(self.file_list)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.semantic_sample_rate = 16000 # almost all of pre-trained SSL models are 16khz
        self.split = split
        self.audio_norm_scale = audio_norm_scale
        self.segment_16k = int((self.segment_size/self.sampling_rate) * self.semantic_sample_rate)

    def get_filelist(self, fpath):
        with open(fpath, 'r') as f:
            flist = [l.strip() for l in f if l.strip()]
        return flist

    def __getitem__(self, index):
        fname = self.file_list[index]
        try:
            audio, sr = torchaudio.load(fname)
            if sr != self.sampling_rate:
                audio_24k = Resample(sr, self.sampling_rate)(audio)
            else:
                audio_24k = audio
            if sr != self.semantic_sample_rate:
                audio_16k = Resample(sr, self.semantic_sample_rate)(audio)
            else:
                audio_16k = audio
            if self.split:
                if audio_24k.size(1) >= self.segment_size:
                    max_audio_start = audio_24k.size(1) - self.segment_size
                    audio_start = random.randint(0, max_audio_start)
                    audio_start_16k = int((audio_start/self.sampling_rate) * self.semantic_sample_rate)
                    audio_24k = audio_24k[:, audio_start:audio_start+self.segment_size]
                    audio_16k = audio_16k[:, audio_start_16k:audio_start_16k+self.segment_16k]
                else:
                    audio_24k = torch.nn.functional.pad(audio_24k, (0, self.segment_size - audio_24k.size(1)), 'constant')
                    audio_16k = torch.nn.functional.pad(audio_16k, (0, self.segment_16k - audio_16k.size(1)), 'constant')
            # in case, audio clip is too short in validation set
            if audio_24k.size(1) < self.segment_size:
                audio_24k = torch.nn.functional.pad(audio_24k, (0, self.segment_size - audio_24k.size(1)), 'constant')
            if audio_16k.size(1) < self.segment_16k:
                audio_16k = torch.nn.functional.pad(audio_16k, (0, self.segment_16k - audio_16k.size(1)), 'constant')
            return audio_24k, audio_16k
        except Exception as e:
            audio_24k = torch.zeros((1, self.segment_size))
            audio_16k = torch.zeros((1, self.segment_16k))
            print('error ', e) # show the error
            return audio_24k, audio_16k

    def __len__(self):
        return len(self.file_list)
