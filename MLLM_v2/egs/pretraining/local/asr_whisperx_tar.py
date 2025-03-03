''' this code helps to obtain the word-level alignment by whisperx 
The input data format is tar. Support re-start: we will record the tokenize information
'''
import argparse
import json
import os
import torch
import torchaudio
import whisperx
import webdataset as wds
from io import BytesIO
import io
import soundfile as sf
import numpy as np 
from scipy.signal import resample

def main(args):
    rank = args.rank - 1 # run.pl starts from 1 but the exact jobid / gpuid starts from 0   
    max_gpu = torch.cuda.device_count()
    rank = (rank % max_gpu)
    device = torch.device(f"cuda:{rank}")
    model = whisperx.load_model("medium.en", device="cuda", device_index=rank, compute_type=args.compute_type) # load the whisper model
    model_a, metadata = whisperx.load_align_model(language_code="en", device=device)
    sr = 16000 # whisperx also use 16k hz
    tar_files = []
    with open(args.input_file, 'r') as f:
        for line in f: # tranvel the tar.scp
            tar_files.append(line.split()[-1])
    save_data = {}
    import time
    st_time = time.time()
    for tar_path in tar_files: 
        tar_dataset = wds.WebDataset(tar_path)
        for sample in tar_dataset: # get the speech info
            flag = True
            json_data = sample['json']
            json_decoded = json.loads(json_data.decode('utf-8')) 
            key = sample['__key__']
            # 解码 FLAC 数据
            flac_data = sample['flac']
            audio, original_sr = sf.read(io.BytesIO(flac_data))
            if original_sr != sr:
                num_samples = int(len(audio) * sr / original_sr)
                audio = resample(audio, num_samples)
            # 如果音频是多声道，则取平均值转换为单声道
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            # 将数据转换为 float32 并标准化到 [-1.0, 1.0]
            audio = audio.astype(np.float32) / np.max(np.abs(audio)) # 
            result = model.transcribe(audio, batch_size=args.batch_size, language="en")
            result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
            word_segments = result["word_segments"]
            segments = result["segments"]
            tmp_save = {}
            tmp_save['word_segments'] = word_segments
            tmp_save['duration'] = audio.shape[-1] / 16000
            tmp_save['segments'] = segments
            save_data[key] = tmp_save
        
        # Save the list of new audio files
        #print('args.output_file ', args.output_file)
        scp_dir = os.path.dirname(args.output_file)
        with open(args.output_file, 'a') as f:
            f.write(f"{tar_path}\n")
    torch.save(save_data, args.alignment_file)
    ed_time = time.time()
    print('ed_time-st_time ', ed_time-st_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--alignment_file", type=str, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--compute-type", type=str, default="float16")
    args = parser.parse_args()
    main(args)
