import json
import os
import sys
import torch
import torch.nn.functional as F
import argparse
import logging
from collections import defaultdict
import torchaudio
import time
from tools.tokenizer.MimiCodec.mimi_tokenizer import MimiTokenizer
from tools.tokenizer.Text2ID.text_tokenizer import TextTokenizer
import webdataset as wds
from io import BytesIO
import io
import soundfile as sf
import numpy as np 
from scipy.signal import resample


def get_parser():
    parser = argparse.ArgumentParser(
        description="convert a data list, do tokenization and save as a torch .pt file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tar-file", type=str, default=None, help="wav.scp in the format <exampe_id> <wav_relative_path>")
    parser.add_argument("--alignment-file", type=str, default=None, help="the alignment file save the word duration information")
    parser.add_argument("--tar-info", type=str, default=None, help="save the process information")
    parser.add_argument("--output-text-file", type=str, help="dict")
    parser.add_argument("--output-audio-file", type=str, help="dict")
    parser.add_argument("--rank", type=int, help="local GPU rank, if applicable")
    parser.add_argument("--llm-ckpt-dir", type=str, required=True, help="Path to the text tokenizer directory")
    parser.add_argument("--log-per", type=str, default=100, help="Log per n examples")
    return parser


def align_audio_text(data_dict, args):
    s_cnt = 0
    start_time = time.time()
    for utt, item in data_dict.items():
        max_len = max([item[key].shape[-1] for key in item.keys()])
        for key in item.keys():
            if item[key].shape[-1] < max_len:
                item[key] = F.pad(item[key], (0, max_len - item[key].shape[-1]))
        s_cnt += 1
        if s_cnt > 0 and s_cnt % args.log_per == 0:
            end_time = time.time()
            logging.info(f"Rank {args.rank} packed {s_cnt} examples @ {args.log_per / (end_time - start_time):.2f}files/s")
            start_time = time.time()
    return data_dict


def main(args):
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG,
        format=f"%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    )
    args = get_parser().parse_args(args)
    args.rank -= 1 # run.pl starts from 1 but the exact jobid / gpuid starts from 0   
    max_gpu = torch.cuda.device_count()
    args.rank = (args.rank % max_gpu)

    data_dict = defaultdict(dict)
    # tokenize audio: [1+8, T]
    device = torch.device(f"cuda:{args.rank}")
    audio_tokenizer = MimiTokenizer(device=device)
    logging.info('Audio tokenizer built')
        #data_dict = tokenize_audio(data_dict, audio_tokenizer, args
    text_tokenizer = TextTokenizer(args.llm_ckpt_dir)
    logging.info('Text tokenizer built')
        #data_dict = tokenize_text(data_dict, text_tokenizer, args)
    f_info = open(args.tar_info, 'w')
    align_word_info = torch.load(args.alignment_file, 'cpu') # 
    for i, line in enumerate(open(args.tar_file,'r')):
        tar_path = line.strip().split(' ')[-1] # get the tar path
        # try:
        tar_dataset = wds.WebDataset(tar_path)
        print('tar_dataset ', tar_dataset)
        for sample in tar_dataset: # get the speech info
            flag = True
            try:
                json_data = sample['json']
                json_decoded = json.loads(json_data.decode('utf-8')) 
                key = sample['__key__']
                if 'flac' in sample.keys():
                    flac_data = sample['flac']
                else:
                    flac_data = sample['wav']
                audio_data, sample_rate = sf.read(io.BytesIO(flac_data))
            except:
                logging.error(f"read error: {key}")
                flag = False
            if flag==False:
                continue
            wav = torch.from_numpy(audio_data).unsqueeze(0).float() # transfer to (1,len)
            # if wav.shape[1] / sample_rate > 80:  # 是否控制最大长度
            #     continue
            text = json_decoded['text']
            # audio tokenize
            value = audio_tokenizer.tokenize(wav, sample_rate)
            if value == None:
                logging.error(f"an error instance: {key} {value}")
                continue
            if isinstance(value, torch.Tensor):
                value = value.cpu()
            data_dict[key]['audio'] = value
            metadata = {}
            metadata['text'] = text
            metadata['duration'] = json_decoded['duration']
            metadata['words'] = json_decoded['force_aligned_text']['transcript']
            word_list = text_tokenizer.tokenize_segment([metadata]) # 
            text_tokens = text_tokenizer.pad_tokens(word_list, metadata["duration"])


            # if the duration information is from alignment.
            # metadata = align_word_info[key]

            data_dict[key]["text"] = text_tokens.unsqueeze(0)
            f_info.write(key+'\n')
            if i > 0 and i % 1 == 0:
                logging.info(f"processed {i} examples")
    data_dict = align_audio_text(data_dict, args)
    # NOTE: We do not add delay pattern here for flexibility
    result_text = {}
    result_audio = {}
    for utt, value in data_dict.items():
        result_text[utt] = value['text']
        result_audio[utt] = value['audio']
    os.makedirs(os.path.dirname(args.output_text_file), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_audio_file), exist_ok=True)
    torch.save(result_text, args.output_text_file)
    torch.save(result_audio, args.output_audio_file)

if __name__ == "__main__":
    main(sys.argv[1:])