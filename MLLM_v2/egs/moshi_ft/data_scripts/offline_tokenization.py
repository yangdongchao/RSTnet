import json
import os
import sys
import torch
import torch.nn.functional as F
import argparse
import logging
from collections import defaultdict
import torchaudio
from tools.tokenizer.MimiCodec.mimi_tokenizer import MimiTokenizer
from tools.tokenizer.Text2ID.moshi_text_tokenizer import Text2IDTokenizer


def get_parser():
    parser = argparse.ArgumentParser(
        description="convert a data list, do tokenization and save as a torch .pt file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-audio-file", type=str, default=None, help="wav.scp in the format <exampe_id> <wav_relative_path>")
    parser.add_argument("--input-text-file", type=str, default=None, help="utt2json in the format <exampe_id> <json_relative_path>")
    parser.add_argument("--output-file", type=str, help="dict")
    parser.add_argument("--root-dir", type=str, default=None, help="root dir for relative paths")
    parser.add_argument("--rank", type=int, help="local GPU rank, if applicable")
    return parser

def tokenize_audio(data_dict, audio_tokenizer, args):
    # prepare audio paths
    for line in open(args.input_audio_file):
        id, audio_path = line.strip().split()
        channel = id.split('_')[-1]
        utt = id.replace(f'_{channel}', '')
        if channel == "ch0":
            data_dict[utt]["ch0_clean"] = str(os.path.join(args.root_dir, "sr_audio", audio_path))
            data_dict[utt]["ch0_noisy"] = str(os.path.join(args.root_dir, "se_audio", audio_path))
        else:
            data_dict[utt]["ch1_clean"] = str(os.path.join(args.root_dir, "sr_audio", audio_path))
            data_dict[utt]["ch1_noisy"] = str(os.path.join(args.root_dir, "se_audio", audio_path))
    
    # tokenize audio
    s_cnt = 0
    for utt, item in data_dict.items():
        key_order = ["ch0_clean", "ch0_noisy", "ch1_clean", "ch1_noisy"]
        wav_batch = []
        for key in key_order:
            wav, orig_sr = torchaudio.load(item[key])
            if len(wav_batch) > 0:
                assert wav.shape == wav_batch[0].shape, \
                    f"wav shape mismatch: {wav.shape} vs {wav_batch[0].shape} @ {utt}_{key}"
            wav_batch.append(wav)
        wav_batch = torch.cat(wav_batch, dim=0)
        wav_batch = torchaudio.transforms.Resample(orig_sr, 24000)(wav_batch)
        wav_batch = wav_batch.unsqueeze(1)  # [B, 1, T]
        wav_batch = wav_batch.to(audio_tokenizer.device)

        codes = audio_tokenizer.tokenize(wav_batch) # [B, 8, T]

        codes = codes.cpu()
        for i, key in enumerate(key_order):
            data_dict[utt][key] = codes[i]  # [8, T]
        
        s_cnt += 1
        if s_cnt > 0 and s_cnt % 1000 == 0:
            logging.info(f"processed {s_cnt} examples")
    
    return data_dict

def tokenize_text(data_dict, text_tokenizer, args):
    # prepare metadata paths
    for line in open(args.input_text_file):
        id, text_path = line.strip().split()
        channel = id.split('_')[-1]
        utt = id.replace(f'_{channel}', '')
        if channel == "ch0":
            data_dict[utt]["ch0_text"] = str(os.path.join(args.root_dir, "metadata", text_path))
        else:
            data_dict[utt]["ch1_text"] = str(os.path.join(args.root_dir, "metadata", text_path))
    
    # tokenize text
    s_cnt = 0
    for utt, item in data_dict.items():
        key_order = ["ch0_text", "ch1_text"]
        for key in key_order:
            with open(item[key]) as f:
                metadata = json.load(f)
            # TODO: Remove this after the data is fixed
            if "duration" not in metadata:
                metadata["duration"] = metadata["segments"][-1]["end"]
            word_list = text_tokenizer.tokenize(metadata["segments"])
            text_tokens = text_tokenizer.pad_tokens(word_list, metadata["duration"])
            data_dict[utt][key] = text_tokens.unsqueeze(0)
        
        s_cnt += 1
        if s_cnt > 0 and s_cnt % 1000 == 0:
            logging.info(f"processed {s_cnt} examples")
    
    return data_dict

def align_audio_text(data_dict, args):
    for utt, item in data_dict.items():
        max_len = max([item[key].shape[-1] for key in item.keys()])
        for key in item.keys():
            if item[key].shape[-1] < max_len:
                item[key] = F.pad(item[key], (0, max_len - item[key].shape[-1]))
        print(f"finished processing {utt}")
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
    if args.input_audio_file is not None:
        device = torch.device(f"cuda:{args.rank}")
        audio_tokenizer = MimiTokenizer(device=device)
        logging.info('Audio tokenizer built')
        data_dict = tokenize_audio(data_dict, audio_tokenizer, args)
    
    # tokenize text: [1, T]
    if args.input_text_file is not None:
        text_tokenizer = Text2IDTokenizer()
        logging.info('Text tokenizer built')
        data_dict = tokenize_text(data_dict, text_tokenizer, args)
        
    # align audio and text
    if args.input_audio_file is not None and \
        args.input_text_file is not None:
        data_dict = align_audio_text(data_dict, args)
    
    # pack and save
    # NOTE: We do not add delay pattern here for flexibility
    result = {}
    for utt, value in data_dict.items():
        # ch0 as Moshi
        new_key = utt + "_ch0"
        result[new_key] = torch.cat([
            value["ch0_text"], 
            value["ch0_clean"], 
            value["ch1_noisy"],
        ], dim=0)   # [1+1+7+1+7, T]
        # ch1 as Moshi
        new_key = utt + "_ch1"
        result[new_key] = torch.cat([
            value["ch1_text"], 
            value["ch1_clean"], 
            value["ch0_noisy"],
        ], dim=0)   # [1+1+7+1+7, T]
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    torch.save(result, args.output_file)

if __name__ == "__main__":
    main(sys.argv[1:])