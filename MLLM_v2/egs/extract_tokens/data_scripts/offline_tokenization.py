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
    if args.tar_file is not None:
        f_info = open(args.tar_info, 'w')
        for i, line in enumerate(open(args.tar_file,'r')):
            tar_path = line.strip().split(' ')[-1] # get the tar path
            tar_dataset = wds.WebDataset(tar_path)
            for sample in tar_dataset: # get the speech info
                flag = True
                json_data = sample['json']
                json_decoded = json.loads(json_data.decode('utf-8')) 
                key = sample['__key__']
                # 解码 FLAC 数据
                flac_data = sample['flac']
                try:
                    audio_data, sample_rate = sf.read(io.BytesIO(flac_data))
                except:
                    logging.error(f"read error: {key}")
                    flag = False
                if flag==False:
                    continue
                wav = torch.from_numpy(audio_data).unsqueeze(0).float() # transfer to (1,len)
                spk = 'libriheavy_' + json_decoded['speaker']
                text = json_decoded['normalized_text']
                value = tokenizer.tokenize(wav, sample_rate)
                if value == None:
                    logging.error(f"an error instance: {key} {value}")
                    continue
                if isinstance(value, torch.Tensor):
                    assert value.dim() == 1
                    value = value.cpu()
                data_dict[key] = value
                f_text.write(key+' '+text+'\n')
                f_utt2spk.write(key+' '+spk+'\n')
                f_info.write(key+'\n')
                if i > 0 and i % 1 == 0:
                    logging.info(f"processed {i} examples")
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