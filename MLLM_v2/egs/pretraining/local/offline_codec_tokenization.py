# Author: # UniAudio Teams

import sys
import torch
import argparse
import logging
import tarfile
import mmap
import pickle
import librosa
from io import BytesIO
import io
from tools.tokenizer.MimiCodec.mimi_tokenizer import MimiTokenizer
# from tools.tokenizer.GLM4V.semantic import SSLTokenizer
import json
import soundfile as sf
# import webdataset as wds
import torchaudio

def get_parser():
    parser = argparse.ArgumentParser(
        description="convert a data list, do tokenization and save as a torch .pt file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-file", type=str, default=None, help="text file in the format <exampe_id> <content>")
    parser.add_argument("--tar-file", type=str, default=None, help="we use tar chunk to save audio information")
    parser.add_argument("--tar-info", type=str, default=None, help="the file to save tar information")
    parser.add_argument("--wav-scp", type=str, default=None, help="kaldi wav.scp file")
    parser.add_argument("--segments", type=str, default=None, help="kaldi segment file")
    parser.add_argument("--output-file", type=str, help="dict")
    parser.add_argument("--output-text", type=str, default=None, help="dict")
    parser.add_argument("--output-utt2spk", type=str, default=None, help="dict")
    parser.add_argument("--tokenizer", type=str, choices=['audio', 'g2p', 'stablecodec', 'semantic', 'mimi', 'ssl'], help="what tokenizer to use")
    parser.add_argument("--rank", type=int, help="local GPU rank, if applicable")
    parser.add_argument("--batch-size", type=int, default=1, help="for batch tokenization")
    return parser

def main(args):
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG,
        format=f"%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    )
    args = get_parser().parse_args(args)
    args.rank -= 1 # run.pl starts from 1 but the exact jobid / gpuid starts from 0   
    max_gpu = torch.cuda.device_count()
    logging.info(f"max gpu {max_gpu}")
    args.rank = (args.rank % max_gpu) #

    if args.tokenizer in ['audio', 'stablecodec', 'encodec', 'mimi', 'ssl']:
        device = torch.device(f"cuda:{args.rank}")
    else:
        device = torch.device('cpu')
    logging.info(f"Using device: {device}")

    # GPU tokenizers 
    if args.tokenizer == "stablecodec":
        tokenizer = StableCodecTokenizer(device=device)
    elif args.tokenizer == 'mimi':
        tokenizer = MimiTokenizer(device=device)
    elif args.tokenizer == 'ssl':
        tokenizer = SSLTokenizer(device=device)
    # CPU tokenizers
    elif args.tokenizer == "phone":
        tokenizer = Text2PhoneTokenizer()
    elif args.tokenizer == "alignment":
        tokenizer = PhoneTokenizer(duplicate=True)
    else:
        raise NotImplementedError
    #tokenizer = tokenizer.to(device)
    logging.info('tokenizer built')
    if args.output_text is not None:
        f_text = open(args.output_text, 'w')
        f_utt2spk = open(args.output_utt2spk, 'w')
    data_dict = {}
    import time
    st_time = time.time()
    assert not (args.input_file is not None and args.wav_scp is not None)
    # TODO: support batch inference
    if args.input_file is not None:
        iterator = open(args.input_file)
        s_cnt = 0
        for i, line in enumerate(open(args.input_file)):
            try:
                line = line.strip().split()
                key, value_path = line[0], " ".join(line[1:])
                audio_data, sample_rate = sf.read(value_path)
                wav = torch.from_numpy(audio_data).unsqueeze(0).float()
                # utts = [(wav, sample_rate)]
                # values = tokenizer.tokenize(utts)
                # for i_r, value in enumerate(values):
                #     value = torch.tensor(value).to(torch.int16)
                #     data_dict[key] = value
                value = tokenizer.tokenize(wav, sample_rate)
                data_dict[key] = value
                # print(value.shape)
                s_cnt += 1
                if i > 0 and i % 1000 == 0:
                    logging.info(f"processed {s_cnt} examples")
            except Exception as e:
                logging.error(f"an error instance: {line}, {e}")
    elif args.tar_file is not None:
        # we use tar as chunk
        f_info = open(args.tar_info, 'w')
        for i, line in enumerate(open(args.tar_file,'r')):
            tar_path = line.strip().split(' ')[-1] # get the tar path
            try:
                tar_dataset = wds.WebDataset(tar_path)
                # print('tar_dataset ', tar_dataset)
                sample_id = 0
                utts = []
                names = []
                for sample in tar_dataset: # get the speech info
                    sample_id += 1
                    # print('sample_id ', sample_id)
                    flag = True
                    try:
                        json_data = sample['json']
                        json_decoded = json.loads(json_data.decode('utf-8')) 
                        key = sample['__key__']
                        flac_data = sample['wav']
                        audio_data, sample_rate = sf.read(io.BytesIO(flac_data))
                    except Exception as e:
                        logging.error(f"read error: {key}, {e}")
                        flag = False
                    if flag==False:
                        continue
                    wav = torch.from_numpy(audio_data).unsqueeze(0).float() # transfer to (1,len)
                    if wav.shape[1] / sample_rate > 40:
                        continue
                    utts.append((wav, sample_rate))
                    #print('json_decoded ', json_decoded)
                    spk = 'mls_' + str(json_decoded['speaker_id'])
                    text = json_decoded['transcript']
                    f_text.write(key+' '+text+'\n')
                    f_utt2spk.write(key+' '+spk+'\n')
                    f_info.write(key+'\n')
                    names.append(key)

                    if sample_id % 10 == 0:
                        if sample_id % 1000 == 0:
                            print('sample_id ', sample_id)
                        values = tokenizer.tokenize(utts)
                        for i_r, value in enumerate(values):
                            value = torch.tensor(value).to(torch.int16)
                            tmp_key = names[i_r]
                            data_dict[tmp_key] = value
                        utts = []
                        names = []
                    if i > 0 and i % 1 == 0:
                        logging.info(f"processed {i} examples")
                
                if len(names) > 0:
                    values = tokenizer.tokenize(utts)
                    for i_r, value in enumerate(values):
                        value = torch.tensor(value).to(torch.int16)
                        tmp_key = names[i_r]
                        data_dict[tmp_key] = value
                    utts = []
                    names = []
            except Exception as e:
                logging.error(f"an error instance: {tar_path}, {e}")
            
    else:
        # kaldiio format
        iterator = ReadHelper('scp:'+args.wav_scp, args.segments)
        count = 0
        for key, (sr, value) in iterator:
            value = torch.from_numpy(value.copy())  / 32768 # [channel, samples]
            value = value.unsqueeze(0)
            value = tokenizer.tokenize(value)
            data_dict[key] = value
            if count > 0 and count % 100 == 0:
                logging.info(f"processed {count} examples")
            count += 1
    torch.save(data_dict, args.output_file)
    ed_time = time.time()
    print('ed_time-st_time ', ed_time-st_time)
    logging.info(f"processed {ed_time-st_time} seconds")

if __name__ == "__main__":
    main(sys.argv[1:])
