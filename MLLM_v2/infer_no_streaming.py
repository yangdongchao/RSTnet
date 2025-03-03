''' The inference code for MLLM. 
    In this inference code, we do not consider the streaming, but our model is designed for streaming.
    We will add the streaming inference code in the future.
    Copyright (c) Dongchao, all rights reserved.
    We use some code from Moshi
'''
from dataclasses import dataclass
from functools import partial
import logging
import typing as tp
import torch
from torch import nn
from utils.sampling import sample_token,sample_token_audio, sample_token_audio_2048
from utils.compile import CUDAGraphed
from modules.streaming import StreamingContainer, StreamingModule
from models.llama_streaming import GPT, CausalSelfAttention
from modules.transformer import StreamingMultiheadAttention
import argparse
from huggingface_hub import hf_hub_download
import torch
from tools.tokenizer.MimiCodec.mimi_tokenizer import MimiTokenizer
from utils.train_utils import resume_for_inference
from moshi.models import loaders
import sys
import yaml
import torchaudio
from pathlib import Path
from models.llama_streaming import Config
from torchaudio.transforms import Resample
from tools.tokenizer.Text2ID.text_tokenizer import TextTokenizer
from models.model import CrossEntropyAndAccuracy
from utils.train_utils import to_device
from utils.dataloader import get_data_iterator_tokenizer_vocabulary

logger = logging.getLogger(__name__)

def get_parser():
    parser = argparse.ArgumentParser()
    # model related: use the resume model if provided; otherwise use the latest in exp_dir
    parser.add_argument('--resume', type=str, default=None, help='model to resume. If None, use the latest checkpoint in exp_dir')
    parser.add_argument('--exp_dir', type=str, default=None, help='experiment directory')
    # inference related: 
    parser.add_argument('--inference_mode', type=str, default='sampling', 
                         choices=['sampling', 'greedy', 'teacher-force'])
    parser.add_argument('--temp', type=float, default=0.8, help='softmax temperature in sampling for audio streaming')
    parser.add_argument('--temp_text', type=float, default=0.7, help='the temperature for text streaming')
    parser.add_argument('--topk', type=int, default=30, help='can only select top-k candidate in sampling')
    parser.add_argument('--n_samples', type=int, default=1, help="number of samples during inference")
    parser.add_argument('--maxlen_ratio', type=float, default=-1, help='max length ratio w.r.t. prefix')
    parser.add_argument('--minlen_ratio', type=float, default=-1, help='min length ratio w.r.t. prefix')
    parser.add_argument('--seed', type=int, default=888, help='random seed')
    # device related
    parser.add_argument('--rank', type=int, default=-1, help='GPU rank. -1 means CPU')
    parser.add_argument('--task_name', type=str, help='the name of task')
    # data related
    parser.add_argument('--data_json', type=str, default=None, help="data jsons for inference")
    parser.add_argument('--output_dir', type=str, help="tag for decoding")
    parser.add_argument('--generate_target', type=str, default="audio", help="the format of the generated target")
    return parser 


def main():
    # (1) arg parsing & train_arg parsing & logging & seed
    parser = get_parser()
    args = parser.parse_args()
    train_config = args.exp_dir + '/config.yaml'
    with open(train_config, "r", encoding="utf-8") as f:
        train_args = yaml.safe_load(f)
        train_args = argparse.Namespace(**train_args)
    if args.rank >= 0:
        args.rank = args.rank % torch.cuda.device_count()
        device = torch.device(f'cuda:{args.rank}') # run.pl index from 1
    else:
        device = torch.device('cpu')
    logging.info(f'inference using {device}')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True) 
    training_dtype = torch.bfloat16 # fix using torch.bfloat16
    train_it, valid_iter = get_data_iterator_tokenizer_vocabulary(
            args=train_args,
            train_jsons=[],
            valid_jsons=[args.data_json],
            delay_step = 1, # default setting
            batch_scale=train_args.batch_scale,
            minibatch_debug=train_args.minibatch_debug,
            max_length=train_args.max_length,
            min_length=train_args.min_length,
            parallel_number= train_args.parallel_number,
            text_empty_token = 128002,
            semantic_empty_token=2048,
            acoustic_empty_token=2048,
            acoustic_pad_token = 2049,
            semantic_pad_token = 2049,
            text_pad_token=128003)
    
    config = Config.from_file(train_args.model_config, lora_r=train_args.lora_r,
                    lora_alpha=train_args.lora_alpha, lora_dropout=train_args.lora_dropout, lora_query=train_args.lora_query, lora_key=train_args.lora_key,
                    lora_value=train_args.lora_value, lora_projection=train_args.lora_projection, lora_mlp=train_args.lora_mlp,
                    lora_head=train_args.lora_head, audio_card=train_args.audio_card, codecformer_dim=train_args.codecformer_dim,
                    n_q=train_args.n_q, dep_q=train_args.dep_q, codecformer_heads=train_args.codecformer_heads, codecformer_layers=train_args.codecformer_layers,
                    codecformer_hidden_scale=train_args.codecformer_hidden_scale, causal=train_args.causal,
                    codecformer_multi_linear=train_args.codecformer_multi_linear, codecformer_weights_per_step=train_args.codecformer_weights_per_step,
                    codecformer_dim_feedforward=train_args.codecformer_dim_feedforward) # load the config parameters
    model = GPT(config).to(device=device, dtype=training_dtype) #
    model = model.eval()
    resume_for_inference(args.resume, args.exp_dir, model, device) # init the model
    audio_tokenizer = MimiTokenizer(device=device) # 
    audio_tokenizer = audio_tokenizer.to(device)
    # print('audio_tokenizer ', audio_tokenizer.device)
    # assert 1==2
    text_tokenizer = TextTokenizer(checkpoint_dir='/home-dongchao/data/checkpoints/meta-llama/Llama-3.2-3B') # set the path of llama
    inference_implementation = InferenceImp(model=model, args=args, mode=args.inference_mode, temp_text=args.temp_text, 
                               top_k_text=25, temp=args.temp, top_k=args.topk, task_name=args.task_name)
    writer = open(output_dir / 'results.scp', 'w')
    input_writer = open(output_dir / 'input.scp', 'w')
    total_audio_loss = 0
    total_text_loss = 0
    cnt = 0
    for b_idx, batch in enumerate(valid_iter):
        seqs , masks, lengths, example_ids = to_device(batch, device=device, non_blocking=False)
        for i_idx in range(len(seqs)):
            if args.inference_mode == 'teacher-force':
                # calculate the ppl
                searched_results, metrics_audio = inference_implementation(seqs[i_idx], masks[i_idx])
                total_audio_loss = total_audio_loss + searched_results
                total_text_loss = total_text_loss 
                cnt += 1
                print('metrics_audio ', metrics_audio)
            else:
                searched_results = inference_implementation(seqs[i_idx], masks[i_idx])
                if args.generate_target == 'audio':
                    detokenized = audio_tokenizer.detokenize(searched_results)
                    detokenized = detokenized.detach().cpu() #.squeeze(0)
                    file_name = f"{example_ids[i_idx]}_sample.wav"
                    file_name = str(output_dir / file_name)
                    logging.info(f"save audio into {file_name}")
                    torchaudio.save(file_name, detokenized, 24000, bits_per_sample=16, encoding='PCM_S')
                    # assert 1==2
                else:
                    pre_text = text_tokenizer.decode(text_results)
                    print('pre_text ', pre_text)
        if b_idx > 5:
            break
    if args.inference_mode == 'teacher-force':
        perplexity_audio = torch.exp(total_audio_loss/cnt)
        print('perplexity_audio, perplexity_text ', total_audio_loss/cnt, perplexity_audio)


class InferenceImp(object):
    def __init__(self, args, model, mode, temp_text, top_k_text, temp, top_k, task_name):
        self.model = model
        self.args = args
        self.n_samples = 1 # we first set it as one
        self.task_name = task_name
        # hyper-params
        # special token-ids
        self.text_pad_token = 128003
        self.acoustic_pad_token = 2049
        self.semantic_pad_token = 2049
        self.text_empty_token = 128002
        self.mode = mode
        self.use_sampling = True
        self.temp_text = temp_text
        self.top_k_text = top_k_text
        self.temp = temp
        self.top_k = top_k
    
    @torch.no_grad()
    def __call__(self, seq, mask):
        device = seq.device
        seq  = seq.unsqueeze(0).expand(self.n_samples, -1, -1) # B, 9, Len
        mask = mask.unsqueeze(0).expand(self.n_samples, -1, -1)
        # (0) full forward like training; for debug only
        if self.mode == "teacher-force":
            label_audio = seq[:,1:,:] #.reshape(seqs.shape[0], -1) # B*n, T : B,n,T
            label_text = seq[:,0,:]
            audio_logits, text_logits = self.model(seq)
            loss_audio, metrics_audio = CrossEntropyAndAccuracy(audio_logits, label_audio, mask[:,1:,:], loss_weights=[1,1,1,1,1,1,1,1], ignore_ids=[2049,2049,2049,2049,2049,2049,2049,2049])
            loss_text, metrics_text = CrossEntropyAndAccuracy(text_logits.unsqueeze(2), label_text.unsqueeze(1), mask[:,0:1,:], loss_weights=[1], ignore_ids=[128003])
            print('metrics_audio ', metrics_audio)
            print('metrics_text ', metrics_text)
            return loss_audio/8, loss_text
        # remove padding
        if self.task_name in ['text_only','word_level_audio_text_alignment', 'ASR']:
            pad_len = seq[0,0:1,:].eq(self.text_pad_token).int().sum().item()
            mask = mask[:, :, :seq.shape[2] - pad_len]
            seq = seq[:, :, :seq.shape[2] - pad_len]
        elif self.task_name in ['audio_only', 'TTS']:
            pad_len = seq[0,1:2,:].eq(self.semantic_pad_token).int().sum().item()
            # print('pad_len ', pad_len)
            mask = mask[:, :, :seq.shape[2] - pad_len]
            seq = seq[:, :, :seq.shape[2] - pad_len]
            # print('seq ', seq[0])
            # assert 1==2
        else:
            raise NotImplementedError
        # (1) prefix inference
        if self.task_name == 'text_only': # for text continue, we set half of text sequence as the prompt
            prefix_len = seq.shape[-1] //2 # 
            prefix = seq[:, :, :prefix_len]
            mask = mask[:, :, :prefix_len]
            maxlen = prefix_len
            minlen = prefix_len
        elif self.task_name == 'audio_only': # for audio continue, we set half of text sequence as the prompt
            # print('seq 2 ', seq.shape)
            prefix_len = seq.shape[2] //2 # 
            #print('prefix_len ', prefix_len)
            prefix = seq[:, :, :prefix_len]
            mask = mask[:, :, :prefix_len]
            maxlen = prefix_len
            minlen = prefix_len
        elif self.task_name == 'TTS':
            prefix_len = seq.shape[2] - seq[0,0,:].eq(self.text_empty_token).int().sum().item()
            prefix = seq[:, :, :prefix_len] # add 1 for the delay pattern
            mask = mask[:, :, :prefix_len]
            maxlen = seq.shape[2]-prefix_len 
            minlen = seq.shape[2]-prefix_len 
            gt_audio = seq[:,1:,prefix_len:]
            # s

        elif self.task_name == 'ASR':
            prefix_len = seq[0,0,:].eq(self.text_empty_token).int().sum().item()
            prefix = seq[:, :, :prefix_len+1]
            mask = mask[:, :, :prefix_len+1]
            maxlen = seq.shape[2]-prefix_len + 13
            minlen = seq.shape[2]-prefix_len - 13

        # (2) search loop
        final_results = []
        text_results = []
        pre_gen_len = prefix.shape[2] 
        for g_idx in range(maxlen):
            g_len = prefix.shape[2] # 
            # (2.1) global inference. AR prediction requires no mask
            next_frame = []
            init_token = self.model._get_initial_token()
            init_token = init_token.expand(prefix.shape[0], -1, -1)
            global_prefix = torch.cat([init_token, prefix], dim =-1) # 
            # print('global_prefix ', global_prefix.shape)
            transformer_out, text_logits = self.model.forward_global(global_prefix) #

            # add pading box for local
            local_pad = torch.ones_like(prefix[:,:,0:1])*self.model.initial_token_id # 
            prefix = torch.cat([prefix, local_pad], dim=-1) # 
            # print('text_logits ', text_logits.shape)
            valid_text_logits = text_logits[:,-1:,:]
            text_token = sample_token(
                    valid_text_logits.float(),
                    self.use_sampling,
                    self.temp_text,
                    self.top_k_text,)
            prefix[:,0,-1] = text_token.squeeze() # update the text token

            # (2.2) local inference
            audio_seq = []
            flag = True
            for l_idx in range(8):
                
                text_indices = prefix[:,0,:] # 
                local_start_token = self.model.codecformer_text_emb(text_indices)
                logits = self.model.forward_local(local_start_token=local_start_token, sequence=prefix[:,1:,:], transformer_out=transformer_out)
                # print('logits ', logits.shape)
                # assert 1==2
                valid_logits = logits[:,-1:,l_idx:l_idx+1,:]
                if g_len == pre_gen_len:
                    next_token = sample_token_audio(
                        valid_logits.float(),
                        self.use_sampling,
                        self.temp,
                        self.top_k,
                    )
                elif l_idx > 0 and g_len > minlen:
                    next_token = sample_token_audio(
                        valid_logits.float(),
                        self.use_sampling,
                        self.temp,
                        self.top_k,
                    )
                else:
                    next_token = sample_token_audio_2048(valid_logits.float(),
                        self.use_sampling,
                        self.temp,
                        self.top_k)
                # print('next_token ', next_token.shape)
                # assert 1==2
                if (g_idx > minlen) and (l_idx >2) and next_token[0,0] >= 2048:
                    flag = False
                    break
                audio_seq.append(next_token.squeeze())
                prefix[:,l_idx+1, g_len] = next_token.squeeze()
                
            if flag==False:
                break
            audio_seq = torch.tensor(audio_seq) # transfor to tensor
            #next_frame = torch.tensor(next_frame).to(prefix.device)
            final_results.append(audio_seq)
        
        final_results = torch.stack(final_results, dim=0).to(device) #.detach().cpu()
        if gt_audio is not None:
            gt_audio = reverse_delay(gt_audio.squeeze())
            #gt_audio = gt_audio #.detach().cpu()

        if self.task_name == 'audio_only':
            prompt_audio = torch.cat([prompt_audio, final_results.transpose(0, 1)], dim=-1)
        if self.task_name == 'TTS':
            final_results = reverse_delay(final_results)
            return final_results
        return prompt_audio


def reverse_delay(x):
    ''' reverse one step delay
        x: should be 8*L
    '''
    if x.shape[0] != 8:
        x = x.transpose(0, 1)
    #print('x ', x)
    x_new = torch.ones_like(x)
    x_new[0,:-1] = x[0,:-1]
    x_new[1:, :-1] = x[1:, 1:] # 
    # print('x_new ', x_new)
    # assert 1==2
    return x_new[:,:-1]


if __name__ == '__main__':
    main()    
