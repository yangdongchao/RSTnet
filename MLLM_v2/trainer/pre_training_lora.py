""" Dongchao Yang
the pre-training code for MLLM (LORA for global transformer. Full-parameter training for local transformer)
"""
# External dependency
import os
import time
import math
import pickle
import numpy as np
import torch
import argparse
import logging
import json
import functools
from pathlib import Path
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import torch.distributed as dist
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader
from utils.dataloader import get_data_iterator_tokenizer_vocabulary
import torch._dynamo
from models.llama_streaming import GPT 
from models.llama_streaming import Config
from models.llama_streaming import mark_only_lora_as_trainable
from lightning.fabric.utilities.load import _lazy_load as lazy_load
from utils.train_utils import to_device
# Local dependency
from utils.train_utils import (
    seed_everything, 
    setup_logging,
    yaml_no_alias_safe_dump,
    save_checkpoint,
    maybe_resume_checkpoint,
    WarmupLR,
    str2bool,
    find_data_jsons,
    save_model
)
from utils.reporter import Reporter
from utils.arguments import get_args
import json
from models.model import LMModel,CrossEntropyAndAccuracy
from safetensors.torch import load_model

def _is_safetensors(path: Path | str) -> bool:
    return Path(path).suffix in (".safetensors", ".sft", ".sfts")

def cuda_prefix_print(msg: str):
    prefix = f"[CUDA_{torch.cuda.current_device()}]"
    print(f"{prefix}\t{msg}")

def print_cuda_mem_info(msg: str):
    cuda_prefix_print(msg)
    cuda_prefix_print(f"[Allocated]\t{torch.cuda.max_memory_allocated()/1024:.1f}\tMiB")
    cuda_prefix_print(f"[Cached]\t{torch.cuda.max_memory_cached()/1024:.1f}\tMiB")
    cuda_prefix_print(f"[Reserved]\t{torch.cuda.max_memory_reserved()/1024:.1f}\tMiB")

def main():
    # (1) use DDP anyway (even for 1 GPU)
    dist.init_process_group(backend="nccl", init_method="env://")
    rank, local_rank, world_size = dist.get_rank(), int(os.environ["LOCAL_RANK"]), dist.get_world_size()
    assert torch.cuda.is_available(), "CUDA is not available"
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    training_dtype = torch.bfloat16 # fix using torch.bfloat16
    # (2) arg parsing and logging
    args = get_args()
    args.local_rank = local_rank
    print('args ', args)
    if rank == 0:
        os.makedirs(args.exp_dir, exist_ok=True)
        os.makedirs(args.exp_dir + '/logs', exist_ok=True)
    else:
        time.sleep(3)
    config = Config.from_file(args.model_config, lora_r=args.lora_r,
                    lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, lora_query=args.lora_query, lora_key=args.lora_key,
                    lora_value=args.lora_value, lora_projection=args.lora_projection, lora_mlp=args.lora_mlp,
                    lora_head=args.lora_head, audio_card=args.audio_card, codecformer_dim=args.codecformer_dim,
                    n_q=args.n_q, dep_q=args.dep_q, codecformer_heads=args.codecformer_heads, codecformer_layers=args.codecformer_layers,
                    codecformer_hidden_scale=args.codecformer_hidden_scale, causal=args.causal,
                    codecformer_multi_linear=args.codecformer_multi_linear, codecformer_weights_per_step=args.codecformer_weights_per_step,
                    codecformer_dim_feedforward=args.codecformer_dim_feedforward) # load the config parameters
    log_file = args.exp_dir + '/logs/RANK.log'
    setup_logging(rank, world_size, log_file)
    reporter = Reporter()
    # (3) randomness & cudnn settings 
    if args.seed is not None or args.cudnn_deterministic:
        seed_everything(args.seed, args.cudnn_deterministic)
    torch.manual_seed(1337 + args.seed)
    # build LLM and init the global transformer
    model = GPT(config).to(device=device, dtype=training_dtype) # 
    mark_only_lora_as_trainable(model.transformer)
    mark_only_lora_as_trainable(model.lm_head) # we need to mark the global transformer part.
    state_dict = lazy_load(args.checkpoint_path)
    state_dict = state_dict.get("model", state_dict)
    model.load_state_dict(state_dict, strict=False)
    model.print_trainable_parameters()
    # (4) data related objects: data iterator, tokenizers, vocabulary
    (train_iter, valid_iter) = \
        get_data_iterator_tokenizer_vocabulary(
            args=args,
            train_jsons=find_data_jsons(args.train_data_jsons),
            valid_jsons=find_data_jsons(args.valid_data_jsons),
            delay_step = 1, # default setting
            batch_scale=args.batch_scale,
            minibatch_debug=args.minibatch_debug,
            max_length=args.max_length,
            min_length=args.min_length,
            n_worker=args.n_worker,
            seed=args.seed,
            parallel_number= args.parallel_number,
            text_empty_token = 128002,
            semantic_empty_token=2048,
            acoustic_empty_token=2048,
            acoustic_pad_token = 2049,
            semantic_pad_token = 2049,
            text_pad_token=128003,
    ) 
    # (5) save config
    if rank == 0:
        with open(args.exp_dir + "/config.yaml", "w", encoding="utf-8") as f:
            logging.warning(f'Saving the configuration in {args.exp_dir}/config.yaml')
            yaml_no_alias_safe_dump(vars(args), f, indent=4, sort_keys=False)
    logging.warning(
        "num. model params: {:,} (num. trained: {:,} ({:.1f}%))".format(
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad),
            sum(p.numel() for p in model.parameters() if p.requires_grad)
            * 100.0
            / sum(p.numel() for p in model.parameters()),
        )
    )
    args.training_dtype = training_dtype # add training_dtype into args
    # (6) model, wrapped in FSDP
    model = FSDP(model, use_orig_params=True) # using bf16 training
    # (7) objects related to optimization: optimizer and scheduler model.out_norm.parameters()
    optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.global_learning_rate,
            betas=(0.9, 0.95),
            weight_decay=1e-3,
            eps=1e-8,
    )
    scheduler = WarmupLR(optimizer, args.warmup_steps)
    # (8) Resume model, optimizer, scaler, etc, if needed.
    # if args.resume is not 
    maybe_resume_checkpoint(args, model, optimizer, scheduler, reporter, train_iter)
    print(f'model arch: {model}')
    # statistics
    logging.info(f'model arch: {model}')
    # (9) training and evaluation
    start_epoch = reporter.get_epoch() + 1
    if start_epoch > args.n_epoch:
        logging.error(f'already reach the maximum training epochs. Done!')
    logging.info("training start ... ")
    print("training start ... ")
    for ep in range(start_epoch, args.n_epoch + 1):
        reporter.set_epoch(ep)
        # (10.1) train
        with reporter.observe("train") as sub_reporter:
            train_one_epoch(
              args=args,
              model=model,
              train_dl=train_iter,
              optimizer=optimizer,
              scheduler=scheduler,
              reporter=sub_reporter,
              parent_reporter=reporter,
            )
        train_iter.sampler.refresh() # refresh the dataloader
        # (10.2) evaluation
        with torch.no_grad():
            with reporter.observe("valid") as sub_reporter:
                validate_model(
                  args=args,
                  model=model,
                  valid_dl=train_iter,
                  reporter=sub_reporter,
                )
        # (10.3) epoch logging. 
        logging.info(reporter.log_message())
        # (10.4) save checkpoint
        checkpoint_path = args.exp_dir + f"/ep{ep}.checkpoint"
        logging.info(f"Saving checkpoint file {checkpoint_path}")
        #save_model(checkpoint_path, model)
        if ep > 100:
            save_checkpoint(checkpoint_path, model, optimizer, scheduler, reporter)

def train_one_epoch(args, model, train_dl, optimizer, scheduler, reporter, parent_reporter):
    model = model.train()
    optimizer.zero_grad()
    for b_idx, batch in enumerate(reporter.measure_iter_time(train_dl, "iter_time"), 1):
        # batch = to_device(batch, "cuda")
        seqs, masks, lengths, example_ids = batch
        data_stats = {
            "batch_size": len(seqs),
            "seq_len": seqs.size(2),
        }
        #print('seqs ', seqs.shape, seqs.dtype, seqs.device)
        # # print(next(model.parameters()).device)
        # assert 1==2
        reporter.register(data_stats)
        with reporter.measure_time("forward_time"):
            label_audio = seqs[:,1:9,:] #.reshape(seqs.shape[0], -1) # B*n, T : B,n,T
            label_text = seqs[:,0,:]  # B, T
            audio_logits, text_logits = model(seqs) # text_logits: B, T, D
            loss_audio, metrics_audio = CrossEntropyAndAccuracy(audio_logits, label_audio, masks[:,1:9,:], loss_weights=[2,1,1,1,1,1,1,1], ignore_ids=[2049,2049,2049,2049,2049,2049,2049,2049])
            loss_text, metrics_text = CrossEntropyAndAccuracy(text_logits.unsqueeze(2), label_text.unsqueeze(1), masks[:,0:1,:], loss_weights=[1], ignore_ids=[128003])
            loss = loss_audio + loss_text
            metrics = {}
            metrics['loss_audio'] = loss_audio.clone().detach()
            metrics['loss_text'] = loss_text.clone().detach()
            metrics['acc_audio'] = metrics_audio['acc_all']
            metrics['acc_text'] = metrics_text['acc_all']
            metrics['acc_audio_tgt'] = metrics_audio['acc_target']
            metrics['acc_text_tgt'] = metrics_text['acc_target']
            for v in metrics.values(): # Cross-GPU statistics
                dist.all_reduce(v, dist.ReduceOp.AVG)
            reporter.register(metrics)

        with reporter.measure_time("backward_time"):
            loss.backward()
        
        with reporter.measure_time("optim_time"):
            if b_idx % args.grad_accum == 0:
                # grad_norm = model.clip_grad_norm_(args.grad_clip)
                # if math.isnan(grad_norm):
                #     logging.warning(f"grad norm is NaN. Discard this gradient")
                #     optimizer.zero_grad()
                optimizer.step() # update the model even with ill gradient - sync the training
                optimizer.zero_grad()
                scheduler.step()
                reporter.register(
                {f'lr_param_{i}': pg['lr'] for i, pg in enumerate(optimizer.param_groups)}
                )

        # must call this here so that the saved checkpoint is valid for reporter
        reporter.next()

        if b_idx % args.print_freq == 0:
            logging.info(reporter.log_message(-args.print_freq))

        if args.save_interval > 0 and b_idx % args.save_interval == 0:
            checkpoint_path = args.exp_dir + f"/ep{reporter.get_epoch()}-iter{b_idx}.checkpoint"
            logging.info(f"Saving checkpoint file within an epoch: {checkpoint_path}")
            #save_model(checkpoint_path, model)
            save_checkpoint(checkpoint_path, model, optimizer, scheduler, parent_reporter)


def validate_model(args, model, valid_dl, reporter):
    model = model.eval()
    for b_idx, batch in enumerate(reporter.measure_iter_time(valid_dl, "iter_time"), 1):
        # batch = to_device(batch, "cuda")
        seqs, masks, lengths, example_ids = batch
        data_stats = {
            "batch_size": len(seqs),
            "seq_len": seqs.size(2),
        }
        reporter.register(data_stats)
        with reporter.measure_time("forward_time"):
            label_audio = seqs[:,1:9,:] #.reshape(seqs.shape[0], -1) # B*n, T : B,n,T
            label_text = seqs[:,0,:]  # B, T
            audio_logits, text_logits = model(seqs) # text_logits: B, T, D
            loss_audio, metrics_audio = CrossEntropyAndAccuracy(audio_logits, label_audio, masks[:,1:9,:], loss_weights=[2,1,1,1,1,1,1,1], ignore_ids=[2048,2048,2048,2048,2048,2048,2048,2048])
            loss_text, metrics_text = CrossEntropyAndAccuracy(text_logits.unsqueeze(2), label_text.unsqueeze(1), masks[:,0:1,:], loss_weights=[1], ignore_ids=[32000])
            loss = loss_audio + loss_text
            metrics = {}
            metrics['loss_audio'] = loss_audio.clone().detach()
            metrics['loss_text'] = loss_text.clone().detach()
            metrics['acc_audio'] = metrics_audio['acc_all']
            metrics['acc_text'] = metrics_text['acc_all']
            metrics['acc_audio_tgt'] = metrics_audio['acc_target']
            metrics['acc_text_tgt'] = metrics_text['acc_target']
            reporter.register(metrics)
        reporter.next()

if __name__ == '__main__':
    main()    

