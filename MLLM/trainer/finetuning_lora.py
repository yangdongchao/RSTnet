"""
Lora fintuing for Moshi
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
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from huggingface_hub import hf_hub_download
import torch.distributed as dist
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.fsdp import ShardingStrategy
# Local dependency
from utils.train_utils import (
    seed_everything, 
    setup_logging,
    yaml_no_alias_safe_dump,
    save_checkpoint,
    save_model,
    maybe_resume_checkpoint,
    WarmupLR,
    str2bool,
    find_data_jsons,
)
from utils.dataloader import get_data_iterator_tokenizer_vocabulary
from utils.reporter import Reporter
import json
from utils.arguments import get_args
from model_lora import LMModel,CrossEntropyAndAccuracy
from safetensors.torch import load_model
from pathlib import Path
TEXT_TOKENIZER_NAME = 'tokenizer_spm_32k_3.model'
MOSHI_NAME = 'model.safetensors'
MIMI_NAME = 'tokenizer-e351c8d8-checkpoint125.safetensors'
DEFAULT_REPO = 'kyutai/moshiko-pytorch-bf16'

def _is_safetensors(path: Path | str) -> bool:
    return Path(path).suffix in (".safetensors", ".sft", ".sfts")

def main():
    # (1) use DDP anyway (even for 1 GPU)
    dist.init_process_group(backend="nccl", init_method="env://")
    rank, local_rank, world_size = dist.get_rank(), int(os.environ["LOCAL_RANK"]), dist.get_world_size()
    assert torch.cuda.is_available(), "CUDA is not available"
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.bfloat16 # fix using torch.bfloat16
    # (2) arg parsing and logging
    args = get_args()
    args.local_rank = local_rank
    # (2) arg parsing and logging
    args = get_args()
    if rank == 0:
        os.makedirs(args.exp_dir, exist_ok=True)
        os.makedirs(args.exp_dir + '/logs', exist_ok=True)
    else:
        time.sleep(3)
    _lm_kwargs = { "dim": 4096, "text_card": 32000, "existing_text_padding_id": 3, "n_q": 16, "dep_q": 8, "card": args.card,
                   "num_heads": 32, "num_layers": 32, "hidden_scale": 4.125, "causal": True, "layer_scale": None, "context": 3000,
                   "max_period": 10000, "gating": "silu", "norm": "rms_norm_f32", "positional_embedding": "rope", "depformer_dim": 1024,
                   "depformer_dim_feedforward": int(4.125 * 1024), "depformer_num_heads": 16, "depformer_num_layers": 6, "depformer_causal": True,
                   "depformer_layer_scale": None, "depformer_multi_linear": True, "depformer_context": 8, "depformer_max_period": 10000,
                   "depformer_gating": "silu", "depformer_pos_emb": "none", "depformer_weights_per_step": True,
                   "delays": [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]}
    log_file = args.exp_dir + '/logs/RANK.log'
    setup_logging(rank, world_size, log_file)
    reporter = Reporter()

    # (3) randomness & cudnn settings 
    # if args.seed is not None or args.cudnn_deterministic:
    #     seed_everything(args.seed, args.cudnn_deterministic)
    torch.manual_seed(1337 + args.seed)

    # build tokenizer from LLM
    moshi_weight = hf_hub_download(DEFAULT_REPO, MOSHI_NAME, cache_dir='exp_data/Moshi')
    # build tokenizer from LLM
    model = LMModel(device=device, dtype=dtype, **_lm_kwargs).to(device=device, dtype=dtype)
    if _is_safetensors(moshi_weight):
        load_model(model, moshi_weight, strict=False)
    else:
        pkg = torch.load(moshi_weight, "cpu",)
        model.load_state_dict(pkg["fsdp_best_state"]["model"])
    # add a lora adaptor
    # see https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/config.py for details
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "lora" in name:
            param.data = param.data.to(dtype)  # Cast to bfloat16
            param.requires_grad = True
    model.print_trainable_parameters()
    # assert 1==2
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
    ) 
    # (5) save config
    if rank == 0:
        with open(args.exp_dir + "/config.yaml", "w", encoding="utf-8") as f:
            logging.warning(
                f'Saving the configuration in {args.exp_dir}/config.yaml'
            )
            yaml_no_alias_safe_dump(vars(args), f, indent=4, sort_keys=False)
    # (6) model, wrapped in FSDP
    
    logging.warning(
        "num. model params: {:,} (num. trained: {:,} ({:.1f}%))".format(
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad),
            sum(p.numel() for p in model.parameters() if p.requires_grad)
            * 100.0
            / sum(p.numel() for p in model.parameters()),
        )
    )
    model = FSDP(model, use_orig_params=True)
    # (7) objects related to optimization: optimizer and scheduler
    optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.global_learning_rate,
            betas=(0.9, 0.95),
            weight_decay=1e-3,
            eps=1e-8,
    )
    scheduler = WarmupLR(optimizer, args.warmup_steps)

    # (8) Resume model, optimizer, scaler, etc, if needed. 
    maybe_resume_checkpoint(args, model, optimizer, scheduler, reporter, train_iter)

    # statistics
    logging.info(f'model arch: {model}')
    # TODO: more model statistics, like param budget? 

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
        print(reporter.log_message())
        # (10.4) save checkpoint
        checkpoint_path = args.exp_dir + f"/ep{ep}.checkpoint"
        logging.info(f"Saving checkpoint file {checkpoint_path}")
        print(f"Saving checkpoint file {checkpoint_path}")
        save_model(checkpoint_path, model)

def train_one_epoch(args, model, train_dl, optimizer, scheduler, reporter, parent_reporter):
    model = model.train()
    optimizer.zero_grad()
    for b_idx, batch in enumerate(reporter.measure_iter_time(train_dl, "iter_time"), 1):
        seqs, masks, lengths, example_ids = batch
        data_stats = {
            "batch_size": len(seqs),
            "seq_len": seqs.size(2),
        }
        reporter.register(data_stats)
        with reporter.measure_time("forward_time"):
            label_audio = seqs[:,1:9,:] #.reshape(seqs.shape[0], -1) # B*n, T : B,n,T
            label_text = seqs[:,0,:]  # B, T
            audio_logits, text_logits = model(seqs, masks) # text_logits: B, T, D
            loss_audio, metrics_audio = CrossEntropyAndAccuracy(audio_logits, label_audio, masks[:,1:9,:], loss_weights=[100,1,1,1,1,1,1,1], ignore_ids=[2048,2048,2048,2048,2048,2048,2048,2048])
            loss_text, metrics_text = CrossEntropyAndAccuracy(text_logits.unsqueeze(2), label_text.unsqueeze(1), masks[:,0:1,:], loss_weights=[1], ignore_ids=[32000])
            loss = loss_audio + loss_text
            metrics = {}
            metrics['loss_audio'] = loss_audio.clone().detach()
            metrics['loss_text'] = loss_text.clone().detach()
            metrics['acc_audio'] = metrics_audio['acc_all']
            metrics['acc_text'] = metrics_text['acc_all']
            for v in metrics.values(): # Cross-GPU statistics
                dist.all_reduce(v, dist.ReduceOp.AVG)
            reporter.register(metrics)

        with reporter.measure_time("backward_time"):
            loss.backward()
        
        with reporter.measure_time("optim_time"):
            if b_idx % args.grad_accum == 0:
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
            print(reporter.log_message(-args.print_freq))

        if args.save_interval > 0 and b_idx % args.save_interval == 0:
            checkpoint_path = args.exp_dir + f"/ep{reporter.get_epoch()}-iter{b_idx}.checkpoint"
            logging.info(f"Saving checkpoint file within an epoch: {checkpoint_path}")
            print(f"Saving checkpoint file within an epoch: {checkpoint_path}")
            save_model(checkpoint_path, model)
            #save_checkpoint(checkpoint_path, model, optimizer, scheduler, parent_reporter)


def validate_model(args, model, valid_dl, reporter):
    model = model.eval()

    for b_idx, batch in enumerate(reporter.measure_iter_time(valid_dl, "iter_time"), 1):
        seqs, masks, lengths, example_ids = batch
        data_stats = {
            "batch_size": len(seqs),
            "seq_len": seqs.size(2),
        }
        reporter.register(data_stats)
        with reporter.measure_time("forward_time"):
            label_audio = seqs[:,1:9,:] #.reshape(seqs.shape[0], -1) # B*n, T : B,n,T
            label_text = seqs[:,0,:]  # B, T
            audio_logits, text_logits = model(seqs, masks) # text_logits: B, T, D
            loss_audio, metrics_audio = CrossEntropyAndAccuracy(audio_logits, label_audio, masks[:,1:9,:], loss_weights=[100,1,1,1,1,1,1,1], ignore_ids=[2048,2048,2048,2048,2048,2048,2048,2048])
            loss_text, metrics_text = CrossEntropyAndAccuracy(text_logits.unsqueeze(2), label_text.unsqueeze(1), masks[:,0:1,:], loss_weights=[1], ignore_ids=[32000])
            loss = loss_audio + loss_text
            metrics = {}
            metrics['loss_audio'] = loss_audio.clone().detach()
            metrics['loss_text'] = loss_text.clone().detach()
            metrics['acc_audio'] = metrics_audio['acc_all']
            metrics['acc_text'] = metrics_text['acc_all']
            reporter.register(metrics)
        reporter.next()


if __name__ == '__main__':
    main()    
