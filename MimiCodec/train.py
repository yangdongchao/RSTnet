""" The training code for MimiCodec
"""
import argparse
import os
from collections import defaultdict
import itertools
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import torch.multiprocessing as mp
from models.MimiCodec import MimiCodec
from modules.discriminators.frequency_discriminator import MultiFrequencyDiscriminator
from dataloaders.base_dataloader import WaveDataset
from utils.utils import (
    seed_everything, Logger, cal_model_size, load_obj, to_device, is_primary,
    save_checkpoint, scan_checkpoint, plot_spectrogram
)
from semantic_features import WavLMFeature

def build_codec_model(config):
    model = eval(config.generator.name)(**config.generator.config)
    return model

def build_d_models(config):
    model_disc = dict()
    for d_name in config.d_list:
        model_disc[d_name] = eval(config[d_name].name)(config[d_name].config)
    return model_disc

def build_semantic_teacher_model(config):
    if config.semantic_feature_type == 'wavlm':
        model = WavLMFeature(ckpt_path=config.semantic_model_path, device=torch.device(f"cuda:{config.local_rank}"))
        return model
    elif config.semantic_feature_type == 'whisper':
        pass 
    elif config.semantic_feature_type == 'hubert':
        pass 
    elif config.semantic_feature_type == 'w2v_bert':
        pass 
    else:
        raise NotImplementedError

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    # model configurations
    parser.add_argument('--basic_model_config', default='config/encodec_16k_6kbps_v1.yaml',
                        help='YAML files for configurations.')
    parser.add_argument('--exp_model_config', default=None, help='YAML files for configurations.')
    parser.add_argument('--training_file', default=None)
    parser.add_argument('--validation_file', default=None)
    parser.add_argument('--log_dir', default='exp_logs', help="Log dir")
    parser.add_argument('--semantic_feature_type', default='wavlm', help="choose a pre-trained SSL model to extract features [wavlm, whisper, hubert, wav2vec2-bert]")
    parser.add_argument('--hop_length', default=2000)
    args = parser.parse_args()
    return args

def make_log_dir(config, log_root, config_name="config.yaml"):
    os.makedirs(log_root, exist_ok=True)
    OmegaConf.save(config, f"{log_root}/{config_name}")

def main():
    args = get_args()
    basic_model_config = OmegaConf.load(args.basic_model_config)
    if args.exp_model_config is not None:
        exp_model_config = OmegaConf.load(args.exp_model_config)
        model_config = OmegaConf.merge(basic_model_config, exp_model_config)
    else:
        model_config = basic_model_config
    args.ngpus_per_node = torch.cuda.device_count()
    if args.training_file is None:
        args.training_file = model_config.training_file
        args.validation_file = model_config.validation_file
    assert args.training_file is not None
    assert args.validation_file is not None
    args = OmegaConf.create(vars(args))
    config = OmegaConf.merge(model_config, args)
    config.sample_rate = config.generator.config.sample_rate
    config.model_ckpt_dir = os.path.join(args.log_dir, 'model_ckpts')
    if config.seed is not None or config.cudnn_deterministic:
        seed_everything(config.seed + config.local_rank, config.cudnn_deterministic)
    make_log_dir(config, config.log_dir)
    main_worker(config)


def main_worker(args):
    if args.ngpus_per_node > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
    rank, local_rank, world_size = dist.get_rank(), int(os.environ["LOCAL_RANK"]), dist.get_world_size()
    args.local_rank = local_rank
    args.distributed = args.ngpus_per_node > 1
    torch.cuda.set_device(local_rank)
    ## Build a logger
    logger = Logger(args) # SummaryWriter is contained in logger
    ## build model
    codec_model = build_codec_model(args)
    disc_models = build_d_models(args)
    semantic_teacher = build_semantic_teacher_model(args)
    logger.log_info("="*10 + f" Codec Model " + "="*10)
    # logger.log_info(codec_model)
    logger.log_info(f"Discriminators: {args.d_list}")
    logger.log_info("Building models successfully.")
    size_info = cal_model_size(codec_model, 'codec-model')
    logger.log_info(size_info)

    for k, v in disc_models.items():
        size_info = cal_model_size(v, k)
        logger.log_info(size_info)
    
    args.hop_length = args.hop_length
    if args.distributed:
        codec_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(codec_model)
        for k, v in disc_models.items():
            disc_models[k] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(v)
    device = torch.device('cuda', args.local_rank)
    codec_model.to(device)
    for k, v in disc_models.items():
        v.to(device)
    semantic_teacher.to(device)

    ## Build optimizers and learning rate schedulers
    optimizer_g = getattr(torch.optim, args.optimizer.g.name)(
        codec_model.parameters(),
        **args.optimizer.g.config
    )
    lr_scheduler_g = getattr(torch.optim.lr_scheduler, args.lr_scheduler.g.name)(
        optimizer_g, **args.lr_scheduler.g.config
    )
    optimizer_d = getattr(torch.optim, args.optimizer.d.name)(
        itertools.chain(*[v.parameters() for k, v in disc_models.items()]),
        **args.optimizer.d.config
    )
    lr_scheduler_d = getattr(torch.optim.lr_scheduler, args.lr_scheduler.d.name)(
        optimizer_d, **args.lr_scheduler.d.config
    )

    if args.distributed:
        print('begin DDP')
        codec_model = DDP(codec_model, device_ids=[torch.cuda.current_device()], find_unused_parameters=True)
        for k, v in disc_models.items():
            disc_models[k] = DDP(v, device_ids=[torch.cuda.current_device()], find_unused_parameters=True)
    ## Build data loader
    train_dataset = WaveDataset(
        flist_file=args.training_file,
        segment_size=args.segment_size,
        sampling_rate=args.sample_rate,
        split=True, # whether or not to get a segment of an audio sample to form the batch
        shuffle=False if args.distributed else True,
        audio_norm_scale=args.audio_norm_scale,
    )
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset, drop_last=True, shuffle=True)
    else:
        train_sampler = None
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              sampler=train_sampler,
                              pin_memory=True,)
    valid_dataset = WaveDataset(
        flist_file=args.validation_file,
        segment_size=args.segment_size,
        sampling_rate=args.sample_rate,
        split=True, # whether or not to get a segment of an audio sample to form the batch
        shuffle=True,
        audio_norm_scale=args.audio_norm_scale,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=1, num_workers=1, shuffle=False, pin_memory=True, drop_last=False)  
    # automatically find the latest ckpt
    if os.path.isdir(args.model_ckpt_dir):
        ckpt_file = scan_checkpoint(args.model_ckpt_dir, prefix='ckpt_')
    else:
        ckpt_file = None
        if logger.is_primary:
            os.makedirs(args.model_ckpt_dir, exist_ok=True)
    global_steps = 0
    if ckpt_file is None:
        args.last_epoch = -1
    else:
        ckpt_state_dict = torch.load(ckpt_file, map_location=device)
        args.last_epoch = ckpt_state_dict['epoch']
        global_steps = ckpt_state_dict['steps'] + 1
        if args.ngpus_per_node > 1:
            codec_model.module.load_state_dict(ckpt_state_dict['codec_model'])
        else:
            codec_model.load_state_dict(ckpt_state_dict['codec_model'])
        for k, v in disc_models.items():
            if args.ngpus_per_node > 1:
                v.module.load_state_dict(ckpt_state_dict[k])
            else:
                v.load_state_dict(ckpt_state_dict[k])
        optimizer_g.load_state_dict(ckpt_state_dict['optimizer_g'])
        lr_scheduler_g.load_state_dict(ckpt_state_dict['lr_scheduler_g'])
        optimizer_d.load_state_dict(ckpt_state_dict['optimizer_d'])
        lr_scheduler_d.load_state_dict(ckpt_state_dict['lr_scheduler_d'])
        logger.log_info(f"Resume from: {ckpt_file}")
    logger.log_info(f"Global steps: {global_steps}")
    ## Build criterion
    criterion = {}
    criterion["generator"] = load_obj(
        args.criterion.g_criterion.name)(args.criterion.g_criterion.config).cuda(args.local_rank)
    for d_name in args.d_list:
        criterion[d_name] = load_obj(
            args.criterion.d_criterion.name)(args.criterion.d_criterion.config).cuda(args.local_rank)

    train(args, device, codec_model, semantic_teacher, disc_models, train_loader, valid_loader,
          optimizer_g, optimizer_d, lr_scheduler_g, lr_scheduler_d, logger, criterion, global_steps)


## Training function
def train(args, device, codec_model, semantic_teacher, disc_models, train_loader, valid_loader, 
          optimizer_g, optimizer_d, lr_scheduler_g, lr_scheduler_d, logger, criterion,
          global_steps):
    plot_gt_once = False
    for epoch in range(max(0, args.last_epoch), args.num_epoches):
        logger.log_info(f"="*10 + f" Epoch: {epoch}, Step: {global_steps} " + f"="*10)
        codec_model.train()
        for k, v in disc_models.items():
            v.train()
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        for i, batch in enumerate(train_loader):
            audio_24k, audio_16k = batch
            y = to_device(audio_24k, device, non_blocking=True)
            audio_16k = to_device(audio_16k, device, non_blocking=True)
            with torch.no_grad():
                features = semantic_teacher.extract(audio_16k)
            
            #######################
            #      Generator      #
            #######################
            optimizer_g.zero_grad()
            y_hat, _, commitment_loss, global_semantic_loss  = codec_model(y, features)
            codec_loss = global_semantic_loss  # only use contristive loss
            # g loss
            output_real, output_fake, fmap_real, fmap_fake = {}, {}, {}, {}
            for d_name in args.d_list:
                output_real[d_name], output_fake[d_name], fmap_real[d_name], fmap_fake[d_name] = \
                    disc_models[d_name](y, y_hat)
            g_loss, g_loss_items = criterion["generator"](
                y, y_hat, output_real, output_fake, fmap_real, fmap_fake,
                use_adv_loss=global_steps>args.discriminator_iter_start)
            g_loss = g_loss + codec_loss
            g_loss_items['Train/g_loss'] = g_loss.item()
            g_loss_items['Train/codec_loss'] = codec_loss.item()
            
            g_loss.backward()
            optimizer_g.step()

            #######################
            #    Discriminator    #
            #######################
            optimizer_d.zero_grad()
            d_loss_items = {}
            d_loss = 0.
            y_hat, _, commitment_loss, global_semantic_loss  = codec_model(y, features) # update y_hat
            for d_name in args.d_list:
                output_real, output_fake, _, _ = disc_models[d_name](y, y_hat.detach())
                cur_d_loss = criterion[d_name](output_real, output_fake)
                d_loss += cur_d_loss
                d_loss_items[f"Train/D_{d_name}"] = cur_d_loss.item()
            
            d_loss_items[f"Train/d_loss"] = d_loss.item()
            d_loss.backward()
            optimizer_d.step()

            global_steps += 1
            
            if logger.is_primary:
                if global_steps % args.print_freq == 0:
                    message = f"epoch: {epoch}, iter: {global_steps}, "
                    for key in sorted(g_loss_items.keys()):
                        message += f"{key}: {g_loss_items[key]}, "
                    for key in sorted(d_loss_items.keys()):
                        message += f"{key}: {d_loss_items[key]}, "
                    logger.log_info(message)

                if global_steps % args.summary_interval == 0:
                    for k, v in g_loss_items.items():
                        logger.tb_writer.add_scalar(k, v, global_steps)
                    for k, v in d_loss_items.items():
                        logger.tb_writer.add_scalar(k, v, global_steps)
                    cur_g_lr = lr_scheduler_g.get_lr()[0]
                    cur_d_lr = lr_scheduler_d.get_lr()[0]
                    logger.tb_writer.add_scalar('lr_g', cur_g_lr, global_steps)
                    logger.tb_writer.add_scalar('lr_d', cur_d_lr, global_steps)

                # checkpointing
                if global_steps % args.checkpoint_interval == 0 and global_steps != 0:
                    checkpoint_path = f"{args.model_ckpt_dir}/ckpt_{global_steps:08d}.pth"
                    state_dict = {
                        'codec_model': (codec_model.module if args.ngpus_per_node > 1 else codec_model).state_dict(),
                        'optimizer_g': optimizer_g.state_dict(),
                        'optimizer_d': optimizer_d.state_dict(),
                        'lr_scheduler_g': lr_scheduler_g.state_dict(),
                        'lr_scheduler_d': lr_scheduler_d.state_dict(),
                        'epoch': epoch,
                        'steps': global_steps,
                    }
                    for k, v in disc_models.items():
                        state_dict[k] = (v.module if args.ngpus_per_node > 1 else v).state_dict()
                    
                    save_checkpoint(
                        checkpoint_path,
                        state_dict,
                        num_ckpt_keep=args.num_ckpt_keep
                    )

            #### Validation
            if global_steps % args.validation_interval == 0:
                codec_model.eval()
                for k, v in disc_models.items():
                    v.eval()
                valid_loss = defaultdict(float)
                with torch.no_grad():
                    for j, batch in enumerate(valid_loader):
                        audio_24k, audio_16k = batch
                        y = to_device(audio_24k, device, non_blocking=True)
                        audio_16k = to_device(audio_16k, device, non_blocking=True)
                        with torch.no_grad():
                            features = semantic_teacher.extract(audio_16k)
                        y_len = y.size(-1)
                        #y = y[..., :int(y_len//args.hop_length * args.hop_length)]
                        y_hat, _, commitment_loss, global_semantic_loss = codec_model(y, features)
                        codec_loss = commitment_loss.mean() + global_semantic_loss 
                        # g loss
                        output_real, output_fake, fmap_real, fmap_fake = {}, {}, {}, {}
                        for d_name in args.d_list:
                            output_real[d_name], output_fake[d_name], fmap_real[d_name], fmap_fake[d_name] = \
                                disc_models[d_name](y, y_hat)
                        g_loss, g_loss_items = criterion["generator"](
                            y, y_hat, output_real, output_fake, fmap_real, fmap_fake)
                        g_loss = g_loss + codec_loss
                        g_loss_items['Valid/g_loss'] = g_loss.item()
                        g_loss_items['Valid/codec_loss'] = codec_loss.item()
                        # d loss
                        d_loss_items = {}
                        d_loss = 0.
                        for d_name in args.d_list:
                            output_real, output_fake, _, _ = disc_models[d_name](y, y_hat.detach())
                            cur_d_loss = criterion[d_name](output_real, output_fake)
                            d_loss += cur_d_loss
                            d_loss_items[f"Valid/D_{d_name}"] = cur_d_loss.item()
                        
                        d_loss_items[f"Valid/d_loss"] = d_loss.item()
                        for key in g_loss_items:
                            valid_loss[key.replace('Train', 'Valid')] += g_loss_items[key]
                        for key in d_loss_items:
                            valid_loss[key.replace('Train', 'Valid')] += d_loss_items[key]

                    # Average validation loss
                    for key in valid_loss:
                        valid_loss[key] /= (j + 1)
                    message = f"epoch: {epoch}, iter: {global_steps}, "
                    for key in sorted(valid_loss.keys()):
                        message += f"{key}: {valid_loss[key]}, "
                    logger.log_info(message)
                    if logger.is_primary:
                        for k, v in valid_loss.items():
                            logger.tb_writer.add_scalar(k, v, global_steps)
                codec_model.train()
                for k, v in disc_models.items():
                    v.train()
                lr_scheduler_g.step()
                lr_scheduler_d.step()

if __name__ == '__main__':
    main()
