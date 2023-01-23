# Cross-View Language Modeling: Towards Unified Cross-Lingual Cross-Modal Pre-training (https://arxiv.org/abs/2206.00621)
# Github: https://github.com/zengyan-97/CCLM
# Copyright (c) 2022, ByteDance Inc.
# All rights reserved.

import argparse
import os
import sys

import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
import math

import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.optim import Optimizer
from torch.cuda.amp import autocast
from accelerators.torch_ddp_accelerator import TorchDDPAccelerator
from models.model_pretrain_mm import UniAlignLM

import utils
from dataset import create_dataset
from scheduler import create_scheduler
from optim import create_optimizer

from utils.checkpointer import Checkpointer
from utils.hdfs_io import hmkdir, hcopy


def reinit_scheduler_properties_mysched(optimizer: Optimizer, scheduler, cfg) -> None:
    """
    with ApexDDP, do re-init to avoid lr_scheduler warning.
    issue: https://github.com/pytorch/pytorch/issues/27595
    issue: https://github.com/PyTorchLightning/pytorch-lightning/issues/841
    """
    args = cfg

    if scheduler.optimizer == optimizer:
        # from transformers import get_linear_schedule_with_warmup
        def lr_lambda(current_step: int):
            if current_step < args.num_warmup_steps:
                return float(current_step) / float(max(1, args.num_warmup_steps))
            return max(
                0.0, float(args.num_training_steps - current_step) / float(
                    max(1, args.num_training_steps - args.num_warmup_steps))
            )

        scheduler.__init__(optimizer, lr_lambda, last_epoch=-1)


def image_multi_iter(model, image_batch, optimizer, accelerator, metric_logger, device):
    image, image_batch = image_batch[0].to(device, non_blocking=True), \
                         [t.to(device) if t is not None else None for t in image_batch[1:]]
    text_ids, text_atts, text_ids_masked, masked_pos, masked_ids = image_batch
    optimizer.zero_grad()
    with autocast():
        loss = model(image, text_ids, text_atts, text_ids_masked, masked_pos,
                     masked_ids)

        loss_in_total = loss['loss_mitc'] + loss['loss_hitm'] + loss['loss_hmlm']
        if 'loss_ttc' in loss.keys():
            loss_in_total += loss['loss_ttc']
            loss_ttc = loss['loss_ttc'].item()
        else:
            loss_ttc = 0.0
    if accelerator != None:
        accelerator.backward_step(loss_in_total, optimizer)

        accelerator_clip_grad_norm = float(config['accelerator']['CLIP_GRAD_NORM'])
        if accelerator_clip_grad_norm > 0:
            accelerator.optimizer_step(optimizer, model, accelerator_clip_grad_norm)
    else:
        loss_in_total.backward()
    accelerator.scaler.step(optimizer)
    accelerator.scaler.update()

    metric_logger.update(loss_mm_img_mitc=loss['loss_mitc'].item())
    metric_logger.update(loss_mm_img_hitm=loss['loss_hitm'].item())
    metric_logger.update(loss_mm_img_hmlm=loss['loss_hmlm'].item())
    metric_logger.update(loss_mm_img_ttc=loss_ttc)
    metric_logger.update(loss_mm_img_invariance=loss['loss_invariance'].item())
    metric_logger.update(loss_mm_img_variance=loss['loss_variance'].item())
    metric_logger.update(loss_mm_img_covariance=loss['loss_covariance'].item())


def train(model, general_loader, optimizer, epoch_info, device, scheduler,
          config, accelerator, checkpointer, global_step):
    model.train()
    start_epoch, _ = epoch_info
    metric_logger = utils.MetricLogger(delimiter="  ")

    # multilingual images
    metric_logger.add_meter('loss_mm_img_mitc', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_mm_img_hitm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_mm_img_hmlm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_mm_img_ttc', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_mm_img_invariance', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_mm_img_variance', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_mm_img_covariance', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))

    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('lr_large', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))

    header = 'Train step: [{}]'.format(start_epoch)
    # assert start_epoch == 0
    print_freq = 50

    world_size = utils.get_world_size()
    step_per_epoch = math.ceil(config['train_dataset_size'] / (config['batch_size'] * world_size))
    assert step_per_epoch > 1
    # global_step = global_step  # start from 0

    # image_iter = iter(general_loader)

    for i, image_batch in enumerate(
            metric_logger.log_every(general_loader, print_freq, header, step_per_epoch, epoch_info)):

        image_multi_iter(model, image_batch, optimizer, accelerator, metric_logger, device)

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(lr_large=optimizer.param_groups[2]["lr"])
        scheduler.step()

        current_epoch = global_step // step_per_epoch

        if (global_step + 1) % step_per_epoch == 0:
            if utils.is_main_process():
                train_stats = {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             'epoch': current_epoch}

                with open("log.txt", "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                if (current_epoch + 1) % config['ckpt_frequent'] == 0:
                    model_without_ddp = model
                    if hasattr(model, 'module'):
                        model_without_ddp = model.module

                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': scheduler.state_dict(),
                        'config': config,
                        'epoch': current_epoch,
                        'random_state': random.getstate(),
                        'np_random_state': np.random.get_state(),
                        'torch_rng_state': torch.get_rng_state(),
                        'global_step': global_step,
                        'scaler': accelerator.scaler.state_dict()
                    }
                    checkpointer.save_checkpoint(model_state=save_obj,
                                                 epoch=current_epoch,
                                                 training_states=optimizer.state_dict())

            dist.barrier()

        if (global_step + 1) % config['ckpt_frequent_step'] == 0:
            if utils.is_main_process():
                model_without_ddp = model
                if hasattr(model, 'module'):
                    model_without_ddp = model.module

                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': scheduler.state_dict(),
                    'config': config,
                    'epoch': current_epoch,
                    'random_state': random.getstate(),
                    'np_random_state': np.random.get_state(),
                    'torch_rng_state': torch.get_rng_state(),
                    'global_step': global_step,
                    'scaler': accelerator.scaler.state_dict()
                }

                checkpointer.save_checkpoint(model_state=save_obj,
                                             epoch=current_epoch, step=global_step,
                                             training_states=optimizer.state_dict())

            dist.barrier()

        global_step += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


def main(args, config):
    checkpoint = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    # torch.autograd.set_detect_anomaly(True)
    config['train_file'] = ','.join(config['train_file'])
    config['train_file_regions'] = ','.join(config['train_file_regions'])

    config['train_file_mono'] = ','.join(config['train_file_mono'])
    config['train_file_text'] = ','.join(config['train_file_text'])  # multilingual parallel texts

    config['batch_size'] = config['images']['batch_size']

    if args.epoch > 0:
        config['schedular']['epochs'] = args.epoch
        print(f"### set epochs to: {args.epoch}", flush=True)
    if checkpoint:
        config = checkpoint['config']
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if checkpoint:
        torch.set_rng_state(checkpoint['torch_rng_state'])
        random.setstate(checkpoint['random_state'])
        np.random.set_state(checkpoint['np_random_state'])

    cudnn.benchmark = True

    print("Creating dataset", flush=True)
    general_dataset, region_dataset, mono_dataset, text_dataset = \
        create_dataset('pretrain_multilingual', config)

    if utils.is_main_process():
        print(f"### train_file: {config['train_file']}", flush=True)
        print(f"### train_file_regions: {config['train_file_regions']}", flush=True)

        print(f"### train_file_mono: {config['train_file_mono']}", flush=True)
        print(f"### train_file_text: {config['train_file_text']}", flush=True)
        print(f"### batch size, {config['batch_size']} x {int(os.environ.get('WORLD_SIZE', 1))}")

    general_loader = torch.utils.data.DataLoader(general_dataset, batch_size=config['images']['batch_size'],
                                                 num_workers=config['images']['num_workers'],
                                                 pin_memory=True,
                                                 drop_last=False,
                                                 collate_fn=general_dataset.collate_fn)

    print("Creating model", flush=True)
    model = UniAlignLM(config=config)
    if args.save0:
        save_obj = {
            'model': model.state_dict(),
            # 'optimizer': optimizer.state_dict(),
            # 'lr_scheduler': scheduler.state_dict(),
            'config': config,
            # 'epoch': current_epoch,
        }
        torch.save(save_obj, "init_model.pth")
        return
        # print(model)
    if checkpoint:
        model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    print("### Total Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad), flush=True)

    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    arg_sche = utils.AttrDict(config['schedular'])
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    arg_sche['step_per_epoch'] = math.ceil(config['train_dataset_size'] / (config['batch_size'] * world_size))
    lr_scheduler = create_scheduler(arg_sche, optimizer)
    if checkpoint:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    arg_acc = utils.AttrDict(config['accelerator'])
    accelerator = TorchDDPAccelerator(arg_acc, rank, logger=None)
    if checkpoint:
        if 'scaler' in checkpoint:
            accelerator.scaler.load_state_dict(checkpoint['scaler'])
    model, optimizer, lr_scheduler = accelerator.set_up(model, optimizer, lr_scheduler, local_rank, world_size)
    # reinit_scheduler_properties_mysched(optimizer, lr_scheduler, arg_sche)

    checkpointer = Checkpointer(args.output_dir)

    print("### output_dir, ", args.output_dir, flush=True)
    start_time = time.time()

    start_epoch = 0
    global_step = 0
    if checkpoint:
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['global_step'] + 1
    max_epoch = config['schedular']['epochs']
    epoch_info = (start_epoch, max_epoch)

    print("Start training", flush=True)
    train(model, general_loader, optimizer, epoch_info, device, lr_scheduler,
          config,
          accelerator, checkpointer, global_step)
    dist.barrier()

    if utils.is_main_process():
        os.system("cat log.txt")
        hcopy('log.txt', args.output_dir)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str), flush=True)

    print('### Time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='output/pretrain')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--epoch', default=-1, type=int, help="for pre-training (debug) only")
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--distributed', action='store_false')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--save0', action='store_true', help="whether save model at the beginning")
    parser.add_argument('--imgbsz', type=int, help='img-text dataset batch size', default=0)
    parser.add_argument('--checkpoint', type=str, help='pretrain model checkpoint', default='')
    parser.add_argument('--train_file', help='train file list use , join', type=str, default='')
    parser.add_argument('--neg_sample_type', help='neg sample type 0,1,2', default=1, type=int)
    parser.add_argument('--sample_lan_file', default='')
    parser.add_argument('--need_divm', default=0, type=int)
    parser.add_argument('--caption_num', default=0, type=int)
    parser.add_argument('--cclm_easy', default=0, type=int)
    parser.add_argument('--itc_allgather', default=0, type=int, help="0--mitc allgather 1--mitc not allgather")
    parser.add_argument('--vicreg', default=1, type=int, help="need vicreg")
    parser.add_argument('--lr', default=1e-4, type=float)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    if args.imgbsz > 0:
        config['images']['batch_size'] = args.imgbsz
    if args.train_file:
        config['train_file'] = args.train_file.split(',')
        print()
    config['neg_sample_type'] = args.neg_sample_type
    config['sample_lan_file'] = args.sample_lan_file
    config['need_divm'] = args.need_divm
    config['caption_num'] = args.caption_num
    config['cclm_easy'] = args.cclm_easy
    config['itc_allgather'] = args.itc_allgather
    config['vicreg'] = args.vicreg
    config['schedular']['lr'] = args.lr
    config['optimizer']['lr'] = args.lr
    if utils.is_main_process():
        print('neg_sample_type:', args.neg_sample_type, 'sample_lan_file:', args.sample_lan_file, 'need_divm:',
              args.need_divm, 'caption_num', args.caption_num, 'lr:', args.lr)
    hmkdir(args.output_dir)

    yaml.dump(config, open('config.yaml', 'w'))
    hcopy('config.yaml', args.output_dir)

    main(args, config)
