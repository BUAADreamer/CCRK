import argparse
import os
import sys
import math

import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from PIL import Image
from models.model_retrieval import RetrievalModel

import utils
from utils.hdfs_io import hopen, hexists, hmkdir
from dataset import create_dataset, create_sampler, create_loader, build_tokenizer
from scheduler import create_scheduler
from optim import create_optimizer


def train(model, data_loader, optimizer, tokenizer, epoch, device, scheduler, config):
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itc', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    for i, (image, text, idx) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)
        text_input = tokenizer(text, padding='longest', max_length=config['max_tokens'], truncation=True,
                               return_tensors="pt").to(device)

        loss_itc, loss_itm = model(image, text_input.input_ids, text_input.attention_mask, idx=idx)
        loss = loss_itc + loss_itm

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(loss_itc=loss_itc.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluation_itm(model, data_loader, tokenizer, device, config):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'

    print('Computing features for evaluation...')
    start_time = time.time()

    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = config['batch_size_test_text']  # 256
    text_feats = []
    text_embeds = []
    text_atts = []
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i + text_bs)]
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=config['max_tokens'],
                               return_tensors="pt").to(device)

        text_feat = model.get_text_embeds(text_input.input_ids, text_input.attention_mask)
        text_embed = model.get_features(text_embeds=text_feat)

        text_embeds.append(text_embed)
        text_feats.append(text_feat)
        text_atts.append(text_input.attention_mask)

    text_embeds = torch.cat(text_embeds, dim=0)
    text_feats = torch.cat(text_feats, dim=0)
    text_atts = torch.cat(text_atts, dim=0)

    image_feats = []
    image_embeds = []
    for image, img_id in data_loader:
        image = image.to(device)

        image_feat, _ = model.get_vision_embeds(image)
        image_embed = model.get_features(image_embeds=image_feat)

        image_feats.append(image_feat)
        image_embeds.append(image_embed)

    image_feats = torch.cat(image_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)

    score_matrix_i2t = torch.full((len(data_loader.dataset.image), len(texts)), -100.0).to(device)

    num_tasks = utils.get_world_size()
    rank = utils.get_rank()
    step = score_matrix_i2t.size(0) // num_tasks + 1
    start = rank * step
    end = min(score_matrix_i2t.size(0), start + step)
    N = 2000
    for i, sims in enumerate(metric_logger.log_every(score_matrix_i2t[start:end], 50, header)):
        for j in range(0, 2000, N):
            topk_idx = torch.arange(j, j + N)
            encoder_output = image_feats[start + i].repeat(N, 1, 1)
            encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)

            output = model.get_cross_embeds(image_embeds=encoder_output, image_atts=encoder_att,
                                            text_embeds=text_feats[topk_idx], text_atts=text_atts[topk_idx])
            score = model.itm_head(output[:, 0, :])[:, 1]
            score_matrix_i2t[start + i, topk_idx] = score

    score_matrix_t2i = torch.full((len(texts), len(data_loader.dataset.image)), -100.0).to(device)

    step = score_matrix_t2i.size(0) // num_tasks + 1
    start = rank * step
    end = min(score_matrix_t2i.size(0), start + step)

    for i, sims in enumerate(metric_logger.log_every(score_matrix_t2i[start:end], 50, header)):
        for j in range(0, 2000, N):
            topk_idx = torch.arange(j, j + N)
            encoder_output = image_feats[topk_idx]
            encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)

            output = model.get_cross_embeds(image_embeds=encoder_output, image_atts=encoder_att,
                                            text_embeds=text_feats[start + i].repeat(N, 1, 1),
                                            text_atts=text_atts[start + i].repeat(N, 1))
            score = model.itm_head(output[:, 0, :])[:, 1]
            score_matrix_t2i[start + i, topk_idx] = score

    if args.distributed:
        dist.barrier()
        torch.distributed.all_reduce(score_matrix_i2t, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))
    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()


@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'

    print('Computing features for evaluation...')
    start_time = time.time()

    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = config['batch_size_test_text']  # 256
    text_feats = []
    text_embeds = []
    text_atts = []
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i + text_bs)]
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=config['max_tokens'],
                               return_tensors="pt").to(device)

        text_feat = model.get_text_embeds(text_input.input_ids, text_input.attention_mask)
        text_embed = model.get_features(text_embeds=text_feat)

        text_embeds.append(text_embed)
        text_feats.append(text_feat)
        text_atts.append(text_input.attention_mask)

    text_embeds = torch.cat(text_embeds, dim=0)
    text_feats = torch.cat(text_feats, dim=0)
    text_atts = torch.cat(text_atts, dim=0)

    image_feats = []
    image_embeds = []
    for image, img_id in data_loader:
        image = image.to(device)

        image_feat, _ = model.get_vision_embeds(image)
        image_embed = model.get_features(image_embeds=image_feat)

        image_feats.append(image_feat)
        image_embeds.append(image_embed)

    image_feats = torch.cat(image_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)

    sims_matrix = image_embeds @ text_embeds.t()
    score_matrix_i2t = torch.full((len(data_loader.dataset.image), len(texts)), -100.0).to(device)

    num_tasks = utils.get_world_size()
    rank = utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)

        encoder_output = image_feats[start + i].repeat(config['k_test'], 1, 1)
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)

        output = model.get_cross_embeds(image_embeds=encoder_output, image_atts=encoder_att,
                                        text_embeds=text_feats[topk_idx], text_atts=text_atts[topk_idx])
        score = model.itm_head(output[:, 0, :])[:, 1]
        score_matrix_i2t[start + i, topk_idx] = score

    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full((len(texts), len(data_loader.dataset.image)), -100.0).to(device)

    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        encoder_output = image_feats[topk_idx]
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)

        output = model.get_cross_embeds(image_embeds=encoder_output, image_atts=encoder_att,
                                        text_embeds=text_feats[start + i].repeat(config['k_test'], 1, 1),
                                        text_atts=text_atts[start + i].repeat(config['k_test'], 1))
        score = model.itm_head(output[:, 0, :])[:, 1]
        score_matrix_t2i[start + i, topk_idx] = score

    if args.distributed:
        dist.barrier()
        torch.distributed.all_reduce(score_matrix_i2t, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))
    if args.eval_not_cross:
        return sims_matrix.t().cpu().numpy(), sims_matrix.cpu().numpy()
    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()


@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
    specls = [564, 422, 1621, 282, 1343, 430, 233, 797, 634, 710,
              629, 237, 1493, 479, 1142, 1140, 135, 546, 722, 1797, 1452, 211, 996, 1773, 1419, 59, 1130, 963, 515,
              1138,
              855, 1936, 1225, 332, 98, 1477, 428, 359, 887, 1326]
    contentls = [0 for _ in range(40)]
    # Images->Text
    ranks = np.zeros(scores_i2t.shape[0])
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    if utils.is_main_process() and args.evaluate:
        np.save(f'ranks/{language}_{model_cap}_tr.npy', ranks)

    # Text->Images
    ranks = np.zeros(scores_t2i.shape[0])

    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        if index in specls:
            temp_i = specls.index(index)
            contentls[temp_i] = int(inds[0])
        ranks[index] = np.where(inds == txt2img[index])[0][0]
    with open(f"ranks/case_{language}.json", "w") as f:
        f.write(json.dumps(contentls, ensure_ascii=False))
    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    if utils.is_main_process() and args.evaluate:
        np.save(f'ranks/{language}_{model_cap}_ir.npy', ranks)

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result = {'txt_r1': tr1,
                   'txt_r5': tr5,
                   'txt_r10': tr10,
                   'txt_r_mean': tr_mean,
                   'img_r1': ir1,
                   'img_r5': ir5,
                   'img_r10': ir10,
                   'img_r_mean': ir_mean,
                   'r_mean': r_mean}
    return eval_result


def analysis_ranks(rtype="tr"):
    global model_cap
    lanls = ['en', 'de', 'ja', 'zh', 'es', 'ru', 'id', 'tr']
    lan2ranks = {
        "en": np.load(f"ranks/en_{model_cap}_{rtype}.npy"),
        "de": np.load(f"ranks/de_{model_cap}_{rtype}.npy"),
        "ja": np.load(f"ranks/ja_{model_cap}_{rtype}.npy"),
        "zh": np.load(f"ranks/zh_{model_cap}_{rtype}.npy")
    }
    rank_stds = []
    for i in range(2000):
        ranks_per = []
        for j in range(4):
            ranks_per.append(lan2ranks[lanls[j]][i])
        rank_stds.append(np.array(ranks_per).std())
    print(model_cap, rtype, "evaluation result:", np.array(rank_stds).mean())


def main(args, config):
    try:
        utils.init_distributed_mode(args)
    except Exception as e:
        print("init_distributed_mode error:", e)
    device = torch.device(args.device)

    world_size = utils.get_world_size()

    if args.epoch > 0:
        config['schedular']['epochs'] = args.epoch
        print(f"### set epochs to: {args.epoch}", flush=True)

    if args.bs > 0:
        config['batch_size_train'] = args.bs // world_size

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    print("Creating xFlickrCO dataset", flush=True)
    train_dataset, val_dataset, test_dataset_dict = create_dataset('xflickrco', config)

    train_dataset_size = len(train_dataset)

    if utils.is_main_process():
        print(f"### data {train_dataset_size}, batch size, {config['batch_size_train']} x {world_size}")
        print(f"### Test: {[(k, len(dataset)) for k, dataset in test_dataset_dict.items()]}")

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None]
    else:
        samplers = [None, None]

    train_loader, val_loader = create_loader([train_dataset, val_dataset], samplers,
                                             batch_size=[config['batch_size_train'], config['batch_size_test']],
                                             num_workers=[4, 4],
                                             is_trains=[True, False],
                                             collate_fns=[None, None])
    test_loader_dict = {}
    for k, v in test_dataset_dict.items():
        test_loader_dict[k] = create_loader([v], [None], batch_size=[config['batch_size_test']],
                                            num_workers=[2], is_trains=[False], collate_fns=[None])[0]

    print("Creating model", flush=True)
    model = RetrievalModel(config=config)
    model.load_pretrained(args.checkpoint, config, is_eval=args.evaluate)
    model = model.to(device)
    print("### Total Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    tokenizer = build_tokenizer(config['text_encoder'])

    start_time = time.time()
    print("### output_dir, ", args.output_dir, flush=True)
    print("### output_hdfs, ", args.output_hdfs, flush=True)
    num = args.checkpoint.split('.')[0].split('_')[-1]
    # log_file = 'epoch' + '_' + num + '.txt'
    log_file = 'log.txt'
    if args.evaluate:
        print("Start evaluating", flush=True)
        global language
        print("language:", language)
        for language, test_loader in test_loader_dict.items():
            if not args.all_lan_eval and not language in ('en', 'de', 'ja', 'zh'):
                continue
            if args.fewshot and language != cur_lan:
                continue
            if args.wo_cl:
                score_test_i2t, score_test_t2i = evaluation_itm(model_without_ddp, test_loader, tokenizer, device,
                                                                config)
            else:
                score_test_i2t, score_test_t2i = evaluation(model_without_ddp, test_loader, tokenizer, device, config)

            if utils.is_main_process():
                test_result = itm_eval(score_test_i2t, score_test_t2i, test_loader.dataset.txt2img,
                                       test_loader.dataset.img2txt)
                log_stats = {**{f'test_{language}_{k}': v for k, v in test_result.items()}}
                print(log_stats)

        dist.barrier()
        if utils.is_main_process() and not args.fewshot:
            analysis_ranks("tr")
            analysis_ranks("ir")
    else:
        print("Start training", flush=True)
        arg_opt = utils.AttrDict(config['optimizer'])
        optimizer = create_optimizer(arg_opt, model)
        arg_sche = utils.AttrDict(config['schedular'])
        arg_sche['step_per_epoch'] = math.ceil(train_dataset_size / (config['batch_size_train'] * world_size))
        lr_scheduler = create_scheduler(arg_sche, optimizer)

        max_epoch = config['schedular']['epochs']
        best = 0
        best_stats = {}
        best_epoch = 0

        if args.zs:
            max_epoch = 1
            config['start_eval'] = 0

        for epoch in range(0, max_epoch):
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, device, lr_scheduler, config)

            if epoch >= config['start_eval'] and (epoch + 1) % config['eval_interval'] == 0:
                log_stats = {}
                score_val_i2t, score_val_t2i, = evaluation(model_without_ddp, val_loader, tokenizer, device, config)
                if utils.is_main_process():
                    val_result = itm_eval(score_val_i2t, score_val_t2i, val_loader.dataset.txt2img,
                                          val_loader.dataset.img2txt)
                    log_stats.update({f'val_{k}': v for k, v in val_result.items()})
                    val_mean = log_stats['val_r_mean']
                if not args.fewshot:
                    for language, test_loader in test_loader_dict.items():
                        score_test_i2t, score_test_t2i = evaluation(model_without_ddp, test_loader, tokenizer, device,
                                                                    config)

                        if utils.is_main_process():
                            test_result = itm_eval(score_test_i2t, score_test_t2i, test_loader.dataset.txt2img,
                                                   test_loader.dataset.img2txt)
                            log_stats.update({f'test_{language}_{k}': v for k, v in test_result.items()})
                            # if args.zs:
                            #     score_zs += test_result['r_mean']
                        dist.barrier()
                        # if args.zs:
                        #     score_zs /= 8
                        #     with open(os.path.join(args.output_dir, "score-epoch.txt"), "a") as f:
                        #         f.write(f'{score_zs} \n')
                if utils.is_main_process():
                    log_stats.update({**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch})
                    # if args.zs:
                    # model_name = args.checkpoint.split('/')[1]
                    with open(os.path.join(args.output_dir, log_file), "a") as f:
                        f.write(json.dumps(log_stats) + "\n")

                    if val_mean > best:
                        save_obj = {
                            'model': model_without_ddp.state_dict(),
                            'config': config,
                            'epoch': epoch,
                        }
                        torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))
                        best = val_mean
                        best_stats = log_stats
                        best_epoch = epoch

                    elif epoch >= max_epoch - 1:
                        save_obj = {
                            'model': model_without_ddp.state_dict(),
                            # 'optimizer': optimizer.state_dict(),
                            # 'lr_scheduler': lr_scheduler.state_dict(),
                            'config': config,
                            # 'epoch': epoch,
                        }
                        torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_latest.pth'))

            dist.barrier()
            torch.cuda.empty_cache()

        if utils.is_main_process():
            with open(os.path.join(args.output_dir, log_file), "a") as f:
                f.write("best epoch: %d" % best_epoch)
            if args.output_hdfs and not args.fewshot:
                os.system(f'hdfs dfs -put {os.path.join(args.output_dir, log_file)} {args.output_hdfs}')
                os.system(f'hdfs dfs -put {os.path.join(args.output_dir, "checkpoint_best.pth")} {args.output_hdfs}')
                os.system(
                    f'hdfs dfs -put {os.path.join(args.output_dir, "checkpoint_{}.pth".format(max_epoch - 1))} {args.output_hdfs}')

            os.system(f"cat {args.output_dir}/{log_file}")
            print(best_stats)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if utils.is_main_process():
        print('### Time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)  # this script works for both mscoco and flickr30k
    parser.add_argument('--output_hdfs', type=str)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', action='store_false')

    parser.add_argument('--epoch', default=-1, type=int)
    parser.add_argument('--bs', default=-1, type=int, help="for each gpu, batch_size = bs // num_gpus")
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--fewshot', default='', type=str, help="IGLUE fewshot. <lang>,<shot_num>, eg: de,10")
    parser.add_argument('--lr', default=0., type=float, help="learning rate")
    parser.add_argument('--gmt', action='store_true', help="whether use google machine translation as test set")
    parser.add_argument('--calcu', action='store_true', help="whether calculate the similarity")
    parser.add_argument('--zs', action='store_true', help="whether calculate zero shot")
    parser.add_argument('--eval_not_cross', action='store_true', help='not use cross embedding to eval')
    parser.add_argument('--wo_cl', action='store_true')
    parser.add_argument("--model_cap", type=str, default='ours3m')
    parser.add_argument("--checkpoint_fmt", type=str)
    parser.add_argument("--all_lan_eval", action='store_true')
    parser.add_argument("--text_encoder", type=str, default='data/xlm-roberta-large')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    lan2ranks = dict()
    model_cap = args.model_cap
    language = 'en'
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.output_hdfs and not hexists(args.output_hdfs):
        hmkdir(args.output_hdfs)

    if args.fewshot:  # fewshot eg: ar,25
        for i, train_file in enumerate(config['train_file']):
            config['train_file'][i] = train_file.format(*args.fewshot.split(','))
        config['val_file'] = config['val_file'].format(args.fewshot.split(',')[0])
        if args.evaluate:
            pass
    if args.lr != 0.:
        config['optimizer']['lr'] = args.lr
        config['schedular']['lr'] = args.lr

    if args.gmt:
        config['test_file'] = config['gmt_test_file']
    config['eval_not_cross'] = args.eval_not_cross
    config['wo_cl'] = args.wo_cl
    config['text_encoder'] = args.text_encoder
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))
    cur_lan = 'en'
    os.makedirs("ranks", exist_ok=True)
    if args.fewshot and args.evaluate:
        if args.all_lan_eval:
            checkpoint_fmt = args.checkpoint_fmt.replace("format", "{}")
            for cur_lan in ['de', 'ja', 'zh', 'es', 'ru', 'id', 'tr']:
                args.checkpoint = checkpoint_fmt.format(cur_lan)
                main(args, config)
        else:
            main(args, config)
            # checkpoint_fmt = 'output/ours-cc-6lan/xflickrco/fewshot/{}/checkpoint_best_ee29.pth'
            checkpoint_fmt = args.checkpoint_fmt.replace("format", "{}")
            for cur_lan in ['de', 'ja', 'zh']:
                args.checkpoint = checkpoint_fmt.format(cur_lan)
                main(args, config)
            if utils.is_main_process():
                analysis_ranks("tr")
                analysis_ranks("ir")
    else:
        main(args, config)
