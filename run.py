import os
import sys
import time
import random
import argparse

from utils.hdfs_io import HADOOP_BIN, hexists, hmkdir, hcopy
from utils.wit_preproc import wit_preproc
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
############ Set it correctly for distributed training across nodes
NNODES = 1  # e.g. 1/2/3/4
NPROC_PER_NODE = 4  # e.g. 8 gpus

MASTER_ADDR = 'SET_IT'
MASTER_PORT = 12345
NODE_RANK = 0  # e.g. 0/1/2


############


def get_nnodes(args):  # when using only part of nodes
    if args.dist == 'all':
        return NNODES
    else:
        return 1


def get_dist_launch(args):  # some examples
    if args.dist == 'all':  # use all nodes
        return "python3 -m torch.distributed.launch --nproc_per_node={:} " \
               "--nnodes={:} --node_rank={:} --master_addr={:} --master_port={:}".format(
            NPROC_PER_NODE, NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT)

    elif args.dist == '1':
        master_port = f'--master_port={args.port}' if args.port > 0 else ''
        return "python3 -m torch.distributed.launch --nproc_per_node={:} " \
               "--nnodes=1 {}".format(NPROC_PER_NODE, master_port)

    elif args.dist == 'f4':
        return "CUDA_VISIBLE_DEVICES=0,1,2,3 WORLD_SIZE=4 python3 -m torch.distributed.launch --nproc_per_node=4 " \
               "--nnodes=1 "

    elif args.dist == 'l4':
        return "CUDA_VISIBLE_DEVICES=4,5,6,7 WORLD_SIZE=4 python3 -m torch.distributed.launch --master_port=12345 --nproc_per_node=4 " \
               "--nnodes=1 "

    elif args.dist.startswith('gpu'):  # use one gpu, --dist "gpu0"
        num = int(args.dist[3:])
        assert 0 <= num <= 8
        return "CUDA_VISIBLE_DEVICES={:} WORLD_SIZE=1 python3 -m torch.distributed.launch --nproc_per_node=1 " \
               "--nnodes=1 ".format(num)

    else:
        raise ValueError


def get_from_hdfs(file_hdfs):
    """
    compatible to HDFS path or local path
    """
    if file_hdfs.startswith('hdfs'):
        file_local = os.path.split(file_hdfs)[-1]

        if os.path.exists(file_local):
            print(f"rm existing {file_local}")
            os.system(f"rm {file_local}")

        hcopy(file_hdfs, file_local)

    else:
        file_local = file_hdfs
        assert os.path.exists(file_local)

    return file_local


def run_pretrain(args):
    print("### Start pre-training", flush=True)

    dist_launch = get_dist_launch(args) + ' --use_env '
    Pretrain_name = f'Pretrain_{args.pret_type}' if args.pret_type else 'Pretrain'
    checkpoint = f'--checkpoint {args.checkpoint}' if args.checkpoint else ''
    pret_para = f'{args.pret_para}' if args.pret_para else ''
    os.system(f"{dist_launch} {Pretrain_name}.py --seed {args.seed} "
              f"--epoch {args.epoch} --config {args.config} --output_dir {args.output_dir} {checkpoint}"
              f" {pret_para}")


def run_pretrain_nlvr(args):
    print("### Start nlvr domain pre-training", flush=True)

    dist_launch = get_dist_launch(args)

    if len(args.load_ckpt_from):
        print(f"### Loading domain pre-trained results from: {args.load_ckpt_from}")
        domain_ckpt = get_from_hdfs(args.load_ckpt_from)

    else:  # domain pre-train
        if not os.path.exists(args.config): args.config = f'configs/{args.model}/NLVR_pretrain_O1.yaml'

        os.system(f"{dist_launch} --use_env NLVR_pretrain.py --seed {args.seed} --config {args.config} "
                  f"--output_dir {args.output_dir} --checkpoint {args.checkpoint}")

        domain_ckpt = get_from_hdfs(f"{args.output_dir}/model_state_epoch_latest.th")

    return domain_ckpt


def run_nlvr2(args, load_nlvr_pretrain=False):
    dist_launch = get_dist_launch(args)

    # assert os.path.exists("images/nlvr2")

    if not os.path.exists('data/marvl'):
        marvl_preproc('iglue/datasets/marvl', 'data/marvl')

    assert os.path.exists("images/marvl-images")
    assert os.path.exists("images/marvl_fewshot")
    assert os.path.exists('data/marvl')

    args.config = f'./configs/{args.model}/NLVR.yaml' if not args.fewshot else f'./configs/{args.model}/NLVR_fewshot.yaml'

    print("### Training NLVR2", flush=True)
    os.system(f"{dist_launch} "
              f"--use_env NLVR.py --config {args.config} "
              f"--output_dir {args.output_dir} --bs {args.bs} --seed {args.seed} --epoch {args.epoch} "
              f"--checkpoint {args.checkpoint} {'--load_nlvr_pretrain' if load_nlvr_pretrain else ''} "
              f"{'--evaluate' if args.evaluate else ''} "
              f"--lr {args.lr} {'--fewshot ' + args.fewshot if args.fewshot else ''}")


def run_itr_flickr(args):
    dist_launch = get_dist_launch(args)

    assert os.path.exists("images/flickr30k-images")

    if not os.path.exists(args.config): args.config = f"configs/{args.model}/Retrieval_multi30k_all_ft.yaml"

    print("### Training Retrieval Flickr", flush=True)
    os.system(f"{dist_launch} "
              f"--use_env 'Retrieval.py' --config {args.config} "
              f"--output_dir {args.output_dir} --bs {args.bs} --seed {args.seed} --checkpoint {args.checkpoint} {'--evaluate' if args.evaluate else ''} " + f" {args.ft_para}")


def run_itr_coco(args):
    dist_launch = get_dist_launch(args)

    assert os.path.exists("images/coco")

    if not os.path.exists(args.config): args.config = f"configs/{args.model}/Retrieval_coco.yaml"

    print("### Training Retrieval COCO", flush=True)
    os.system(f"{dist_launch} "
              f"--use_env 'Retrieval.py' --config {args.config} "
              f"--output_dir {args.output_dir} --bs {args.bs} --seed {args.seed} --epoch {args.epoch} "
              f"--checkpoint {args.checkpoint} {'--evaluate' if args.evaluate else ''}")


def run_vqa(args, load_vqa_pretrain=False):
    dist_launch = get_dist_launch(args)

    assert os.path.exists("images/gqa")

    print("### Training VQA", flush=True)
    args.config = f"configs/{args.model}/GQA_fewshot.yaml" if args.fewshot else f"configs/{args.model}/GQA.yaml"

    os.system(f"{dist_launch} "
              f"--use_env VQA.py --config {args.config} {'--load_vqa_pretrain' if load_vqa_pretrain else ''}"
              f"{f'--output_hdfs {args.output_hdfs}' if len(args.output_hdfs) else ''} --output_dir {args.output_dir} "
              f"--bs {args.bs} --seed {args.seed} --checkpoint {args.checkpoint} {'--evaluate' if args.evaluate else ''} "
              f"{'--load_vqa_pretrain --fewshot ' + args.fewshot if args.fewshot else ''} --lr {args.lr}")


def run_xvnli(args):
    dist_launch = get_dist_launch(args)
    print("### Training xvnli", flush=True)

    assert os.path.exists("images/flickr30k-images")

    evaluate = ' --evaluate' if args.evaluate else ''

    if args.fewshot:
        args.config = f'./configs/cclm-base-ft/XVNLI_fewshot.yaml'
        os.system(f"{dist_launch} "
                  f"--use_env XVNLI.py --config {args.config} "
                  f"--output_dir {args.output_dir} {f'--output_hdfs {args.output_hdfs}' if len(args.output_hdfs) else ''} --bs {args.bs} --seed {args.seed} --checkpoint {args.checkpoint} "
                  f"--fewshot {args.fewshot} --lr {args.lr}" + evaluate)
    else:
        args.config = f'./configs/cclm-base-ft/XVNLI.yaml'
        trans_test = ' --gmt' if args.gmt else ''
        os.system(f"{dist_launch} "
                  f"--use_env XVNLI.py --config {args.config} "
                  f"--output_dir {args.output_dir} {f'--output_hdfs {args.output_hdfs}' if len(args.output_hdfs) else ''} --bs {args.bs} --seed {args.seed} --checkpoint {args.checkpoint} "
                  f"--lr {args.lr}" + trans_test + evaluate)


def run_flickrco(args):
    dist_launch = get_dist_launch(args)
    print("### Training xFlickr&CO", flush=True)

    assert os.path.exists("images/val2014")
    assert os.path.exists("images/flickr30k-images")

    evaluate = ' --evaluate' if args.evaluate else ''

    if args.fewshot:
        args.config = f"configs/cclm-base-ft/xFlickrCO_fewshot.yaml"
        os.system(f"{dist_launch} "
                  f"--use_env xFlickrCO.py --config {args.config} "
                  f"--output_dir {args.output_dir} --bs {args.bs} --seed {args.seed} --epoch {args.epoch} "
                  f"{f'--output_hdfs {args.output_hdfs}' if len(args.output_hdfs) else ''} --checkpoint {args.checkpoint} "
                  f"--fewshot {args.fewshot} --lr {args.lr}" + evaluate + f" {args.ft_para}")
    else:
        args.config = f"configs/cclm-base-ft/xFlickrCO.yaml"
        trans_test = ' --gmt' if args.gmt else ''
        os.system(f"{dist_launch} "
                  f"--use_env xFlickrCO.py --config {args.config} "
                  f"--output_dir {args.output_dir} --bs {args.bs} --seed {args.seed} --epoch {args.epoch} "
                  f"{f'--output_hdfs {args.output_hdfs}' if len(args.output_hdfs) else ''} --checkpoint {args.checkpoint} "
                  f"--lr {args.lr} " + trans_test + evaluate + f" {args.ft_para}")


def run_wit(args):
    dist_launch = get_dist_launch(args)
    print("### Training WIT", flush=True)

    if not os.path.exists("data/wit"):
        wit_preproc("test", "iglue/datasets/wit/annotations", "images/wit_test", "data/wit/annotations-bs64")
        wit_preproc("train", "iglue/datasets/wit/annotations", "images/image_data_train/image_pixels",
                    "data/wit/annotations-bs64")

    evaluate = ' --evaluate' if args.evaluate else ''

    args.config = f"configs/cclm-base-ft/WIT.yaml"
    trans_test = ' --gmt' if args.gmt else ''
    os.system(f"{dist_launch} "
              f"--use_env WIT.py --config {args.config} "
              f"--output_dir {args.output_dir} --bs {args.bs} --seed {args.seed} --epoch {args.epoch} "
              f"{f'--output_hdfs {args.output_hdfs}' if len(args.output_hdfs) else ''} --checkpoint {args.checkpoint}" + trans_test + evaluate)


def run(args):
    if args.task == 'itr_coco':
        run_itr_coco(args)

    elif args.task == 'itr_multi30k':
        run_itr_flickr(args)

    elif args.task == 'gqa':
        run_vqa(args)

    elif args.task == 'nlvr_domain':
        args.config = f'configs/{args.model}/NLVR_multilingual_pretrain_O1.yaml'
        domain_ckpt = run_pretrain_nlvr(args)

        # run fine-tune, reset args
        args.checkpoint = domain_ckpt
        if hexists(args.output_dir): args.output_dir = os.path.join(args.output_dir, 'nlvr_ft')
        args.config = f'./configs/{args.model}/NLVR.yaml'
        run_nlvr2(args, load_nlvr_pretrain=True)

    elif args.task == 'nlvr':
        run_nlvr2(args)

    elif args.task == 'xvnli':
        run_xvnli(args)

    elif args.task == 'xflickrco':
        run_flickrco(args)

    elif args.task == 'wit':
        run_wit(args)

    elif args.task == 'pretrain':
        run_pretrain(args)

    else:
        raise NotImplementedError(f"task == {args.task}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--dist', type=str, required=True, help="see func get_dist_launch for details")

    parser.add_argument('--config', default='configs/Pretrain.yaml', type=str, help="if not given, use default")
    parser.add_argument('--model', default='cclm-base-ft', type=str, help="to set default fine-tuning configs")

    parser.add_argument('--epoch', default=-1, type=int, help="for pre-training (debug) only")
    parser.add_argument('--bs', default=-1, type=int, help="for each gpu, batch_size = bs // num_gpus; "
                                                           "this option only works for fine-tuning scripts.")

    parser.add_argument('--checkpoint', default='', type=str, help="for fine-tuning")
    parser.add_argument('--load_ckpt_from', default='', type=str, help="load domain pre-trained params")

    # write path: local or HDFS
    parser.add_argument('--output_dir', type=str, required=True, help='for fine-tuning, local path; '
                                                                      'for pre-training, local and HDFS are both allowed.')
    parser.add_argument('--output_hdfs', type=str, default='', help="HDFS path required by VQA and Refcoco, "
                                                                    "to collect eval results among nodes")

    parser.add_argument('--evaluate', action='store_true', help="evaluation on downstream tasks")
    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--fewshot', default='', type=str, help="IGLUE fewshot. <lang>,<shot_num>, eg: ar,25")
    parser.add_argument('--lr', default=0., type=float, help="learning rate")
    parser.add_argument('--gmt', action='store_true', help="whether use google machine translation as test set")

    parser.add_argument('--device', default=None, type=str, help="which gpu can be visible")
    parser.add_argument('--pret_type', type=str, default='', help='use which pretrain code, dafault is ours')
    parser.add_argument('--rank', type=int, help='node rank', default=0)
    parser.add_argument('--nnodes', type=int, help='node num', default=1)
    parser.add_argument('--master_addr', type=str, help='node rank', default='')
    parser.add_argument('--debug', help='debug enable', action='store_true')
    parser.add_argument('--pret_para', help='other params for pretraining code', type=str, default='')
    parser.add_argument('--ft_para', help='other params for finetune code', type=str, default='')
    parser.add_argument('--port', type=int, help='dist main port', default=-1)

    args = parser.parse_args()
    if args.device:
        if args.device.startswith('c'):
            num = int(args.device[1:])
            int_devices = [str(i) for i in range(num)]
            args.device = ','.join(int_devices)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        NPROC_PER_NODE = len(args.device.split(","))
    if args.nnodes > 1:
        NNODES = args.nnodes
        NODE_RANK = args.rank
    if args.master_addr != '':
        MASTER_ADDR = args.master_addr
    if args.port > 0:
        MASTER_PORT = args.port
    if args.debug:
        import torch

        torch.autograd.set_detect_anomaly(True)
    print("NNODES, ", NNODES)
    print("NPROC_PER_NODE, ", NPROC_PER_NODE)
    print("MASTER_ADDR, ", MASTER_ADDR)
    print("MASTER_PORT, ", MASTER_PORT)
    print("NODE_RANK, ", NODE_RANK)
    if MASTER_ADDR == 'SET_IT':
        print("### warning: the settings for distributed training is not filled (ignore this if you only use one node)")

    if '/SET/PATH/TO/hadoop/bin/hdfs' in HADOOP_BIN:
        print("### warning: you have not set the path to hadoop_bin (ignore this if you don't use HDFS)")
    os.makedirs(args.output_dir, exist_ok=True)
    assert hexists(os.path.dirname(args.output_dir))
    hmkdir(args.output_dir)

    if len(args.output_hdfs):
        assert hexists(os.path.dirname(args.output_hdfs))

    if len(args.config):
        assert hexists(args.config)

        if args.config.startswith('hdfs://'):
            args.config = get_from_hdfs(args.config)

    if args.checkpoint.startswith('hdfs://'):
        args.checkpoint = get_from_hdfs(args.checkpoint)

    run(args)
