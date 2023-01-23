# -*- coding: utf-8 -*-
import os
import random
import sys
from typing import Tuple, Union, Optional, Any
import numpy as np
import torch
import torch.distributed as distributed
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

Net = torch.nn.Module

from .accelerator import Accelerator

from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel
from torch.nn import SyncBatchNorm


class TorchDDPAccelerator(Accelerator):
    """
    ApexDDPAccelerator, use apex DistributedDataParallel
    """

    def __init__(self, cfg, rank, logger):
        super().__init__(cfg, logger)
        self.accelerator_rng_seed = self.cfg.RNG_SEED
        self.accelerator_syncbn = self.cfg.SYNCBN
        self.accelerator_fp16_opt_level = self.cfg.FP16_OPT_LEVEL
        self.accelerator_fp16_loss_scale = self.cfg.FP16_LOSS_SCALE
        self.scaler = amp.GradScaler()
        self.rank = rank

    def set_up(self, model: Net, optimizer: Optimizer, lr_scheduler: LambdaLR,
               local_rank: int, world_size: int) -> Tuple[DistributedDataParallel, Optimizer, LambdaLR]:
        """
        set up TorchDDPAccelerator, including process_group and torch_ddp
        """
        rank = self.rank
        torch.backends.cudnn.benchmark = False
        random.seed(self.accelerator_rng_seed)
        np.random.seed(self.accelerator_rng_seed)
        torch.random.manual_seed(self.accelerator_rng_seed)
        torch.cuda.manual_seed_all(self.accelerator_rng_seed)
        master_address = os.environ.get('MASTER_ADDR', "127.0.0.1")
        master_port = int(os.environ.get('MASTER_PORT', 34171))

        torch.cuda.set_device(local_rank)
        model = model.cuda()
        if not torch.distributed.is_initialized():
            distributed.init_process_group(
                backend='nccl',
                init_method='tcp://{}:{}'.format(master_address, master_port),
                world_size=world_size,
                rank=rank,
                group_name='mtorch')
            print(
                f'TorchDDPAccelerator distributed, size: {world_size}, rank: {rank}, local rank: {local_rank}')
            sys.stdout.flush()

        self.broadcast(model)
        model, optimizer = self.configure_ddp(model, optimizer)

        return model, optimizer, lr_scheduler

    def broadcast(self, model: Net, src=0) -> None:
        for v in model.state_dict().values():
            distributed.broadcast(v, src)

    def configure_ddp(self, model: Net, optimizer: Optimizer) -> Tuple[DistributedDataParallel, Optimizer]:
        torch_model = DistributedDataParallel(model, device_ids=[self.rank], output_device=self.rank,
                                              find_unused_parameters=True)
        self.ddp_model = torch_model
        return torch_model, optimizer

    def backward_step(self, loss: torch.Tensor, optimizer: Optimizer):
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(optimizer)

    def optimizer_step(self, optimizer: Optimizer, model: Net, grad_norm: float) -> float:
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                    grad_norm)
        return float(total_norm)
