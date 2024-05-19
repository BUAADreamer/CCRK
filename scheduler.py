from torch.optim.lr_scheduler import LambdaLR
import math


def create_scheduler(args, optimizer):
    # base_warm_up = 2500
    # base_step = 2772 * 30  # math.ceil(2838361 / 1024)*30
    # base_rate = base_warm_up / base_step  # 0.03006253006253006
    if 'num_training_steps' not in args:
        args['num_training_steps'] = args['epochs'] * args['step_per_epoch']
    print("### num_training_steps, ", args['num_training_steps'], flush=True)

    if isinstance(args['num_warmup_steps'], float):
        assert 0 <= args['num_warmup_steps'] < 1
        args['num_warmup_steps'] = int(args['num_training_steps'] * args['num_warmup_steps'])
    # args['num_warmup_steps'] = int(args['num_training_steps'] * base_rate)
    print("### num_warmup_steps, ", args['num_warmup_steps'], flush=True)

    if args.sched == 'linear':
        class lr_lambda_class:
            def __init__(self):
                pass

            def __call__(self, current_step):
                if current_step < args.num_warmup_steps:
                    return float(current_step) / float(max(1, args.num_warmup_steps))
                return max(
                    0.0, float(args.num_training_steps - current_step) / float(
                        max(1, args.num_training_steps - args.num_warmup_steps))
                )

        # def lr_lambda(current_step: int):
        #     if current_step < args.num_warmup_steps:
        #         return float(current_step) / float(max(1, args.num_warmup_steps))
        #     return max(
        #         0.0, float(args.num_training_steps - current_step) / float(
        #             max(1, args.num_training_steps - args.num_warmup_steps))
        #     )

        lr_scheduler = LambdaLR(optimizer, lr_lambda_class(), last_epoch=-1)

    else:
        raise NotImplementedError(f"args.sched == {args.sched}")

    return lr_scheduler
