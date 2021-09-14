import torch

# Local imports
from train_net import train
from test_net import test

from utils.parser import load_config, parse_args
import utils.multiprocessing as multiprocessing

def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)

    # Perform training.
    if cfg.TRAIN.ENABLE:
        if cfg.NUM_GPUS > 1:
            torch.multiprocessing.spawn(
                multiprocessing.run,
                nprocs=cfg.NUM_GPUS,
                args=(
                    cfg.NUM_GPUS,
                    train,
                    args.init_method,
                    cfg.SHARD_ID,
                    cfg.NUM_SHARDS,
                    cfg.DIST_BACKEND,
                    cfg,
                ),
                daemon=False,
            )
        else:
            train(cfg=cfg)

    # Perform testing.
    if cfg.TEST.ENABLE:
        if cfg.NUM_GPUS > 1:
            torch.multiprocessing.spawn(
                multiprocessing.run,
                nprocs=cfg.NUM_GPUS,
                args=(
                    cfg.NUM_GPUS,
                    test,
                    args.init_method,
                    cfg.SHARD_ID,
                    cfg.NUM_SHARDS,
                    cfg.DIST_BACKEND,
                    cfg,
                ),
                daemon=False,
            )
        else:
            test(cfg=cfg)

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("forkserver")
    main()
