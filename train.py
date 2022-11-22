import logging
import os
import sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from typing import List

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf, ListConfig

import run_seed_fn
from helpers.utils import create_obs_config

import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')
from torch.multiprocessing import set_start_method, get_start_method

try:
    if get_start_method() != 'spawn':
        set_start_method('spawn', force=True)
except RuntimeError:
    print("Could not set start method to spawn")
    pass


@hydra.main(config_name='config', config_path='conf')
def main(cfg: DictConfig) -> None:
    cfg_yaml = OmegaConf.to_yaml(cfg)
    logging.info('\n' + cfg_yaml)

    os.environ['MASTER_ADDR'] = cfg.ddp.master_addr
    os.environ['MASTER_PORT'] = cfg.ddp.master_port

    cfg.rlbench.cameras = cfg.rlbench.cameras \
        if isinstance(cfg.rlbench.cameras, ListConfig) else [cfg.rlbench.cameras]
    obs_config = create_obs_config(cfg.rlbench.cameras,
                                   cfg.rlbench.camera_resolution,
                                   cfg.method.name)
    multi_task = len(cfg.rlbench.tasks) > 1

    cwd = os.getcwd()
    logging.info('CWD:' + os.getcwd())

    if cfg.framework.start_seed >= 0:
        # seed specified
        start_seed = cfg.framework.start_seed
    elif cfg.framework.start_seed == -1 and \
            len(list(filter(lambda x: 'seed' in x, os.listdir(cwd)))) > 0:
        # unspecified seed; use largest existing seed plus one
        largest_seed =  max([int(n.replace('seed', ''))
                             for n in list(filter(lambda x: 'seed' in x, os.listdir(cwd)))])
        start_seed = largest_seed + 1
    else:
        # start with seed 0
        start_seed = 0

    seed_folder = os.path.join(os.getcwd(), 'seed%d' % start_seed)
    os.makedirs(seed_folder, exist_ok=True)

    with open(os.path.join(seed_folder, 'config.yaml'), 'w') as f:
        f.write(cfg_yaml)

    # check if previous checkpoints already exceed the number of desired training iterations
    # if so, exit the script
    weights_folder = os.path.join(seed_folder, 'weights')
    if os.path.isdir(weights_folder) and len(os.listdir(weights_folder)) > 0:
        weights = os.listdir(weights_folder)
        latest_weight = sorted(map(int, weights))[-1]
        if latest_weight >= cfg.framework.training_iterations:
            logging.info('Agent was already trained for %d iterations. Exiting.' % latest_weight)
            sys.exit(0)

    # run train jobs with multiple seeds (sequentially)
    for seed in range(start_seed, start_seed + cfg.framework.seeds):
        logging.info('Starting seed %d.' % seed)

        world_size = cfg.ddp.num_devices
        mp.spawn(run_seed_fn.run_seed,
                 args=(cfg,
                       obs_config,
                       cfg.rlbench.cameras,
                       multi_task,
                       seed,
                       world_size,),
                 nprocs=world_size,
                 join=True)


if __name__ == '__main__':
    main()
