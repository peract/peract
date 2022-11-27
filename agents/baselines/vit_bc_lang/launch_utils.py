# Adapted from ARM
# Source: https://github.com/stepjam/ARM
# License: https://github.com/stepjam/ARM/LICENSE

import logging
from typing import List

import numpy as np
from omegaconf import DictConfig
from rlbench.backend.observation import Observation
from rlbench.observation_config import ObservationConfig
import rlbench.utils as rlbench_utils
from rlbench.demo import Demo
from yarr.replay_buffer.prioritized_replay_buffer import \
    PrioritizedReplayBuffer, ObservationElement
from yarr.replay_buffer.replay_buffer import ReplayElement, ReplayBuffer
from yarr.replay_buffer.uniform_replay_buffer import UniformReplayBuffer
from yarr.replay_buffer.task_uniform_replay_buffer import TaskUniformReplayBuffer

from helpers import demo_loading_utils, utils
from agents.baselines.vit_bc_lang.vit_bc_lang_agent import ViTBCLangAgent
from helpers.custom_rlbench_env import CustomRLBenchEnv
from helpers.network_utils import ViTLangAndFcsNet, ViT
from helpers.preprocess_agent import PreprocessAgent

import torch
from torch.multiprocessing import Process, Value, Manager
from helpers.clip.core.clip import build_model, load_clip, tokenize

LOW_DIM_SIZE = 4


def create_replay(batch_size: int, timesteps: int,
                  prioritisation: bool, task_uniform: bool,
                  save_dir: str, cameras: list,
                  image_size=[128, 128],
                  replay_size=3e5):
    lang_feat_dim = 1024

    # low_dim_state
    observation_elements = []
    observation_elements.append(
        ObservationElement('low_dim_state', (LOW_DIM_SIZE,), np.float32))

    # rgb, depth, point cloud, intrinsics, extrinsics
    for cname in cameras:
        observation_elements.append(
            ObservationElement('%s_rgb' % cname, (3, *image_size,), np.float32))
        observation_elements.append(
            ObservationElement('%s_point_cloud' % cname, (3, *image_size),
                               np.float32))  # see pyrep/objects/vision_sensor.py on how pointclouds are extracted from depth frames
        observation_elements.append(
            ObservationElement('%s_camera_extrinsics' % cname, (4, 4,), np.float32))
        observation_elements.append(
            ObservationElement('%s_camera_intrinsics' % cname, (3, 3,), np.float32))

    observation_elements.extend([
        ReplayElement('lang_goal_emb', (lang_feat_dim,),
                      np.float32),
        ReplayElement('task', (),
                      str),
        ReplayElement('lang_goal', (1,),
                      object),  # language goal string for debugging and visualization
    ])

    extra_replay_elements = [
        ReplayElement('demo', (), np.bool),
    ]

    replay_buffer = TaskUniformReplayBuffer(
        save_dir=save_dir,
        batch_size=batch_size,
        timesteps=timesteps,
        replay_capacity=int(replay_size),
        action_shape=(8,),
        action_dtype=np.float32,
        reward_shape=(),
        reward_dtype=np.float32,
        update_horizon=1,
        observation_elements=observation_elements,
        extra_replay_elements=extra_replay_elements
    )
    return replay_buffer


def _get_action(obs_tp1: Observation):
    quat = utils.normalize_quaternion(obs_tp1.gripper_pose[3:])
    if quat[-1] < 0:
        quat = -quat
    return np.concatenate([obs_tp1.gripper_pose[:3], quat,
                           [float(obs_tp1.gripper_open)]])


def _add_keypoints_to_replay(
        cfg: DictConfig,
        task: str,
        replay: ReplayBuffer,
        inital_obs: Observation,
        demo: Demo,
        episode_keypoints: List[int],
        cameras: List[str],
        description: str = '',
        clip_model = None,
        device = 'cpu'):
    prev_action = None
    obs = inital_obs
    all_actions = []
    for k, keypoint in enumerate(episode_keypoints):
        obs_tp1 = demo[keypoint]
        action = _get_action(obs_tp1)
        all_actions.append(action)
        terminal = (k == len(episode_keypoints) - 1)
        reward = float(terminal) if terminal else 0

        obs_dict = utils.extract_obs(obs, t=k, prev_action=prev_action,
                                     cameras=cameras, episode_length=cfg.rlbench.episode_length)
        del obs_dict['ignore_collisions']
        tokens = tokenize([description]).numpy()
        token_tensor = torch.from_numpy(tokens).to(device)
        lang_feats, lang_embs = clip_model.encode_text_with_embeddings(token_tensor)
        obs_dict['lang_goal_emb'] = lang_feats[0].float().detach().cpu().numpy()

        final_obs = {
            'task': task,
            'lang_goal': np.array([description], dtype=object),
        }

        prev_action = np.copy(action)
        others = {'demo': True}
        others.update(final_obs)
        others.update(obs_dict)
        timeout = False
        replay.add(action, reward, terminal, timeout, **others)
        obs = obs_tp1  # Set the next obs
    # Final step
    obs_dict_tp1 = utils.extract_obs(obs_tp1, t=k + 1, prev_action=prev_action,
                                     cameras=cameras, episode_length=cfg.rlbench.episode_length)
    obs_dict_tp1['lang_goal_emb'] = lang_feats[0].float().detach().cpu().numpy()
    # del obs_dict_tp1['lang_goal_tokens']
    del obs_dict_tp1['ignore_collisions']
    # obs_dict_tp1['task'] = task
    obs_dict_tp1.update(final_obs)
    replay.add_final(**obs_dict_tp1)
    return all_actions


def fill_replay(cfg: DictConfig,
                obs_config: ObservationConfig,
                rank: int,
                replay: ReplayBuffer,
                task: str,
                num_demos: int,
                demo_augmentation: bool,
                demo_augmentation_every_n: int,
                cameras: List[str],
                clip_model = None,
                device = 'cpu'):
    logging.getLogger().setLevel(cfg.framework.logging_level)

    if clip_model is None:
        model, _ = load_clip('RN50', jit=False, device=device)
        clip_model = build_model(model.state_dict())
        clip_model.to(device)
        del model

    logging.debug('Filling %s replay ...' % task)
    all_actions = []
    for d_idx in range(num_demos):
        # load demo from disk
        demo = rlbench_utils.get_stored_demos(
            amount=1, image_paths=False,
            dataset_root=cfg.rlbench.demo_path,
            variation_number=-1, task_name=task,
            obs_config=obs_config,
            random_selection=False,
            from_episode_number=d_idx)[0]

        descs = demo._observations[0].misc['descriptions']

        # extract keypoints (a.k.a keyframes)
        episode_keypoints = demo_loading_utils.keypoint_discovery(demo)

        if rank == 0:
            logging.info(f"Loading Demo({d_idx}) - found {len(episode_keypoints)} keypoints - {task}")

        for i in range(len(demo) - 1):
            if not demo_augmentation and i > 0:
                break
            if i % demo_augmentation_every_n != 0:
                continue

            obs = demo[i]
            desc = descs[0]
            # if our starting point is past one of the keypoints, then remove it
            while len(episode_keypoints) > 0 and i >= episode_keypoints[0]:
                episode_keypoints = episode_keypoints[1:]
            if len(episode_keypoints) == 0:
                break
            all_actions.extend(_add_keypoints_to_replay(
                cfg, task, replay, obs, demo, episode_keypoints, cameras,
                description=desc, clip_model=clip_model, device=device))
    logging.debug('Replay filled with demos.')
    return all_actions


def fill_multi_task_replay(cfg: DictConfig,
                           obs_config: ObservationConfig,
                           rank: int,
                           replay: ReplayBuffer,
                           tasks: List[str],
                           num_demos: int,
                           demo_augmentation: bool,
                           demo_augmentation_every_n: int,
                           cameras: List[str],
                           clip_model = None):
    manager = Manager()
    store = manager.dict()

    # create a MP dict for storing indicies
    # TODO(mohit): this shouldn't be initialized here
    del replay._task_idxs
    task_idxs = manager.dict()
    replay._task_idxs = task_idxs
    replay._create_storage(store)
    replay.add_count = Value('i', 0)

    # fill replay buffer in parallel across tasks
    max_parallel_processes = cfg.replay.max_parallel_processes
    processes = []
    n = np.arange(len(tasks))
    split_n = utils.split_list(n, max_parallel_processes)
    for split in split_n:
        for e_idx, task_idx in enumerate(split):
            task = tasks[int(task_idx)]
            model_device = torch.device('cuda:%s' % (e_idx % torch.cuda.device_count())
                                        if torch.cuda.is_available() else 'cpu')
            p = Process(target=fill_replay,
                        args=(cfg,
                              obs_config,
                              rank,
                              replay,
                              task,
                              num_demos,
                              demo_augmentation,
                              demo_augmentation_every_n,
                              cameras,
                              clip_model,
                              model_device))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    logging.debug('Replay filled with multi demos.')


def create_agent(camera_name: str,
                 activation: str,
                 lr: float,
                 weight_decay: float,
                 image_resolution: list,
                 grad_clip: float):

    vit = ViT(
        image_size=128,
        patch_size=8,
        num_classes=16,
        dim=64,
        depth=6,
        heads=8,
        mlp_dim=64,
        dropout=0.1,
        emb_dropout=0.1,
        channels=6,
    )

    actor_net = ViTLangAndFcsNet(
        vit=vit,
        input_resolution=image_resolution,
        filters=[64, 96, 128],
        kernel_sizes=[1, 1, 1],
        strides=[1, 1, 1],
        norm=None,
        activation=activation,
        fc_layers=[128, 64, 3 + 4 + 1],
        low_dim_state_len=LOW_DIM_SIZE)

    bc_agent = ViTBCLangAgent(
        actor_network=actor_net,
        camera_name=camera_name,
        lr=lr,
        weight_decay=weight_decay,
        grad_clip=grad_clip)

    return PreprocessAgent(pose_agent=bc_agent)
