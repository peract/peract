import copy
import logging
from typing import List

import numpy as np
import torch
import torch.nn as nn
from rlbench.backend.observation import Observation
from rlbench.demo import Demo
from yarr.replay_buffer.prioritized_replay_buffer import \
    PrioritizedReplayBuffer, ObservationElement
from yarr.replay_buffer.replay_buffer import ReplayElement, ReplayBuffer
from yarr.replay_buffer.uniform_replay_buffer import UniformReplayBuffer

from helpers import demo_loading_utils, utils
from helpers.custom_rlbench_env import CustomRLBenchEnv
from helpers.network_utils import SiameseNet, DenseBlock, Conv2DBlock, \
    Conv2DUpsampleBlock
from helpers.preprocess_agent import PreprocessAgent
from agents.arm.next_best_pose_agent import NextBestPoseAgent
from agents.arm.qattention_agent import QAttentionAgent

REWARD_SCALE = 100.0


def create_replay(batch_size: int, timesteps: int, prioritisation: bool,
                  save_dir: str, cameras: list, env: CustomRLBenchEnv):
    observation_elements = env.observation_elements
    for cname in cameras:
        observation_elements.extend([
            ObservationElement('%s_pixel_coord' % cname, (2,), np.int32),
        ])

    replay_class = UniformReplayBuffer
    if prioritisation:
        replay_class = PrioritizedReplayBuffer
    replay_buffer = replay_class(
        save_dir=save_dir,
        batch_size=batch_size,
        timesteps=timesteps,
        replay_capacity=int(1e5),
        action_shape=(8,),
        action_dtype=np.float32,
        reward_shape=(),
        reward_dtype=np.float32,
        update_horizon=1,
        observation_elements=observation_elements,
        extra_replay_elements=[ReplayElement('demo', (), np.bool)]
    )
    return replay_buffer


def _point_to_pixel_index(
        point: np.ndarray,
        extrinsics: np.ndarray,
        intrinsics: np.ndarray):
    point = np.array([point[0], point[1], point[2], 1])
    world_to_cam = np.linalg.inv(extrinsics)
    point_in_cam_frame = world_to_cam.dot(point)
    px, py, pz = point_in_cam_frame[:3]
    px = 2 * intrinsics[0, 2] - int(-intrinsics[0, 0] * (px / pz) + intrinsics[0, 2])
    py = 2 * intrinsics[1, 2] - int(-intrinsics[1, 1] * (py / pz) + intrinsics[1, 2])
    return px, py


def _get_action(obs_tp1: Observation):
    quat = utils.normalize_quaternion(obs_tp1.gripper_pose[3:])
    if quat[-1] < 0:
        quat = -quat
    return np.concatenate([obs_tp1.gripper_pose[:3], quat,
                           [float(obs_tp1.gripper_open)]])


def _add_keypoints_to_replay(
        replay: ReplayBuffer,
        inital_obs: Observation,
        demo: Demo,
        env: CustomRLBenchEnv,
        episode_keypoints: List[int],
        cameras: List[str]):
    prev_action = None
    obs = inital_obs
    all_actions = []
    for k, keypoint in enumerate(episode_keypoints):
        obs_tp1 = demo[keypoint]
        action = _get_action(obs_tp1)
        all_actions.append(action)
        terminal = (k == len(episode_keypoints) - 1)
        reward = float(terminal) * REWARD_SCALE if terminal else 0
        obs_dict = env.extract_obs(obs, t=k, prev_action=prev_action)
        prev_action = np.copy(action)
        others = {'demo': True}
        final_obs = {}
        for name in cameras:
            px, py = _point_to_pixel_index(
                obs_tp1.gripper_pose[:3],
                obs_tp1.misc['%s_camera_extrinsics' % name],
                obs_tp1.misc['%s_camera_intrinsics' % name])
            final_obs['%s_pixel_coord' % name] = [py, px]
        others.update(final_obs)
        others.update(obs_dict)
        timeout = False
        replay.add(action, reward, terminal, timeout, **others)
        obs = obs_tp1  # Set the next obs
    # Final step
    obs_dict_tp1 = env.extract_obs(
        obs_tp1, t=k + 1, prev_action=prev_action)
    obs_dict_tp1.update(final_obs)
    replay.add_final(**obs_dict_tp1)
    return all_actions


def fill_replay(replay: ReplayBuffer,
                task: str,
                env: CustomRLBenchEnv,
                num_demos: int,
                demo_augmentation: bool,
                demo_augmentation_every_n: int,
                cameras: List[str]):
    logging.info('Filling replay with demos...')
    all_actions = []
    for d_idx in range(num_demos):
        demo = env.env.get_demos(
            task, 1, variation_number=0, random_selection=False,
            from_episode_number=d_idx)[0]
        episode_keypoints = demo_loading_utils.keypoint_discovery(demo)

        for i in range(len(demo) - 1):
            if not demo_augmentation and i > 0:
                break
            if i % demo_augmentation_every_n != 0:
                continue
            obs = demo[i]
            # If our starting point is past one of the keypoints, then remove it
            while len(episode_keypoints) > 0 and i >= episode_keypoints[0]:
                episode_keypoints = episode_keypoints[1:]
            if len(episode_keypoints) == 0:
                break
            all_actions.extend(_add_keypoints_to_replay(
                replay, obs, demo, env, episode_keypoints, cameras))
    logging.info('Replay filled with demos.')
    return all_actions


class SharedNet(nn.Module):

    def __init__(self,
                 activation: str,
                 norm: str = None):
        super(SharedNet, self).__init__()
        self._activation = activation
        self._norm = norm

    def build(self):
        self._rgb_pre = nn.Sequential(
            Conv2DBlock(3, 32, 3, 1, activation=self._activation, norm=self._norm),
        )
        self._pcd_pre = nn.Sequential(
            Conv2DBlock(3, 32, 3, 1, activation=self._activation, norm=self._norm),
        )

    def forward(self, observations):
        x_rgb, x_pcd = self._rgb_pre(observations[0]), self._pcd_pre(observations[1])
        x = torch.cat([x_rgb, x_pcd], dim=1)
        return x


class ActorNet(nn.Module):

    def __init__(self,
                 activation: str,
                 low_dim_size: int,
                 norm: str = None):
        super(ActorNet, self).__init__()
        self._activation = activation
        self._low_dim_size = low_dim_size
        self._norm = norm

    def build(self):
        self._convs = nn.Sequential(
            Conv2DBlock(64 + self._low_dim_size, 64, 1, 1, activation=self._activation, norm=self._norm),
            Conv2DBlock(64, 64, 3, 1, activation=self._activation, norm=self._norm),
        )
        self._fcs = nn.Sequential(
           DenseBlock(64, 64, activation=self._activation),
           DenseBlock(64, 64, activation=self._activation),
           DenseBlock(64, 8*2),
        )
        self._maxp = nn.AdaptiveMaxPool2d(1)

    def forward(self, observation_feats, low_dim_ins):
        low_dim_feats = low_dim_ins
        _, _, h, w = observation_feats.shape
        low_dim_feats = low_dim_feats.unsqueeze(
            -1).unsqueeze(-1).repeat(1, 1, h, w)
        x = torch.cat([observation_feats, low_dim_feats], dim=1)
        x = self._convs(x)
        x = self._maxp(x).squeeze(-1).squeeze(-1)
        x = self._fcs(x)
        return x


class CriticNet(nn.Module):

    def __init__(self,
                 activation: str,
                 low_dim_size: int,
                 norm: str = None,
                 q_conf: bool = True):
        super(CriticNet, self).__init__()
        self._activation = activation
        self._low_dim_size = low_dim_size
        self._norm = norm
        self._q_conf = q_conf

    def build(self):
        self._convs = nn.Sequential(
            Conv2DBlock(64 + self._low_dim_size, 128, 3, 1, self._norm, self._activation),
            Conv2DBlock(128, 128, 3, 1, self._norm, self._activation),
            Conv2DBlock(128, 128, 3, 1, self._norm, self._activation),
            Conv2DBlock(128, 128, 3, 1, self._norm, self._activation)
        )
        if self._q_conf:
            self._final_conv = Conv2DBlock(128, 2, 3, 1)
        else:
            self._maxp = nn.AdaptiveMaxPool2d(1)
            self._fcs = nn.Sequential(
                DenseBlock(128, 64, activation=self._activation),
                DenseBlock(64, 1),
            )

    def forward(self, observation_feats, low_dim_ins):
        low_dim_feats = low_dim_ins
        _, _, h, w = observation_feats.shape
        low_dim_feats = low_dim_feats.unsqueeze(
            -1).unsqueeze(-1).repeat(1, 1, h, w)
        x = torch.cat([observation_feats, low_dim_feats], dim=1)
        x = self._convs(x)
        if self._q_conf:
            x = self._final_conv(x)
            x[:, 1] = torch.sigmoid(x[:, 1])
        else:
            x = self._maxp(x).squeeze(-1).squeeze(-1)
            x = self._fcs(x)
        return x


class Qattention2DNet(nn.Module):

    def __init__(self,
                 siamese_net: SiameseNet,
                 filters: List[int],
                 kernel_sizes: List[int],
                 strides: List[int],
                 low_dim_state_len: int,
                 norm: str = None,
                 activation: str = 'relu',
                 output_channels: int = 1,
                 skip_connections: bool = True):
        super(Qattention2DNet, self).__init__()
        self._siamese_net = copy.deepcopy(siamese_net)
        self._input_channels = self._siamese_net.output_channels + low_dim_state_len
        self._filters = filters
        self._kernel_sizes = kernel_sizes
        self._strides = strides
        self._norm = norm
        self._activation = activation
        self._output_channels = output_channels
        self._skip_connections = skip_connections
        self._build_calls = 0

    def build(self):

        self._build_calls += 1
        if self._build_calls != 1:
            raise RuntimeError('Build needs to be called once.')
        self._siamese_net.build()
        self._down = []
        ch = self._input_channels
        for filt, ksize, stride in zip(
                self._filters, self._kernel_sizes, self._strides):
            conv_block = Conv2DBlock(
                ch, filt, ksize, stride, self._norm, self._activation,
                padding_mode='replicate')
            ch = filt
            self._down.append(conv_block)
        self._down = nn.ModuleList(self._down)

        reverse_conv_data = list(zip(self._filters, self._kernel_sizes,
                                     self._strides))
        reverse_conv_data.reverse()

        self._up = []
        for i, (filt, ksize, stride) in enumerate(reverse_conv_data):
            if i > 0 and self._skip_connections:
                ch += reverse_conv_data[-i-1][0]
            convt_block = Conv2DUpsampleBlock(
                ch, filt, ksize, stride, self._norm, self._activation)
            ch = filt
            self._up.append(convt_block)
        self._up = nn.ModuleList(self._up)

        self._final_conv = Conv2DBlock(ch, self._output_channels, 3, 1,
                                       padding_mode='replicate')

    def forward(self, observations, low_dim_ins):
        x = self._siamese_net(observations)
        _, _, h, w = x.shape
        if low_dim_ins is not None:
            low_dim_latents = low_dim_ins.unsqueeze(
                -1).unsqueeze(-1).repeat(1, 1, h, w)
            x = torch.cat([x, low_dim_latents], dim=1)
        self.ups = []
        self.downs = []
        layers_for_skip = []
        for l in self._down:
            x = l(x)
            layers_for_skip.append(x)
            self.downs.append(x)
        self.latent = x
        layers_for_skip.reverse()
        for i, l in enumerate(self._up):
            if i > 0 and self._skip_connections:
                # Skip connections. Skip the first up layer.
                x = torch.cat([layers_for_skip[i], x], 1)
            x = l(x)
            self.ups.append(x)
        x = self._final_conv(x)
        return x


def create_agent(camera_name: str,
                 activation: str,
                 q_conf: bool,
                 action_min_max,
                 alpha,
                 alpha_lr,
                 alpha_auto_tune,
                 critic_lr,
                 actor_lr,
                 next_best_pose_critic_weight_decay,
                 next_best_pose_actor_weight_decay,
                 crop_shape,
                 next_best_pose_tau,
                 next_best_pose_critic_grad_clip,
                 next_best_pose_actor_grad_clip,
                 qattention_tau,
                 qattention_lr,
                 qattention_weight_decay,
                 qattention_lambda_qreg,
                 low_dim_state_len,
                 qattention_grad_clip,
                 ):

    siamese_net = SiameseNet(
        input_channels=[3, 3],
        filters=[8],
        kernel_sizes=[5],
        strides=[1],
        activation=activation,
        norm=None,
    )
    qattention_net = Qattention2DNet(
        siamese_net=siamese_net,
        filters=[16, 16],
        kernel_sizes=[5, 5],
        strides=[2, 2],
        output_channels=1,
        norm=None,
        activation=activation,
        skip_connections=True,
        low_dim_state_len=0)

    qattention_agent = QAttentionAgent(
        pixel_unet=qattention_net,
        tau=qattention_tau,
        camera_name=camera_name,
        lr=qattention_lr,
        weight_decay=qattention_weight_decay,
        lambda_qreg=qattention_lambda_qreg,
        include_low_dim_state=False,
        grad_clip=qattention_grad_clip)

    shared_net = SharedNet(activation, norm='layer')
    critic_net = CriticNet(activation, low_dim_state_len + 8,
                           norm='layer', q_conf=q_conf)
    actor_net = ActorNet(activation, low_dim_state_len)

    next_best_pose_agent = NextBestPoseAgent(
        qattention_agent=qattention_agent,
        shared_network=shared_net,
        critic_network=critic_net,
        actor_network=actor_net,
        action_min_max=action_min_max,
        camera_name=camera_name,
        alpha=alpha,
        alpha_lr=alpha_lr,
        alpha_auto_tune=alpha_auto_tune,
        critic_lr=critic_lr,
        actor_lr=actor_lr,
        critic_weight_decay=next_best_pose_critic_weight_decay,
        actor_weight_decay=next_best_pose_actor_weight_decay,
        crop_shape=crop_shape,
        critic_tau=next_best_pose_tau,
        critic_grad_clip=next_best_pose_critic_grad_clip,
        actor_grad_clip=next_best_pose_actor_grad_clip,
        q_conf=q_conf)

    return PreprocessAgent(pose_agent=next_best_pose_agent)
