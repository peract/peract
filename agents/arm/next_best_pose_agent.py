import copy
import logging
import os
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from yarr.agents.agent import Agent, Summary, ActResult, \
    ScalarSummary, ImageSummary, HistogramSummary

from helpers import utils
from helpers.utils import stack_on_channel
from agents.arm.qattention_agent import QAttentionAgent

NAME = 'NextBestPoseAgent'
LOG_STD_MAX = 4
LOG_STD_MIN = -40
REPLAY_ALPHA = 0.7
REPLAY_BETA = 0.5


class QFunction(nn.Module):

    def __init__(self, critic: nn.Module, shared: nn.Module, q_conf: bool):
        super(QFunction, self).__init__()
        self._q_conf = q_conf
        self._q1 = copy.deepcopy(critic)
        self._q2 = copy.deepcopy(critic)
        self.shared = copy.deepcopy(shared)
        self._q1.build()
        self._q2.build()
        self.shared.build()

    def forward(self, observations, robot_state, action):
        obs_feats = self.shared(observations)
        combined = torch.cat([robot_state, action.float()], dim=1)
        q1 = self._q1(obs_feats, combined)
        q2 = self._q2(obs_feats, combined)
        if self._q_conf:
            b = q1.shape[0]
            q1 = q1.view(b, 2, -1)
            q2 = q2.view(b, 2, -1)
            q1v, q1c = q1[:, 0], q1[:, 1]
            q1_best = q1v.gather(1, q1c.argmax(dim=1).unsqueeze(-1))
            q2v, q2c = q2[:, 0], q2[:, 1]
            q2_best = q2v.gather(1, q2c.argmax(dim=1).unsqueeze(-1))
            return q1, q2, q1_best, q2_best
        else:
            q1, q2 = q1.unsqueeze(1), q2.unsqueeze(1)
            return q1, q2, q1, q2


class Actor(nn.Module):

    def __init__(self, actor_network: nn.Module, action_min_max: torch.tensor):
        super(Actor, self).__init__()
        self._action_min_max = action_min_max
        self._actor_network = copy.deepcopy(actor_network)
        self._actor_network.build()

    def _rescale_actions(self, x):
        return (0.5 * (x + 1.) * (
                self._action_min_max[1] - self._action_min_max[0]) +
                self._action_min_max[0])

    def _normalize(self, x):
        return x / x.square().sum(dim=1).sqrt().unsqueeze(-1)

    def _gaussian_logprob(self, noise, log_std):
        residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
        return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)

    def forward(self, observations, robot_state):
        mu_and_logstd = self._actor_network(observations, robot_state)
        mu, log_std = torch.split(mu_and_logstd, 8, dim=1)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        std = log_std.exp()
        noise = torch.randn_like(mu)
        pi = mu + noise * std
        log_pi = self._gaussian_logprob(noise, log_std)
        mu = torch.tanh(mu)
        pi = torch.tanh(pi)
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)

        pi = self._rescale_actions(pi)
        mu = self._rescale_actions(mu)

        pi = torch.cat(
            [pi[:, :3], self._normalize(pi[:, 3:7]), pi[:, 7:]], dim=-1)
        mu = torch.cat(
            [mu[:, :3], self._normalize(mu[:, 3:7]), mu[:, 7:]], dim=-1)
        return mu, pi, log_pi, log_std


class NextBestPoseAgent(Agent):

    def __init__(self,
                 qattention_agent: QAttentionAgent,
                 shared_network: nn.Module,
                 critic_network: nn.Module,
                 actor_network: nn.Module,
                 action_min_max: tuple,
                 camera_name: str,
                 alpha: float = 0.2,
                 alpha_auto_tune: bool = True,
                 alpha_lr: float = 0.001,
                 critic_lr: float = 0.01,
                 actor_lr: float = 0.01,
                 critic_weight_decay: float = 1e-5,
                 actor_weight_decay: float = 1e-5,
                 crop_shape: tuple = (16, 16),
                 critic_tau: float = 0.005,
                 critic_grad_clip: float = 20.0,
                 actor_grad_clip: float = 20.0,
                 gamma: float = 0.99,
                 nstep: int = 1,
                 q_conf: bool = True):
        self._qattention_agent = qattention_agent
        self._alpha = alpha
        self._alpha_auto_tune = alpha_auto_tune
        self._crop_shape = crop_shape
        self._critic_tau = critic_tau
        self._critic_grad_clip = critic_grad_clip
        self._actor_grad_clip = actor_grad_clip
        self._camera_name = camera_name
        self._gamma = gamma
        self._nstep = nstep
        self._target_entropy = -8
        self._shared_network = shared_network
        self._critic_network = critic_network
        self._actor_network = actor_network
        self._action_min_max = action_min_max
        self._critic_lr = critic_lr
        self._actor_lr = actor_lr
        self._alpha_lr = alpha_lr
        self._critic_weight_decay = critic_weight_decay
        self._actor_weight_decay = actor_weight_decay
        self._q_conf = q_conf
        self._crop_augmentation = False

    def build(self, training: bool, device: torch.device = None):
        if device is None:
            device = torch.device('cpu')
        self._qattention_agent.build(training, device)
        action_min_max = torch.tensor(self._action_min_max).to(device)
        self._actor = Actor(self._actor_network, action_min_max).to(
            device).train(training)

        self._action_min_max_t = torch.tensor(self._action_min_max).to(device)

        grid_for_crop = torch.arange(
            0, self._crop_shape[0], device=device).unsqueeze(0).repeat(
            self._crop_shape[0], 1).unsqueeze(-1)
        self._grid_for_crop = torch.cat([grid_for_crop.transpose(1, 0),
                                         grid_for_crop], dim=2).unsqueeze(0)
        self._q = QFunction(self._critic_network, self._shared_network, self._q_conf).to(
            device).train(training)
        if training:
            self._q_target = QFunction(self._critic_network, self._shared_network, self._q_conf).to(device).train(False)
            utils.soft_updates(self._q, self._q_target, 1.0)

            self._crop_shape_t = torch.tensor(
                [list(self._crop_shape)], dtype=torch.int32, device=device)

            # Freeze target critic.
            for p in self._q_target.parameters():
                p.requires_grad = False

            self._log_alpha = 0
            if self._alpha_auto_tune:
                self._log_alpha = torch.tensor(
                    (np.log(self._alpha)), dtype=torch.float,
                    requires_grad=True, device=device)
                if training:
                    self._alpha_optimizer = torch.optim.Adam(
                        [self._log_alpha], lr=self._alpha_lr)

            self._critic_optimizer = torch.optim.Adam(
                self._q.parameters(), lr=self._critic_lr,
                weight_decay=self._critic_weight_decay)
            self._actor_optimizer = torch.optim.Adam(
                self._actor.parameters(), lr=self._actor_lr,
                weight_decay=self._actor_weight_decay)

            logging.info('# NBP Critic Params: %d' % sum(
                p.numel() for p in self._q.parameters() if p.requires_grad))
            logging.info('# NBP Actor Params: %d' % sum(
                p.numel() for p in self._actor.parameters() if p.requires_grad))
        else:
            for p in self._actor.parameters():
                p.requires_grad = False

        self._device = device

    @property
    def alpha(self):
        return self._log_alpha.exp() if self._alpha_auto_tune else self._alpha

    def _extract_crop(self, pixel_action, observation):
        # Pixel action will now be (B, 2)
        observation = stack_on_channel(observation)
        h = observation.shape[-1]
        top_left_corner = torch.clamp(
            pixel_action - self._crop_shape[0] // 2, 0, h - self._crop_shape[1])
        grid = self._grid_for_crop + top_left_corner.unsqueeze(1).unsqueeze(1)
        grid = ((grid / float(h)) * 2.0) - 1.0
        grid = torch.cat((grid[:, :, :, 1:2], grid[:, :, :, 0:1]), dim=-1)
        crop = F.grid_sample(observation, grid, mode='nearest',
                             align_corners=True)
        return crop

    def _preprocess_inputs(self, replay_sample, pixel_action, pixel_action_tp1):
        observations = [
            self._extract_crop(pixel_action,
                               replay_sample['%s_rgb' % self._camera_name]),
            self._extract_crop(pixel_action, replay_sample[
                '%s_point_cloud' % self._camera_name]),
        ]
        tp1_observations = [
            self._extract_crop(pixel_action_tp1,
                               replay_sample['%s_rgb_tp1' % self._camera_name]),
            self._extract_crop(pixel_action_tp1, replay_sample[
                '%s_point_cloud_tp1' % self._camera_name]),
        ]
        return observations, tp1_observations

    def _clip_action(self, a):
        return torch.min(torch.max(a, self._action_min_max_t[0:1]),
                         self._action_min_max_t[1:2])

    def _update_critic(self, replay_sample: dict) -> None:
        action = replay_sample['action']
        reward = replay_sample['reward']

        robot_state = stack_on_channel(replay_sample['low_dim_state'][:, -1:])
        robot_state_tp1 = stack_on_channel(
            replay_sample['low_dim_state_tp1'][:, -1:])

        # Get last of time stack and first of plan stack
        pixel_action = replay_sample[
                           '%s_pixel_coord' % self._camera_name][:, -1]
        pixel_action_tp1 = replay_sample[
                               '%s_pixel_coord_tp1' % self._camera_name][:, -1]

        if self._crop_augmentation:
            shifted = ((torch.rand_like(pixel_action.float())
                        * self._crop_shape_t).int() - self._crop_shape_t // 2
                       ) * replay_sample['demo'].int().unsqueeze(1)
            pixel_action += shifted
            pixel_action_tp1 += shifted

        # Don't want timeouts to be classed as terminals
        terminal = replay_sample['terminal'].float() - replay_sample['timeout'].float()

        observations, tp1_observations = self._preprocess_inputs(
            replay_sample, pixel_action, pixel_action_tp1)

        q1, q2, _, _ = self._q(observations, robot_state, action)

        with torch.no_grad():
            obs_feats = self._q.shared(tp1_observations)
            _, pi_tp1, logp_pi_tp1, _ = self._actor(
                obs_feats, robot_state_tp1)

            q1_pi_tp1_targ, q2_pi_tp1_targ, _, _ = self._q_target(
                tp1_observations, robot_state_tp1, pi_tp1)

            min_q_pi_targ = torch.min(q1_pi_tp1_targ[:, 0], q2_pi_tp1_targ[:, 0])
            next_value = (min_q_pi_targ - self.alpha * logp_pi_tp1)
            q_backup = (reward.unsqueeze(-1) + (
                        self._gamma ** self._nstep) * (
                            1. - terminal.unsqueeze(-1)) * next_value)

        loss_weights = utils.loss_weights(replay_sample, REPLAY_BETA)

        self._critic_summaries = {}
        if self._q_conf:
            w = 1.0
            q1_delta = F.smooth_l1_loss(q1[:, 0], q_backup, reduction='none') * q1[:, 1] - w * q1[:, 1].log()
            q2_delta = F.smooth_l1_loss(q2[:, 0], q_backup, reduction='none') * q2[:, 1] - w * q2[:, 1].log()
            self._critic_summaries = {
                'q_conf_loss': -(w * q1[:, 1].log()).mean(),
                'q_conf_mean': q1[:, 1].mean(),
            }
        else:
            q1_delta = F.smooth_l1_loss(q1[:, 0], q_backup, reduction='none')
            q2_delta = F.smooth_l1_loss(q2[:, 0], q_backup, reduction='none')

        q1_delta, q2_delta = q1_delta.mean(1), q2_delta.mean(1)
        q1_bellman_loss = (q1_delta * loss_weights).mean()
        q2_bellman_loss = (q2_delta * loss_weights).mean()

        critic_loss = q1_bellman_loss + q2_bellman_loss

        self._critic_summaries.update({
            'q1_bellman_loss': q1_bellman_loss,
            'q2_bellman_loss': q2_bellman_loss,
            'q1_mean': q1[:, 0].mean().item(),
            'q2_mean': q2[:, 0].mean().item(),
            'alpha': self.alpha,
        })
        self._crop_summary = observations
        self._crop_summary_tp1 = tp1_observations

        new_pri = torch.sqrt((q1_delta + q2_delta) / 2. + 1e-10)
        self._new_priority = (new_pri / torch.max(new_pri)).detach()
        self._grad_step(critic_loss, self._critic_optimizer,
                        self._q.parameters(), self._critic_grad_clip)

    def _update_actor(self, replay_sample: dict) -> None:

        robot_state = stack_on_channel(replay_sample['low_dim_state'][:, -1:])
        pixel_action = replay_sample['%s_pixel_coord' % self._camera_name][:, -1]

        if self._crop_augmentation:
            shifted = ((torch.rand_like(pixel_action.float())
                        * self._crop_shape_t).int() - self._crop_shape_t // 2
                       ) * replay_sample['demo'].int().unsqueeze(1)
            pixel_action += shifted

        # Crop the observations
        observations = [
            self._extract_crop(pixel_action,
                               replay_sample['%s_rgb' % self._camera_name]),
            self._extract_crop(pixel_action, replay_sample[
                '%s_point_cloud' % self._camera_name]),
        ]

        with torch.no_grad():
            obs_feats = self._q.shared(observations)

        mu, pi, self._logp_pi, log_scale_diag = self._actor(
            obs_feats, robot_state)

        _, _, q1_pi, q2_pi = self._q(observations, robot_state, pi)

        min_q_pi = torch.min(q1_pi, q2_pi)[:, 0]
        pi_loss = (self.alpha * self._logp_pi - min_q_pi)

        loss_weights = utils.loss_weights(replay_sample, REPLAY_BETA)
        pi_loss = (pi_loss * loss_weights).mean()

        self._actor_summaries = {
            'pi/loss': pi_loss,
            'pi/q1_pi_mean': q1_pi.mean(),
            'pi/q2_pi_mean': q2_pi.mean(),
            'pi/mu': mu.mean(),
            'pi/pi': pi.mean(),
            'pi/log_pi': self._logp_pi.mean(),
            'pi/log_scale_diag': log_scale_diag.mean()
        }
        self._grad_step(pi_loss, self._actor_optimizer,
                        self._actor.parameters(), self._actor_grad_clip)

    def _update_alpha(self):
        alpha_loss = -(self.alpha * (
                self._logp_pi + self._target_entropy).detach()).mean()
        self._grad_step(alpha_loss, self._alpha_optimizer)

    def _grad_step(self, loss, opt, model_params=None, clip=None):
        opt.zero_grad()
        loss.backward()
        if clip is not None and model_params is not None:
            nn.utils.clip_grad_value_(model_params, clip)
        opt.step()

    def update(self, step: int, replay_sample: dict) -> dict:
        info = self._qattention_agent.update(step, replay_sample)

        self._update_critic(replay_sample)

        # Freeze critic so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self._q.parameters():
            p.requires_grad = False

        self._update_actor(replay_sample)
        if self._alpha_auto_tune:
            self._update_alpha()

        # UnFreeze critic.
        for p in self._q.parameters():
            p.requires_grad = True

        utils.soft_updates(self._q, self._q_target, self._critic_tau)
        pixel_agent_priority = info['priority']
        return {
            'priority': ((self._new_priority +
                         pixel_agent_priority) / 2.0) ** REPLAY_ALPHA
        }

    def act(self, step: int, observation: dict,
            deterministic=False) -> ActResult:
        with torch.no_grad():
            act_res = self._qattention_agent.act(step, observation, deterministic)
            observations = [
                self._extract_crop(
                    act_res.action.unsqueeze(0),
                    observation['%s_rgb' % self._camera_name]),
                self._extract_crop(
                    act_res.action.unsqueeze(0),
                    observation['%s_point_cloud' % self._camera_name]),
            ]
            self._act_crop_summaries = observations
            robot_state = stack_on_channel(observation['low_dim_state'][:, -1:])
            obs_feats = self._q.shared(observations)
            mu, pi, _, _ = self._actor(obs_feats, robot_state)
            act_res.action = (mu if deterministic else pi)[0]
            act_res.info.update({
                'rgb_crop': observations[0]
            })
            return act_res

    def update_summaries(self) -> List[Summary]:

        summaries = [
            ImageSummary('%s/crops/rgb' % NAME,
                         (self._crop_summary[0] + 1.0) / 2.0),
            ImageSummary('%s/crops/point_cloud' % NAME,
                         self._crop_summary[1]),
            ImageSummary('%s/crops_tp1/rgb' % NAME,
                         (self._crop_summary_tp1[0] + 1.0) / 2.0),
            ImageSummary('%s/crops_tp1/point_cloud' % NAME,
                         self._crop_summary_tp1[1]),
        ]

        for n, v in list(self._critic_summaries.items()) + list(
                self._actor_summaries.items()):
            summaries.append(ScalarSummary('%s/%s' % (NAME, n), v))

        for tag, param in list(self._q.named_parameters()) + list(
                self._actor.named_parameters()):
            summaries.append(
                HistogramSummary('%s/gradient/%s' % (NAME, tag), param.grad))
            summaries.append(
                HistogramSummary('%s/weight/%s' % (NAME, tag), param.data))

        pixel_summaries = self._qattention_agent.update_summaries()
        return pixel_summaries + summaries

    def act_summaries(self) -> List[Summary]:
        summaries = [
            ImageSummary('%s/crops/act/rgb' % NAME,
                         (self._act_crop_summaries[0] + 1.0) / 2.0),
            ImageSummary('%s/crops/act/point_cloud' % NAME,
                         self._act_crop_summaries[1]),
        ]
        return summaries + self._qattention_agent.act_summaries()

    def load_weights(self, savedir: str):
        self._qattention_agent.load_weights(savedir)
        self._actor.load_state_dict(
            torch.load(os.path.join(savedir, 'pose_actor.pt'),
                       map_location=torch.device('cpu')))
        self._q.load_state_dict(
            torch.load(os.path.join(savedir, 'pose_q.pt'),
                       map_location=torch.device('cpu')))

    def save_weights(self, savedir: str):
        self._qattention_agent.save_weights(savedir)
        torch.save(self._actor.state_dict(),
                   os.path.join(savedir, 'pose_actor.pt'))
        torch.save(self._q.state_dict(),
                   os.path.join(savedir, 'pose_q.pt'))
