import copy
import logging
import os
from typing import List

import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from yarr.agents.agent import Agent, ActResult, ScalarSummary, \
    HistogramSummary, ImageSummary, Summary

from helpers import utils
from helpers.utils import stack_on_channel

NAME = 'QAttentionAgent'
REPLAY_BETA = 1.0


class QFunction(nn.Module):

    def __init__(self,
                 unet: nn.Module):
        super(QFunction, self).__init__()
        self._qnet = copy.deepcopy(unet)
        self._qnet2 = copy.deepcopy(unet)
        self._qnet.build()
        self._qnet2.build()

    def _argmax_2d(self, tensor):
        t_shape = tensor.shape
        m = tensor.view(t_shape[0], -1).argmax(1).view(-1, 1)
        indices = torch.cat((m // t_shape[-1], m % t_shape[-1]), dim=1)
        return indices

    def forward(self, x, robot_state):
        q = self._qnet(x, robot_state)[:, 0]
        q2 = self._qnet2(x, robot_state)[:, 0]
        coords = self._argmax_2d(torch.min(q, q2))
        return q, q2, coords


class QAttentionAgent(Agent):

    def __init__(self,
                 pixel_unet: nn.Module,
                 camera_name: str,
                 tau: float = 0.005,
                 gamma: float = 0.99,
                 nstep: int = 1,
                 lr: float = 0.0001,
                 weight_decay: float = 1e-5,
                 lambda_qreg: float = 1e-6,
                 grad_clip: float = 20.,
                 include_low_dim_state: bool = False):
        self._pixel_unet = pixel_unet
        self._camera_name = camera_name
        self._tau = tau
        self._gamma = gamma
        self._nstep = nstep
        self._lr = lr
        self._weight_decay = weight_decay
        self._lambda_qreg = lambda_qreg
        self._grad_clip = grad_clip
        self._include_low_dim_state = include_low_dim_state

    def build(self, training: bool, device: torch.device = None):
        if device is None:
            device = torch.device('cpu')
        self._q = QFunction(self._pixel_unet).to(device).train(training)
        self._q_target = None
        if training:
            self._q_target = QFunction(self._pixel_unet).to(device).train(False)
            for p in self._q_target.parameters():
                p.requires_grad = False
            utils.soft_updates(self._q, self._q_target, 1.0)
            self._optimizer = torch.optim.Adam(
                self._q.parameters(), lr=self._lr,
                weight_decay=self._weight_decay)
            logging.info('# Q-attention Params: %d' % sum(
                p.numel() for p in self._q.parameters() if p.requires_grad))
        else:
            for p in self._q.parameters():
                p.requires_grad = False
        self._device = device

    def _get_q_from_pixel_coord(self, q, coord):
        b, h, w = q.shape
        flat_indicies = (coord[:, 0] * w + coord[:, 1])[:, None].long()
        return q.view(b, h * w).gather(1, flat_indicies)

    def _preprocess_inputs(self, replay_sample):
        observations = [
            stack_on_channel(replay_sample['%s_rgb' % self._camera_name]),
            stack_on_channel(
                replay_sample['%s_point_cloud' % self._camera_name])
        ]
        tp1_observations = [
            stack_on_channel(replay_sample['%s_rgb_tp1' % self._camera_name]),
            stack_on_channel(
                replay_sample['%s_point_cloud_tp1' % self._camera_name])
        ]
        return observations, tp1_observations

    def update(self, step: int, replay_sample: dict) -> dict:
        pixel_action = replay_sample['%s_pixel_coord' % self._camera_name][:, -1].int()
        reward = replay_sample['reward']
        reward = torch.where(reward > 0, reward, torch.zeros_like(reward))

        robot_state = robot_state_tp1 = None
        if self._include_low_dim_state:
            robot_state = stack_on_channel(replay_sample['low_dim_state'])
            robot_state_tp1 = stack_on_channel(replay_sample['low_dim_state_tp1'])

        # Don't want timeouts to be classed as terminals
        terminal = replay_sample['terminal'].float() - replay_sample['timeout'].float()

        obs, obs_tp1 = self._preprocess_inputs(replay_sample)
        q, q2, coords = self._q(obs, robot_state)

        with torch.no_grad():
            # (B, h, w)
            _, _, coords_tp1 = self._q(obs_tp1, robot_state_tp1)
            q_tp1_targ, q2_tp1_targ, _ = self._q_target(obs_tp1, robot_state_tp1)
            q_tp1_targ = torch.min(q_tp1_targ, q2_tp1_targ)
            q_tp1_targ = self._get_q_from_pixel_coord(q_tp1_targ, coords_tp1)
            target = reward.unsqueeze(1) + (self._gamma ** self._nstep) * (
                    1 - terminal.unsqueeze(1)) * q_tp1_targ
            target = torch.clamp(target, 0.0, 100.0)

        q_pred = self._get_q_from_pixel_coord(q, pixel_action)
        delta = F.smooth_l1_loss(q_pred, target, reduction='none').mean(1)

        delta += F.smooth_l1_loss(self._get_q_from_pixel_coord(q2, pixel_action), target, reduction='none').mean(1)
        q_reg = ((0.5 * torch.sum(q ** 2)) + (0.5 * torch.sum(q2 ** 2))) * self._lambda_qreg

        loss_weights = utils.loss_weights(replay_sample, REPLAY_BETA)
        total_loss = ((delta) * loss_weights).mean() + q_reg
        new_priority = ((delta) + 1e-10).sqrt()
        new_priority /= new_priority.max()

        self._summaries = {
            'losses/bellman': delta.mean(),
            'losses/qreg': q_reg.mean(),
            'q/mean': q.mean(),
            'q/action_q': q_pred.mean(),
        }
        self._qvalues = q[:1]
        self._rgb_observation = replay_sample['front_rgb'][0, -1]
        self._optimizer.zero_grad()
        total_loss.backward()
        if self._grad_clip is not None:
            nn.utils.clip_grad_value_(self._q.parameters(), self._grad_clip)
        self._optimizer.step()
        utils.soft_updates(self._q, self._q_target, self._tau)

        return {
            'priority': new_priority,
        }

    def act(self, step: int, observation: dict,
            deterministic=False) -> ActResult:
        with torch.no_grad():
            observations = [
                stack_on_channel(observation['%s_rgb' % self._camera_name]),
                stack_on_channel(observation['%s_point_cloud' % self._camera_name])
            ]
            robot_state = None
            if self._include_low_dim_state:
                robot_state = stack_on_channel(observation['low_dim_state'])
            # Coords are stored as (y, x)
            q, q2, coords = self._q(observations, robot_state)
            self._act_qvalues = torch.min(q, q2)[:1]
            self._rgb_observation = observation['front_rgb'][0, -1]
            return ActResult(
                coords[0],
                observation_elements={
                    '%s_pixel_coord' % self._camera_name: coords[0],
                },
                info={'q_values': self._act_qvalues}
            )

    @staticmethod
    def generate_heatmap(q_values, rgb_obs):
        norm_q = torch.clamp(q_values / 100.0, 0, 1)
        heatmap = torch.cat(
            [norm_q, torch.zeros_like(norm_q), torch.zeros_like(norm_q)])
        img = transforms.functional.to_pil_image(rgb_obs)
        h_img = transforms.functional.to_pil_image(heatmap).convert("RGB")
        ret = PIL.Image.blend(img, h_img, 0.75)
        return transforms.ToTensor()(ret).unsqueeze_(0)

    def update_summaries(self) -> List[Summary]:
        summaries = [
            ImageSummary('%s/Q' % NAME, QAttentionAgent.generate_heatmap(
                self._qvalues.cpu(), ((self._rgb_observation + 1) / 2.0).cpu()))
        ]
        for n, v in self._summaries.items():
            summaries.append(ScalarSummary('%s/%s' % (NAME, n), v))

        for tag, param in self._q.named_parameters():
            assert not torch.isnan(param.grad.abs() <= 1.0).all()
            summaries.append(
                HistogramSummary('%s/gradient/%s' % (NAME, tag), param.grad))
            summaries.append(
                HistogramSummary('%s/weight/%s' % (NAME, tag), param.data))
        return summaries

    def act_summaries(self) -> List[Summary]:
        return [
            ImageSummary('%s/Q_act' % NAME, QAttentionAgent.generate_heatmap(
                self._act_qvalues.cpu(), ((self._rgb_observation + 1) / 2.0).cpu()))
        ]

    def load_weights(self, savedir: str):
        self._q.load_state_dict(
            torch.load(os.path.join(savedir, 'pixel_agent_q.pt'),
                       map_location=torch.device('cpu')))

    def save_weights(self, savedir: str):
        torch.save(
            self._q.state_dict(), os.path.join(savedir, 'pixel_agent_q.pt'))
