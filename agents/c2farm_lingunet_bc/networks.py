import torch
import torch.nn as nn

from helpers.network_utils import Conv3DInceptionBlock, DenseBlock, SpatialSoftmax3D, \
    Conv3DInceptionBlockUpsampleBlock, Conv3DBlock


class QattentionLingU3DNet(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 out_dense: int,
                 voxel_size: int,
                 low_dim_size: int,
                 kernels: int,
                 norm: str = None,
                 activation: str = 'relu',
                 dense_feats: int = 32,
                 include_prev_layer = False,
                 depth = 0,
                 lingunet_dropout = 0.0,):
        super(QattentionLingU3DNet, self).__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._norm = norm
        self._activation = activation
        self._kernels = kernels
        self._low_dim_size = low_dim_size
        self._build_calls = 0
        self._voxel_size = voxel_size
        self._dense_feats = dense_feats
        self._out_dense = out_dense
        self._include_prev_layer = include_prev_layer
        self._depth = depth

        self._lingunet_dropout = lingunet_dropout
        self._clip_lang_feat_dim = 1024

        if self._voxel_size < 16:
            raise Exception("Voxel size for C2FARM_LINGUNET_BC should be at least 16 or higher")

    def build(self):
        use_residual = False
        self._build_calls += 1
        if self._build_calls != 1:
            raise RuntimeError('Build needs to be called once.')

        spatial_size = self._voxel_size
        self._input_preprocess = Conv3DInceptionBlock(
            self._in_channels, self._kernels, norm=self._norm,
            activation=self._activation)

        d0_ins = self._input_preprocess.out_channels
        if self._include_prev_layer:
            PREV_VOXEL_CHANNELS = 0
            d0_ins += self._input_preprocess.out_channels * self._depth

        if self._low_dim_size > 0:
            self._proprio_preprocess = DenseBlock(
                self._low_dim_size, self._kernels, None, self._activation)
            d0_ins += self._kernels

        self._down0 = Conv3DInceptionBlock(
            d0_ins, self._kernels, norm=self._norm,
            activation=self._activation, residual=use_residual)
        self._ss0 = SpatialSoftmax3D(
            spatial_size, spatial_size, spatial_size,
            self._down0.out_channels)
        spatial_size //= 2
        self._down1 = Conv3DInceptionBlock(
            self._down0.out_channels, self._kernels * 2, norm=self._norm,
            activation=self._activation, residual=use_residual)
        self._ss1 = SpatialSoftmax3D(
            spatial_size, spatial_size, spatial_size,
            self._down1.out_channels)
        spatial_size //= 2

        flat_size = self._down0.out_channels * 4 + self._down1.out_channels * 4

        k1 = self._down1.out_channels
        if self._voxel_size > 8:
            k1 += self._kernels
            self._down2 = Conv3DInceptionBlock(
                self._down1.out_channels, self._kernels * 4, norm=self._norm,
                activation=self._activation,  residual=use_residual)
            self._lang_proj2 = DenseBlock(
                self._clip_lang_feat_dim, self._down2.out_channels, None, None)
            self._dropout2 = nn.Dropout(self._lingunet_dropout)
            flat_size += self._down2.out_channels * 4
            self._ss2 = SpatialSoftmax3D(
                spatial_size, spatial_size, spatial_size,
                self._down2.out_channels)
            spatial_size //= 2
            k2 = self._down2.out_channels
            if self._voxel_size > 16:
                k2 *= 2
                self._down3 = Conv3DInceptionBlock(
                    self._down2.out_channels, self._kernels, norm=self._norm,
                    activation=self._activation, residual=use_residual)
                self._lang_proj3 = DenseBlock(
                    self._clip_lang_feat_dim, self._down3.out_channels, None, None)
                self._dropout3 = nn.Dropout(self._lingunet_dropout)
                flat_size += self._down3.out_channels * 4
                self._ss3 = SpatialSoftmax3D(
                    spatial_size, spatial_size, spatial_size,
                    self._down3.out_channels)
                self._up3 = Conv3DInceptionBlockUpsampleBlock(
                    self._kernels, self._kernels * 4, 2, norm=self._norm,
                    activation=self._activation, residual=use_residual)
            self._up2 = Conv3DInceptionBlockUpsampleBlock(
                k2, self._kernels, 2, norm=self._norm,
                activation=self._activation, residual=use_residual)

        self._up1 = Conv3DInceptionBlockUpsampleBlock(
            k1, self._kernels, 2, norm=self._norm,
            activation=self._activation, residual=use_residual)

        self._global_maxp = nn.AdaptiveMaxPool3d(1)
        self._local_maxp = nn.MaxPool3d(3, 2, padding=1)
        self._final = Conv3DBlock(
            self._kernels * 2, self._kernels, kernel_sizes=3,
            strides=1, norm=self._norm, activation=self._activation)
        self._final2 = Conv3DBlock(
            self._kernels, self._out_channels, kernel_sizes=3,
            strides=1, norm=None, activation=None)

        self._ss_final = SpatialSoftmax3D(
            self._voxel_size, self._voxel_size, self._voxel_size,
            self._kernels)
        flat_size += self._kernels * 4

        if self._out_dense > 0:
            self._dense0 = DenseBlock(
                flat_size, self._dense_feats, None, self._activation)
            self._dense1 = DenseBlock(
                self._dense_feats, self._dense_feats, None, self._activation)
            self._dense2 = DenseBlock(
                self._dense_feats, self._out_dense, None, None)

    def _proj_feature(self, x, spatial_size, proj_fn):
        x = proj_fn(x)
        x = x.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        x = x.repeat(1, 1, spatial_size, spatial_size, spatial_size)
        return x

    def forward(self, ins, proprio,
                lang_goal_embs, lang_token_embs,
                bounds, prev_bounds,
                prev_layer_voxel_grid):
        b, _, d, h, w = ins.shape
        x = self._input_preprocess(ins)

        if self._include_prev_layer:
            for voxel_grid in prev_layer_voxel_grid:
                y = self._input_preprocess(voxel_grid)
                x = torch.cat([x, y], dim=1)

        if self._low_dim_size > 0:
            p = self._proprio_preprocess(proprio)
            p = p.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(
                1, 1, d, h, w)
            x = torch.cat([x, p], dim=1)

        l_feat = lang_goal_embs
        l_feat = l_feat.to(dtype=x.dtype)

        d0 = self._down0(x)
        # l0 = self._proj_feature(l_feat, d0.shape[-1], self._lang_proj0)
        # d0 = self._dropout0(d0 * l0)
        ss0 = self._ss0(d0)
        maxp0 = self._global_maxp(d0).view(b, -1)

        d1 = u = self._down1(self._local_maxp(d0))
        # l1 = self._proj_feature(l_feat, d1.shape[-1], self._lang_proj1)
        # d1 = self._dropout1(d1 * l1)
        ss1 = self._ss1(d1)
        maxp1 = self._global_maxp(d1).view(b, -1)

        feats = [ss0, maxp0, ss1, maxp1]

        if self._voxel_size > 8:
            d2 = u = self._down2(self._local_maxp(d1))
            l2 = self._proj_feature(l_feat, d2.shape[-1], self._lang_proj2)
            d2 = self._dropout2(d2 * l2)
            feats.extend([self._ss2(d2), self._global_maxp(d2).view(b, -1)])
            if self._voxel_size > 16:
                d3 = self._down3(self._local_maxp(d2))
                l3 = self._proj_feature(l_feat, d3.shape[-1], self._lang_proj3)
                d3 = self._dropout3(d3 * l3)
                feats.extend([self._ss3(d3), self._global_maxp(d3).view(b, -1)])
                u3 = self._up3(d3)
                u = torch.cat([d2, u3], dim=1)
            u2 = self._up2(u)
            u = torch.cat([d1, u2], dim=1)

        u1 = self._up1(u)
        f1 = self._final(torch.cat([d0, u1], dim=1))
        trans = self._final2(f1)

        feats.extend([self._ss_final(f1), self._global_maxp(f1).view(b, -1)])

        self.latent_dict = {
            'd0': d0.mean(-1).mean(-1).mean(-1),
            'd1': d1.mean(-1).mean(-1).mean(-1),
            'u1': u1.mean(-1).mean(-1).mean(-1),
            'trans_out': trans,
        }

        rot_and_grip_out, collision_out = None, None
        if self._out_dense > 0:
            dense0 = self._dense0(torch.cat(feats, 1))
            dense1 = self._dense1(dense0)
            rot_and_grip_collision_out = self._dense2(dense1)
            rot_and_grip_out = rot_and_grip_collision_out[:, :-2]
            collision_out = rot_and_grip_collision_out[:, -2:]
            self.latent_dict.update({
                'dense0': dense0,
                'dense1': dense1,
                'dense2': rot_and_grip_collision_out,
            })

        if self._voxel_size > 8:
            self.latent_dict.update({
                'd2': d2.mean(-1).mean(-1).mean(-1),
                'u2': u2.mean(-1).mean(-1).mean(-1),
            })
        if self._voxel_size > 16:
            self.latent_dict.update({
                'd3': d3.mean(-1).mean(-1).mean(-1),
                'u3': u3.mean(-1).mean(-1).mean(-1),
            })

        return trans, rot_and_grip_out, collision_out