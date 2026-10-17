# Regression test for https://github.com/peract/peract/issues/24
#
# VoxelGrid registers all of its buffers with the default persistent=True and
# several of them are shaped by `batch_size` (_ones_max_coords, _flat_output,
# _tiled_batch_indices, _index_grid). save_weights therefore persists
# batch-size-dependent buffers, and loading such a checkpoint into a model
# built with a different batch size (e.g. eval.py after training with
# batch_size > 1) fails with:
#     RuntimeError: Error(s) in loading state_dict for QFunction:
#         size mismatch for _voxelizer._ones_max_coords: ...
#
# The buffers are fully derived from the constructor arguments, so they do not
# need to be persisted (register_buffer(..., persistent=False)); checkpoints
# become batch-size-agnostic and both agents' load_weights already skip
# checkpoint-only `_voxelizer.*` keys, so old checkpoints keep working.

import os
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from voxel.voxel_grid import VoxelGrid  # noqa: E402

COORD_BOUNDS = [-0.3, -0.5, 0.6, 0.7, 0.5, 1.6]
VOXEL_SIZE = 20  # small grid to keep the test light
FEATURE_SIZE = 3
MAX_NUM_COORDS = 64


class _QFunctionStub(nn.Module):
    """Mirrors how the agents embed the voxelizer inside the Q network."""

    def __init__(self, batch_size: int):
        super().__init__()
        self._voxelizer = VoxelGrid(coord_bounds=COORD_BOUNDS,
                                    voxel_size=VOXEL_SIZE,
                                    device='cpu',
                                    batch_size=batch_size,
                                    feature_size=FEATURE_SIZE,
                                    max_num_coords=MAX_NUM_COORDS)
        # stand-in for the real learned weights; voxel output features =
        # 4 + feature_size (voxel grid) + 3 (appended index grid)
        self._head = nn.Linear(4 + FEATURE_SIZE + 3, 2)

    def forward(self, coords):
        # mirrors the agents' call: 3 coord dims + feature_size + 1 occupancy
        features = torch.rand(*coords.shape[:-1], FEATURE_SIZE)
        return self._head(self._voxelizer.coords_to_bounding_voxel_grid(
            coords, coord_features=features))


def _random_coords(batch_size: int, num_coords: int = MAX_NUM_COORDS):
    lo = torch.tensor(COORD_BOUNDS[0:3], dtype=torch.float)
    hi = torch.tensor(COORD_BOUNDS[3:6], dtype=torch.float)
    coords = lo + (hi - lo) * torch.rand(batch_size, num_coords, 3)
    return coords


def _agent_style_load(target: nn.Module, checkpoint_state_dict: dict):
    """Reproduces agents/*/load_weights: merge checkpoint keys into the
    current model's state_dict, then load strictly."""
    merged_state_dict = target.state_dict()
    for k, v in checkpoint_state_dict.items():
        if k in merged_state_dict:
            merged_state_dict[k] = v
    target.load_state_dict(merged_state_dict)


class TestVoxelGridPersistentBuffers(unittest.TestCase):

    def test_state_dict_has_no_batch_size_dependent_buffers(self):
        """Buffers are reconstruction artifacts; none should be persisted."""
        sd = _QFunctionStub(batch_size=2).state_dict()
        voxel_keys = [k for k in sd if k.startswith('_voxelizer.')]
        self.assertEqual(
            voxel_keys, [],
            'VoxelGrid buffers must not be persisted into state_dict; '
            'got %d persistent buffer keys: %s' % (len(voxel_keys), voxel_keys))

    def test_eval_batch_size_one_loads_checkpoint_trained_with_batch_size_two(self):
        """The exact #24 scenario: train with batch_size: 2, eval with 1."""
        trained = _QFunctionStub(batch_size=2)
        with torch.no_grad():
            trained._head.weight.fill_(0.5)
            trained._head.bias.fill_(-0.25)

        checkpoint = trained.state_dict()  # what save_weights would write
        eval_model = _QFunctionStub(batch_size=1)  # eval.py builds batch_size: 1

        _agent_style_load(eval_model, checkpoint)  # must not raise

        # learned weights actually transferred
        self.assertTrue(torch.allclose(eval_model._head.weight,
                                       torch.full_like(eval_model._head.weight, 0.5)))
        self.assertTrue(torch.allclose(eval_model._head.bias,
                                       torch.full_like(eval_model._head.bias, -0.25)))

        # forward still works on the eval batch size
        out = eval_model(_random_coords(1))
        self.assertEqual(out.shape[0], 1)

    def test_old_checkpoint_with_voxelizer_keys_still_loads(self):
        """Backward compat: checkpoints saved by the old code contain
        `_voxelizer.*` buffer keys; agents' load_weights skip keys that are
        not in the current model, so they must keep loading."""
        legacy = _QFunctionStub(batch_size=2)
        checkpoint = dict(legacy.state_dict())
        # synthesize an old-format checkpoint by adding back the (now
        # non-persistent) voxelizer buffers from the model's live buffers
        for name, buf in legacy._voxelizer.named_buffers():
            checkpoint['_voxelizer.%s' % name] = buf.clone()
        self.assertIn('_voxelizer._index_grid', checkpoint)

        target = _QFunctionStub(batch_size=1)
        merged_state_dict = target.state_dict()
        for k, v in checkpoint.items():
            if k in merged_state_dict:  # voxelizer keys are not -> skipped
                merged_state_dict[k] = v
        target.load_state_dict(merged_state_dict)  # must not raise

    def test_buffers_still_functional_after_becoming_non_persistent(self):
        for batch_size in (1, 2):
            model = _QFunctionStub(batch_size=batch_size)
            out = model(_random_coords(batch_size))
            self.assertEqual(out.shape[0], batch_size)
            # buffers exist as buffers even though they are not persisted
            names = dict(model._voxelizer.named_buffers()).keys()
            self.assertIn('_ones_max_coords', names)
            self.assertIn('_index_grid', names)


if __name__ == '__main__':
    unittest.main()
