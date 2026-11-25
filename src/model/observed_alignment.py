"""Frozen, real-observation transport to the current frame's coordinates.

The missing lane's feature source is always a synchronized, both-real key.
Independent last-real indices are tracked separately; they never form a
synthetic paired key. Preparing a transport does not advance causal state.
"""
from collections.abc import Mapping

import torch
from torch import nn
import torch.nn.functional as F


MODALITIES = ('visible', 'infrared')


def _observation(image, *, batched, channels):
    shape_ok = (isinstance(image, torch.Tensor)
                and image.ndim == (4 if batched else 3))
    if not shape_ok:
        raise ValueError('A real observation must be a CHW or single-frame BCHW tensor')
    channel_axis = 1 if batched else 0
    if ((batched and image.shape[0] != 1)
            or image.shape[channel_axis] not in channels
            or min(image.shape[-2:]) < 2
            or not image.is_floating_point()
            or not torch.isfinite(image).all()
            or (image < 0).any() or (image > 1).any()):
        raise ValueError('Real observations must be finite floating images in [0,1]')


def _signature(tensor):
    return (tuple(tensor.shape[-2:]), tensor.device, tensor.dtype)


def _flow_tensor(flow, reference):
    if (not isinstance(flow, torch.Tensor)
            or tuple(flow.shape) != (1, 2, *reference.shape[-2:])
            or not flow.is_floating_point() or not torch.isfinite(flow).all()
            or flow.device != reference.device or flow.dtype != reference.dtype):
        raise ValueError('Flow must be finite native-grid B2HW with matching dtype/device')


def _backward_warp(source, flow):
    """Output (x,y) samples source at (x+flow_x,y+flow_y), border-padded.

    This is the same current-to-source convention as utils.flow_warp. A typed
    grid keeps the observation/feature dtype contract explicit here.
    """
    height, width = source.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.arange(height, device=source.device, dtype=source.dtype),
        torch.arange(width, device=source.device, dtype=source.dtype), indexing='ij')
    sample_x, sample_y = xx[None] + flow[:, 0], yy[None] + flow[:, 1]
    grid = torch.stack((2 * sample_x / (width - 1) - 1,
                        2 * sample_y / (height - 1) - 1), dim=-1)
    warped = F.grid_sample(source, grid, mode='bilinear',
                           padding_mode='border', align_corners=True)
    inbounds = ((sample_x >= 0) & (sample_x <= width - 1)
                & (sample_y >= 0) & (sample_y <= height - 1))[:, None]
    return warped, inbounds


class FrozenObservedFlow(nn.Module):
    """SEA-RAFT adapter for two real, single-frame BCHW observations in [0,1].

    IR is repeated to three channels. The network sees 0..255 and runs frozen
    in test mode. Native x/y displacement scales use actual resized width and
    height separately, including rounding of a noninteger resize.
    """

    def __init__(self, network, max_res=320):
        super().__init__()
        if type(max_res) is not int or max_res <= 0:
            raise ValueError('max_res must be a positive integer')
        self.network = network.eval().requires_grad_(False)
        self.max_res = max_res

    def train(self, mode=True):
        super().train(mode)
        self.network.eval()
        return self

    @torch.no_grad()
    def forward(self, current, key):
        _observation(current, batched=True, channels=(1, 3))
        _observation(key, batched=True, channels=(1, 3))
        if _signature(current) != _signature(key):
            raise ValueError('Current/key observation geometry, dtype and device must agree')
        self.network.eval()
        height, width = current.shape[-2:]
        scale = min(1., self.max_res / max(height, width))
        resized = (max(2, round(height * scale)), max(2, round(width * scale)))
        parameter = next(self.network.parameters(), None)
        network_dtype = parameter.dtype if parameter is not None else current.dtype
        images = []
        for image in (current, key):
            if image.shape[1] == 1:
                image = image.repeat(1, 3, 1, 1)
            image = image.to(dtype=network_dtype)
            if resized != (height, width):
                image = F.interpolate(image, size=resized, mode='bilinear', align_corners=False)
            images.append(image * 255.)
        output = self.network(images[0], images[1], test_mode=True)
        if not isinstance(output, Mapping) or 'final' not in output:
            raise ValueError('SEA-RAFT must return a final displacement tensor')
        flow = output['final']
        _flow_tensor(flow, images[0])
        if resized != (height, width):
            flow = F.interpolate(flow, size=(height, width), mode='bilinear', align_corners=False)
            factors = flow.new_tensor([width / resized[1], height / resized[0]])
            flow = flow * factors[None, :, None, None]
        flow = flow.to(dtype=current.dtype).detach().clone()
        _flow_tensor(flow, current)
        return flow


class PairedKeyAligner(nn.Module):
    """Causal synchronized real key, independent target ages, and raw transport.

    Reliability is source-inbounds and
      ||f + warp(b,f)|| <= 0.5 + 0.01 * (||f|| + ||warp(b,f)||).
    All norms are Euclidean pixel displacements. The diagnostic exposes this
    mask without replacing unreliable warped pixels with held features.
    """

    def __init__(self, flow, dim=32):
        super().__init__()
        if type(dim) is not int or dim <= 0:
            raise ValueError('dim must be a positive integer')
        self.flow = flow.eval().requires_grad_(False)
        self.dim = dim
        self.reset()

    def train(self, mode=True):
        super().train(mode)
        self.flow.eval()
        return self

    def reset(self):
        self.next_index = 0
        self.keyframe_index = None
        self.key_images, self.key_features = {}, {}
        self.latest_real_indices = {}
        self.geometry = None

    def _inputs(self, index, observed, available_mask):
        if type(index) is not int or index < 0 or index != self.next_index:
            raise ValueError('A sequence must start at zero and advance contiguously')
        try:
            mask = torch.as_tensor(available_mask)
        except (TypeError, RuntimeError, ValueError) as error:
            raise ValueError('An observation mask must be binary [visible,infrared]') from error
        if (mask.shape != (2,) or not ((mask == 0) | (mask == 1)).all()
                or not mask.any()):
            raise ValueError('An observation mask needs at least one real modality')
        if not isinstance(observed, Mapping):
            raise ValueError('observed must map available modalities to real images')
        flags = tuple(bool(value) for value in mask)
        images = {}
        for present, name in zip(flags, MODALITIES):
            if not present:
                continue  # Masked placeholders may be poison objects; never inspect them.
            if name not in observed:
                raise ValueError('An available real observation is absent')
            image = observed[name]
            _observation(image, batched=False, channels=(3,) if name == 'visible' else (1, 3))
            images[name] = image
        geometry = _signature(next(iter(images.values())))
        if any(_signature(image) != geometry for image in images.values()):
            raise ValueError('Available observations must share geometry, device and dtype')
        if self.geometry is not None and self.geometry != geometry:
            raise ValueError('Video geometry, dtype or device changed without reset')
        if self.keyframe_index is None and not all(flags):
            raise ValueError('Missing transport requires an earlier synchronized both-real key')
        return flags, images, geometry

    @torch.no_grad()
    def prepare(self, index, observed, available_mask):
        flags, images, _ = self._inputs(index, observed, available_mask)
        if all(flags):
            return None
        missing = MODALITIES[flags.index(False)]
        cue = MODALITIES[flags.index(True)]
        current = images[cue].detach().clone()[None]
        key = self.key_images[cue].detach().clone()[None]
        self.flow.eval()
        forward = self.flow(current.clone(), key.clone())
        _flow_tensor(forward, current)
        # The estimator may reuse an output buffer on its reverse call.
        forward = forward.detach().clone()
        backward = self.flow(key.clone(), current.clone())
        _flow_tensor(backward, current)
        backward = backward.detach().clone()
        transported, valid = _backward_warp(self.key_features[missing], forward)
        sampled_reverse, _ = _backward_warp(backward, forward)
        cycle = torch.linalg.vector_norm(forward + sampled_reverse, dim=1, keepdim=True)
        threshold = .5 + .01 * (
            torch.linalg.vector_norm(forward, dim=1, keepdim=True)
            + torch.linalg.vector_norm(sampled_reverse, dim=1, keepdim=True))
        cue_warped, _ = _backward_warp(key, forward)
        return {
            'warped_features': transported.detach().clone(),
            'held_key_features': self.key_features[missing].detach().clone(),
            'cue_key_features': self.key_features[cue].detach().clone(),
            'cue_key_image': key[0].detach().clone(),
            'cue_warped_image': cue_warped[0].detach().clone(),
            'flow_current_to_key': forward,
            'flow_key_to_current': backward,
            'valid_mask': valid.detach().clone(),
            'reliable_mask': (valid & (cycle <= threshold)).detach().clone(),
            'cycle_error': cycle.detach().clone(),
            'missing_modality': missing,
            'cue_modality': cue,
            'keyframe_index': self.keyframe_index,
            'keyframe_age': index - self.keyframe_index,
            'latest_target_index': self.latest_real_indices[missing],
            'latest_target_age': index - self.latest_real_indices[missing],
        }

    @torch.no_grad()
    def commit(self, index, observed, available_mask, real_features):
        flags, images, geometry = self._inputs(index, observed, available_mask)
        if not isinstance(real_features, Mapping):
            raise ValueError('real_features must map observed modalities to BCHW features')
        next_images, next_features = {}, {}
        # Validate and copy everything before mutating any causal state.
        for present, name in zip(flags, MODALITIES):
            if not present:
                continue
            if name not in real_features:
                raise ValueError('An observed modality has no real encoded feature')
            feature = real_features[name]
            if (not isinstance(feature, torch.Tensor)
                    or tuple(feature.shape) != (1, self.dim, *images[name].shape[-2:])
                    or not feature.is_floating_point() or not torch.isfinite(feature).all()
                    or _signature(feature) != geometry):
                raise ValueError('Real features must match the full-resolution feature space')
            next_images[name] = images[name].detach().clone()
            next_features[name] = feature.detach().clone()
        next_indices = dict(self.latest_real_indices)
        next_indices.update({name: index for name in next_features})
        if all(flags):
            self.key_images, self.key_features = next_images, next_features
            self.keyframe_index = index
        self.latest_real_indices = next_indices
        self.geometry = geometry
        self.next_index = index + 1
