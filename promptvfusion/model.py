"""Learn motion-aware missing-feature corrections before video fusion."""
import torch
from torch import nn
from torch.nn import functional as F

from .candidate_screen import reliable
from prompt_gate.geometry import warp
from prompt_gate.streams import LANES
from .repair import RepairStream
from .velocity import project


class MotionCorrection(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.input = nn.Conv2d(4 * dim + 2, dim, 1)
        self.spatial = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1, groups=dim), nn.SiLU(),
                                     nn.Conv2d(dim, dim, 3, padding=3, dilation=3, groups=dim), nn.SiLU())
        self.output = nn.Conv2d(dim, dim, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, base, proposal, cue, mismatch, confidence, available, own_scale, cue_scale):
        prior = torch.where(available, (proposal - base) / own_scale, 0.)
        context = torch.cat((base / own_scale, prior, cue / cue_scale, mismatch / cue_scale,
                             confidence * available, available.to(base.dtype)), dim=1)
        hidden = F.silu(self.input(context))
        return base + own_scale * self.output(hidden + self.spatial(hidden))


class MotionGuidedStream(RepairStream):
    def __init__(self, backbone, prompt, structures, flow, calibration, *, use_correction=True):
        super().__init__(backbone, prompt, structures, flow, calibration)
        self.use_correction = use_correction
        self.motion_prior = {}

    def _read_history(self, lane, current_key, pair_key, pair_value, current_other, pair_flow, **kwargs):
        result = super()._read_history(lane, current_key, pair_key, pair_value, current_other, pair_flow, **kwargs)
        if not self.use_correction:
            return result

        idx = LANES.index(lane)
        cue = LANES[1 - idx]
        recent_index, latest = self.latest[lane]
        age = self.index - recent_index
        base = result['completed_features']
        if age == 1:
            self.motion_prior.pop(lane, None)
            history = self.real_history[lane]
            if len(history) == 2 and history[0][0] == recent_index - 1:
                older = history[0][1]
                backward = self.history_flow(latest, older)
                reverse = self.history_flow(older, latest)
                speed = torch.linalg.vector_norm(backward, dim=1, keepdim=True)
                # Threshold speed in native-equivalent pixels at both train and test sizes.
                valid = reliable(backward, reverse) & (speed > 2 * latest.shape[-1] / 640)
                pair_is_recent = self.rollout.pair['source_index'] == recent_index
                if pair_is_recent:
                    source_cue = self.pair_images[cue]
                    source_feature = pair_value
                else:
                    source_cue, inside = warp(self.current_cue, -backward)
                    valid &= inside
                    source_feature = self.recent_links[lane][1]
                self.motion_prior[lane] = -backward, valid, torch.cat((source_feature, source_cue), 1)

        available = torch.zeros_like(base[:, :1], dtype=torch.bool)
        confidence = torch.zeros_like(base[:, :1])
        candidate = base
        if lane in self.motion_prior:
            velocity, valid, source = self.motion_prior[lane]
            warped, coverage = project(source, age * velocity, valid)
            candidate, projected_cue = warped.split((base.shape[1], self.current_cue.shape[1]), dim=1)
            error = (projected_cue - self.current_cue).abs().mean(dim=1, keepdim=True)
            available = coverage > .5
            confidence = 1 - error

        anchor_cue = F.grid_sample(kwargs['pair_other_value'], result['sampling_grid'],
                                   padding_mode='border', align_corners=True)
        result['completed_features'] = self.prompt.motion_correction[lane](
            base, candidate, current_other, current_other - anchor_cue, confidence, available,
            self.prompt.scales[idx], self.prompt.scales[1 - idx])
        return result

    def detach_state(self):
        super().detach_state()
        self.motion_prior = {lane: (motion, valid, source.detach())
                             for lane, (motion, valid, source) in self.motion_prior.items()}
        self.recent_links = {lane: (key, feature.detach(), proxy, mapping, valid)
                             for lane, (key, feature, proxy, mapping, valid) in self.recent_links.items()}
