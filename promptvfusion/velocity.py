"""Recover a moving feature from the last reliable same-modality velocity."""
import torch
from torch.nn import functional as F

from .candidate_screen import reliable
from prompt_gate.geometry import warp
from prompt_gate.streams import LANES
from .repair import RepairStream, cue_error


def project(source, displacement, valid):
    height, width = source.shape[-2:]
    y, x = torch.meshgrid(torch.arange(height, device=source.device),
                          torch.arange(width, device=source.device), indexing='ij')
    x = x + displacement[:, 0]
    y = y + displacement[:, 1]
    x0, y0 = x.floor().long(), y.floor().long()
    fx, fy = x - x0, y - y0
    batch, channels = source.shape[:2]
    source = source.flatten(2)
    total = source.new_zeros(batch, 1, height * width)
    fused = torch.zeros_like(source)
    for dx, dy, weight in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                           (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        sx, sy = x0 + dx, y0 + dy
        inside = (sx >= 0) & (sx < width) & (sy >= 0) & (sy < height)
        destination = (sy.clamp(0, height - 1) * width + sx.clamp(0, width - 1)).flatten(1)[:, None]
        amount = (weight * inside * valid[:, 0]).flatten(1)[:, None]
        total.scatter_add_(2, destination, amount)
        fused.scatter_add_(2, destination.expand(-1, channels, -1), source * amount)
    return (fused / total.clamp_min(1e-6)).reshape(batch, channels, height, width), total.reshape(batch, 1, height, width)


class VelocityStream(RepairStream):
    def __init__(self, backbone, prompt, structures, flow, calibration):
        super().__init__(backbone, prompt, structures, flow, calibration)
        self.velocity = {}

    def _read_history(self, lane, current_key, pair_key, pair_value, current_other, pair_flow, **kwargs):
        result = super()._read_history(lane, current_key, pair_key, pair_value, current_other, pair_flow, **kwargs)
        if 'recent_candidate' not in result:
            return result
        key, feature, _, _, chain_valid = self.recent_links[lane]
        age = self.index - result['recent_index']
        if age == 1:
            (_, older), (_, latest) = self.real_history[lane]
            backward = self.history_flow(latest, older)
            forward = self.history_flow(older, latest)
            moving = torch.linalg.vector_norm(backward, dim=1, keepdim=True) > 2
            support = reliable(backward, forward) & moving
            source_cue, inside = warp(self.current_cue, -backward)
            self.velocity[lane] = -backward, support & inside, torch.cat((feature, source_cue, latest), 1)
            return result

        velocity, support, source = self.velocity[lane]
        values, occupied = project(source, age * velocity, support)
        candidate, cue, raw_own = values.split((feature.shape[1], self.current_cue.shape[1],
                                               self.latest[lane][1].shape[1]), 1)
        difference = (self.current_cue - cue).abs().mean(1, keepdim=True)
        anchor_error = cue_error(self.current_cue, warp(self.pair_images[LANES[1 - LANES.index(lane)]], pair_flow)[0])
        accepted = ((occupied > .5) & ~chain_valid & (difference < .07)
                    & (difference <= anchor_error + .01) & (result['weight'] == 0))
        idx = LANES.index(lane)
        own, other = self.prompt.scales[idx], self.prompt.scales[1 - idx]
        reader_input = torch.cat((current_other / other, candidate / own), 1)
        candidate = candidate + own * self.prompt.readers[lane](reader_input).tanh()
        w = F.avg_pool2d(accepted.float(), 3, 1, 1) * accepted
        result['completed_features'] = torch.lerp(result['completed_features'], candidate, w)
        result['fallback_weight'] = torch.maximum(result['fallback_weight'], w)
        if lane == 'visible':
            result['direct_vis'] = torch.where(accepted, raw_own, result['direct_vis'])
        return result
