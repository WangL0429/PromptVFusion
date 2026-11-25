"""Recover failed feature transport and restore VIS chroma in a causal stream."""
import torch
from torch.nn import functional as F

from .candidate_screen import reliable
from prompt_gate.geometry import warp
from prompt_gate.streams import LANES
from .short_step import advance_mapping
from .stream import ScoringStream


def luminance(rgb):
    weights = rgb.new_tensor([.299, .587, .114]).view(1, 3, 1, 1)
    return (rgb * weights).sum(1, keepdim=True)


def preserve_luma(fused, visible, weight):
    base = fused.clamp(0, 1)
    y = luminance(base)
    chroma = torch.lerp(base - y, visible - luminance(visible), weight)
    extent = chroma.abs().clamp_min(1e-8)
    limits = torch.where(chroma > 0, 1 - y, y) / extent
    return (y + chroma * limits.amin(1, keepdim=True).clamp(max=1)).clamp(0, 1)


def motion_fallback(link, reverse, initial, initial_valid, chain_valid, cue_error):
    mapping, valid = advance_mapping(link, reverse, initial, initial_valid)
    return mapping, valid & ~chain_valid & (cue_error < .07)


def cue_error(current, aligned):
    return F.avg_pool2d((current - aligned).abs().mean(1, keepdim=True), 5, 1, 2)


class RepairStream(ScoringStream):
    def __init__(self, backbone, prompt, structures, flow, calibration):
        super().__init__(backbone, prompt, structures, flow, calibration)
        self.gap_anchors = {}

    def _read_history(self, lane, current_key, pair_key, pair_value, current_other, pair_flow, **kwargs):
        result = super()._read_history(lane, current_key, pair_key, pair_value, current_other, pair_flow, **kwargs)
        flow = self.history_flow
        cue = LANES[1 - LANES.index(lane)]
        fallback = torch.zeros_like(result['recent_weight'])
        if 'recent_candidate' in result:
            key, feature, _, chain_mapping, chain_valid = self.recent_links[lane]
            if key != self.gap_anchors.get(lane, (None,))[0]:
                self.gap_anchors[lane] = key, self.current_cue, chain_mapping, chain_valid
            else:
                _, gap_cue, initial, initial_valid = self.gap_anchors[lane]
                direct = flow(self.current_cue, gap_cue)
                reverse = flow(gap_cue, self.current_cue)
                matched, _ = warp(gap_cue, direct)
                mapping, valid = motion_fallback(direct, reverse, initial, initial_valid, chain_valid,
                                                 cue_error(self.current_cue, matched))
                valid &= result['weight'] == 0
                fallback = F.avg_pool2d(valid.float(), 3, 1, 1) * valid
                warped, _ = warp(feature, mapping)
                idx = LANES.index(lane)
                own, other = self.prompt.scales[idx], self.prompt.scales[1 - idx]
                reader_input = torch.cat((current_other / other, warped / own), 1)
                candidate = warped + own * self.prompt.readers[lane](reader_input).tanh()
                result['completed_features'] = torch.lerp(result['completed_features'], candidate, fallback)
                if lane == 'visible':
                    result['direct_vis'] = warp(self.latest[lane][1], mapping)[0]
            if lane == 'visible':
                result['recent_vis'] = warp(self.latest[lane][1], chain_mapping)[0]
        result['fallback_weight'] = fallback
        if lane == 'visible':
            pair_ir = self.pair_images[cue]
            reverse = flow(pair_ir, self.current_cue)
            matched_ir, _ = warp(pair_ir, pair_flow)
            valid = reliable(pair_flow, reverse) & (cue_error(self.current_cue, matched_ir) < .07)
            source = F.grid_sample(self.pair_images[lane], result['sampling_grid'], padding_mode='border', align_corners=True)
            result['pair_color'] = source, valid.float()
        return result

    def color(self, output, observed, mask, *, source_completion, use_direct):
        fused = output['fused'][:, 0]
        if mask[0]:
            return preserve_luma(fused, observed['visible'], .65)
        completion = source_completion
        source, confidence = completion['pair_color']
        if 'recent_vis' in completion:
            recent = completion['recent_weight'] > 0
            source = torch.where(recent, completion['recent_vis'], source)
            confidence = torch.where(recent, completion['recent_weight'], confidence)
        if use_direct and 'direct_vis' in completion:
            moved = completion['fallback_weight'] > 0
            source = torch.where(moved, completion['direct_vis'], source)
            confidence = torch.where(moved, completion['fallback_weight'], confidence)
        return preserve_luma(fused, source, .65 * confidence)
