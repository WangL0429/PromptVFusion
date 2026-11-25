"""Inference probe for a newer real-frame candidate after a modality switch."""
import torch
from torch.nn import functional as F

from prompt_gate.geometry import warp
from prompt_gate.streams import LANES, StudentStream


def reliable(forward, backward):
    reverse, support = warp(backward, forward)
    norm = lambda value: torch.linalg.vector_norm(value, dim=1, keepdim=True)
    return support & (norm(forward + reverse) <= .5 + .01 * (norm(forward) + norm(reverse)))


def align_recent(feature, recent_cue, to_pair, from_pair, to_recent, from_recent):
    first_ok = reliable(to_pair, from_pair)
    second_ok = reliable(to_recent, from_recent)
    tail, _ = warp(to_recent, to_pair)
    total = to_pair + tail
    aligned_feature, support = warp(feature, total)
    aligned_cue, _ = warp(recent_cue, total)
    second_ok_here, _ = warp(second_ok.float(), to_pair)
    return aligned_feature, aligned_cue, first_ok & (second_ok_here > .999) & support


class CandidateStream(StudentStream):
    def __init__(self, backbone, prompt, structures, flow, *, use_recent):
        self.use_recent = use_recent
        self.latest = {}
        self.recent_links = {}
        super().__init__(backbone, prompt, structures, history_flow=flow)

    def step(self, index, observed, mask, motion):
        for lane, present in zip(LANES, mask):
            if present:
                self.latest[lane] = (index, observed[lane])
        return super().step(index, observed, mask, motion)

    def _read_history(self, lane, current_key, pair_key, pair_value, current_other, pair_flow, **kwargs):
        output = self.prompt(lane, current_key, pair_key, pair_value, current_other, pair_flow, **kwargs)
        index = LANES.index(lane)
        cue = LANES[1 - index]
        pair_cue = self.pair_images[cue]
        first_cue = self.first_pair_images[cue]
        aligned_cue = F.grid_sample(pair_cue, output['sampling_grid'], padding_mode='border', align_corners=True)
        first_flow = self.history_flow(self.current_cue, first_cue)
        earlier_cue, support = warp(first_cue, first_flow)

        def cue_error(candidate):
            return F.avg_pool2d((self.current_cue - candidate).abs().mean(1, keepdim=True), 5, 1, 2)

        base_error = cue_error(aligned_cue)
        held_error = cue_error(pair_cue)
        earlier_error = cue_error(earlier_cue)
        prefer_earlier = (earlier_error < held_error) & support
        candidate_error = torch.where(prefer_earlier, earlier_error, held_error)
        replace = (base_error > .07) & (base_error > candidate_error + .03) & (candidate_error < .07)
        weight = F.avg_pool2d(replace.float(), 3, 1, 1)
        encode = self.rollout.backbone.encode_modality
        earlier_value, _ = warp(encode(self.first_pair_images[lane], lane), first_flow)
        own, other = self.prompt.scales[index], self.prompt.scales[1 - index]

        def read(feature):
            reader_input = torch.cat((current_other / other, feature / own), 1)
            return feature + own * self.prompt.readers[lane](reader_input).tanh()

        original = output['completed_features']
        # Keep the existing spatial choice before the reader to reproduce the baseline.
        candidate = read(torch.where(prefer_earlier, earlier_value, pair_value))
        output['completed_features'] = torch.lerp(original, candidate, weight)
        output['weight'] = weight

        if not self.use_recent:
            return output

        valid = torch.ones_like(support)
        output['candidates'] = {'aligned': (original, valid), 'held': (read(pair_value), valid),
                                'first': (read(earlier_value), support)}
        recent_index, image = self.latest[lane]
        pair_index = self.rollout.pair['source_index']
        output['recent_index'] = recent_index
        output['recent_weight'] = torch.zeros_like(weight)
        if recent_index == pair_index:
            return output

        key = (pair_index, recent_index)
        if lane not in self.recent_links or self.recent_links[lane][0] != key:
            pair = self.pair_images[lane]
            to_recent = self.history_flow(pair, image)
            from_recent = self.history_flow(image, pair)
            cue_proxy, _ = warp(pair_cue, from_recent)
            self.recent_links[lane] = (key, encode(image, lane), cue_proxy, to_recent, from_recent)
        _, feature, cue_proxy, to_recent, from_recent = self.recent_links[lane]
        from_pair = self.history_flow(pair_cue, self.current_cue)
        recent, aligned_proxy, valid = align_recent(feature, cue_proxy, pair_flow, from_pair, to_recent, from_recent)
        recent = read(recent)
        output['candidates']['recent'] = (recent, valid)
        # The proxy checks motion consistency against the observed stream. There
        # is no real paired cue at the newer source during a single-modality gap.
        error = cue_error(aligned_proxy)
        eligible = valid & (weight == 0) & (error < .07) & (error <= base_error + .01)
        recent_weight = F.avg_pool2d(eligible.float(), 3, 1, 1) * eligible
        output['recent_weight'] = recent_weight
        output['completed_features'] = torch.lerp(output['completed_features'], recent, recent_weight)
        return output
