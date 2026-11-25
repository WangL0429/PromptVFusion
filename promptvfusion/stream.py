"""Short-step transport with calibrated candidate scoring."""
from collections import deque

import torch
from torch.nn import functional as F
from prompt_gate.geometry import warp
from prompt_gate.streams import LANES
from .candidate_screen import CandidateStream, reliable
from .short_step import advance_mapping
from .scores import legacy_eligibility, measure_scores


class ScoringStream(CandidateStream):
    def __init__(self, backbone, prompt, structures, flow, calibration):
        super().__init__(backbone, prompt, structures, flow, use_recent=False)
        self.calibration = calibration
        self.real_history = {lane: deque(maxlen=2) for lane in LANES}
        self.prior_observed = {}

    def step(self, index, observed, mask, motion):
        for lane, present in zip(LANES, mask):
            if present:
                self.real_history[lane].append((index, observed[lane]))
        self.index = index
        self.link = motion['flow_prev']
        result = super().step(index, observed, mask, motion)
        self.prior_observed = observed
        return result

    def _read_history(self, lane, current_key, pair_key, pair_value, current_other, pair_flow, **kwargs):
        output = super()._read_history(lane, current_key, pair_key, pair_value, current_other, pair_flow, **kwargs)
        recent_index, recent_image = self.latest[lane]
        pair_index = self.rollout.pair['source_index']
        output['recent_index'] = recent_index
        output['recent_weight'] = torch.zeros_like(output['weight'])
        if recent_index == pair_index:
            return output

        index = LANES.index(lane)
        cue = LANES[1 - index]
        flow = self.history_flow
        key = (pair_index, recent_index)
        if lane not in self.recent_links or self.recent_links[lane][0] != key:
            (older_index, older_image), _ = self.real_history[lane]
            assert older_index == recent_index - 1 and recent_index == self.index - 1
            backward = flow(recent_image, older_image)
            forward = flow(older_image, recent_image)
            # Carry the last observed velocity one step. Its consistency is measured on the observed pair.
            initial = warp(backward, backward)[0]
            initial_valid = (warp(reliable(backward, forward).float(), initial)[0] > .999)
            initial_valid &= warp(backward, initial)[1]
            recent_feature = self.rollout.backbone.encode_modality(recent_image, lane)
            # Both experiments use the same proxy for the unobserved cue at the source.
            cue_proxy = warp(self.pair_images[cue], flow(recent_image, self.pair_images[lane]))[0]
            self.recent_links[lane] = (key, recent_feature, cue_proxy, initial, initial_valid)
        else:
            _, feature, proxy, last_mapping, last_valid = self.recent_links[lane]
            mapping, valid = advance_mapping(self.link, flow(self.prior_observed[cue], self.current_cue),
                                             last_mapping, last_valid)
            self.recent_links[lane] = (key, feature, proxy, mapping, valid)

        _, feature, proxy, mapping, valid = self.recent_links[lane]
        candidate, _ = warp(feature, mapping)
        aligned_proxy, _ = warp(proxy, mapping)
        scale, cue_scale = self.prompt.scales[index], self.prompt.scales[1 - index]
        read = self.prompt.readers[lane](torch.cat((current_other / cue_scale, candidate / scale), 1)).tanh()
        candidate = candidate + scale * read
        aligned_cue = F.grid_sample(self.pair_images[cue], output['sampling_grid'], padding_mode='border', align_corners=True)

        def cue_error(value):
            return F.avg_pool2d((self.current_cue - value).abs().mean(1, keepdim=True), 5, 1, 2)

        proxy_error, anchor_error = cue_error(aligned_proxy), cue_error(aligned_cue)
        eligible = valid & (output['weight'] == 0)
        absolute, relative, original = legacy_eligibility(eligible, proxy_error, anchor_error)
        pixel, structure = measure_scores(self.structure_encoders[cue], current_key, pair_key, aligned_proxy,
                                          output['sampling_grid'], cue_scale, proxy_error, anchor_error)
        selected = original
        if self.calibration is not None:
            policy = self.calibration[lane]
            choice = policy['scores'][policy['selected']]
            a, b = choice['coefficients']
            selected = eligible & ((a * pixel + b * structure) > choice['threshold'])
        output['before_recent'] = output['completed_features']
        output['score_filters'] = absolute, relative, original
        output['scores'] = pixel, structure
        weight = F.avg_pool2d(selected.float(), 3, 1, 1) * selected
        output['recent_weight'] = weight
        output['recent_candidate'] = candidate, valid
        output['completed_features'] = torch.lerp(output['completed_features'], candidate, weight)
        return output
