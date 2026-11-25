"""Causal fusion with the newest synchronized real structure/value pair."""
from collections import deque

import torch
from torch.nn import functional as F

from .geometry import compose_flow


LANES = ('visible', 'infrared')


def _window(values):
    return torch.stack([values[0]] * (3 - len(values)) + values, 1)


class CoreRollout:
    def __init__(self, backbone, prompt, use_memory=True):
        self.backbone, self.prompt = backbone, prompt
        self.use_memory = use_memory
        self.reset()

    def reset(self):
        self.next_index = 0
        self.pair = None
        self.history = {lane: deque(maxlen=2) for lane in LANES}
        self.previous_flow = None

    def detach_state(self):
        """End a training graph while retaining every causal state value."""
        self.pair = {'source_index': self.pair['source_index'],
                     'keys': {lane: value.detach() for lane, value in self.pair['keys'].items()},
                     'values': {lane: value.detach() for lane, value in self.pair['values'].items()}}
        self.history = {lane: deque((value.detach() for value in values), maxlen=2)
                        for lane, values in self.history.items()}
        self.previous_flow = self.previous_flow.detach()

    def step(self, index, encoded_available, structure_available, mask, flow_prev, pair_flow,
             fuse=True):
        flags = [bool(value) for value in mask]
        # A missing dictionary entry can be absent or poisonous. Only real
        # features cross this boundary; completion resolves the other lane.
        features = {lane: encoded_available[lane]
                    for lane, present in zip(LANES, flags) if present}
        flow = flow_prev.detach()
        completion = None
        for lane, present in zip(LANES, flags):
            if present:
                continue
            other = LANES[1 - LANES.index(lane)]
            # The anchor's cue-modality value rides along so the prompt can test
            # whether the warped history still agrees with the present modality.
            completion = self.prompt(lane, structure_available[other].detach(),
                                     self.pair['keys'][other], self.pair['values'][lane],
                                     features[other], pair_flow, use_memory=self.use_memory,
                                     pair_other_value=self.pair['values'][other])
            features[lane] = completion['completed_features']

        fused = None
        if fuse:
            windows = {lane: _window(list(self.history[lane]) + [features[lane]])
                       for lane in LANES}
            old = flow if self.previous_flow is None else compose_flow(flow, self.previous_flow)
            magnitude = torch.maximum(torch.linalg.vector_norm(flow, dim=1, keepdim=True),
                                      torch.linalg.vector_norm(old, dim=1, keepdim=True))
            motion = F.max_pool2d((magnitude > .5).to(flow.dtype), 3, 1, 1)
            fused = self.backbone.forward_features(windows['infrared'], windows['visible'],
                                                   flow, old, motion)
        for lane in LANES:
            self.history[lane].append(features[lane])
        if all(flags):
            self.pair = {'source_index': index,
                         'keys': {lane: structure_available[lane].detach() for lane in LANES},
                         'values': {lane: features[lane] for lane in LANES}}
        self.previous_flow = flow
        self.next_index += 1
        return {'features': features, 'fused': fused, 'completion': completion,
                'pair': self.pair}
