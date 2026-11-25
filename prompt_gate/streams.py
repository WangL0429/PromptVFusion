"""Separate causal student and complete-input teacher video state."""
from collections import deque
import torch
from torch.nn import functional as F

from .rollout import CoreRollout,LANES,_window
from .geometry import compose_flow, warp


class StudentStream:
    def __init__(self,backbone,prompt,structure_encoders,use_memory=True,history_flow=None):
        self.history_flow = history_flow
        self.prompt = prompt
        reader = self._read_history if history_flow is not None and use_memory else prompt
        self.rollout=CoreRollout(backbone,reader,use_memory=use_memory)
        self.structure_encoders=structure_encoders.eval().requires_grad_(False)

    def step(self,index,observed,mask,motion):
        if index==0 and not all(mask):
            raise ValueError('Initialize memory with a synchronized real pair')
        encoded={lane:self.rollout.backbone.encode_modality(observed[lane],lane)
                 for lane,present in zip(LANES,mask) if present}
        if self.history_flow is not None:
            if all(mask):
                self.pair_images = {lane: observed[lane] for lane in LANES}
                if index == 0:
                    self.first_pair_images = self.pair_images
            else:
                self.current_cue = observed[LANES[mask.index(True)]]
        self.structure_encoders.eval()
        with torch.no_grad():
            structures={lane:self.structure_encoders[lane](
                observed[lane].repeat(1,3,1,1) if observed[lane].shape[1]==1 else observed[lane])
                for lane,present in zip(LANES,mask) if present}
        return self.rollout.step(index,encoded,structures,mask,motion['flow_prev'],motion['pair_flow'])

    def _read_history(self, lane, current_key, pair_key, pair_value, current_other, pair_flow, **kwargs):
        output = self.prompt(lane, current_key, pair_key, pair_value, current_other, pair_flow, **kwargs)
        index = LANES.index(lane)
        cue_lane = LANES[1 - index]
        pair_cue = self.pair_images[cue_lane]
        first_cue = self.first_pair_images[cue_lane]
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

        # Encode the earlier real frame with the current weights, including during training.
        earlier_value = self.rollout.backbone.encode_modality(self.first_pair_images[lane], lane)
        earlier_value, _ = warp(earlier_value, first_flow)
        candidate = torch.where(prefer_earlier, earlier_value, pair_value)
        own, other = self.prompt.scales[index], self.prompt.scales[1 - index]
        reader_input = torch.cat((current_other / other, candidate / own), 1)
        candidate = candidate + own * self.prompt.readers[lane](reader_input).tanh()
        output['completed_features'] = torch.lerp(output['completed_features'], candidate, weight)
        return output

    def detach_state(self):
        self.rollout.detach_state()


class TeacherStream:
    def __init__(self,backbone):
        self.backbone=backbone.eval().requires_grad_(False)
        self.history={lane:deque(maxlen=2) for lane in LANES}
        self.previous_flow=None

    @torch.no_grad()
    def target_feature(self,complete,lane):
        return self.backbone.encode_modality(complete[lane],lane)

    @torch.no_grad()
    def step(self,index,complete,motion):
        features={lane:self.backbone.encode_modality(complete[lane],lane) for lane in LANES}
        windows={lane:_window(list(self.history[lane])+[features[lane]]) for lane in LANES}
        flow=motion['flow_prev']
        old=flow if self.previous_flow is None else compose_flow(flow,self.previous_flow)
        magnitude=torch.maximum(torch.linalg.vector_norm(flow,dim=1,keepdim=True),
                                torch.linalg.vector_norm(old,dim=1,keepdim=True))
        mask=F.max_pool2d((magnitude>.5).to(flow.dtype),3,1,1)
        output=self.backbone.forward_features(windows['infrared'],windows['visible'],flow,old,mask)
        for lane in LANES:self.history[lane].append(features[lane])
        self.previous_flow=flow
        return output
