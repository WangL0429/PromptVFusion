"""Track connected moving features and clear their vacated positions."""
import numpy as np
from scipy import ndimage
import torch
from torch.nn import functional as F

from .candidate_screen import reliable
from prompt_gate.geometry import warp
from prompt_gate.streams import LANES
from .repair import RepairStream, cue_error
from .model import MotionGuidedStream


def coherent_tracks(motion, valid):
    vectors = motion[0].permute(1, 2, 0).cpu().numpy()
    seeds = valid[0, 0].cpu().numpy()
    speed = np.linalg.norm(vectors, axis=-1)
    direction = np.floor((np.arctan2(vectors[..., 1], vectors[..., 0]) + np.pi / 4) / (np.pi / 2)).astype(int) % 4
    tracks = np.zeros_like(seeds, dtype=np.int32)
    offsets = {}
    for heading in range(4):
        moving = seeds & (direction == heading)
        closed = ndimage.binary_closing(moving, iterations=2) | moving
        labels, _ = ndimage.label(closed)
        for component, bounds in enumerate(ndimage.find_objects(labels), start=1):
            region = labels[bounds] == component
            good = region & moving[bounds]
            if np.count_nonzero(good) < 32:
                continue
            sample = vectors[bounds][good]
            median = np.median(sample, axis=0)
            residual = np.linalg.norm(sample - median, axis=1)
            if np.median(residual) > max(1., .25 * np.median(speed[bounds][good])):
                continue
            region = ndimage.binary_fill_holes(region) & (tracks[bounds] == 0)
            track = len(offsets) + 1
            tracks[bounds][region] = track
            offsets[track] = tuple(float(v) for v in median)
    return torch.from_numpy(tracks).to(device=motion.device), offsets


def translate(value, dx, dy):
    moved = torch.roll(value, shifts=(dy, dx), dims=(-2, -1))
    if dx > 0:
        moved[..., :dx] = 0
    elif dx < 0:
        moved[..., dx:] = 0
    if dy > 0:
        moved[..., :dy, :] = 0
    elif dy < 0:
        moved[..., dy:, :] = 0
    return moved


class CoherentStream(RepairStream):
    def __init__(self, backbone, prompt, structures, flow, calibration):
        super().__init__(backbone, prompt, structures, flow, calibration)
        self.tracks = {}
        self.first_features = {}

    def _read_history(self, lane, current_key, pair_key, pair_value, current_other, pair_flow, **kwargs):
        result = super()._read_history(lane, current_key, pair_key, pair_value, current_other, pair_flow, **kwargs)
        if 'recent_candidate' not in result:
            return result
        _, feature, _, _, chain_valid = self.recent_links[lane]
        age = self.index - result['recent_index']
        if age == 1:
            (_, older), (_, latest) = self.real_history[lane]
            backward = self.history_flow(latest, older)
            forward = self.history_flow(older, latest)
            speed = torch.linalg.vector_norm(backward, dim=1, keepdim=True)
            labels, vectors = coherent_tracks(-backward, reliable(backward, forward) & (speed > 2))
            source_cue, _ = warp(self.current_cue, -backward)
            self.tracks[lane] = labels, vectors, source_cue
            return result

        labels, vectors, source_cue = self.tracks[lane]
        idx = LANES.index(lane)
        cue = LANES[1 - idx]
        own, other = self.prompt.scales[idx], self.prompt.scales[1 - idx]
        feature_source = torch.zeros_like(feature)
        own_source = torch.zeros_like(self.latest[lane][1])
        assigned = torch.zeros_like(chain_valid)
        swept = torch.zeros_like(assigned)
        recent_visible = self.latest[lane][1]
        missing_history = result['recent_index'] - self.rollout.pair['source_index']
        for track, (vx, vy) in vectors.items():
            region = (labels == track)[None, None]
            dx, dy = round(age * vx), round(age * vy)
            current = translate(region, dx, dy)
            needs_recovery = current & ~chain_valid
            if not needs_recovery.any():
                continue
            cue_aligned = translate(source_cue, dx, dy)
            agrees = (self.current_cue - cue_aligned).abs().mean(1, keepdim=True) < .07
            if (current & agrees).sum() < .65 * current.sum():
                continue
            # A track's common displacement prevents different parts of the object from drifting apart.
            selected = current & agrees & ~assigned & (result['weight'] == 0)
            feature_source = torch.where(selected, translate(feature, dx, dy), feature_source)
            own_source = torch.where(selected, translate(recent_visible, dx, dy), own_source)
            assigned |= selected
            for offset in range(-missing_history, age):
                swept |= translate(region, round(offset * vx), round(offset * vy))
        if not assigned.any():
            return result

        if lane == 'visible':
            first = self.first_pair_images[cue]
            back = self.history_flow(self.current_cue, first)
            rev = self.history_flow(first, self.current_cue)
            matched, _ = warp(first, back)
            clean = swept & ~F.max_pool2d(assigned.float(), 7, 1, 3).bool()
            clean &= reliable(back, rev) & (cue_error(self.current_cue, matched) < .04)
            if lane not in self.first_features:
                self.first_features[lane] = self.rollout.backbone.encode_modality(self.first_pair_images[lane], lane)
            background, _ = warp(self.first_features[lane], back)
            feature_source = torch.where(clean, background, feature_source)
            own_source = torch.where(clean, warp(self.first_pair_images[lane], back)[0], own_source)
            assigned |= clean
        reader_input = torch.cat((current_other / other, feature_source / own), 1)
        candidate = feature_source + own * self.prompt.readers[lane](reader_input).tanh()
        weight = F.avg_pool2d(assigned.float(), 3, 1, 1) * assigned
        result['completed_features'] = torch.lerp(result['completed_features'], candidate, weight)
        result['fallback_weight'] = torch.maximum(result['fallback_weight'], weight)
        if lane == 'visible':
            result['direct_vis'] = torch.where(assigned, own_source, result['direct_vis'])
        return result


class CoherentGuidedStream(MotionGuidedStream, CoherentStream):
    """Experimental coherent candidates before learned feature correction."""
