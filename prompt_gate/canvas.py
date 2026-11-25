"""Flow-guided local retrieval plus a consistency gate that can reject stale history.

The only structural difference from ``prompt_core.canvas`` is the gate. Retrieval
(frozen flow + 3x3 native-pixel matching) is unchanged. After retrieval, the
canvas also samples the anchor's *cue-modality* features and structure at the same
positions. Where the warped history of the present modality disagrees with the
present modality itself (a moving object left, or arrived), the retrieved value
of the missing modality is stale. A gate then blends in a history-free
prediction made from the current observation only.
"""
import torch
from torch import nn
from torch.nn import functional as F

LANES = ('visible', 'infrared')
GATE_BIAS = -5.0  # sigmoid(-5)=0.0067: starts almost exactly as prompt_core


def _resize(value, size):
    size = tuple(size)
    if tuple(value.shape[-2:]) == size:
        return value
    return F.interpolate(value, size, mode='bilinear', align_corners=True)


class ConsistencyGate(nn.Module):
    """Dilated context at the canonical grid + local mismatch at native size.

    Outputs a gate logit (weight on the history-free prediction) and that
    history-free feature prediction, both at the input resolution.
    """

    def __init__(self, dim, canonical_size, width=48, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.canonical_size = tuple(canonical_size)
        self.stem = nn.Sequential(nn.Conv2d(4 * dim, width, 1), nn.SiLU())
        self.blocks = nn.ModuleList(nn.Sequential(
            nn.Conv2d(width, width, 3, padding=d, dilation=d, groups=width),
            nn.Conv2d(width, width, 1), nn.SiLU()) for d in dilations)
        self.gate_head = nn.Conv2d(width, 1, 1)
        self.feature_head = nn.Conv2d(width, dim, 1)
        self.local = nn.Sequential(nn.Conv2d(2, 8, 3, padding=1), nn.SiLU(), nn.Conv2d(8, 1, 1))
        for layer in (self.gate_head, self.feature_head, self.local[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.constant_(self.gate_head.bias, GATE_BIAS)

    def forward(self, context, mismatch):
        x = self.stem(F.adaptive_avg_pool2d(context, self.canonical_size))
        for block in self.blocks:
            x = x + block(x)
        size = context.shape[-2:]
        logit = _resize(self.gate_head(x), size) + self.local(mismatch)
        return logit, _resize(self.feature_head(x), size)


class PromptCanvas(nn.Module):
    def __init__(self, scales, dim=32, canonical_size=(60, 80), temperature=.02,
                 gate_width=48, gate_dilations=(1, 2, 4, 8)):
        super().__init__()
        if (not isinstance(scales, torch.Tensor) or scales.shape != (2, 1, dim, 1, 1)
                or not scales.is_floating_point() or not bool(torch.isfinite(scales).all())
                or not bool((scales > 0).all())):
            raise ValueError('Fixed scales must be positive finite [2,1,dim,1,1]')
        if temperature <= 0:
            raise ValueError('Attention temperature must be positive')
        self.dim, self.canonical_size, self.temperature = dim, tuple(canonical_size), temperature
        self.search_radius = 1
        self.register_buffer('scales', scales.detach().clone())
        self.qk = nn.ModuleDict({lane: nn.Conv2d(dim, dim, 1, bias=False) for lane in LANES})
        self.readers = nn.ModuleDict({lane: nn.Sequential(
            nn.Conv2d(2 * dim, dim, 1), nn.SiLU(),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim), nn.SiLU(),
            nn.Conv2d(dim, dim, 1)) for lane in LANES})
        self.gates = nn.ModuleDict({lane: ConsistencyGate(dim, canonical_size, gate_width, gate_dilations)
                                    for lane in LANES})
        # Per-channel mean of real features: the history-free prediction is
        # own_scale*head + mean, so a nearly closed gate barely perturbs history
        # (features sit several scales away from zero). Set from data at start-up.
        self.gen_mean = nn.ParameterDict({lane: nn.Parameter(torch.zeros(1, dim, 1, 1)) for lane in LANES})
        self.to(scales)
        for projection in self.qk.values():
            nn.init.zeros_(projection.weight)
        for reader in self.readers.values():
            nn.init.zeros_(reader[-1].weight)
            nn.init.zeros_(reader[-1].bias)
        self.configuration = {'architecture': 'FlowGuidedSpatialPrompt+ConsistencyGate', 'dim': dim,
            'canonical_size': list(self.canonical_size), 'lanes': list(LANES),
            'temperature': temperature, 'search_radius': self.search_radius,
            'gate': {'width': gate_width, 'dilations': list(gate_dilations), 'init_bias': GATE_BIAS,
                     'history_free': 'own_scale*head + learned per-channel mean'},
            'retrieval': 'Zero-start bilinear query learns local native-pixel offsets around frozen pair flow.'}

    # ------------------------------------------------------------------ retrieval
    def _match_residual(self, cue, current_structure, pair_structure, current_other, pair_flow):
        other_scale = self.scales[LANES.index(cue)]
        canonical = lambda value: F.interpolate(value, self.canonical_size,
                                                 mode='bilinear', align_corners=True)
        historical = pair_structure / other_scale
        center = historical.mean((-2, -1), keepdim=True)
        query = self.qk[cue](F.normalize(canonical(current_structure / other_scale) - center, dim=1))
        key = F.normalize(historical - center, dim=1)
        height, width = self.canonical_size
        native_height, native_width = current_other.shape[-2:]
        y, x = torch.meshgrid(torch.linspace(-1, 1, height, device=query.device, dtype=query.dtype),
                              torch.linspace(-1, 1, width, device=query.device, dtype=query.dtype),
                              indexing='ij')
        unit = pair_flow.new_tensor([2 / (native_width - 1), 2 / (native_height - 1)])
        prior_grid = torch.stack((x, y), -1)[None] + canonical(pair_flow).permute(0, 2, 3, 1) * unit
        offsets = query.new_tensor([(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
        candidate_grid = prior_grid[:, None] + offsets[None, :, None, None] * unit
        candidates = F.grid_sample(key, candidate_grid.reshape(query.shape[0], 9 * height, width, 2),
                                    mode='bilinear', padding_mode='border', align_corners=True)
        candidates = candidates.reshape(query.shape[0], self.dim, 9, height, width)
        similarity = (query[:, :, None] * F.normalize(candidates, dim=1)).sum(1) / self.temperature
        weights = similarity.softmax(1)
        # Pair opposite candidates so uniform weights produce exact zero.
        dx = (weights[:, 2] - weights[:, 0]) + (weights[:, 5] - weights[:, 3]) + (weights[:, 8] - weights[:, 6])
        dy = (weights[:, 6] - weights[:, 0]) + (weights[:, 7] - weights[:, 1]) + (weights[:, 8] - weights[:, 2])
        residual = torch.stack((dx, dy), 1)
        return F.interpolate(residual, current_other.shape[-2:], mode='bilinear', align_corners=True)

    def forward(self, lane, current_structure, pair_structure, pair_value,
                current_other, pair_flow, use_memory=True, pair_other_value=None,
                retrieval='matched', use_gate=True):
        """retrieval: 'matched' (learned offsets), 'flow' (frozen flow only), 'hold' (no motion)."""
        if retrieval not in ('matched', 'flow', 'hold'):
            raise ValueError('retrieval must be matched, flow or hold')
        index = LANES.index(lane)
        cue = LANES[1 - index]
        own_scale, other_scale = self.scales[index], self.scales[1 - index]
        size = current_other.shape[-2:]
        if use_memory:
            if pair_other_value is None:
                raise ValueError('pair_other_value (anchor cue-modality features) is required')
            flow = torch.zeros_like(pair_flow) if retrieval == 'hold' else pair_flow
            residual = (self._match_residual(cue, current_structure, pair_structure, current_other, pair_flow)
                        if retrieval == 'matched' else torch.zeros_like(pair_flow))
            height, width = size
            yy, xx = torch.meshgrid(torch.linspace(-1, 1, height, device=flow.device, dtype=flow.dtype),
                                    torch.linspace(-1, 1, width, device=flow.device, dtype=flow.dtype),
                                    indexing='ij')
            sampling_grid = torch.stack((xx, yy), -1)[None] + (flow + residual).permute(0, 2, 3, 1) * flow.new_tensor(
                [2 / (width - 1), 2 / (height - 1)])
            sample = lambda value: F.grid_sample(value, sampling_grid, mode='bilinear',
                                                 padding_mode='border', align_corners=True)
            retrieved = sample(pair_value)
            history_other = sample(pair_other_value)
            history_structure = sample(_resize(pair_structure, size))
        else:
            retrieved = torch.zeros_like(current_other)
            history_other = torch.zeros_like(current_other)
            history_structure = torch.zeros_like(current_other)
            sampling_grid = current_other.new_zeros(current_other.shape[0], *size, 2)

        cue_now = current_other / other_scale
        completed = retrieved
        gate = None
        if use_gate:
            difference_value = cue_now - history_other / other_scale
            difference_structure = _resize(current_structure, size) / other_scale - history_structure / other_scale
            context = torch.cat((cue_now, retrieved / own_scale, difference_value, difference_structure), 1)
            mismatch = torch.stack((difference_value.abs().mean(1), difference_structure.abs().mean(1)), 1)
            logit, generated = self.gates[lane](context, mismatch)
            gate = torch.sigmoid(logit)  # weight of the history-free prediction
            completed = retrieved + gate * (own_scale * generated + self.gen_mean[lane] - retrieved)
        reader_input = torch.cat((cue_now, completed / own_scale), 1)
        completed = completed + own_scale * self.readers[lane](reader_input).tanh()
        return {'completed_features': completed, 'retrieved_features': retrieved,
                'sampling_grid': sampling_grid, 'gate': gate}
