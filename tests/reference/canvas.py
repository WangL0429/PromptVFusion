"""Local spatial retrieval around frozen current-to-pair observed motion."""
import torch
from torch import nn
from torch.nn import functional as F

LANES = ('visible', 'infrared')


class PromptCanvas(nn.Module):
    def __init__(self, scales, dim=32, canonical_size=(60, 80), temperature=.02):
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
        self.to(scales)
        for projection in self.qk.values():
            nn.init.zeros_(projection.weight)
        for reader in self.readers.values():
            nn.init.zeros_(reader[-1].weight)
            nn.init.zeros_(reader[-1].bias)
        self.configuration = {'architecture': 'FlowGuidedSpatialPrompt', 'dim': dim,
            'canonical_size': list(self.canonical_size), 'lanes': list(LANES),
            'temperature': temperature, 'search_radius': self.search_radius,
            'retrieval': 'Zero-start bilinear query learns local native-pixel offsets around frozen pair flow.'}

    def forward(self, lane, current_structure, pair_structure, pair_value,
                current_other, pair_flow, use_memory=True):
        index = LANES.index(lane)
        cue = LANES[1 - index]
        own_scale, other_scale = self.scales[index], self.scales[1 - index]
        if use_memory:
            canonical = lambda value: F.interpolate(value, self.canonical_size,
                                                     mode='bilinear', align_corners=True)
            historical = pair_structure / other_scale
            center = historical.mean((-2,-1),keepdim=True)
            query = self.qk[cue](F.normalize(canonical(current_structure / other_scale)-center, dim=1))
            key = F.normalize(historical-center, dim=1)
            height, width = self.canonical_size
            native_height, native_width = current_other.shape[-2:]
            y, x = torch.meshgrid(torch.linspace(-1, 1, height, device=query.device, dtype=query.dtype),
                                  torch.linspace(-1, 1, width, device=query.device, dtype=query.dtype),
                                  indexing='ij')
            prior_grid = torch.stack((x, y), -1)[None] + canonical(pair_flow).permute(0, 2, 3, 1) * pair_flow.new_tensor(
                [2 / (native_width - 1), 2 / (native_height - 1)])
            offsets = query.new_tensor([(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
            candidate_grid = prior_grid[:, None] + offsets[None, :, None, None] * query.new_tensor(
                [2 / (native_width - 1), 2 / (native_height - 1)])
            candidates = F.grid_sample(key, candidate_grid.reshape(query.shape[0], 9 * height, width, 2),
                                        mode='bilinear', padding_mode='border', align_corners=True)
            candidates = candidates.reshape(query.shape[0], self.dim, 9, height, width)
            # The zero query yields uniform symmetric offsets and exactly the
            # frozen prior; do not normalize away its zero-start amplitude.
            similarity = (query[:, :, None] * F.normalize(candidates, dim=1)).sum(1) / self.temperature
            weights = similarity.softmax(1)
            # Pair opposite candidates so uniform weights produce exact zero.
            dx=(weights[:,2]-weights[:,0])+(weights[:,5]-weights[:,3])+(weights[:,8]-weights[:,6])
            dy=(weights[:,6]-weights[:,0])+(weights[:,7]-weights[:,1])+(weights[:,8]-weights[:,2])
            residual = torch.stack((dx,dy),1)
            residual = F.interpolate(residual, current_other.shape[-2:], mode='bilinear', align_corners=True)
            yy, xx = torch.meshgrid(torch.linspace(-1, 1, native_height, device=query.device, dtype=query.dtype),
                                    torch.linspace(-1, 1, native_width, device=query.device, dtype=query.dtype),
                                    indexing='ij')
            sampling_grid = torch.stack((xx, yy), -1)[None] + (pair_flow + residual).permute(0, 2, 3, 1) * pair_flow.new_tensor(
                [2 / (native_width - 1), 2 / (native_height - 1)])
            retrieved = F.grid_sample(pair_value, sampling_grid, mode='bilinear',
                                       padding_mode='border', align_corners=True)
        else:
            retrieved = torch.zeros_like(current_other)
            sampling_grid = current_other.new_zeros(current_other.shape[0], *current_other.shape[-2:], 2)
        reader_input = torch.cat((current_other / other_scale, retrieved / own_scale), 1)
        completed = retrieved + own_scale * self.readers[lane](reader_input).tanh()
        return {'completed_features': completed, 'retrieved_features': retrieved,
                'sampling_grid': sampling_grid}
