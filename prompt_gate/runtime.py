"""Original MAVFusion, the gated Prompt core and frozen real-observation flow."""
import json
from copy import deepcopy
from pathlib import Path

import torch

from .canvas import PromptCanvas
from .motion import ObservedMotion
from .streams import StudentStream, TeacherStream

BACKBONE_GROUPS = ('feat_extractor_ir', 'feat_extractor_vi', 'aam_ir', 'aam_vi', 'fusion_net')


def load_prompt_state(prompt, state):
    """Warm start from a prompt_core checkpoint: only the new gate parts may be missing."""
    missing, unexpected = prompt.load_state_dict(state, strict=False)
    if unexpected or any(not key.startswith(('gates.', 'gen_mean.')) for key in missing):
        raise RuntimeError('Incompatible prompt checkpoint: missing=%s unexpected=%s'
                           % (list(missing), list(unexpected)))
    return missing


def load_models(config, device, warm_start=True):
    from src.model.net import VideoFusion
    from src.model.observed_alignment import FrozenObservedFlow
    teacher = VideoFusion({'model': {'dim': config['dim'], 'causal': True, 'causal_current_flow': 'zero'}})
    state = torch.load(config['fusion_checkpoint'], map_location='cpu', weights_only=False)
    teacher.load_state_dict(state, strict=True)
    teacher.to(device).eval().requires_grad_(False)
    backbone = deepcopy(teacher)
    for name in BACKBONE_GROUPS:
        getattr(backbone, name).requires_grad_(True)
    raw = json.loads(Path(config['channel_scales']).read_text())
    scales = torch.stack([torch.tensor(raw[lane], device=device).reshape(1, config['dim'], 1, 1)
                          for lane in ('visible', 'infrared')])
    gate = config.get('gate', {})
    prompt = PromptCanvas(scales, config['dim'], config['canonical_size'], config['temperature'],
                          gate.get('width', 48), tuple(gate.get('dilations', (1, 2, 4, 8)))).to(device)
    if config.get('channel_means'):
        means = json.loads(Path(config['channel_means']).read_text())
        with torch.no_grad():
            for lane in ('visible', 'infrared'):
                prompt.gen_mean[lane].copy_(torch.tensor(means[lane], device=device).reshape(1, config['dim'], 1, 1))
    if warm_start and config.get('init_checkpoint'):
        value = torch.load(config['init_checkpoint'], map_location='cpu', weights_only=False)
        backbone.load_state_dict(value['backbone'], strict=True)
        load_prompt_state(prompt, value['prompt'])
        backbone.to(device)
        prompt.to(device)
    flow = FrozenObservedFlow(teacher.flow_net, config['flow_max_res']).to(device)
    return teacher, backbone, prompt, flow


def optimizer_for(backbone, prompt, config):
    gate_parameters = list(prompt.gates.parameters())
    gate_ids = {id(p) for p in gate_parameters}
    return torch.optim.Adam([
        {'params': [p for p in prompt.parameters() if id(p) not in gate_ids], 'lr': config['prompt_lr']},
        {'params': gate_parameters, 'lr': config.get('gate_lr', config['prompt_lr'])},
        {'params': [p for p in backbone.parameters() if p.requires_grad],
         'lr': config['backbone_lr']}])


def streams(teacher, backbone, prompt, flow, *, reject_stale_history=False):
    structure_encoders = torch.nn.ModuleDict({'visible': teacher.feat_extractor_vi,
                                              'infrared': teacher.feat_extractor_ir})
    student = StudentStream(backbone, prompt, structure_encoders,
                            history_flow=flow if reject_stale_history else None)
    return student, TeacherStream(teacher), ObservedMotion(flow)
