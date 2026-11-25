"""Fuse a paired or partially observed frame sequence in causal order."""
import argparse
import json
from pathlib import Path

from PIL import Image
import torch

from prompt_gate.data import load_images, videos
from prompt_gate.motion import ObservedMotion
from .coherent import CoherentGuidedStream
from .fit import load_models
from .model import MotionGuidedStream


@torch.inference_mode()
def run(config, checkpoint, manifest, output, device, *, coherent=True):
    teacher, backbone, prompt, flow = load_models(config, device, warm_start=False)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    backbone.load_state_dict(state['backbone'], strict=True)
    prompt.load_state_dict(state['prompt'], strict=True)
    backbone.eval()
    prompt.eval()
    calibration = json.loads(Path(config['calibration']).read_text())
    structures = torch.nn.ModuleDict({'visible': teacher.feat_extractor_vi,
                                      'infrared': teacher.feat_extractor_ir})
    stream_type = CoherentGuidedStream if coherent else MotionGuidedStream
    for sequence, rows in videos(manifest).items():
        if Path(sequence).name != sequence or sequence in ('', '.', '..'):
            raise ValueError(f'Invalid sequence name: {sequence!r}')
        directory = output / sequence
        directory.mkdir(parents=True, exist_ok=True)
        student = stream_type(backbone, prompt, structures, flow, calibration)
        motion = ObservedMotion(flow)
        for row in rows:
            index, mask = row['frame_index'], row['available_mask']
            observed = load_images(row['input_paths'], config['data_root'], config['native_size_hw'], device)
            result = student.step(index, observed, mask, motion.step(index, observed, mask))
            rgb = student.color(result, observed, mask, source_completion=result['completion'], use_direct=True)
            pixels = rgb[0].clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
            Image.fromarray(pixels).save(directory / f'{index:05d}.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('config/train.json'))
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--no-coherent', action='store_true', help='Disable motion and background candidates for the same-weight comparison')
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.checkpoint, args.manifest, args.output,
        torch.device(args.device), coherent=not args.no_coherent)
