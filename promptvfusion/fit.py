"""End-to-end motion-aware feature correction using the existing full M3SVD curriculum.

Differences from prompt_core.full_train: the curriculum comes from the config
(so a warm-started run can use only the long-gap stages), the loss may weight
stale/changed regions, and per-update logs record the mean gate.
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch

from .curriculum import mask_video
from prompt_gate.data import videos, load_images
from prompt_gate.distillation import distill_step
from prompt_gate.runtime import load_models as load_base_models, optimizer_for, streams
from .model import MotionGuidedStream, MotionCorrection



def load_models(config, device, warm_start=True):
    teacher, backbone, prompt, flow = load_base_models(config, device, warm_start)
    prompt.motion_correction = torch.nn.ModuleDict({
        lane: MotionCorrection(config['dim']).to(device) for lane in ('visible', 'infrared')
    })
    return teacher, backbone, prompt, flow


def training_units(rows, complete_chunk):
    start = 0
    while start < len(rows):
        missing = start
        while missing < len(rows) and all(rows[missing]['available_mask']):
            missing += 1
        stop = len(rows) if missing == len(rows) else missing - 1
        while start < stop:
            end = min(start + complete_chunk, stop)
            yield start, end
            start = end
        if missing == len(rows):
            break
        end = missing
        while end < len(rows) and not all(rows[end]['available_mask']):
            end += 1
        yield missing - 1, end
        start = end


def train_video(rows, teacher, backbone, prompt, flow, optimizer, config, device, emit):
    backbone.train()
    prompt.train()
    if hasattr(backbone, 'flow_net'):
        backbone.flow_net.eval()
    _, reference, motion_stream = streams(teacher, backbone, prompt, flow)
    structures = torch.nn.ModuleDict({'visible': teacher.feat_extractor_vi, 'infrared': teacher.feat_extractor_ir})
    calibration = json.loads(Path(config['calibration']).read_text())
    student = MotionGuidedStream(backbone, prompt, structures, flow, calibration)
    totals = {'frames': 0, 'missing_frames': 0, 'updates': 0, 'completion_sum': 0., 'fusion_sum': 0.,
              'gate_sum': 0., 'missing_by_lane': {'visible': 0, 'infrared': 0}}
    roi = config['loss'].get('roi')
    for start, end in training_units(rows, config['complete_chunk']):
        optimizer.zero_grad(set_to_none=True)
        feature_terms, fusion_terms, gate_terms, fraction_terms = [], [], [], []
        for row in rows[start:end]:
            mask = row['available_mask']
            observed = load_images(row['input_paths'], config['data_root'], config['training_size_hw'], device)
            motion = motion_stream.step(row['frame_index'], observed, mask)
            result = distill_step(student, reference, row['frame_index'], observed, mask, motion,
                lambda row=row: load_images(row['target_paths'], config['data_root'],
                                            config['training_size_hw'], device), prompt.scales,
                config['loss']['completion'], config['loss']['fusion'], roi=roi)
            fusion_terms.append(result['terms']['fusion'])
            if not all(mask):
                feature_terms.append(result['terms']['completion'])
                gate_terms.append(result['terms']['gate'])
                fraction_terms.append(result['terms']['roi_fraction'])
                totals['missing_by_lane'][('visible', 'infrared')[mask.index(False)]] += 1
        fusion = torch.stack(fusion_terms).mean()
        feature = torch.stack(feature_terms).mean() if feature_terms else fusion.new_zeros(())
        objective = config['loss']['completion'] * feature + config['loss']['fusion'] * fusion
        objective.backward()
        torch.nn.utils.clip_grad_norm_([p for group in optimizer.param_groups for p in group['params']],
                                       config['grad_clip'], error_if_nonfinite=True)
        optimizer.step()
        student.detach_state()
        gate = float(torch.stack(gate_terms).mean()) if gate_terms else 0.
        record = {'frames': end - start, 'missing_frames': len(feature_terms),
                  'loss': float(objective.detach()), 'completion': float(feature.detach()),
                  'fusion': float(fusion.detach()), 'gate': gate,
                  'roi_fraction': float(torch.stack(fraction_terms).mean()) if fraction_terms else 0.}
        emit(record)
        totals['frames'] += end - start
        totals['missing_frames'] += len(feature_terms)
        totals['updates'] += 1
        totals['completion_sum'] += record['completion'] * record['missing_frames']
        totals['fusion_sum'] += record['fusion'] * record['frames']
        totals['gate_sum'] += gate * record['missing_frames']
    return totals


def save_checkpoint(path, backbone, prompt, optimizer, cursor, config):
    value = {'backbone': backbone.state_dict(), 'prompt': prompt.state_dict(),
             'optimizer': optimizer.state_dict(), 'cursor': cursor, 'config': config,
             'torch_rng': torch.get_rng_state(), 'python_rng': random.getstate()}
    if next(backbone.parameters()).is_cuda:
        value['cuda_rng'] = torch.cuda.get_rng_state()
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def restore_checkpoint(path, backbone, prompt, optimizer, device):
    value = torch.load(path, map_location=device, weights_only=False)
    backbone.load_state_dict(value['backbone'], strict=True)
    prompt.load_state_dict(value['prompt'], strict=True)
    optimizer.load_state_dict(value['optimizer'])
    torch.set_rng_state(value['torch_rng'].cpu())
    random.setstate(value['python_rng'])
    if 'cuda_rng' in value:
        torch.cuda.set_rng_state(value['cuda_rng'].cpu())
    return value


def fresh_cursor(epoch, updates):
    return {'epoch': epoch, 'next_video': 0, 'updates': updates, 'epoch_frames': 0,
            'epoch_missing_frames': 0, 'epoch_complete_sum': 0., 'epoch_fusion_sum': 0., 'epoch_gate_sum': 0.,
            'missing_by_lane': {'visible': 0, 'infrared': 0}, 'seen_videos': []}


def run(config, output, device, resume=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=resume is not None)
    torch.set_num_threads(4)
    torch.manual_seed(config['seed'])
    torch.cuda.set_device(device)
    teacher, backbone, prompt, flow = load_models(config, device, warm_start=resume is None)
    optimizer = optimizer_for(backbone, prompt, config)
    dataset = videos(config['train_manifest'])
    expected_frames = sum(len(rows) for rows in dataset.values())
    assert len(dataset) == config['expected_train_videos'] and expected_frames == config['expected_train_frames']
    assert config['epochs'] == len(config['curriculum'])
    cursor = fresh_cursor(1, 0)
    if resume is not None:
        value = restore_checkpoint(resume, backbone, prompt, optimizer, device)
        assert value['config'] == config
        cursor = value['cursor']
        # The checkpoint commits whole videos. Remove logs from a partially
        # completed video before replaying it from the committed weights.
        path = output / 'updates.jsonl'
        temporary = path.with_suffix('.tmp')
        with path.open() as source, temporary.open('w') as destination:
            if cursor['updates']:
                for line in source:
                    destination.write(line)
                    if json.loads(line)['update'] == cursor['updates']:
                        break
        temporary.replace(path)
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    started = time.monotonic()
    startup_written = (output / 'startup_verified.json').exists()
    with (output / 'updates.jsonl').open('a') as log:
        while cursor['epoch'] <= config['epochs']:
            epoch = cursor['epoch']
            stage = dict(config['curriculum'][epoch - 1])
            names = sorted(dataset)
            random.Random(config['seed'] + epoch).shuffle(names)
            epoch_start = time.monotonic()
            for video_index in range(cursor['next_video'], len(names)):
                sequence = names[video_index]
                rows = mask_video(dataset[sequence], seed=config['seed'] + epoch * 100000 + sorted(dataset).index(sequence),
                    max_gap=stage['max_gap'], complete_ratio=stage['complete'], warmup=config['warmup_frames'],
                    switch_ratio=stage.get('switch_ratio', 0))

                def emit(record):
                    nonlocal startup_written
                    cursor['updates'] += 1
                    log.write(json.dumps({'epoch': epoch, 'sequence': sequence,
                        'update': cursor['updates'], **record}) + '\n')
                    if not startup_written and record['missing_frames']:
                        marker = output / 'startup_verified.json'
                        temporary = marker.with_suffix('.tmp')
                        temporary.write_text(json.dumps({
                            'epoch': epoch, 'sequence': sequence, 'update': cursor['updates'],
                            'first_missing_episode': record, 'max_gap': stage['max_gap'],
                            'peak_gpu_gib': torch.cuda.max_memory_allocated() / 1024 ** 3,
                            'train_videos': len(dataset), 'train_frames': expected_frames}, indent=2) + '\n')
                        temporary.replace(marker)
                        startup_written = True
                totals = train_video(rows, teacher, backbone, prompt, flow, optimizer, config, device, emit)
                log.flush()
                cursor['epoch_frames'] += totals['frames']
                cursor['epoch_missing_frames'] += totals['missing_frames']
                cursor['epoch_complete_sum'] += totals['completion_sum']
                cursor['epoch_fusion_sum'] += totals['fusion_sum']
                cursor['epoch_gate_sum'] += totals['gate_sum']
                for lane, count in totals['missing_by_lane'].items():
                    cursor['missing_by_lane'][lane] += count
                cursor['seen_videos'].append(sequence)
                cursor['next_video'] = video_index + 1
                save_checkpoint(output / 'latest.pth', backbone, prompt, optimizer, cursor, config)
            assert cursor['epoch_frames'] == expected_frames and len(set(cursor['seen_videos'])) == len(dataset)
            summary = {'epoch': epoch, 'max_gap': stage['max_gap'], 'complete_target': stage['complete'],
                'videos': len(dataset), 'frames': cursor['epoch_frames'], 'updates_total': cursor['updates'],
                'missing_frames': cursor['epoch_missing_frames'], 'missing_by_lane': cursor['missing_by_lane'],
                'completion': cursor['epoch_complete_sum'] / cursor['epoch_missing_frames'],
                'fusion': cursor['epoch_fusion_sum'] / cursor['epoch_frames'],
                'gate_mean': cursor['epoch_gate_sum'] / cursor['epoch_missing_frames'],
                'epoch_seconds_this_session': time.monotonic() - epoch_start,
                'peak_gpu_gib': torch.cuda.max_memory_allocated() / 1024 ** 3}
            (output / ('epoch_' + str(epoch) + '.json')).write_text(json.dumps(summary, indent=2) + '\n')
            save_checkpoint(output / ('epoch_' + str(epoch) + '.pth'), backbone, prompt, optimizer, cursor, config)
            print(json.dumps({'event': 'epoch_complete', **summary}), flush=True)
            cursor = fresh_cursor(epoch + 1, cursor['updates'])
            save_checkpoint(output / 'latest.pth', backbone, prompt, optimizer, cursor, config)
    assert all(not p.requires_grad and p.grad is None for p in teacher.parameters())
    assert all(not p.requires_grad and p.grad is None for p in backbone.flow_net.parameters())
    (output / 'result.json').write_text(json.dumps({'status': 'completed', 'epochs': config['epochs'],
        'train_frames_per_epoch': expected_frames, 'updates': cursor['updates'],
        'seconds_this_session': time.monotonic() - started, 'teacher_and_raft_frozen': True,
        'peak_gpu_gib': torch.cuda.max_memory_allocated() / 1024 ** 3}, indent=2) + '\n')
    print(json.dumps({'event': 'completed', 'updates': cursor['updates']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume')
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()), args.output, torch.device(args.device), args.resume)
