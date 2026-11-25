"""Fixed scores and development-only threshold calibration."""

import numpy as np
import torch
from torch.nn import functional as F


def legacy_eligibility(valid, proxy, anchor):
    absolute = proxy < .07
    relative = proxy <= anchor + .01
    return absolute, relative, valid & absolute & relative


def roc_auc(score, positive):
    order = np.argsort(score, kind='stable')
    groups = np.r_[0, np.flatnonzero(np.diff(score[order])) + 1]
    positives = np.add.reduceat(positive[order].astype(np.int64), groups)
    negatives = np.diff(np.r_[groups, len(score)]) - positives
    pairs = int(positives.sum()) * int(negatives.sum())
    return float((positives * (np.cumsum(negatives) - .5 * negatives)).sum() / pairs) if pairs else None


def measure_scores(encoder, key, pair_key, proxy_image, grid, cue_scale, proxy_error, anchor_error):
    if proxy_image.shape[1] == 1:
        proxy_image = proxy_image.repeat(1, 3, 1, 1)
    proxy = encoder(proxy_image) / cue_scale
    anchor = F.grid_sample(pair_key, grid, padding_mode='border', align_corners=True) / cue_scale
    query = key / cue_scale
    mean = query.mean(dim=(-2, -1), keepdim=True)
    anchor_cos = F.cosine_similarity(query - mean, anchor - mean, dim=1)[:, None]
    proxy_cos = F.cosine_similarity(query - mean, proxy - mean, dim=1)[:, None]
    structure = F.avg_pool2d(proxy_cos - anchor_cos, 5, 1, 2)
    pixel = torch.minimum(.07 - proxy_error, anchor_error + .01 - proxy_error)
    return pixel, structure


def audit(completion, truth, scale, old_edge):
    candidate, valid = completion['recent_candidate']
    before = completion['before_recent']
    eligible = valid & (completion['weight'] == 0)
    region = old_edge & (completion['weight'] == 0)
    roi = region & eligible
    absolute, relative, original = completion['score_filters']
    actual = completion['recent_weight'] > 0
    error = lambda value: ((value - truth) / scale).abs().mean(1, keepdim=True)
    original_error = error(before)
    delta = original_error - error(candidate)
    good = (delta > 1e-6) & eligible
    counts = {'region': region.sum(), 'valid': roi.sum(), 'better': (region & good).sum(),
              'absolute_only': (region & good & ~absolute & relative).sum(),
              'relative_only': (region & good & absolute & ~relative).sum(),
              'both': (region & good & ~absolute & ~relative).sum(),
              'original_selected': (roi & original).sum(),
              'original_selected_good': (region & good & original).sum(),
              'actual_selected': (roi & actual).sum(), 'actual_selected_good': (region & good & actual).sum()}
    counts = dict(zip(counts, map(int, torch.stack(tuple(counts.values())).tolist())))
    pixel, structure = completion['scores']
    sampled = eligible.flatten().nonzero().flatten()[::23]
    scene = torch.stack((pixel.flatten()[sampled], structure.flatten()[sampled], original.flatten()[sampled]), 1).cpu().numpy()
    old = torch.stack((pixel[roi], structure[roi], delta[roi], original[roi], actual[roi]), 1).cpu().numpy()
    errors = {}
    if counts['region']:
        errors = {'without_recent': original_error[region].mean().item(),
                  'output': error(completion['completed_features'])[region].mean().item(),
                  'oracle': (original_error - torch.where(eligible, delta.clamp_min(0), 0))[region].mean().item()}
    return {'counts': counts, 'feature_mae': errors}, scene, old


def calibrate(scenes, edges):
    calibration = {}
    for lane, values in scenes.items():
        scene = np.concatenate(values)
        edge = np.concatenate([values for values in edges[lane].values()])
        coverage = float(scene[:, 2].mean())
        quartiles = np.quantile(scene[:, :2], [.25, .75], axis=0)
        ratio = float(np.diff(quartiles, axis=0)[0, 0] / max(np.diff(quartiles, axis=0)[0, 1], 1e-6))
        metrics = {}
        for name, coefficients in (('pixel', (1., 0.)), ('structure', (0., 1.)), ('mixed', (1., ratio))):
            score = scene[:, :2] @ coefficients
            threshold = float(np.quantile(score, 1 - coverage)) if name != 'pixel' else 0.
            edge_score = edge[:, :2] @ coefficients
            gains = []
            for rows in edges[lane].values():
                take = (rows[:, :2] @ coefficients) > threshold if name != 'pixel' else rows[:, 3] > 0
                gains.append(float(np.where(take, rows[:, 2], 0).mean()))
            metrics[name] = {'coefficients': coefficients, 'threshold': threshold,
                             'auc': roc_auc(edge_score, edge[:, 2] > 1e-6),
                             'mean_feature_gain_per_valid_pixel': float(np.mean(gains)),
                             'sampled_coverage': float((score > threshold).mean())}
        selected = max(('structure', 'mixed'), key=lambda name: metrics[name]['mean_feature_gain_per_valid_pixel'])
        calibration[lane] = {'selected': selected, 'scores': metrics, 'original_coverage': coverage}
    return calibration
