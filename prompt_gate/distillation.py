"""Conditional missing-feature supervision and complete-input RGB distillation.

Optional ROI weighting: where the retrieved history disagrees with the hidden
truth (stale / ghosted content, newly arrived objects) the per-pixel error is
weighted up. Weights are normalised to mean 1 per frame so the overall loss
scale is unchanged; the mask is built from loss-only targets and never reaches
the student.
"""
import torch
from torch.nn import functional as F


def roi_mask(truth, retrieved, scale, quantile=.75, dilate=5):
    """[B,1,H,W] float mask: top quantile of own-scale |truth - retrieved|."""
    change = ((truth.detach() - retrieved.detach()) / scale).abs().mean(1, keepdim=True)
    threshold = change.flatten(1).quantile(quantile, dim=1)[:, None, None, None]
    mask = (change > threshold).to(change.dtype)  # ties (e.g. zero change) select nothing
    if dilate and dilate > 1:
        mask = F.max_pool2d(mask, dilate, 1, dilate // 2)
    return mask


def _weighted_mean(error, mask, roi_weight):
    """error [B,1,H,W]. Without a mask or weight this is the plain mean."""
    if mask is None or not roi_weight:
        return error.mean()
    if mask.shape[-2:] != error.shape[-2:]:
        mask = F.interpolate(mask, error.shape[-2:], mode='nearest')
    weight = 1 + roi_weight * mask
    weight = weight / weight.mean((1, 2, 3), keepdim=True)
    return (error * weight).mean()


def distillation_loss(student_rgb, teacher_rgb, mask=None, roi_weight=0.):
    error = (student_rgb - teacher_rgb.detach()).abs().mean((1, 2))[:, None]
    return _weighted_mean(error, mask, roi_weight)


def completion_loss(prediction, target, scale, mask=None, roi_weight=0.):
    error = ((prediction - target.detach()) / scale).abs().mean(1, keepdim=True)
    return _weighted_mean(error, mask, roi_weight)


def distill_step(student, teacher, index, observed, mask, motion, load_targets, scales,
                 completion_weight, fusion_weight, roi=None):
    """roi: None or {'completion': w, 'fusion': w, 'quantile': q, 'dilate': k}."""
    result = student.step(index, observed, mask, motion)
    # Complete sources are opened only after masked student state commits.
    complete = load_targets()
    target = teacher.step(index, complete, motion)
    roi = roi or {}
    if all(mask):
        fusion = distillation_loss(result['fused'], target)
        feature = fusion.new_zeros(())
        gate = fusion.new_zeros(())
        fraction = fusion.new_zeros(())
    else:
        lane = ('visible', 'infrared')[mask.index(False)]
        scale = scales[('visible', 'infrared').index(lane)]
        truth = teacher.target_feature(complete, lane)
        region = None
        if roi.get('completion') or roi.get('fusion'):
            region = roi_mask(truth, result['completion']['retrieved_features'], scale,
                              roi.get('quantile', .75), roi.get('dilate', 5))
        fusion = distillation_loss(result['fused'], target, region, roi.get('fusion', 0.))
        feature = completion_loss(result['features'][lane], truth, scale, region,
                                  roi.get('completion', 0.))
        gate_map = result['completion'].get('gate')
        gate = gate_map.detach().mean() if gate_map is not None else fusion.new_zeros(())
        fraction = region.mean() if region is not None else fusion.new_zeros(())
    return {**result, 'loss': completion_weight * feature + fusion_weight * fusion,
            'terms': {'completion': feature, 'fusion': fusion, 'gate': gate, 'roi_fraction': fraction}}
