"""Gate keeps prompt_core behaviour at init, can reject history, and trains."""

import pytest
import torch
from torch.nn import functional as F

torch.set_num_threads(2)
LANES = ('visible', 'infrared')
HW = (16, 24)


def scales():
    return torch.stack((torch.linspace(.1, .3, 32), torch.linspace(.02, .2, 32))).reshape(2, 1, 32, 1, 1)


def canvases():
    from reference.canvas import PromptCanvas as Old
    from prompt_gate.canvas import PromptCanvas as New
    torch.manual_seed(0)
    old = Old(scales(), canonical_size=(4, 6))
    new = New(scales(), canonical_size=(4, 6), gate_width=8)
    for lane in LANES:  # make the pretrained part non-trivial
        torch.nn.init.normal_(old.qk[lane].weight, 0, .5)
        torch.nn.init.normal_(old.readers[lane][-1].weight, 0, .05)
    missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
    assert not unexpected and all(k.startswith(('gates.', 'gen_mean.')) for k in missing) and missing
    return old, new


def inputs(seed=1):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)
    return dict(current_structure=r(1, 32, *HW), pair_structure=r(1, 32, *HW), pair_value=r(1, 32, *HW),
                current_other=r(1, 32, *HW), pair_flow=r(1, 2, *HW) * 1.5, pair_other_value=r(1, 32, *HW))


def call(canvas, lane, kw, **extra):
    return canvas(lane, kw['current_structure'], kw['pair_structure'], kw['pair_value'],
                  kw['current_other'], kw['pair_flow'], pair_other_value=kw['pair_other_value'], **extra)


@pytest.mark.parametrize('lane', LANES)
def test_gate_off_reproduces_prompt_core_exactly(lane):
    old, new = canvases()
    kw = inputs()
    expected = old(lane, kw['current_structure'], kw['pair_structure'], kw['pair_value'],
                   kw['current_other'], kw['pair_flow'])
    got = call(new, lane, kw, use_gate=False)
    assert torch.allclose(got['completed_features'], expected['completed_features'], atol=1e-6)
    assert torch.allclose(got['retrieved_features'], expected['retrieved_features'], atol=1e-6)


@pytest.mark.parametrize('lane', LANES)
def test_gate_starts_nearly_closed(lane):
    _, new = canvases()
    kw = inputs()
    out = call(new, lane, kw)
    assert float(out['gate'].max()) < .03
    off = call(new, lane, kw, use_gate=False)['completed_features']
    own = new.scales[LANES.index(lane)]
    assert float(((out['completed_features'] - off) / own).abs().mean()) < .1 * float((off / own).abs().mean()) + 1e-3


def test_saturated_gate_makes_output_independent_of_history():
    _, new = canvases()
    for lane in LANES:
        torch.nn.init.constant_(new.gates[lane].gate_head.bias, 30.)
    kw = inputs()
    a = call(new, 'infrared', kw)
    kw2 = dict(kw, pair_value=torch.randn(1, 32, *HW) * 5)
    b = call(new, 'infrared', kw2)
    assert torch.allclose(a['completed_features'], b['completed_features'], atol=1e-4)
    assert not torch.allclose(call(new, 'infrared', kw, use_gate=False)['completed_features'],
                              call(new, 'infrared', kw2, use_gate=False)['completed_features'])


def test_every_gate_parameter_gets_a_gradient_after_one_step():
    _, new = canvases()
    kw = inputs()
    target = torch.randn(1, 32, *HW)
    opt = torch.optim.SGD(new.gates['visible'].parameters(), lr=1e-2)
    for _ in range(2):  # zero-initialised heads open after the first update
        opt.zero_grad()
        loss = (call(new, 'visible', kw)['completed_features'] - target).abs().mean()
        loss.backward()
        opt.step()
    for name, p in new.gates['visible'].named_parameters():
        assert p.grad is not None and bool(torch.isfinite(p.grad).all()), name
    assert sum(float(p.grad.abs().sum()) for p in new.gates['visible'].parameters()) > 0


def test_gate_learns_to_remove_stale_history_in_toy_region():
    """History is wrong inside a box; the gate must close there and stay open elsewhere."""
    _, new = canvases()
    lane, cue = 'visible', 'infrared'
    for p in new.parameters():
        p.requires_grad_(False)
    for p in new.gates[lane].parameters():
        p.requires_grad_(True)
    torch.manual_seed(3)
    own = new.scales[0]
    truth = torch.randn(1, 32, *HW) * own
    base = inputs(5)
    base['pair_flow'] = torch.zeros(1, 2, *HW)
    box = torch.zeros(1, 1, *HW)
    box[..., 4:12, 6:18] = 1
    stale = truth + box * 3 * own * torch.randn(1, 32, *HW)
    other = torch.randn(1, 32, *HW) * new.scales[1]
    base.update(pair_value=stale, current_other=other,
                pair_other_value=other + box * 3 * new.scales[1])  # history of cue disagrees inside the box
    base['pair_structure'] = torch.randn(1, 32, *HW)
    base['current_structure'] = base['pair_structure'] + box * 3
    opt = torch.optim.Adam(new.gates[lane].parameters(), lr=3e-3)
    for _ in range(300):
        opt.zero_grad()
        out = call(new, lane, base, retrieval='flow')
        loss = ((out['completed_features'] - truth) / own).abs().mean()
        loss.backward()
        opt.step()
    gate = call(new, lane, base, retrieval='flow')['gate']
    inside, outside = float(gate[box.bool()].mean()), float(gate[~box.bool()].mean())
    assert inside > .5 and outside < .25 and inside > outside + .3, (inside, outside)


def test_use_memory_false_and_missing_anchor_value():
    _, new = canvases()
    kw = inputs()
    assert call(new, 'visible', kw, use_memory=False)['completed_features'].shape == (1, 32, *HW)
    with pytest.raises(ValueError):
        new('visible', kw['current_structure'], kw['pair_structure'], kw['pair_value'],
            kw['current_other'], kw['pair_flow'])


def test_warm_start_loader_accepts_old_checkpoint_and_rejects_foreign_keys():
    from prompt_gate.runtime import load_prompt_state
    old, new = canvases()
    missing = load_prompt_state(new, old.state_dict())
    assert missing and all(k.startswith(('gates.', 'gen_mean.')) for k in missing)
    bad = dict(old.state_dict(), **{'alien.weight': torch.zeros(1)})
    with pytest.raises(RuntimeError):
        load_prompt_state(new, bad)
    bad2 = {k: v for k, v in old.state_dict().items() if not k.startswith('readers.visible.0')}
    with pytest.raises(RuntimeError):
        load_prompt_state(new, bad2)


def test_roi_weighted_losses():
    from prompt_gate.distillation import roi_mask, completion_loss, distillation_loss
    torch.manual_seed(0)
    scale = torch.ones(1, 32, 1, 1)
    truth = torch.zeros(1, 32, 8, 8)
    retrieved = torch.zeros(1, 32, 8, 8)
    retrieved[..., :2, :2] = 5.
    mask = roi_mask(truth, retrieved, scale, quantile=.9, dilate=1)
    assert mask[..., :2, :2].min() == 1 and mask.sum() == mask[..., :2, :2].sum()
    pred = torch.zeros(1, 32, 8, 8)
    pred[..., :2, :2] = 1.  # only the wrong region has error
    plain = completion_loss(pred, truth, scale)
    off = completion_loss(pred, truth, scale, mask, 0.)
    on = completion_loss(pred, truth, scale, mask, 4.)
    assert torch.isclose(plain, off)
    assert on > plain * 2  # error concentrated in the ROI is upweighted
    pred2 = torch.zeros(1, 32, 8, 8)
    pred2[..., 4:, 4:] = 1.  # error outside the ROI is downweighted
    assert completion_loss(pred2, truth, scale, mask, 4.) < completion_loss(pred2, truth, scale)
    # uniform error: normalised weights leave the loss unchanged
    uniform = torch.ones(1, 32, 8, 8)
    assert torch.isclose(completion_loss(uniform, truth, scale, mask, 4.), completion_loss(uniform, truth, scale))
    rgb_s, rgb_t = torch.zeros(1, 1, 3, 8, 8), torch.zeros(1, 1, 3, 8, 8)
    rgb_s[..., :2, :2] = 1.
    assert distillation_loss(rgb_s, rgb_t, mask, 2.) > distillation_loss(rgb_s, rgb_t)
