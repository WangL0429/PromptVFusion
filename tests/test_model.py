import unittest
import torch

from promptvfusion.model import MotionCorrection


class MotionCorrectionTest(unittest.TestCase):
    def test_identity_initialization_and_unavailable_candidate_invariance(self):
        torch.manual_seed(7)
        layer = MotionCorrection(4)
        cue = torch.randn(1, 4, 10, 12)
        base = torch.randn_like(cue, requires_grad=True)
        proposal = torch.randn_like(cue)
        confidence = torch.ones_like(cue[:, :1])
        available = confidence.bool()
        scale = torch.ones(1, 4, 1, 1)
        out = layer(base, proposal, cue, cue, confidence, available, scale, scale)
        torch.testing.assert_close(out, base, atol=0, rtol=0)
        (out - (base.detach() + .1)).abs().mean().backward()
        self.assertGreater(layer.output.weight.grad.abs().sum().item(), 0.)
        with torch.no_grad():
            layer.output.weight.uniform_(-.2, .2)
            mask = ~available
            a = layer(base, proposal, cue, cue, confidence, mask, scale, scale)
            b = layer(base, proposal * 50, cue, cue, confidence, mask, scale, scale)
        torch.testing.assert_close(a, b, atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
