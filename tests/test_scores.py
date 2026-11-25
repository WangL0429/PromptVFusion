import unittest

import numpy as np
import torch

from promptvfusion.scores import legacy_eligibility, roc_auc


class ScoreTest(unittest.TestCase):
    def test_threshold_rejections_distinguish_absolute_relative_and_geometry(self):
        valid = torch.tensor([1, 1, 0, 1], dtype=torch.bool)
        proxy = torch.tensor([.03, .08, .01, .06])
        anchor = torch.tensor([.03, .10, .10, .04])
        absolute, relative, selected = legacy_eligibility(valid, proxy, anchor)
        self.assertEqual(absolute.tolist(), [True, False, True, True])
        self.assertEqual(relative.tolist(), [True, True, True, False])
        self.assertEqual(selected.tolist(), [True, False, False, False])

    def test_auc_awards_half_credit_for_ties(self):
        self.assertAlmostEqual(roc_auc(np.array([0., 1., 1., 2.]), np.array([0, 0, 1, 1])), .875)


if __name__ == '__main__':
    unittest.main()
