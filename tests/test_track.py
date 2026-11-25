import unittest

import torch

from promptvfusion.coherent import coherent_tracks


class TrackTest(unittest.TestCase):
    def test_nearby_consistent_pixels_share_velocity_across_a_small_hole(self):
        motion = torch.zeros(1, 2, 20, 26)
        valid = torch.zeros(1, 1, 20, 26, dtype=torch.bool)
        valid[..., 5:14, 5:18] = True
        valid[..., 8:10, 10] = False
        motion[:, 0, 5:14, 5:18] = 8
        motion[:, 0, 5:14, 6] = 9
        labels, vectors = coherent_tracks(motion, valid)
        self.assertEqual(vectors, {1: (8., 0.)})
        self.assertEqual(int(labels[8, 10]), 1)
        self.assertEqual(int(labels[1, 1]), 0)


if __name__ == '__main__':
    unittest.main()
