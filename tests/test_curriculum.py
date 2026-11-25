from collections import Counter
from itertools import groupby
import unittest

from promptvfusion.curriculum import mask_video


def source_rows(count):
    return [{'frame_index': i, 'target_paths': {'visible': f'vis/{i}.png', 'infrared': f'ir/{i}.png'}}
            for i in range(count)]


def mask_runs(rows):
    return [(mask, list(group)) for mask, group in groupby(rows, key=lambda row: tuple(row['available_mask']))]


class SwitchCurriculumTest(unittest.TestCase):
    def test_switches_preserve_recovery_budget_and_cap_the_combined_gap(self):
        source = source_rows(2010)
        settings = {'seed': 2025, 'max_gap': 7, 'complete_ratio': .2}
        original = mask_video(source, **settings)
        rows = mask_video(source, **settings, switch_ratio=.25)

        self.assertEqual(rows, mask_video(source, **settings, switch_ratio=.25))
        self.assertEqual([row['frame_index'] for row in rows], list(range(len(source))))
        self.assertEqual(Counter(tuple(row['available_mask']) for row in rows),
                         Counter(tuple(row['available_mask']) for row in original))
        self.assertTrue(all(row['available_mask'] == [1, 1] for row in rows[:10]))
        self.assertTrue(all(any(row['available_mask']) for row in rows))
        for before, after in zip(source, rows):
            self.assertEqual(after['target_paths'], before['target_paths'])
            for lane, present in zip(('visible', 'infrared'), after['available_mask']):
                self.assertEqual(after['input_paths'][lane], before['target_paths'][lane] if present else None)

        masks = mask_runs(rows)
        direct_switches = [(a, b) for ((a, _), (b, _)) in zip(masks, masks[1:]) if not all(a) and not all(b)]
        self.assertEqual(set(direct_switches), {((1, 0), (0, 1)), ((0, 1), (1, 0))})
        for available, group in groupby(rows, key=lambda row: all(row['available_mask'])):
            episode = list(group)
            if not available:
                self.assertLessEqual(len(mask_runs(episode)), 2)
                self.assertLessEqual(len(episode), 14)
        self.assertTrue(any(not all(a) and all(b) and not all(c)
                            for (a, _), (b, _), (c, _) in zip(masks, masks[1:], masks[2:])))
        for mask, segment in masks:
            if all(mask):
                continue
            self.assertLessEqual(len(segment), 7)
            self.assertEqual([row['gap_age'] for row in segment], list(range(1, len(segment) + 1)))
            self.assertTrue(all(row['gap_length'] == len(segment) for row in segment))
            self.assertEqual([row['gap_end'] for row in segment], [False] * (len(segment) - 1) + [True])

    def test_short_video_and_disabled_switch_preserve_previous_masks(self):
        for count in (0, 1, 10, 11, 12, 24, 40):
            with self.subTest(count=count):
                source = source_rows(count)
                options = {'seed': 17, 'max_gap': 7, 'complete_ratio': .2}
                original = mask_video(source, **options)
                self.assertEqual(mask_video(source, **options, switch_ratio=0), original)
                switched = mask_video(source, **options, switch_ratio=.25)
                self.assertEqual(len(switched), count)
                self.assertTrue(all(row['available_mask'] == [1, 1] for row in switched[:10]))
