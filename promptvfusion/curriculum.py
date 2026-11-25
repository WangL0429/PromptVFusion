"""Seeded short-to-long outages over every frame of an original video."""
import random


LANES = ('visible', 'infrared')
STAGES = [{'max_gap': 1, 'complete': .5}, {'max_gap': 2, 'complete': .4},
          {'max_gap': 4, 'complete': .3}, {'max_gap': 7, 'complete': .2}]


def stage_for_epoch(epoch):
    return dict(STAGES[epoch - 1])


def mask_video(rows, seed, max_gap, complete_ratio, warmup=10, gap_lengths=None, *, switch_ratio=0):
    rng = random.Random(seed)
    prefix = min(warmup, len(rows))
    count = len(rows) - prefix
    # The mandatory recovery between outages sets a minimum feasible budget;
    # the declared four stages all meet it without changing their ratios.
    complete = max(round(count * complete_ratio), count // (max_gap + 1))
    remaining = count - complete
    choices = tuple(range(1, max_gap + 1)) if gap_lengths is None else tuple(gap_lengths)
    lengths = []
    while remaining:
        slots_after = complete - len(lengths)
        minimum = max(1, remaining - slots_after * max_gap)
        length = min(remaining, rng.choice([value for value in choices if value >= minimum]))
        lengths.append(length)
        remaining -= length

    # One real frame separates every outage. Extra real frames can appear
    # before, between or after them without restarting the video warmup.
    real_slots = [0] + [1] * max(0, len(lengths) - 1) + [0]
    for _ in range(complete - max(0, len(lengths) - 1)):
        real_slots[rng.randrange(len(lengths) + 1)] += 1
    missing_counts = [0, 0]
    missing_lanes = []
    for length in lengths:
        if missing_counts[0] == missing_counts[1]:
            lane = rng.randrange(2)
        else:
            lane = int(missing_counts[1] < missing_counts[0])
        missing_counts[lane] += length
        missing_lanes.append(lane)

    if switch_ratio:
        # Join disjoint pairs of opposite outages. Move the recovery frames
        # after the pair so the coverage budget and outage lengths stay fixed.
        boundaries = [i for i in range(1, len(lengths)) if missing_lanes[i - 1] != missing_lanes[i]]
        rng.shuffle(boundaries)
        limit = round(max(0, len(lengths) - 1) * switch_ratio)
        joined = set()
        for boundary in boundaries:
            if len(joined) == limit:
                break
            if boundary - 1 in joined or boundary + 1 in joined:
                continue
            real_slots[boundary + 1] += real_slots[boundary]
            real_slots[boundary] = 0
            joined.add(boundary)

    states = [(0, 0, 0, False)] * prefix
    for index, (length, lane) in enumerate(zip(lengths, missing_lanes)):
        states.extend([(0, 0, 0, False)] * real_slots[index])
        state = 2 if lane == 0 else 1
        states.extend((state, length, age, age == length) for age in range(1, length + 1))
    states.extend([(0, 0, 0, False)] * real_slots[len(lengths)])

    result = []
    for row, (state, length, age, end) in zip(rows, states):
        mask = ([1, 1], [1, 0], [0, 1])[state]
        result.append({**row, 'available_mask': list(mask),
                       'missing_type': ('complete', 'missing_infrared', 'missing_visible')[state],
                       'missing_type_id': state,
                       'input_paths': {lane: row['target_paths'][lane] if present else None
                                       for lane, present in zip(LANES, mask)},
                       'gap_length': length, 'gap_age': age, 'gap_end': end})
    return result
