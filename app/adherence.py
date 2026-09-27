"""Optimal matching of raw samples against a planned sampling schedule.

Given planned instants ``p_i`` on an even grid and raw samples whose
time lies within a symmetric tolerance of at least one point, a matching
pairs every planned point with at most one sample and every sample with
at most one point. The objective is, in order:

1. maximize the number of matched (hit) planned points;
2. among those matchings, minimize the sum of absolute deviations
   ``|sampled_at - planned_at|``;
3. among those, take the lexicographically smallest match list when the
   matches are sorted by ``(planned_at, sampled_at, sequence)``.

Both points and samples are totally ordered in time (samples share an
instant only when sequences differ), and the cost ``|s - p|`` is a Monge
matrix, so an optimal matching never crosses: if point ``i < k`` is
matched to sample ``j < l`` then ``s_j`` is not later than ``s_l``. The
problem is therefore an ordered sequence alignment solved by dynamic
programming. To keep that DP small even when a device holds millions of
raw samples, candidates are pruned to a provably sufficient set: for
each planned point only its best few in-window samples (ordered by
absolute deviation, then by the (instant, sequence) tie key) are kept,
because in a non-crossing matching a sample closer to a point than its
matched sample can only have been taken by one of the point's immediate
neighbors.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from heapq import nsmallest

# How many closest eligible samples per planned point survive pruning.
# In a non-crossing optimum, when point i is matched, every eligible
# sample ranked better than its own match that is used elsewhere can
# only be matched to an immediate neighbor: a sample matched to a point
# further away would cross one of the two adjacent matched edges. Each
# neighbor (i-1 and i+1) consumes a single sample, so at most two
# better-ranked samples of i can be blocked; i's third-best eligible
# sample is therefore always reachable in some optimum. Same-instant
# duplicates do not raise this bound -- a blocked sample is counted per
# neighbor, not per instant.
RETAIN_PER_POINT = 3


@dataclass(frozen=True)
class AdherenceCandidate:
    """A retained candidate sample.

    ``first_plan``/``last_plan`` delimit the consecutive planned points
    whose closed tolerance window contains the sample's instant.
    """

    instant_us: int
    sequence: int
    turbidity: float
    first_plan: int
    last_plan: int


@dataclass(frozen=True)
class AdherenceMatch:
    plan_index: int
    planned_us: int
    instant_us: int
    sequence: int
    turbidity: float
    deviation_us: int


def select_candidates(
    samples: list[tuple[int, int, float]],
    points: list[int],
    tolerance_us: int,
) -> list[AdherenceCandidate]:
    """Reduce all samples to the set an optimal matching may need.

    ``samples`` holds ``(instant_us, sequence, turbidity)`` triples in
    non-decreasing (instant_us, sequence) order. For every planned point
    the closest ``RETAIN_PER_POINT`` (=3) samples inside its closed
    tolerance window -- ranked by (absolute deviation, instant,
    sequence) -- are retained; the union of those per-point shortlists is
    returned in (instant, sequence) order.

    Why three suffice. Fix an optimal non-crossing matching (the
    absolute-distance cost is a Monge matrix, so uncrossing a pair of
    edges never lowers the hit count or raises the deviation sum; equal
    deviations uncross toward the smaller (instant, sequence) key, so the
    lexicographically smallest optimum is non-crossing as well) and a
    point i it matches. Any eligible sample of i ranked better than i's
    own matched sample that is itself matched to another point can only
    be matched to an immediate neighbor i-1 or i+1: matched to a point
    two or more grid steps away it would either be outside that point's
    tolerance window (tolerance <= interval leaves at least one whole
    interval between the windows) or it would cross the edge of an
    adjacent matched point. Each neighbor consumes one sample, so at most
    two better-ranked samples of i can be unavailable; the third-best is
    unused and may replace i's match without changing the hit count and
    without worsening the deviation sum or the lexicographic key (the
    ranking's tie breakers are exactly the key's first two coordinates).
    Shortlisting three per point therefore leaves an optimum intact,
    including when several samples share an instant (a neighbor still
    consumes a single matched sample).
    """
    if not points:
        return []

    retained: dict[tuple[int, int], AdherenceCandidate] = {}

    def plan_span(instant_us: int) -> tuple[int, int]:
        # Consecutive planned points whose closed window contains the
        # sample instant.
        first = bisect_right(points, instant_us + tolerance_us) - 1
        last = bisect_left(points, instant_us - tolerance_us)
        return last, first

    # Samples arrive in (instant, sequence) order and the planned points
    # are chronological and evenly spaced, so each point's window is a
    # contiguous slice reached by advancing two cursors; the total scan
    # over all points is linear in the number of raw samples.
    low = 0
    high = 0
    for planned in points:
        while low < len(samples) and samples[low][0] < planned - tolerance_us:
            low += 1
        if high < low:
            high = low
        while high < len(samples) and samples[high][0] <= planned + tolerance_us:
            high += 1
        # nsmallest keeps the three best entries during a single scan of
        # the window, avoiding a sort when the device holds many raw
        # samples.
        shortlist = nsmallest(
            RETAIN_PER_POINT,
            samples[low:high],
            key=lambda sample: (
                abs(sample[0] - planned),
                sample[0],
                sample[1],
            ),
        )
        for instant_us, sequence, turbidity in shortlist:
            last, first = plan_span(instant_us)
            retained[(instant_us, sequence)] = AdherenceCandidate(
                instant_us=instant_us,
                sequence=sequence,
                turbidity=turbidity,
                first_plan=last,
                last_plan=first,
            )

    candidates = list(retained.values())
    candidates.sort(key=lambda item: (item.instant_us, item.sequence))
    return candidates


def match_schedule(
    samples: list[tuple[int, int, float]],
    points: list[int],
    tolerance_us: int,
) -> list[AdherenceMatch]:
    """Return the lexicographically optimal one-to-one matching.

    See the module docstring for the objective order. Candidates are
    ordered by (instant, sequence); a suffix DP over points x candidates
    stores (hit count, -total absolute deviation) to maximize, and the
    answer is reconstructed greedily in plan order, which realizes the
    final lexicographic tie break: at each point a hit is tried before a
    miss and then the earliest feasible sample is tried first.
    """
    candidates = select_candidates(samples, points, tolerance_us)
    point_count = len(points)
    candidate_count = len(candidates)

    # best[i][j] = (hits, -cost_us) optimal value using points i..n-1
    # and candidates j..m-1. Three moves leave this ordered-alignment
    # state:
    #
    # - miss point i (it gets no sample), keeping every candidate for
    #   the later points: continue at (i+1, j);
    # - skip candidate j (leave it unused), keeping point i open for a
    #   later sample: continue at (i, j+1);
    # - match point i with candidate j when its window contains i,
    #   consuming both: continue at (i+1, j+1) with the deviation cost.
    #
    # The last row and last column are zero boundaries; the two value
    # maximands are hit count first, negative total deviation second.
    best: list[list[tuple[int, int]]] = [
        [(0, 0)] * (candidate_count + 1) for _ in range(point_count + 1)
    ]
    for i in range(point_count - 1, -1, -1):
        row = best[i]
        next_row = best[i + 1]
        for j in range(candidate_count - 1, -1, -1):
            candidate = candidates[j]
            value = next_row[j]  # miss point i, keep candidate j
            skipped = row[j + 1]  # skip candidate j, keep point i
            if skipped > value:
                value = skipped
            if candidate.first_plan <= i <= candidate.last_plan:
                deviation = candidate.instant_us - points[i]
                hits, neg_cost = next_row[j + 1]
                matched = (hits + 1, neg_cost - abs(deviation))
                if matched > value:
                    value = matched
            row[j] = value

    matches: list[AdherenceMatch] = []
    i = 0
    j = 0
    while i < point_count:
        target = best[i][j]
        # Tertiary tie break, applied per point: a hit at point i always
        # outranks a miss, and among hits the earliest (instant, sequence)
        # eligible candidate that still reaches the optimum is chosen,
        # which is the lexicographically smallest match list.
        chosen = -1
        for k in range(j, candidate_count):
            candidate = candidates[k]
            if candidate.first_plan <= i <= candidate.last_plan:
                deviation = candidate.instant_us - points[i]
                hits, neg_cost = best[i + 1][k + 1]
                if (hits + 1, neg_cost - abs(deviation)) == target:
                    chosen = k
                    break
        if chosen >= 0:
            candidate = candidates[chosen]
            deviation = candidate.instant_us - points[i]
            matches.append(
                AdherenceMatch(
                    plan_index=i,
                    planned_us=points[i],
                    instant_us=candidate.instant_us,
                    sequence=candidate.sequence,
                    turbidity=candidate.turbidity,
                    deviation_us=deviation,
                )
            )
            # Candidates j..chosen-1 are left unused.
            j = chosen + 1
            i += 1
        else:
            # Point i cannot be hit at the optimum, so miss it while
            # keeping every remaining candidate for the later points;
            # skipping a candidate here would only discard choices that
            # the later points' lexicographic minimum may need.
            i += 1
    return matches
