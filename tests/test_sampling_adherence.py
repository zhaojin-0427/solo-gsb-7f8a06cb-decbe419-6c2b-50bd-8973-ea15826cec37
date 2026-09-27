from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.conftest import ack_event, get_events, post_records


def sample(sequence: int, at: str, turbidity: float = 12.0, device: str = "sensor-a"):
    return {
        "device_id": device,
        "sequence": sequence,
        "sampled_at": at,
        "turbidity": turbidity,
    }


def query_adherence(
    client: TestClient,
    *,
    device: str = "sensor-a",
    window_from: str = "2026-09-23T08:00:00Z",
    window_to: str = "2026-09-23T09:00:00Z",
    interval: int = 60,
    tolerance: int = 5,
) -> dict:
    response = client.post(
        f"/devices/{device}/sampling-adherence",
        json={
            "from": window_from,
            "to": window_to,
            "interval_seconds": interval,
            "tolerance_seconds": tolerance,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_exactly_on_schedule_hits_every_point(client):
    post_records(
        client,
        [
            sample(1, "2026-09-23T08:00:00Z"),
            sample(2, "2026-09-23T08:01:00Z"),
            sample(3, "2026-09-23T08:02:00Z"),
        ],
    )
    body = query_adherence(
        client, window_to="2026-09-23T08:03:00Z", interval=60, tolerance=5
    )
    assert body["planned_count"] == 3
    assert body["hit_count"] == 3
    assert [point["planned_at"] for point in body["points"]] == [
        "2026-09-23T08:00:00Z",
        "2026-09-23T08:01:00Z",
        "2026-09-23T08:02:00Z",
    ]
    assert all(point["deviation_seconds"] == 0 for point in body["points"])
    assert all(point["sample"] is not None for point in body["points"])
    assert all(point["missed"] is False for point in body["points"])


def test_missed_point_is_marked_without_sample(client):
    post_records(
        client,
        [
            sample(1, "2026-09-23T08:00:00Z"),
            sample(3, "2026-09-23T08:02:00Z"),
        ],
    )
    body = query_adherence(
        client, window_to="2026-09-23T08:03:00Z", interval=60, tolerance=5
    )
    assert body["planned_count"] == 3
    assert body["hit_count"] == 2
    missed = body["points"][1]
    assert missed["planned_at"] == "2026-09-23T08:01:00Z"
    assert missed["missed"] is True
    assert missed["sample"] is None
    assert missed["deviation_seconds"] is None


def test_signed_deviation_early_and_late(client):
    post_records(
        client,
        [
            sample(1, "2026-09-23T07:59:57Z"),
            sample(2, "2026-09-23T08:01:03Z"),
        ],
    )
    body = query_adherence(
        client, window_to="2026-09-23T08:02:00Z", interval=60, tolerance=5
    )
    assert body["hit_count"] == 2
    assert body["points"][0]["deviation_seconds"] == -3
    assert body["points"][1]["deviation_seconds"] == 3


def test_tolerance_bound_is_inclusive(client):
    post_records(
        client,
        [
            sample(1, "2026-09-23T07:59:55Z"),
            sample(2, "2026-09-23T08:01:05Z"),
            sample(3, "2026-09-23T08:02:06Z"),
        ],
    )
    body = query_adherence(
        client, window_to="2026-09-23T08:03:00Z", interval=60, tolerance=5
    )
    assert body["hit_count"] == 2
    assert body["points"][0]["deviation_seconds"] == -5
    assert body["points"][1]["deviation_seconds"] == 5
    # 08:02:06 is one second outside the tolerance.
    assert body["points"][2]["missed"] is True


def test_zero_tolerance_requires_exact_instant(client):
    post_records(
        client,
        [
            sample(1, "2026-09-23T08:00:00Z"),
            sample(2, "2026-09-23T08:01:01Z"),
        ],
    )
    body = query_adherence(
        client,
        window_to="2026-09-23T08:02:00Z",
        interval=60,
        tolerance=0,
    )
    assert body["hit_count"] == 1
    assert body["points"][0]["sample"]["sequence"] == 1
    assert body["points"][1]["missed"] is True


def test_samples_outside_window_but_within_tolerance_are_used(client):
    # A sample up to tolerance seconds before "from" still serves the first
    # planned point; the raw samples are not restricted to [from, to).
    post_records(client, [sample(1, "2026-09-23T07:59:57Z")])
    body = query_adherence(
        client,
        window_from="2026-09-23T08:00:00Z",
        window_to="2026-09-23T08:02:00Z",
        interval=60,
        tolerance=5,
    )
    assert body["hit_count"] == 1
    assert body["points"][0]["sample"]["sequence"] == 1
    assert body["points"][1]["missed"] is True


def test_last_planned_instant_must_be_strictly_before_to(client):
    post_records(
        client,
        [
            sample(1, "2026-09-23T08:00:00Z"),
            sample(2, "2026-09-23T08:01:00Z"),
        ],
    )
    # from + one interval == to, so only the first point is generated.
    body = query_adherence(
        client, window_to="2026-09-23T08:01:00Z", interval=60, tolerance=5
    )
    assert body["planned_count"] == 1
    assert [point["planned_at"] for point in body["points"]] == [
        "2026-09-23T08:00:00Z"
    ]


def test_each_sample_matches_at_most_one_point(client):
    # One sample sits within tolerance of two adjacent planned points; it
    # can serve only one of them, so the other is missed.
    post_records(client, [sample(1, "2026-09-23T08:00:30Z")])
    body = query_adherence(
        client,
        window_to="2026-09-23T08:02:00Z",
        interval=60,
        tolerance=60,
    )
    assert body["planned_count"] == 2
    assert body["hit_count"] == 1
    sequences = [
        point["sample"]["sequence"]
        for point in body["points"]
        if point["sample"] is not None
    ]
    assert sequences == [1]


def test_matching_maximizes_hits_before_minimizing_deviation(client):
    # Samples at 08:00:10 and 08:01:00. Greedy-by-nearest would give the
    # 08:00:10 sample to point 08:00 (dev 10) and point 08:01 takes
    # 08:01:00 (dev 0): two hits. A single shared sample at 08:00:30 is
    # closer on average but cannot hit both, so an extra farther sample
    # is preferred when it raises the hit count.
    post_records(
        client,
        [
            sample(1, "2026-09-23T08:00:30Z"),
            sample(2, "2026-09-23T08:00:55Z"),
        ],
    )
    body = query_adherence(
        client,
        window_to="2026-09-23T08:02:00Z",
        interval=60,
        tolerance=30,
    )
    assert body["planned_count"] == 2
    assert body["hit_count"] == 2
    assert body["points"][0]["sample"]["sequence"] == 1
    assert body["points"][0]["deviation_seconds"] == 30
    assert body["points"][1]["sample"]["sequence"] == 2
    assert body["points"][1]["deviation_seconds"] == -5


def test_matching_minimizes_total_absolute_deviation(client):
    # Points 08:00 and 08:01; samples 08:00:05 and 08:00:55, both inside
    # both tolerance windows. The crossed assignment (55->08:00, 5->08:01)
    # costs 55+56=111; the ordered assignment costs 5+5=10.
    post_records(
        client,
        [
            sample(1, "2026-09-23T08:00:05Z"),
            sample(2, "2026-09-23T08:00:55Z"),
        ],
    )
    body = query_adherence(
        client,
        window_to="2026-09-23T08:02:00Z",
        interval=60,
        tolerance=60,
    )
    assert body["hit_count"] == 2
    assert body["points"][0]["sample"]["sequence"] == 1
    assert body["points"][1]["sample"]["sequence"] == 2
    assert body["points"][0]["deviation_seconds"] == 5
    assert body["points"][1]["deviation_seconds"] == -5


def test_lexicographic_tie_break_prefers_earlier_sample(client):
    # One planned point, two samples equally 5 seconds off: the earlier
    # (sampled_at, sequence) pair wins.
    post_records(
        client,
        [
            sample(1, "2026-09-23T08:00:55Z"),
            sample(2, "2026-09-23T08:01:05Z"),
        ],
    )
    body = query_adherence(
        client,
        window_from="2026-09-23T08:01:00Z",
        window_to="2026-09-23T08:02:00Z",
        interval=60,
        tolerance=5,
    )
    assert body["hit_count"] == 1
    assert body["points"][0]["sample"]["sequence"] == 1
    assert body["points"][0]["deviation_seconds"] == -5


def test_lexicographic_tie_break_same_instant_uses_sequence(client):
    post_records(
        client,
        [
            sample(5, "2026-09-23T08:00:00Z"),
            sample(9, "2026-09-23T08:00:00Z"),
        ],
    )
    body = query_adherence(
        client, window_to="2026-09-23T08:01:00Z", interval=60, tolerance=5
    )
    assert body["hit_count"] == 1
    assert body["points"][0]["sample"]["sequence"] == 5


def test_lexicographic_tie_break_is_determined_per_match_list(client):
    # Points 08:00 and 08:02 with a 60s tolerance. Two samples share the
    # 08:01 instant, which is exactly on both windows' boundary; either
    # assignment yields two hits with the same total deviation (120s), so
    # the lexicographically smallest match list -- sequences in planned
    # order -- decides: seq 2 goes to the earlier point.
    post_records(
        client,
        [
            sample(2, "2026-09-23T08:01:00Z"),
            sample(3, "2026-09-23T08:01:00Z"),
        ],
    )
    body = query_adherence(
        client,
        window_to="2026-09-23T08:03:00Z",
        interval=120,
        tolerance=60,
    )
    assert body["planned_count"] == 2
    assert body["hit_count"] == 2
    assert body["points"][0]["sample"]["sequence"] == 2
    assert body["points"][0]["deviation_seconds"] == 60
    assert body["points"][1]["sample"]["sequence"] == 3
    assert body["points"][1]["deviation_seconds"] == -60


def test_unknown_device_returns_all_misses(client):
    body = query_adherence(
        client,
        device="sensor-unknown",
        window_to="2026-09-23T08:03:00Z",
        interval=60,
        tolerance=5,
    )
    assert body["device_id"] == "sensor-unknown"
    assert body["planned_count"] == 3
    assert body["hit_count"] == 0
    assert all(point["missed"] for point in body["points"])


def test_timezone_offsets_are_normalized_to_utc(client):
    post_records(client, [sample(1, "2026-09-23T08:00:00Z")])
    body = query_adherence(
        client,
        window_from="2026-09-23T16:00:00+08:00",
        window_to="2026-09-23T16:01:00+08:00",
        interval=60,
        tolerance=0,
    )
    assert body["from"] == "2026-09-23T08:00:00Z"
    assert body["to"] == "2026-09-23T08:01:00Z"
    assert body["points"][0]["planned_at"] == "2026-09-23T08:00:00Z"
    assert body["hit_count"] == 1


def test_fractional_sample_seconds_are_preserved(client):
    post_records(client, [sample(1, "2026-09-23T08:00:00.500000Z")])
    body = query_adherence(
        client, window_to="2026-09-23T08:01:00Z", interval=60, tolerance=1
    )
    assert body["hit_count"] == 1
    assert body["points"][0]["deviation_seconds"] == 0.5


def test_more_than_200_planned_instants_returns_422(client):
    # 201 points: 200 intervals of 60s == 12000s; window strictly larger.
    response = client.post(
        "/devices/sensor-a/sampling-adherence",
        json={
            "from": "2026-09-23T08:00:00Z",
            "to": "2026-09-23T11:20:01Z",
            "interval_seconds": 60,
            "tolerance_seconds": 5,
        },
    )
    assert response.status_code == 422


def test_exactly_200_planned_instants_is_accepted(client):
    response = client.post(
        "/devices/sensor-a/sampling-adherence",
        json={
            "from": "2026-09-23T08:00:00Z",
            "to": "2026-09-23T11:20:00Z",
            "interval_seconds": 60,
            "tolerance_seconds": 5,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["planned_count"] == 200
    assert body["hit_count"] == 0
    assert body["points"][-1]["planned_at"] == "2026-09-23T11:19:00Z"


def test_partial_last_interval_still_generates_a_point(client):
    # The final planned instant only needs to be strictly before to, so a
    # half-filled last interval still adds a point.
    response = client.post(
        "/devices/sensor-a/sampling-adherence",
        json={
            "from": "2026-09-23T08:00:00Z",
            "to": "2026-09-23T08:00:30Z",
            "interval_seconds": 20,
            "tolerance_seconds": 0,
        },
    )
    assert response.status_code == 200, response.text
    assert [point["planned_at"] for point in response.json()["points"]] == [
        "2026-09-23T08:00:00Z",
        "2026-09-23T08:00:20Z",
    ]


def test_boundary_at_201_points_depends_on_strict_before_to(client):
    # 60s interval, 12000s window: exactly 200 points. One extra second
    # makes the 201st instant (from + 12000s) strictly before to -> 422.
    ok = client.post(
        "/devices/sensor-a/sampling-adherence",
        json={
            "from": "2026-09-23T08:00:00Z",
            "to": "2026-09-23T11:20:00Z",
            "interval_seconds": 60,
            "tolerance_seconds": 60,
        },
    )
    assert ok.status_code == 200
    too_many = client.post(
        "/devices/sensor-a/sampling-adherence",
        json={
            "from": "2026-09-23T08:00:00Z",
            "to": "2026-09-23T11:20:00.000001Z",
            "interval_seconds": 60,
            "tolerance_seconds": 60,
        },
    )
    assert too_many.status_code == 422


def test_late_writes_are_reflected_in_the_next_query(client):
    body = query_adherence(
        client, window_to="2026-09-23T08:02:00Z", interval=60, tolerance=5
    )
    assert body["hit_count"] == 0
    post_records(
        client,
        [
            sample(1, "2026-09-23T08:00:00Z"),
            sample(2, "2026-09-23T08:01:00Z"),
        ],
    )
    body = query_adherence(
        client, window_to="2026-09-23T08:02:00Z", interval=60, tolerance=5
    )
    assert body["hit_count"] == 2


def test_corrections_are_reflected_in_the_next_query(client):
    post_records(
        client,
        [
            sample(1, "2026-09-23T08:00:00Z", 12.0),
            sample(2, "2026-09-23T08:09:00Z", 12.0),
        ],
    )
    body = query_adherence(
        client, window_to="2026-09-23T08:02:00Z", interval=60, tolerance=5
    )
    assert body["hit_count"] == 1
    assert body["points"][1]["missed"] is True

    response = client.post(
        "/records/correction",
        json={
            "device_id": "sensor-a",
            "sequence": 2,
            "old_sampled_at": "2026-09-23T08:09:00Z",
            "old_turbidity": 12.0,
            "new_sampled_at": "2026-09-23T08:01:00Z",
            "new_turbidity": 12.0,
        },
    )
    assert response.status_code == 200
    body = query_adherence(
        client, window_to="2026-09-23T08:02:00Z", interval=60, tolerance=5
    )
    assert body["hit_count"] == 2
    assert body["points"][1]["sample"]["sequence"] == 2


def test_query_does_not_modify_events_acks_or_change_stream(client):
    post_records(
        client,
        [
            sample(1, "2026-09-23T08:00:00Z", 12.0),
            sample(2, "2026-09-23T08:05:00Z", 13.0),
        ],
    )
    events = get_events(client, "sensor-a")
    assert ack_event(client, events["events"][0]).status_code == 200
    events_before = get_events(client, "sensor-a")
    changes_before = client.get(
        "/event-changes", params={"after": 0, "limit": 1000}
    ).json()

    body = query_adherence(
        client, window_to="2026-09-23T08:06:00Z", interval=60, tolerance=5
    )
    assert body["hit_count"] == 2

    assert get_events(client, "sensor-a") == events_before
    assert (
        client.get("/event-changes", params={"after": 0, "limit": 1000}).json()
        == changes_before
    )


def test_device_isolation_between_queries(client):
    post_records(
        client, [sample(1, "2026-09-23T08:00:00Z", device="sensor-b")]
    )
    body = query_adherence(
        client,
        device="sensor-a",
        window_to="2026-09-23T08:01:00Z",
        interval=60,
        tolerance=0,
    )
    assert body["hit_count"] == 0
    body = query_adherence(
        client,
        device="sensor-b",
        window_to="2026-09-23T08:01:00Z",
        interval=60,
        tolerance=0,
    )
    assert body["hit_count"] == 1


@pytest.mark.parametrize(
    "payload",
    [
        # from >= to
        {
            "from": "2026-09-23T09:00:00Z",
            "to": "2026-09-23T08:00:00Z",
            "interval_seconds": 60,
            "tolerance_seconds": 5,
        },
        # equal instants
        {
            "from": "2026-09-23T08:00:00Z",
            "to": "2026-09-23T08:00:00Z",
            "interval_seconds": 60,
            "tolerance_seconds": 5,
        },
        # naive timestamps
        {
            "from": "2026-09-23T08:00:00",
            "to": "2026-09-23T09:00:00Z",
            "interval_seconds": 60,
            "tolerance_seconds": 5,
        },
        {
            "from": "2026-09-23T08:00:00Z",
            "to": "2026-09-23T09:00:00",
            "interval_seconds": 60,
            "tolerance_seconds": 5,
        },
        # non-positive interval
        {
            "from": "2026-09-23T08:00:00Z",
            "to": "2026-09-23T09:00:00Z",
            "interval_seconds": 0,
            "tolerance_seconds": 0,
        },
        {
            "from": "2026-09-23T08:00:00Z",
            "to": "2026-09-23T09:00:00Z",
            "interval_seconds": -60,
            "tolerance_seconds": 5,
        },
        # negative tolerance
        {
            "from": "2026-09-23T08:00:00Z",
            "to": "2026-09-23T09:00:00Z",
            "interval_seconds": 60,
            "tolerance_seconds": -1,
        },
        # tolerance above interval
        {
            "from": "2026-09-23T08:00:00Z",
            "to": "2026-09-23T09:00:00Z",
            "interval_seconds": 60,
            "tolerance_seconds": 61,
        },
    ],
)
def test_invalid_requests_return_422(client, payload):
    response = client.post(
        "/devices/sensor-a/sampling-adherence", json=payload
    )
    assert response.status_code == 422, payload


def test_missing_and_extra_fields_return_422(client):
    base = {
        "from": "2026-09-23T08:00:00Z",
        "to": "2026-09-23T09:00:00Z",
        "interval_seconds": 60,
        "tolerance_seconds": 5,
    }
    for key in ("from", "to", "interval_seconds", "tolerance_seconds"):
        payload = {name: value for name, value in base.items() if name != key}
        response = client.post(
            "/devices/sensor-a/sampling-adherence", json=payload
        )
        assert response.status_code == 422, key
    response = client.post(
        "/devices/sensor-a/sampling-adherence", json={**base, "extra": 1}
    )
    assert response.status_code == 422
