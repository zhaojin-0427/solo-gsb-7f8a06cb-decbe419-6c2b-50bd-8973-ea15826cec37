from __future__ import annotations

from fastapi.testclient import TestClient

from tests.conftest import ack_event, get_events, post_records

PREVIEW_PATH = "/maintenance/alarm-rule-preview"


def sample(sequence: int, at: str, turbidity: float, device: str = "sensor-a"):
    return {
        "device_id": device,
        "sequence": sequence,
        "sampled_at": at,
        "turbidity": turbidity,
    }


def preview(
    client: TestClient,
    device_ids: list[str] | None = None,
    *,
    alarm: float = 12.0,
    recovery: float = 6.0,
    gap: int = 600,
    expected: int = 200,
    **extra,
) -> dict:
    payload: dict = {
        "device_ids": device_ids if device_ids is not None else ["sensor-a"],
        "alarm_threshold": alarm,
        "recovery_threshold": recovery,
        "max_sample_gap_seconds": gap,
    }
    payload.update(extra)
    response = client.post(PREVIEW_PATH, json=payload)
    assert response.status_code == expected, response.text
    return response.json()


def event_ids(events: list[dict]) -> list[str]:
    return [event["event_id"] for event in events]


def test_field_change_returns_both_old_and_new_event(client):
    # Current rules 10/5: the 4.0 sample is a recovery. Candidate 8/2:
    # 4.0 lies in the hysteresis band, so the same event stays active.
    post_records(
        client,
        [
            sample(1, "2026-09-23T08:00:00Z", 12.0),
            sample(2, "2026-09-23T08:05:00Z", 4.0),
        ],
    )
    body = preview(client, alarm=8.0, recovery=2.0)

    assert body["device_ids"] == ["sensor-a"]
    assert body["current_rules"] == {
        "alarm_threshold": 10.0,
        "recovery_threshold": 5.0,
        "max_sample_gap_seconds": 600,
    }
    assert body["candidate_rules"] == {
        "alarm_threshold": 8.0,
        "recovery_threshold": 2.0,
        "max_sample_gap_seconds": 600,
    }
    device = body["devices"][0]
    assert device["device_id"] == "sensor-a"
    assert len(device["current_events"]) == 1
    assert len(device["candidate_events"]) == 1
    assert device["added"] == []
    assert device["removed"] == []

    changes = device["field_changes"]
    assert len(changes) == 1
    change = changes[0]
    assert change["event_id"] == "evt:sensor-a:1"
    old = change["current_event"]
    new = change["candidate_event"]
    assert old["status"] == "recovered"
    assert old["end_sequence"] == 2
    assert old["end_at"] == "2026-09-23T08:05:00Z"
    assert new["status"] == "active"
    assert new["end_sequence"] is None
    assert new["end_at"] is None
    # Fields shared by both versions stay identical.
    for field in (
        "event_id",
        "device_id",
        "start_sequence",
        "start_at",
        "peak_turbidity",
        "peak_sequence",
        "sample_count",
    ):
        assert old[field] == new[field]


def test_added_and_removed_events_when_trigger_moves(client):
    # 7.0 is nothing under alarm 10 but triggers under alarm 6.
    post_records(
        client,
        [
            sample(1, "2026-09-23T08:00:00Z", 7.0),
            sample(2, "2026-09-23T08:05:00Z", 12.0),
            sample(3, "2026-09-23T08:10:00Z", 4.0),
        ],
    )
    body = preview(client, alarm=6.0, recovery=3.0)
    device = body["devices"][0]

    assert event_ids(device["current_events"]) == ["evt:sensor-a:2"]
    assert event_ids(device["candidate_events"]) == ["evt:sensor-a:1"]
    assert event_ids(device["added"]) == ["evt:sensor-a:1"]
    assert event_ids(device["removed"]) == ["evt:sensor-a:2"]
    assert device["field_changes"] == []

    added = device["added"][0]
    assert added["status"] == "active"
    assert added["start_sequence"] == 1
    assert added["sample_count"] == 3
    removed = device["removed"][0]
    assert removed["status"] == "recovered"
    assert removed["start_sequence"] == 2
    assert removed["sample_count"] == 2


def test_candidate_equal_to_current_rules_lists_no_differences(client):
    post_records(
        client,
        [
            sample(1, "2026-09-23T08:00:00Z", 7.0),
            sample(2, "2026-09-23T08:05:00Z", 12.0),
            sample(3, "2026-09-23T08:10:00Z", 4.0),
        ],
    )
    body = preview(client, alarm=10.0, recovery=5.0, gap=600)
    device = body["devices"][0]
    assert device["current_events"] == device["candidate_events"]
    assert device["added"] == []
    assert device["field_changes"] == []
    assert device["removed"] == []
    assert body["current_rules"] == body["candidate_rules"]


def test_different_rules_that_derive_equal_events_list_no_changes(client):
    # 12.0 satisfies alarm >= and 4.0 satisfies recovery <= under both
    # (10, 5) and (12, 4); the derived events are identical field by field.
    post_records(
        client,
        [
            sample(1, "2026-09-23T08:00:00Z", 12.0),
            sample(2, "2026-09-23T08:05:00Z", 4.0),
        ],
    )
    body = preview(client, alarm=12.0, recovery=4.0)
    device = body["devices"][0]
    assert device["current_events"] == device["candidate_events"]
    assert device["added"] == []
    assert device["field_changes"] == []
    assert device["removed"] == []


def test_wider_gap_merges_interrupted_segments(client):
    # The 1200s gap interrupts under the current 600s limit.
    post_records(
        client,
        [
            sample(1, "2026-09-23T08:00:00Z", 12.0),
            sample(2, "2026-09-23T08:20:00Z", 12.0),
        ],
    )
    body = preview(client, alarm=10.0, recovery=5.0, gap=1500)
    device = body["devices"][0]

    assert event_ids(device["current_events"]) == [
        "evt:sensor-a:1",
        "evt:sensor-a:2",
    ]
    assert event_ids(device["candidate_events"]) == ["evt:sensor-a:1"]
    assert device["added"] == []
    assert event_ids(device["removed"]) == ["evt:sensor-a:2"]

    change = device["field_changes"][0]
    assert change["event_id"] == "evt:sensor-a:1"
    assert change["current_event"]["status"] == "interrupted"
    assert change["current_event"]["end_sequence"] == 1
    assert change["current_event"]["end_at"] == "2026-09-23T08:00:00Z"
    assert change["current_event"]["sample_count"] == 1
    assert change["candidate_event"]["status"] == "active"
    assert change["candidate_event"]["end_sequence"] is None
    assert change["candidate_event"]["sample_count"] == 2


def test_narrower_gap_splits_one_event_into_two(client):
    # 600s is continuous under the current limit; 300s is not.
    post_records(
        client,
        [
            sample(1, "2026-09-23T08:00:00Z", 12.0),
            sample(2, "2026-09-23T08:10:00Z", 12.0),
        ],
    )
    body = preview(client, alarm=10.0, recovery=5.0, gap=300)
    device = body["devices"][0]

    assert event_ids(device["candidate_events"]) == [
        "evt:sensor-a:1",
        "evt:sensor-a:2",
    ]
    assert event_ids(device["current_events"]) == ["evt:sensor-a:1"]
    assert event_ids(device["added"]) == ["evt:sensor-a:2"]
    assert device["removed"] == []
    change = device["field_changes"][0]
    assert change["current_event"]["status"] == "active"
    assert change["current_event"]["sample_count"] == 2
    assert change["candidate_event"]["status"] == "interrupted"
    assert change["candidate_event"]["sample_count"] == 1
    assert change["candidate_event"]["end_at"] == "2026-09-23T08:00:00Z"


def test_gap_equal_to_candidate_limit_stays_continuous(client):
    post_records(
        client,
        [
            sample(1, "2026-09-23T08:00:00Z", 12.0),
            sample(2, "2026-09-23T08:10:00Z", 12.0),
        ],
    )
    body = preview(client, alarm=10.0, recovery=5.0, gap=600)
    device = body["devices"][0]
    assert event_ids(device["candidate_events"]) == ["evt:sensor-a:1"]
    assert device["candidate_events"][0]["status"] == "active"
    assert device["field_changes"] == []
    assert device["added"] == []
    assert device["removed"] == []


def test_unknown_device_returns_empty_lists(client):
    body = preview(client, ["sensor-ghost"])
    assert body["device_ids"] == ["sensor-ghost"]
    device = body["devices"][0]
    assert device == {
        "device_id": "sensor-ghost",
        "current_events": [],
        "candidate_events": [],
        "added": [],
        "field_changes": [],
        "removed": [],
    }


def test_device_with_only_low_samples_returns_empty_events(client):
    post_records(
        client,
        [
            sample(1, "2026-09-23T08:00:00Z", 1.0),
            sample(2, "2026-09-23T08:05:00Z", 2.0),
        ],
    )
    body = preview(client, alarm=10.0, recovery=5.0)
    device = body["devices"][0]
    assert device["current_events"] == []
    assert device["candidate_events"] == []
    assert device["added"] == []
    assert device["field_changes"] == []
    assert device["removed"] == []


def test_multiple_devices_are_sorted_and_isolated(client):
    post_records(client, [sample(1, "2026-09-23T08:00:00Z", 12.0)])
    post_records(
        client, [sample(1, "2026-09-23T08:00:00Z", 20.0, device="sensor-b")]
    )
    body = preview(client, ["  sensor-b ", "sensor-a", "ghost"], alarm=15.0)

    assert body["device_ids"] == ["ghost", "sensor-a", "sensor-b"]
    assert [device["device_id"] for device in body["devices"]] == [
        "ghost",
        "sensor-a",
        "sensor-b",
    ]

    ghost_preview, a_preview, b_preview = body["devices"]
    # sensor-a's 12.0 sample stops triggering at the candidate alarm 15.
    assert event_ids(a_preview["current_events"]) == ["evt:sensor-a:1"]
    assert a_preview["candidate_events"] == []
    assert event_ids(a_preview["removed"]) == ["evt:sensor-a:1"]
    assert a_preview["added"] == []
    # sensor-b's 20.0 still triggers; the event is unchanged.
    assert event_ids(b_preview["current_events"]) == ["evt:sensor-b:1"]
    assert b_preview["candidate_events"] == b_preview["current_events"]
    assert b_preview["field_changes"] == []
    assert ghost_preview["current_events"] == []
    assert ghost_preview["candidate_events"] == []


def test_events_are_ordered_by_event_id_string(client):
    # Event start sequences 2 and 10: the string order puts :10 before :2.
    post_records(
        client,
        [
            sample(2, "2026-09-23T08:00:00Z", 12.0),
            sample(3, "2026-09-23T08:05:00Z", 4.0),
            sample(10, "2026-09-23T08:10:00Z", 12.0),
            sample(11, "2026-09-23T08:15:00Z", 4.0),
        ],
    )
    body = preview(client, alarm=10.0, recovery=5.0)
    device = body["devices"][0]
    expected_order = ["evt:sensor-a:10", "evt:sensor-a:2"]
    assert event_ids(device["current_events"]) == expected_order
    assert event_ids(device["candidate_events"]) == expected_order


def test_zero_recovery_threshold_is_allowed(client):
    post_records(
        client,
        [
            sample(1, "2026-09-23T08:00:00Z", 12.0),
            sample(2, "2026-09-23T08:05:00Z", 0.0),
        ],
    )
    body = preview(client, alarm=1.0, recovery=0.0)
    candidate = body["devices"][0]["candidate_events"]
    assert len(candidate) == 1
    assert candidate[0]["status"] == "recovered"
    assert candidate[0]["end_sequence"] == 2


def test_boundary_device_counts_are_accepted(client):
    body = preview(client, ["sensor-only"])
    assert body["device_ids"] == ["sensor-only"]

    twenty = [f"sensor-{index:02d}" for index in range(20)]
    body = preview(client, twenty)
    assert len(body["devices"]) == 20


def test_preview_does_not_modify_events_acks_changes_or_settings(client):
    post_records(
        client,
        [
            sample(1, "2026-09-23T08:00:00Z", 12.0),
            sample(2, "2026-09-23T08:05:00Z", 4.0),
        ],
    )
    events_before = get_events(client, "sensor-a")
    assert ack_event(client, events_before["events"][0]).status_code == 200
    events_before = get_events(client, "sensor-a")
    changes_before = client.get(
        "/event-changes", params={"after": 0, "limit": 1000}
    ).json()
    thresholds_before = client.get("/thresholds").json()

    body = preview(client, alarm=8.0, recovery=2.0, gap=300)
    assert body["devices"][0]["field_changes"]

    assert get_events(client, "sensor-a") == events_before
    assert (
        client.get(
            "/event-changes", params={"after": 0, "limit": 1000}
        ).json()
        == changes_before
    )
    assert client.get("/thresholds").json() == thresholds_before


def test_invalid_device_lists_return_422(client):
    base = {
        "alarm_threshold": 12.0,
        "recovery_threshold": 6.0,
        "max_sample_gap_seconds": 600,
    }
    # Empty list and 21 devices.
    response = client.post(PREVIEW_PATH, json={**base, "device_ids": []})
    assert response.status_code == 422
    twenty_one = [f"sensor-{index}" for index in range(21)]
    response = client.post(
        PREVIEW_PATH, json={**base, "device_ids": twenty_one}
    )
    assert response.status_code == 422
    # Duplicates, including duplicates revealed by whitespace stripping.
    for device_ids in (
        ["sensor-a", "sensor-a"],
        [" sensor-a ", "sensor-a"],
        ["   ", "sensor-b"],
    ):
        response = client.post(
            PREVIEW_PATH, json={**base, "device_ids": device_ids}
        )
        assert response.status_code == 422, device_ids


def test_invalid_thresholds_return_422(client):
    for alarm, recovery in (
        (5.0, 10.0),  # alarm below recovery
        (10.0, 10.0),  # equal thresholds
        (-0.1, 5.0),
        (10.0, -0.1),
    ):
        response = client.post(
            PREVIEW_PATH,
            json={
                "device_ids": ["sensor-a"],
                "alarm_threshold": alarm,
                "recovery_threshold": recovery,
                "max_sample_gap_seconds": 600,
            },
        )
        assert response.status_code == 422, (alarm, recovery)


def test_non_finite_thresholds_return_422(client):
    # httpx refuses to serialize non-finite floats, so send the raw JSON
    # literals (Python's parser accepts them) as a strict client would.
    for literal in ("NaN", "Infinity", "-Infinity"):
        response = client.post(
            PREVIEW_PATH,
            content=(
                '{"device_ids": ["sensor-a"], '
                f'"alarm_threshold": {literal}, '
                '"recovery_threshold": 5.0, "max_sample_gap_seconds": 600}'
            ),
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422, literal
        response = client.post(
            PREVIEW_PATH,
            content=(
                '{"device_ids": ["sensor-a"], '
                '"alarm_threshold": 10.0, '
                f'"recovery_threshold": {literal}, '
                '"max_sample_gap_seconds": 600}'
            ),
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422, literal


def test_invalid_gap_returns_422(client):
    base = {"device_ids": ["sensor-a"], "alarm_threshold": 10.0,
            "recovery_threshold": 5.0}
    for gap in (0, -1, 1.5, "ten"):
        response = client.post(
            PREVIEW_PATH, json={**base, "max_sample_gap_seconds": gap}
        )
        assert response.status_code == 422, gap


def test_missing_and_extra_fields_return_422(client):
    base = {
        "device_ids": ["sensor-a"],
        "alarm_threshold": 12.0,
        "recovery_threshold": 6.0,
        "max_sample_gap_seconds": 600,
    }
    for key in (
        "device_ids",
        "alarm_threshold",
        "recovery_threshold",
        "max_sample_gap_seconds",
    ):
        payload = {name: value for name, value in base.items() if name != key}
        response = client.post(PREVIEW_PATH, json=payload)
        assert response.status_code == 422, key
    response = client.post(PREVIEW_PATH, json={**base, "extra": 1})
    assert response.status_code == 422
