from __future__ import annotations

import math

import pytest
from fastapi.testclient import TestClient

from tests.conftest import get_events, post_records, strip_acknowledged


def sample(sequence: int, at: str, turbidity: float, device: str = "sensor-a"):
    return {
        "device_id": device,
        "sequence": sequence,
        "sampled_at": at,
        "turbidity": turbidity,
    }


def preview(
    client: TestClient,
    device_ids: list[str],
    *,
    alarm: float = 8.0,
    recovery: float = 4.0,
    gap: int = 600,
):
    return client.post(
        "/maintenance/alarm-rule-preview",
        json={
            "device_ids": device_ids,
            "alarm_threshold": alarm,
            "recovery_threshold": recovery,
            "max_sample_gap_seconds": gap,
        },
    )


def device_body(body: dict, device: str) -> dict:
    return next(
        entry for entry in body["devices"] if entry["device_id"] == device
    )


def test_preview_reports_rules_and_empty_device_for_unknown_device(client):
    response = preview(client, ["sensor-unknown"])
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["device_ids"] == ["sensor-unknown"]
    assert body["current_rule"] == {
        "alarm_threshold": 10.0,
        "recovery_threshold": 5.0,
        "max_sample_gap_seconds": 600,
    }
    assert body["candidate_rule"] == {
        "alarm_threshold": 8.0,
        "recovery_threshold": 4.0,
        "max_sample_gap_seconds": 600,
    }
    assert body["devices"] == [
        {
            "device_id": "sensor-unknown",
            "current_events": [],
            "candidate_events": [],
            "added": [],
            "changed": [],
            "removed": [],
        }
    ]


def test_preview_with_current_rule_reports_no_differences(client):
    post_records(
        client,
        [
            sample(1, "2026-09-23T08:00:00Z", 12),
            sample(2, "2026-09-23T08:01:00Z", 6),
            sample(3, "2026-09-23T08:02:00Z", 5),
        ],
    )
    current = strip_acknowledged(get_events(client)["events"])

    response = preview(
        client, ["sensor-a"], alarm=10.0, recovery=5.0, gap=600
    )
    assert response.status_code == 200
    entry = device_body(response.json(), "sensor-a")
    assert entry["current_events"] == current
    assert entry["candidate_events"] == entry["current_events"]
    assert entry["added"] == []
    assert entry["changed"] == []
    assert entry["removed"] == []


def test_higher_alarm_threshold_marks_events_removed(client):
    post_records(client, [sample(1, "2026-09-23T08:00:00Z", 12)])
    current = strip_acknowledged(get_events(client)["events"])

    response = preview(client, ["sensor-a"], alarm=15.0, recovery=5.0)
    assert response.status_code == 200
    entry = device_body(response.json(), "sensor-a")
    assert entry["current_events"] == current
    assert entry["candidate_events"] == []
    assert entry["added"] == []
    assert entry["changed"] == []
    assert entry["removed"] == [
        {
            "change_type": "removed",
            "event_id": "evt:sensor-a:1",
            "old_event": current[0],
            "new_event": None,
        }
    ]


def test_lower_alarm_threshold_marks_events_added(client):
    post_records(client, [sample(1, "2026-09-23T08:00:00Z", 8)])
    assert get_events(client)["events"] == []

    response = preview(client, ["sensor-a"], alarm=7.0, recovery=4.0)
    assert response.status_code == 200
    entry = device_body(response.json(), "sensor-a")
    assert entry["current_events"] == []
    assert len(entry["candidate_events"]) == 1
    new_event = entry["candidate_events"][0]
    assert new_event["event_id"] == "evt:sensor-a:1"
    assert new_event["status"] == "active"
    assert entry["removed"] == []
    assert entry["changed"] == []
    assert entry["added"] == [
        {
            "change_type": "added",
            "event_id": "evt:sensor-a:1",
            "old_event": None,
            "new_event": new_event,
        }
    ]


def test_same_event_id_with_changed_fields_embeds_bold_old_and_new(client):
    post_records(
        client,
        [
            sample(1, "2026-09-23T08:00:00Z", 12),
            sample(2, "2026-09-23T08:01:00Z", 6),
            sample(3, "2026-09-23T08:02:00Z", 20),
            sample(4, "2026-09-23T08:03:00Z", 5),
        ],
    )
    current = strip_acknowledged(get_events(client)["events"])
    assert len(current) == 1
    old_event = current[0]
    assert old_event["status"] == "recovered"
    assert old_event["end_at"] == "2026-09-23T08:03:00Z"

    # A lower recovery threshold keeps the same trigger/event_id but the
    # sample at 5.0 no longer recovers: the event stays active.
    response = preview(client, ["sensor-a"], alarm=10.0, recovery=3.0)
    assert response.status_code == 200
    entry = device_body(response.json(), "sensor-a")
    assert entry["added"] == []
    assert entry["removed"] == []
    assert len(entry["changed"]) == 1
    change = entry["changed"][0]
    assert change["event_id"] == "evt:sensor-a:1"
    assert change["old_event"] == old_event
    new_event = change["new_event"]
    assert new_event is not None
    assert new_event["status"] == "active"
    assert new_event["end_at"] is None
    assert new_event["end_sequence"] is None
    assert new_event["sample_count"] == 4
    assert new_event["peak_sequence"] == 3
    assert entry["candidate_events"] == [new_event]
    # Equal values never show up as changes.
    assert change["old_event"] != change["new_event"]


def test_wider_gap_merges_interrupted_segments_into_changed_and_removed(client):
    post_records(
        client,
        [
            sample(1, "2026-09-23T08:00:00Z", 12),
            sample(2, "2026-09-23T08:20:00Z", 15),
        ],
    )
    current = strip_acknowledged(get_events(client)["events"])
    assert [event["event_id"] for event in current] == [
        "evt:sensor-a:1",
        "evt:sensor-a:2",
    ]
    assert current[0]["status"] == "interrupted"

    response = preview(client, ["sensor-a"], alarm=10.0, recovery=5.0, gap=1800)
    assert response.status_code == 200
    entry = device_body(response.json(), "sensor-a")
    assert [event["event_id"] for event in entry["candidate_events"]] == [
        "evt:sensor-a:1"
    ]
    merged = entry["candidate_events"][0]
    assert merged["status"] == "active"
    assert merged["peak_turbidity"] == 15.0
    assert merged["sample_count"] == 2
    assert [change["event_id"] for change in entry["changed"]] == [
        "evt:sensor-a:1"
    ]
    assert [change["event_id"] for change in entry["removed"]] == [
        "evt:sensor-a:2"
    ]
    assert entry["added"] == []
    changed = entry["changed"][0]
    assert changed["old_event"] == current[0]
    assert changed["new_event"] == merged


def test_narrower_gap_splits_events_into_changed_and_added(client):
    # 600 s apart: continuous under the current 600 s limit (equal limit
    # stays continuous), so one active event exists.
    post_records(
        client,
        [
            sample(1, "2026-09-23T08:00:00Z", 12),
            sample(2, "2026-09-23T08:10:00Z", 15),
        ],
    )
    assert [e["event_id"] for e in get_events(client)["events"]] == [
        "evt:sensor-a:1"
    ]

    response = preview(client, ["sensor-a"], alarm=10.0, recovery=5.0, gap=300)
    entry = device_body(response.json(), "sensor-a")
    candidate_ids = [e["event_id"] for e in entry["candidate_events"]]
    assert candidate_ids == ["evt:sensor-a:1", "evt:sensor-a:2"]
    assert [c["event_id"] for c in entry["changed"]] == ["evt:sensor-a:1"]
    assert [c["event_id"] for c in entry["added"]] == ["evt:sensor-a:2"]
    assert entry["removed"] == []
    # The interrupted first event ends at the first sample (no invented
    # recovery sample).
    first = entry["candidate_events"][0]
    assert first["status"] == "interrupted"
    assert first["end_sequence"] == 1
    assert first["end_at"] == "2026-09-23T08:00:00Z"


def test_devices_sorted_and_unknown_devices_return_empty(client):
    post_records(
        client,
        [
            sample(1, "2026-09-23T08:00:00Z", 12, "sensor-c"),
            sample(1, "2026-09-23T08:00:00Z", 8, "sensor-a"),
        ],
    )
    response = preview(
        client, ["sensor-c", "ghost", "sensor-a"], alarm=7.0, recovery=4.0
    )
    assert response.status_code == 200
    body = response.json()
    assert body["device_ids"] == ["ghost", "sensor-a", "sensor-c"]
    assert [entry["device_id"] for entry in body["devices"]] == [
        "ghost",
        "sensor-a",
        "sensor-c",
    ]
    ghost = device_body(body, "ghost")
    assert ghost["current_events"] == []
    assert ghost["candidate_events"] == []
    assert ghost["added"] == []
    sensor_a = device_body(body, "sensor-a")
    assert len(sensor_a["added"]) == 1
    sensor_c = device_body(body, "sensor-c")
    assert sensor_c["changed"] == sensor_c["added"] == sensor_c["removed"] == []


def test_event_lists_sorted_by_event_id_string(client):
    # Trigger samples with sequences 10 (08:00) and 2 (08:20): the wide
    # gap produces evt:...:10 and evt:...:2, which only sort in this
    # order lexicographically as event_id strings.
    post_records(
        client,
        [
            sample(10, "2026-09-23T08:00:00Z", 12),
            sample(2, "2026-09-23T08:20:00Z", 12),
        ],
    )
    response = preview(client, ["sensor-a"], alarm=10.0, recovery=5.0)
    assert response.status_code == 200
    entry = device_body(response.json(), "sensor-a")
    expected_ids = ["evt:sensor-a:10", "evt:sensor-a:2"]
    assert [e["event_id"] for e in entry["current_events"]] == expected_ids
    assert [e["event_id"] for e in entry["candidate_events"]] == expected_ids


def test_preview_is_read_only_across_config_events_acks_and_changes(client):
    post_records(
        client,
        [
            sample(1, "2026-09-23T08:00:00Z", 12),
            sample(2, "2026-09-23T08:01:00Z", 5),
        ],
    )
    event = get_events(client)["events"][0]
    ack_response = client.post(
        "/devices/sensor-a/events/evt:sensor-a:1/ack",
        json={
            "expected_event": {
                key: value
                for key, value in event.items()
                if key != "acknowledged"
            }
        },
    )
    assert ack_response.status_code == 200
    head = client.get("/event-changes").json()["head_change_id"]
    thresholds_before = client.get("/thresholds").json()

    # A dramatically different candidate rule must leave nothing changed.
    response = preview(client, ["sensor-a"], alarm=99.0, recovery=0.0, gap=1)
    assert response.status_code == 200

    assert client.get("/thresholds").json() == thresholds_before
    events_after = get_events(client)["events"]
    assert events_after == [{**event, "acknowledged": True}]
    changes = client.get("/event-changes").json()
    assert changes["head_change_id"] == head
    assert len(changes["changes"]) == 1


@pytest.mark.parametrize(
    "payload",
    [
        # No devices / too many devices.
        {"device_ids": [], "alarm_threshold": 8, "recovery_threshold": 4,
         "max_sample_gap_seconds": 600},
        {"device_ids": [f"sensor-{i}" for i in range(21)],
         "alarm_threshold": 8, "recovery_threshold": 4,
         "max_sample_gap_seconds": 600},
        # Duplicate devices, also after whitespace normalization.
        {"device_ids": ["sensor-a", " sensor-a "], "alarm_threshold": 8,
         "recovery_threshold": 4, "max_sample_gap_seconds": 600},
        # Blank device id.
        {"device_ids": ["   "], "alarm_threshold": 8,
         "recovery_threshold": 4, "max_sample_gap_seconds": 600},
        # Threshold ordering.
        {"device_ids": ["sensor-a"], "alarm_threshold": 4,
         "recovery_threshold": 8, "max_sample_gap_seconds": 600},
        {"device_ids": ["sensor-a"], "alarm_threshold": 5,
         "recovery_threshold": 5, "max_sample_gap_seconds": 600},
        # Negative thresholds and non-positive/non-integer gaps.
        {"device_ids": ["sensor-a"], "alarm_threshold": -1.0,
         "recovery_threshold": 4, "max_sample_gap_seconds": 600},
        {"device_ids": ["sensor-a"], "alarm_threshold": 8,
         "recovery_threshold": -0.1, "max_sample_gap_seconds": 600},
        {"device_ids": ["sensor-a"], "alarm_threshold": 8,
         "recovery_threshold": 4, "max_sample_gap_seconds": 0},
        {"device_ids": ["sensor-a"], "alarm_threshold": 8,
         "recovery_threshold": 4, "max_sample_gap_seconds": -10},
        {"device_ids": ["sensor-a"], "alarm_threshold": 8,
         "recovery_threshold": 4, "max_sample_gap_seconds": 1.5},
        # Missing fields.
        {"device_ids": ["sensor-a"], "recovery_threshold": 4,
         "max_sample_gap_seconds": 600},
        {"device_ids": ["sensor-a"], "alarm_threshold": 8,
         "max_sample_gap_seconds": 600},
        {"device_ids": ["sensor-a"], "alarm_threshold": 8,
         "recovery_threshold": 4},
        # Extra field.
        {"device_ids": ["sensor-a"], "alarm_threshold": 8,
         "recovery_threshold": 4, "max_sample_gap_seconds": 600,
         "unexpected": True},
    ],
)
def test_invalid_requests_return_422(client, payload):
    response = client.post("/maintenance/alarm-rule-preview", json=payload)
    assert response.status_code == 422, response.text


def test_non_finite_thresholds_return_422(client):
    for literal in ("NaN", "Infinity", "-Infinity"):
        response = client.post(
            "/maintenance/alarm-rule-preview",
            content=(
                '{"device_ids": ["sensor-a"], "alarm_threshold": '
                f"{literal}"
                ', "recovery_threshold": 4.0, '
                '"max_sample_gap_seconds": 600}'
            ),
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422, literal
        # The non-finite value must not break the strict JSON renderer.
        assert response.headers["content-type"] == "application/json"
    assert math.isnan(float("nan"))


def test_preview_supports_twenty_distinct_devices(client):
    response = preview(client, [f"sensor-{i:02d}" for i in range(20)])
    assert response.status_code == 200
    body = response.json()
    assert len(body["devices"]) == 20
    assert body["device_ids"] == [f"sensor-{i:02d}" for i in range(20)]


def test_zero_threshold_pair_is_valid_and_gate_keeps_hysteresis(client):
    # alarm 1.0 > recovery 0.0 is allowed; a zero sample stays below the
    # gate and no event is derived.
    post_records(client, [sample(1, "2026-09-23T08:00:00Z", 0.0)])
    response = preview(client, ["sensor-a"], alarm=1.0, recovery=0.0)
    assert response.status_code == 200
    entry = device_body(response.json(), "sensor-a")
    assert entry["candidate_events"] == []
    assert entry["removed"] == []


def test_preview_applies_same_peak_first_hit_rule(client):
    post_records(
        client,
        [
            sample(1, "2026-09-23T08:00:00Z", 12),
            sample(2, "2026-09-23T08:01:00Z", 12),
            sample(3, "2026-09-23T08:02:00Z", 4),
        ],
    )
    # Raising the alarm gate above the samples removes the event; lowering
    # only the recovery threshold keeps trigger and peak identical, so the
    # equal peak is attributed to the first hit either way.
    response = preview(client, ["sensor-a"], alarm=10.0, recovery=3.0)
    entry = device_body(response.json(), "sensor-a")
    new_event = entry["candidate_events"][0]
    assert new_event["peak_sequence"] == 1
    assert new_event["peak_turbidity"] == 12.0
    change = entry["changed"][0]
    assert change["old_event"]["peak_sequence"] == 1
