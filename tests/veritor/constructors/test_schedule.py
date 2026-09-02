"""Continual-batching schedules: canonical bytes, derived occupancy, FCFS."""

from __future__ import annotations

import pytest

from veritor.constructors.schedule import (
    Join,
    Occupant,
    Request,
    Schedule,
    ScheduleError,
    schedule_fcfs,
)

REQUESTS = (
    Request((5, 6, 7), max_new=3),
    Request((1,), max_new=2),
    Request((9, 9), max_new=5),
    Request((2, 3), max_new=1),
)


def test_fcfs_fills_free_slots_in_order_and_reuses_them() -> None:
    schedule = schedule_fcfs(REQUESTS, pods=1, slots=2, steps=6)

    assert schedule.joins == (
        Join(0, 0, 0, 0),  # request 0 takes slot 0 for steps 0..2
        Join(0, 0, 1, 1),  # request 1 takes slot 1 for steps 0..1
        Join(0, 2, 1, 2),  # slot 1 frees at step 2: request 2 joins
        Join(0, 3, 0, 3),  # slot 0 frees at step 3: request 3 joins
    )
    assert schedule.active_steps(REQUESTS) == {0: 3, 1: 2, 2: 4, 3: 1}  # request 2 is cut by the run's end
    occupancy = schedule.occupancy(REQUESTS)
    assert occupancy[(0, 0)] == (Occupant(0, 0, 0), Occupant(1, 1, 0))
    assert occupancy[(0, 2)] == (Occupant(0, 0, 2), Occupant(1, 2, 0))
    assert occupancy[(0, 3)] == (Occupant(0, 3, 0), Occupant(1, 2, 1))
    assert occupancy[(0, 5)] == (Occupant(1, 2, 3),)
    assert sorted(occupancy) == [(0, t) for t in range(6)]


def test_fcfs_spreads_over_pods_and_reports_unscheduled_requests() -> None:
    schedule = schedule_fcfs(REQUESTS, pods=2, slots=1, steps=2)

    assert schedule.joins == (Join(0, 0, 0, 0), Join(1, 0, 0, 1))
    with pytest.raises(ScheduleError, match=r"never scheduled: \[2, 3\]"):
        schedule.validate(REQUESTS)


def test_a_reassignment_cuts_the_previous_request_short() -> None:
    schedule = Schedule(1, 1, 4, (Join(0, 0, 0, 0), Join(0, 2, 0, 1)))
    requests = (Request((1,), max_new=10), Request((2,), max_new=1))

    assert schedule.active_steps(requests) == {0: 2, 1: 1}
    assert schedule.occupancy(requests) == {
        (0, 0): (Occupant(0, 0, 0),),
        (0, 1): (Occupant(0, 0, 1),),
        (0, 2): (Occupant(0, 1, 0),),
    }


def test_encoding_is_canonical_and_round_trips() -> None:
    schedule = schedule_fcfs(REQUESTS, pods=2, slots=2, steps=5)
    data = schedule.encode()

    assert Schedule.decode(data) == schedule
    assert len(data) == 20 + 16 + 16 * len(schedule.joins)
    for corrupt in (data[:-1], data + b"\0", b"x" + data[1:], data[:20] + (1).to_bytes(4, "big") + data[24:]):
        with pytest.raises(ScheduleError):
            Schedule.decode(corrupt)


@pytest.mark.parametrize(
    "joins",
    [
        (Join(0, 0, 0, 0), Join(0, 0, 0, 1)),  # the same slot twice at one step
        (Join(0, 1, 0, 0), Join(0, 0, 0, 1)),  # unsorted
        (Join(0, 0, 3, 0),),  # slot out of range
        (Join(2, 0, 0, 0),),  # pod out of range
        (Join(0, 9, 0, 0),),  # step out of range
    ],
)
def test_malformed_schedules_are_rejected(joins) -> None:
    with pytest.raises(ScheduleError):
        Schedule(2, 2, 4, joins)


def test_validate_checks_requests_join_exactly_once() -> None:
    twice = Schedule(1, 2, 2, (Join(0, 0, 0, 0), Join(0, 0, 1, 0)))
    with pytest.raises(ScheduleError, match="joins more than once"):
        twice.validate(REQUESTS[:1])
    unknown = Schedule(1, 1, 1, (Join(0, 0, 0, 7),))
    with pytest.raises(ScheduleError, match="unknown request 7"):
        unknown.validate(REQUESTS)


def test_requests_are_validated() -> None:
    with pytest.raises(ScheduleError):
        Request((), max_new=1)
    with pytest.raises(ScheduleError):
        Request((1, -1), max_new=1)
    with pytest.raises(ScheduleError):
        Request((1,), max_new=0)
