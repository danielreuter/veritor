"""Continual-batching schedules: canonical bytes, derived occupancy, FCFS, restarts."""

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
        Join(0, 0, 0, 0, 3),  # request 0 takes slot 0 for steps 0..2
        Join(0, 0, 1, 1, 2),  # request 1 takes slot 1 for steps 0..1
        Join(0, 2, 1, 2, 4),  # slot 1 frees at step 2: request 2 joins, cut by the run's end
        Join(0, 3, 0, 3, 1),  # slot 0 frees at step 3: request 3 joins
    )
    assert schedule.active_steps(REQUESTS) == {0: 3, 1: 2, 2: 4, 3: 1}
    occupancy = schedule.occupancy(REQUESTS)
    assert occupancy[(0, 0)] == (Occupant(0, 0, 0, 0), Occupant(1, 1, 0, 1))
    assert occupancy[(0, 2)] == (Occupant(0, 0, 2, 0), Occupant(1, 2, 0, 2))
    assert occupancy[(0, 3)] == (Occupant(0, 3, 0, 3), Occupant(1, 2, 1, 2))
    assert occupancy[(0, 5)] == (Occupant(1, 2, 3, 2),)
    assert sorted(occupancy) == [(0, t) for t in range(6)]
    assert schedule.streamed_before(REQUESTS) == (0, 0, 0, 0)


def test_fcfs_spreads_over_pods_and_reports_unscheduled_requests() -> None:
    schedule = schedule_fcfs(REQUESTS, pods=2, slots=1, steps=2)

    assert schedule.joins == (Join(0, 0, 0, 0, 2), Join(1, 0, 0, 1, 2))
    with pytest.raises(ScheduleError, match=r"never scheduled: \[2, 3\]"):
        schedule.validate(REQUESTS)


def test_an_explicit_length_cuts_a_request_short() -> None:
    schedule = Schedule(1, 1, 4, (Join(0, 0, 0, 0, 2), Join(0, 2, 0, 1, 1)))
    requests = (Request((1,), max_new=10), Request((2,), max_new=1))

    assert schedule.active_steps(requests) == {0: 2, 1: 1}
    assert schedule.occupancy(requests) == {
        (0, 0): (Occupant(0, 0, 0, 0),),
        (0, 1): (Occupant(0, 0, 1, 0),),
        (0, 2): (Occupant(0, 1, 0, 1),),
    }


def test_a_restart_recomputes_and_streams_only_new_positions() -> None:
    """Request 0 is aborted on pod 0 after two tokens and restarted on pod 1 at step 3."""

    requests = (Request((1, 2), max_new=5), Request((3,), max_new=2))
    schedule = Schedule(2, 1, 8, (Join(0, 0, 0, 0, 2), Join(1, 0, 0, 1, 2), Join(1, 3, 0, 0, 5)))

    assert schedule.attempts(requests) == {0: (0, 2), 1: (1,)}
    assert schedule.streamed_before(requests) == (0, 0, 2)
    assert schedule.active_steps(requests) == {0: 5, 1: 2}
    occupancy = schedule.occupancy(requests)
    assert occupancy[(0, 1)] == (Occupant(0, 0, 1, 0),)
    assert occupancy[(1, 3)] == (Occupant(0, 0, 0, 2),)  # the restart prefills again
    assert occupancy[(1, 7)] == (Occupant(0, 0, 4, 2),)
    assert (0, 2) not in occupancy and (1, 2) not in occupancy  # pod 0 is down, pod 1 idle
    # a second abort may end before the first attempt's progress: nothing new is streamed
    twice = Schedule(2, 1, 8, (Join(0, 0, 0, 0, 3), Join(1, 0, 0, 1, 2), Join(1, 3, 0, 0, 2), Join(1, 6, 0, 0, 2)))
    assert twice.streamed_before(requests) == (0, 0, 3, 3)
    assert twice.active_steps(requests) == {0: 3, 1: 2}


def test_a_restart_may_not_overlap_the_attempt_it_replaces() -> None:
    requests = (Request((1, 2), max_new=5),)
    overlapping = Schedule(2, 1, 8, (Join(0, 0, 0, 0, 3), Join(1, 2, 0, 0, 5)))
    with pytest.raises(ScheduleError, match="restarts at step 2 while its earlier attempt"):
        overlapping.validate(requests)
    simultaneous = Schedule(2, 1, 8, (Join(0, 0, 0, 0, 3), Join(1, 0, 0, 0, 3)))
    with pytest.raises(ScheduleError, match="restarts at step 0"):
        simultaneous.validate(requests)
    Schedule(2, 1, 8, (Join(0, 0, 0, 0, 3), Join(1, 3, 0, 0, 5))).validate(requests)  # back to back is fine


def test_encoding_is_canonical_and_round_trips() -> None:
    schedule = schedule_fcfs(REQUESTS, pods=2, slots=2, steps=5)
    data = schedule.encode()

    assert Schedule.decode(data) == schedule
    assert len(data) == 20 + 16 + 20 * len(schedule.joins)
    for corrupt in (data[:-1], data + b"\0", b"x" + data[1:], data[:20] + (1).to_bytes(4, "big") + data[24:]):
        with pytest.raises(ScheduleError):
            Schedule.decode(corrupt)


@pytest.mark.parametrize(
    "joins",
    [
        (Join(0, 0, 0, 0, 1), Join(0, 0, 0, 1, 1)),  # the same slot twice at one step
        (Join(0, 1, 0, 0, 1), Join(0, 0, 0, 1, 1)),  # unsorted
        (Join(0, 0, 0, 0, 2), Join(0, 1, 0, 1, 1)),  # double-booked: the first still holds the slot
        (Join(0, 0, 3, 0, 1),),  # slot out of range
        (Join(2, 0, 0, 0, 1),),  # pod out of range
        (Join(0, 9, 0, 0, 1),),  # step out of range
        (Join(0, 0, 0, 0, 0),),  # an empty attempt
        (Join(0, 3, 0, 0, 2),),  # runs past the end
    ],
)
def test_malformed_schedules_are_rejected(joins) -> None:
    with pytest.raises(ScheduleError):
        Schedule(2, 2, 4, joins)


def test_validate_checks_requests_and_lengths() -> None:
    unknown = Schedule(1, 1, 1, (Join(0, 0, 0, 7, 1),))
    with pytest.raises(ScheduleError, match="unknown request 7"):
        unknown.validate(REQUESTS)
    long = Schedule(1, 1, 4, (Join(0, 0, 0, 3, 2),))  # request 3 wants one token
    with pytest.raises(ScheduleError, match="more than max_new"):
        long.validate(REQUESTS)


def test_requests_are_validated() -> None:
    with pytest.raises(ScheduleError):
        Request((), max_new=1)
    with pytest.raises(ScheduleError):
        Request((1, -1), max_new=1)
    with pytest.raises(ScheduleError):
        Request((1,), max_new=0)
    with pytest.raises(ScheduleError, match="one word per generated position"):
        Request((1,), max_new=2, randomness=(3,))
    with pytest.raises(ScheduleError, match="nonnegative"):
        Request((1,), max_new=1, randomness=(-1,))
    assert Request((1,), max_new=2, randomness=(3, 0)).randomness == (3, 0)
