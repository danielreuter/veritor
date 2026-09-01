"""PoComp prototype.

Security invariants:

INV-PUBLIC-PARAMS:
    Public budgets are nonnegative and the per-output-byte sampling rate is in
    [0, 1].

INV-EVENT-SIZE:
    Recorded artifact sizes are nonnegative. We assume event sizes are derived
    from the hashed artifact preimage, so checking the hash suffices when an
    artifact is opened.

INV-ARTIFACT-OPENING:
    Opened artifact bytes match the requested hash.

INV-OUTPUT-VALIDITY:
    Every task output is a valid monitored event handle.

INV-OUTPUT-OWNERSHIP:
    No event is claimed as the output of more than one task.

INV-OUTPUT-COVERAGE:
    Every monitored output event is claimed by exactly one task.

INV-INPUT-VALIDITY:
    Every task transcript input is a valid event handle.

INV-INPUT-CAUSALITY:
    Every task transcript input strictly precedes that task's first output.

INV-TASK-LOCALITY:
    A task's outputs are emitted by one site, and its transcript inputs were
    received by that same site.

INV-ENVIRONMENT-VALIDITY:
    Environment event artifacts can be opened.

INV-EPOCH-ADVICE:
    Interpreter runs consume at most the epoch advice budget.

INV-REPLAY-COMPUTE:
    Sampled task replay consumes at most the per-task compute budget.

INV-REPLAY-CORRECTNESS:
    Sampled task replay reproduces the recorded output bytes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Generic, Protocol, Sequence, TypeVar, overload

Hash = int
EventId = int
SiteId = int
T = TypeVar("T")

EXTERNAL: SiteId = -1


@dataclass
class NetworkEvent:
    sender: SiteId
    receiver: SiteId
    artifact_hash: Hash
    artifact_size: int


@dataclass
class NetworkEventLog:
    events: tuple[NetworkEvent, ...]


@dataclass
class TaskRecord:
    input_events: tuple[EventId, ...]
    output_events: tuple[EventId, ...]


@dataclass
class ArtifactStore:
    artifacts: dict[Hash, bytes]

    def read(self, artifact_hash: Hash) -> bytes:
        artifact = self.artifacts[artifact_hash]
        assert hash(artifact) == artifact_hash, "violated INV-ARTIFACT-OPENING"
        return artifact

    def read_many(self, artifact_hashes: Sequence[Hash]) -> tuple[bytes, ...]:
        return tuple(self.read(artifact_hash) for artifact_hash in artifact_hashes)


class BaselineCommitment(Protocol):
    def contains(self, artifact_hash: Hash) -> bool: ...


@dataclass
class Task:
    record: TaskRecord
    inputs: tuple[bytes, ...]
    outputs: tuple[bytes, ...]

    @classmethod
    def from_record(
        cls,
        record: TaskRecord,
        artifact_store: ArtifactStore,
        event_log: NetworkEventLog,
    ) -> Task:
        event_input_hashes = tuple(
            event_log.events[i].artifact_hash for i in record.input_events
        )
        output_hashes = tuple(
            event_log.events[i].artifact_hash for i in record.output_events
        )
        return cls(
            record=record,
            inputs=artifact_store.read_many(event_input_hashes),
            outputs=artifact_store.read_many(output_hashes),
        )


@dataclass
class AccountingTask:
    event_log: NetworkEventLog


@dataclass
class PublicParams:
    max_compute_per_task: int
    max_advice_per_epoch: int
    sample_rate_per_output_byte: float


@dataclass
class RunResult(Generic[T]):
    value: T
    compute_cost: int
    advice_consumed: int = 0


class Interpreter(Protocol):
    """Runs accounting jobs and generative task replays under metering."""

    @overload
    def run(
        self,
        task: AccountingTask,
    ) -> RunResult[tuple[TaskRecord, ...]]: ...

    @overload
    def run(
        self,
        task: Task,
    ) -> RunResult[tuple[bytes, ...]]: ...

    def run(
        self,
        task: AccountingTask | Task,
    ) -> RunResult[tuple[TaskRecord, ...]] | RunResult[tuple[bytes, ...]]: ...


def is_strictly_increasing(values: Sequence[int]) -> bool:
    return all(left < right for left, right in zip(values, values[1:]))


def is_external_input(event: NetworkEvent) -> bool:
    return event.sender == EXTERNAL


def is_monitored_output(event: NetworkEvent) -> bool:
    return not is_external_input(event)


def validate_epoch_advice(advice_consumed: int, params: PublicParams) -> None:
    """Establishes INV-EPOCH-ADVICE."""
    assert advice_consumed >= 0, "violated INV-EPOCH-ADVICE"
    assert advice_consumed <= params.max_advice_per_epoch, "violated INV-EPOCH-ADVICE"


def validate_accounting(
    event_log: NetworkEventLog,
    task_records: Sequence[TaskRecord],
    artifact_store: ArtifactStore,
    baseline: BaselineCommitment,
    params: PublicParams,
) -> None:
    assert params.max_compute_per_task >= 0, "violated INV-PUBLIC-PARAMS"
    assert params.max_advice_per_epoch >= 0, "violated INV-PUBLIC-PARAMS"
    assert 0 <= params.sample_rate_per_output_byte <= 1, "violated INV-PUBLIC-PARAMS"

    for event in event_log.events:
        assert event.artifact_size >= 0, "violated INV-EVENT-SIZE"

    task_output_events: set[EventId] = set()
    task_sites: list[SiteId] = []
    event_count = len(event_log.events)

    for task in task_records:
        assert task.output_events, "violated INV-OUTPUT-VALIDITY"
        assert is_strictly_increasing(task.output_events), (
            "violated INV-OUTPUT-VALIDITY"
        )

        output_senders: set[SiteId] = set()
        for output_event in task.output_events:
            assert 0 <= output_event < event_count, "violated INV-OUTPUT-VALIDITY"
            event = event_log.events[output_event]
            assert is_monitored_output(event), "violated INV-OUTPUT-VALIDITY"
            output_senders.add(event.sender)
            assert output_event not in task_output_events, (
                "violated INV-OUTPUT-OWNERSHIP"
            )
            task_output_events.add(output_event)

        assert len(output_senders) == 1, "violated INV-TASK-LOCALITY"
        task_sites.append(next(iter(output_senders)))

    monitored_outputs = {
        event_id
        for event_id, event in enumerate(event_log.events)
        if is_monitored_output(event)
    }
    unclaimed_outputs = monitored_outputs - task_output_events
    assert not unclaimed_outputs, "violated INV-OUTPUT-COVERAGE"

    for task, site in zip(task_records, task_sites):
        assert is_strictly_increasing(task.input_events), "violated INV-INPUT-VALIDITY"

        first_output_event = task.output_events[0]
        for input_event in task.input_events:
            assert 0 <= input_event < event_count, "violated INV-INPUT-VALIDITY"
            event = event_log.events[input_event]
            assert event.receiver == site, "violated INV-TASK-LOCALITY"
            assert input_event < first_output_event, "violated INV-INPUT-CAUSALITY"

    environment_events = {
        event_id
        for event_id, event in enumerate(event_log.events)
        if is_external_input(event)
    }
    for event_id in environment_events:
        artifact_store.read(event_log.events[event_id].artifact_hash)


def sample_by_weight(
    items: Sequence[T],
    weight: Callable[[T], int | float],
    sampling_prob: float,
    beacon: bytes,
) -> tuple[T, ...]:
    rng = random.Random(beacon)
    sample: list[T] = []
    for item in items:
        if rng.random() < min(1, sampling_prob * weight(item)):
            sample.append(item)
    return tuple(sample)


def audit_epoch(
    interpreter: Interpreter,
    event_log: NetworkEventLog,
    artifact_store: ArtifactStore,
    baseline: BaselineCommitment,
    params: PublicParams,
    beacon: bytes,
) -> tuple[TaskRecord, ...]:
    accounting_result = interpreter.run(AccountingTask(event_log))
    advice_consumed = accounting_result.advice_consumed
    validate_epoch_advice(advice_consumed, params)
    task_records = accounting_result.value
    validate_accounting(event_log, task_records, artifact_store, baseline, params)
    sample = sample_by_weight(
        task_records,
        lambda task: sum(event_log.events[i].artifact_size for i in task.output_events),
        params.sample_rate_per_output_byte,
        beacon,
    )
    for task_record in sample:
        task = Task.from_record(task_record, artifact_store, event_log)
        result = interpreter.run(task)
        assert result.advice_consumed >= 0, "violated INV-EPOCH-ADVICE"
        advice_consumed += result.advice_consumed
        validate_epoch_advice(advice_consumed, params)
        assert result.compute_cost <= params.max_compute_per_task, (
            "violated INV-REPLAY-COMPUTE"
        )
        assert result.value == task.outputs, "violated INV-REPLAY-CORRECTNESS"
    return task_records
