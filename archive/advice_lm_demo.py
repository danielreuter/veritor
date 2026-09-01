"""Tiny demo of interpreter-pulled advice.

This is not protocol machinery. It is a small executable sketch of one clean
way the advice object could behave:

1. The task program tries to generate the recorded output with no advice.
2. If that fails, it yields an AdviceRequest and pauses.
3. The interpreter reads advice bytes, meters them, and resumes the program.
4. The resumed program uses those bytes to recover the recorded output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generator, Generic, TypeVar

T = TypeVar("T")
AdviceProgram = Generator["AdviceRequest", bytes, T]


@dataclass(frozen=True)
class AdviceRequest:
    site: str
    nbytes: int


@dataclass(frozen=True)
class AdviceUse:
    site: str
    payload: bytes


@dataclass(frozen=True)
class RunResult(Generic[T]):
    value: T
    compute_cost: int
    advice_consumed: int
    advice_trace: tuple[AdviceUse, ...]


class AdviceUnavailable(Exception):
    pass


class AdviceTape:
    """Replay-side bytes consumed only when the program asks for them."""

    def __init__(self, chunks: tuple[bytes, ...] = ()) -> None:
        self._chunks = chunks
        self._cursor = 0

    def read(self, request: AdviceRequest) -> bytes:
        if self._cursor >= len(self._chunks):
            raise AdviceUnavailable(f"no advice available for {request.site}")

        payload = self._chunks[self._cursor]
        self._cursor += 1
        if len(payload) != request.nbytes:
            raise ValueError(f"{request.site} requested {request.nbytes} bytes")
        return payload


class MeteredInterpreter:
    def __init__(self, advice_tape: AdviceTape) -> None:
        self.advice_tape = advice_tape

    def run(self, program_factory: Callable[[], AdviceProgram[T]]) -> RunResult[T]:
        program = program_factory()
        compute_cost = 0
        advice_trace: list[AdviceUse] = []
        resume_value: bytes | None = None

        while True:
            compute_cost += 1
            try:
                if resume_value is None:
                    request = next(program)
                else:
                    request = program.send(resume_value)
                    resume_value = None
            except StopIteration as done:
                return RunResult(
                    value=done.value,
                    compute_cost=compute_cost,
                    advice_consumed=sum(
                        len(advice.payload) for advice in advice_trace
                    ),
                    advice_trace=tuple(advice_trace),
                )

            payload = self.advice_tape.read(request)
            advice_trace.append(AdviceUse(site=request.site, payload=payload))
            resume_value = payload


def render_completion(prompt: bytes, token: bytes) -> bytes:
    return prompt + b" " + token + b"."


def dummy_lm_task(prompt: bytes, recorded_output: bytes) -> AdviceProgram[bytes]:
    """A toy LM replay that only asks for advice after its first guess fails."""

    candidates = (b"London", b"Berlin", b"Paris", b"Rome")

    first_guess = candidates[0]
    output = render_completion(prompt, first_guess)
    if output == recorded_output:
        return output

    advice = yield AdviceRequest(site="dummy_lm_task.fallback_token", nbytes=1)
    fallback_guess = candidates[advice[0] % len(candidates)]
    output = render_completion(prompt, fallback_guess)
    if output != recorded_output:
        raise ValueError("advice did not recover the recorded output")
    return output


def main() -> None:
    prompt = b"The capital of France is"
    recorded_output = b"The capital of France is Paris."
    program = lambda: dummy_lm_task(prompt, recorded_output)

    try:
        MeteredInterpreter(AdviceTape()).run(program)
    except AdviceUnavailable as exc:
        print(f"without advice: failed after first guess ({exc})")

    result = MeteredInterpreter(AdviceTape((bytes([2]),))).run(program)
    print(f"with advice: {result.value.decode()}")
    print(f"compute cost: {result.compute_cost}")
    print(f"advice consumed: {result.advice_consumed} byte")
    print(f"advice trace: {result.advice_trace}")


if __name__ == "__main__":
    main()
