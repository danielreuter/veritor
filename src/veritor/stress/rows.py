"""Rows of ``docs/data/stress*.json``: one priced scenario each, merged by ID.

The file is a JSON object keyed by scenario ID.  Several test processes may
record rows into it, so :func:`record` takes an exclusive lock on the data
directory, merges its rows into whatever is on disk, and replaces the file
atomically; rows it did not produce are never dropped.  The text is canonical
(one row per line, sorted keys, catalogue order) so that diffs stay small.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows has no advisory file locks
    fcntl = None  # type: ignore[assignment]

__all__ = ["Recorder", "Row", "dump", "load", "record", "row_key"]

# Catalogue order of the ID prefixes (docs/stress-tests.md, section 2; ``H`` is
# the honest-prover programme, docs/honest-prover.md).
_SECTIONS = "SCNWEH"
_IDENTIFIER = re.compile(r"([A-Z])(\d+)([a-z]*)")


@dataclass(frozen=True, slots=True)
class Row:
    """One priced scenario: what was built, how its complexity entered, what it cost."""

    id: str
    what: str
    mechanism: str
    advice_bits: int
    capacity_bits: int
    overhead: float
    description_bytes: int
    verdict: str
    notes: str = ""
    extra: Mapping[str, object] = field(default_factory=dict)
    """Fields other recorders keep in their rows (``gates`` ...): carried through, never rendered."""

    def __post_init__(self) -> None:
        if not _IDENTIFIER.fullmatch(self.id):
            raise ValueError(f"scenario id {self.id!r} is not like 'S1' or 'C6b'")
        for name in ("advice_bits", "capacity_bits", "description_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"{name} must be a non-negative integer, got {value!r}"
                )
        if not isinstance(self.overhead, (int, float)) or self.overhead < 0:
            raise ValueError(
                f"overhead must be a non-negative number, got {self.overhead!r}"
            )
        object.__setattr__(self, "overhead", round(float(self.overhead), 6))
        for name in ("what", "mechanism", "verdict", "notes"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"{name} must be a string")
        if not isinstance(self.extra, Mapping) or set(self.extra) & _NAMES:
            raise ValueError(
                "extra fields must be a mapping of fields the row does not have"
            )

    def to_json(self) -> dict[str, object]:
        body = asdict(self)
        del body["id"]
        del body["extra"]
        body.update(self.extra)
        return body

    @classmethod
    def from_json(cls, identifier: str, body: Mapping[str, object]) -> Row:
        known = {name: value for name, value in body.items() if name in _NAMES}
        extra = {name: value for name, value in body.items() if name not in _NAMES}
        return cls(id=identifier, extra=extra, **known)  # type: ignore[arg-type]


_NAMES = {name.name for name in fields(Row)} - {"id", "extra"}


@dataclass
class Recorder:
    """Rows recorded by one scenario, each ID once; a test fixture writes them when the test passes.

    Keyword arguments beyond the row's own fields are kept in ``extra``: the
    honest-prover table renders ``declarations``, ``charge_bits`` and
    ``recompute`` from there.
    """

    rows: list[Row] = field(default_factory=list)

    def record(
        self,
        *,
        id: str,
        what: str,
        mechanism: str,
        advice_bits: int,
        capacity_bits: int,
        overhead: float,
        description_bytes: int,
        verdict: str,
        notes: str = "",
        **extra: object,
    ) -> Row:
        if any(row.id == id for row in self.rows):
            raise ValueError(f"row {id!r} recorded twice by one test")
        row = Row(
            id=id,
            what=what,
            mechanism=mechanism,
            advice_bits=advice_bits,
            capacity_bits=capacity_bits,
            overhead=overhead,
            description_bytes=description_bytes,
            verdict=verdict,
            notes=notes,
            extra=extra,
        )
        self.rows.append(row)
        return row


def row_key(identifier: str) -> tuple[int, int, str]:
    """Sort key placing IDs in catalogue order: S before C before N ..., then numerically."""
    match = _IDENTIFIER.fullmatch(identifier)
    if match is None:
        raise ValueError(f"scenario id {identifier!r} is not like 'S1' or 'C6b'")
    letter, number, suffix = match.groups()
    section = (
        _SECTIONS.index(letter) if letter in _SECTIONS else len(_SECTIONS) + ord(letter)
    )
    return (section, int(number), suffix)


def dump(rows: Mapping[str, Row]) -> str:
    """Canonical text of ``rows``: one row per line in catalogue order."""
    lines = ["{"]
    ordered = sorted(rows, key=row_key)
    for position, identifier in enumerate(ordered):
        body = json.dumps(
            rows[identifier].to_json(), sort_keys=True, ensure_ascii=False
        )
        comma = "," if position + 1 < len(ordered) else ""
        lines.append(f' "{identifier}": {body}{comma}')
    lines.append("}")
    return "\n".join(lines) + "\n"


def load(path: Path) -> dict[str, Row]:
    """Rows stored at ``path`` (an empty mapping when the file does not exist)."""
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return {}
    data = json.loads(text)
    if not isinstance(data, dict):
        raise TypeError(f"{path} does not hold a JSON object")
    rows = {
        identifier: Row.from_json(identifier, body) for identifier, body in data.items()
    }
    return _ordered(rows)


def _ordered(rows: Mapping[str, Row]) -> dict[str, Row]:
    return {identifier: rows[identifier] for identifier in sorted(rows, key=row_key)}


def record(path: Path, rows: Iterable[Row]) -> dict[str, Row]:
    """Merge ``rows`` into the file at ``path`` by ID and return the merged rows.

    The merge holds an exclusive lock on the parent directory and writes through
    a temporary file, so concurrent recorders neither interleave nor lose rows.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        if fcntl is not None:
            fcntl.flock(directory, fcntl.LOCK_EX)
        merged = load(path)
        for row in rows:
            merged[row.id] = row
        merged = _ordered(merged)
        handle, temporary = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as out:
                out.write(dump(merged))
            os.replace(temporary, path)
        except BaseException:
            if os.path.exists(temporary):
                os.unlink(temporary)
            raise
    finally:
        os.close(directory)  # releases the lock
    return merged
