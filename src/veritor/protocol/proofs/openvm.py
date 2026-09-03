"""The OpenVM adapter: the same statement/witness bytes, a different host.

``openvm-tc-bench`` (OpenVM v2.0.2 with the out-of-tree ``TC_DOT`` and
``TC_MATMUL_4X4X16`` extension for the pinned
``hawkeye_ampere_groupsum_fp8e4m3_v0`` contract) is the recommended
production backend: ~1.2-2.5 M MAC/s of GPU proving on an RTX 4090 against
~1.0 M MAC/s for SP1's ``TC_DOT`` fork, a cost of verifiability of ~7.5e7
relative to native fp8 tensor cores at the matmul plateau, and a fully
out-of-tree extension against unmodified OpenVM.  GPU proving is remote-only,
so this module is an *adapter*: it fixes how our bytes reach an OpenVM host
and is tested for the wire mapping alone.

Mapping.  An OpenVM guest reads one hint-stream item per ``read_vec`` and
reveals ``u32`` public values by index.  Our statement and witness are two
``read_vec`` items, byte for byte the same canonical encodings the SP1 guest
consumes (``zk/sp1/common/src/codec.rs`` is the parser to port; OpenVM's
guest crate is ``no_std`` and the parser is allocation-light).  The public
values are ``sha256(statement)`` as eight little-endian ``u32`` words at
indices ``0..8`` (``reveal_u32``), then the verdict at index ``8``; that is
the same 33 bytes SP1 commits, re-chunked the way OpenVM reveals them.  The
host CLI speaks the subprocess protocol of ``veritor-zk-host``
(``info | execute | prove | verify``, one JSON object on stdout), so
:class:`OpenVMBackend` and :class:`SP1Backend` differ only in the binary.

What the production backend adds beyond the toy ISA is a *tile relation*: a
kind whose gates are ``TC_DOT`` groups over fp8 e4m3 operands (the
``tc-dot-batch`` guest's ``u32 dot_count, (u32 acc, u32 tile_count, tiles)*``
payload is the witness shape for one such kind).  That waits on a gate set
carrying the fp8 contract; nothing here presumes it.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

from ..messages import ProtocolError
from .sp1 import ProveMode, SP1Backend
from .statement import Statement, Witness
from .wire import encode_statement, encode_witness, statement_digest

OPENVM_BACKEND = "openvm"
OPENVM_SEMANTICS_ID = "hawkeye_ampere_groupsum_fp8e4m3_v0"
"""The pinned fp8 contract of ``openvm-tc-bench``'s ``TC_DOT`` extension."""
OPENVM_VERSION = "2.0.2"


@dataclass(frozen=True, slots=True)
class OpenVMInput:
    """The two hint-stream items and the expected reveals for one batch."""

    statement: bytes
    witness: bytes
    reveals: tuple[int, ...]
    """``u32`` public values by index: ``sha256(statement)`` as 8 LE words, then the verdict."""

    @property
    def hint_stream(self) -> tuple[bytes, ...]:
        return (self.statement, self.witness)


def openvm_input(
    statement: Statement, witness: Witness, verdict: bool = True
) -> OpenVMInput:
    """Map a batch onto the OpenVM guest's inputs and expected public values."""

    witness.for_statement(statement)
    encoded = encode_statement(statement)
    digest = statement_digest(encoded)
    words = struct.unpack("<8I", digest)
    return OpenVMInput(encoded, encode_witness(witness), (*words, int(verdict)))


def reveals_to_public_values(reveals: tuple[int, ...]) -> bytes:
    """Re-chunk OpenVM's ``u32`` reveals into SP1's 33-byte public values."""

    if len(reveals) != 9 or any(
        type(item) is not int or not 0 <= item < 1 << 32 for item in reveals
    ):
        raise ProtocolError("expected 8 digest words and a verdict word")
    if reveals[8] > 1:
        raise ProtocolError("the verdict word must be 0 or 1")
    return struct.pack("<8I", *reveals[:8]) + bytes((reveals[8],))


class OpenVMBackend(SP1Backend):
    """Shell out to an OpenVM host speaking the ``veritor-zk-host`` protocol.

    The binary is remote-built (GPU proving); ``host`` must be given and no
    build is attempted.  Everything else -- the statement/witness bytes, the
    three modes, the public-value check -- is inherited.
    """

    backend_id = OPENVM_BACKEND

    def __init__(
        self,
        host: Path,
        *,
        mode: ProveMode = "core",
        vk_hash: str | None = None,
        timeout: float | None = None,
    ) -> None:
        super().__init__(
            host=Path(host),
            workspace=Path(host).parent,
            mode=mode,
            build=False,
            vk_hash=vk_hash,
            timeout=timeout,
        )


__all__ = [
    "OPENVM_BACKEND",
    "OPENVM_SEMANTICS_ID",
    "OPENVM_VERSION",
    "OpenVMBackend",
    "OpenVMInput",
    "openvm_input",
    "reveals_to_public_values",
]
