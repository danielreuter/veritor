"""The trusted compiler: description bytes in, ``Compiled(circuit, index, digest)`` out.

:class:`Constructor` is what the verifier requires of a client's ``G`` and
:class:`Compilation` the record of running it; ``Compile(G, x, a)`` itself is
:func:`veritor.research.Compile`.
"""

from .compiler import Compiler
from .constructor import (
    CONSTRUCTOR_DIGEST_TAG,
    Compilation,
    Constructor,
    constructor_digest,
)
from .description import (
    FORMAT_VERSION,
    CompileError,
    Description,
    canonical_description,
    definition_digest,
    description_digest,
    parse_description,
)

__all__ = [
    "CONSTRUCTOR_DIGEST_TAG",
    "FORMAT_VERSION",
    "Compilation",
    "CompileError",
    "Compiler",
    "Constructor",
    "Description",
    "canonical_description",
    "constructor_digest",
    "definition_digest",
    "description_digest",
    "parse_description",
]
