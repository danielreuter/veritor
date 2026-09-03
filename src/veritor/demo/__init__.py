"""The end-to-end demonstration: a simulated inference datacenter through Verity.

``python -m veritor.demo.datacenter`` simulates a production-shaped serving
run of the toy decoder -- Poisson arrivals, continual batching, early stops
at the end-of-sequence token, pod failures with restarts, and sampled
tokens over published randomness -- compiles the whole run into one
verifiable circuit, runs the protocol honestly and against a server that
exfiltrates a secret through the tokens, and prices the run with ``Bound``
and ``Cost``.  :func:`veritor.demo.datacenter.run` returns a
:class:`~veritor.demo.datacenter.Summary` with every number the report
prints.
"""
