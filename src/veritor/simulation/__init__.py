"""A simulated inference datacenter through Verity: the workload, the run, the adversary.

``python -m veritor.simulation.datacenter`` simulates a production-shaped serving
run of the toy decoder -- Poisson arrivals, continual batching, early stops
at the end-of-sequence token, pod failures with restarts, and sampled
tokens over published randomness -- compiles the whole run into one
verifiable circuit, runs the protocol honestly and against a server that
exfiltrates a secret through the tokens, and prices the run with ``Bound``
and ``Cost``.  :func:`veritor.simulation.datacenter.run` returns a
:class:`~veritor.simulation.datacenter.Summary` with every number the report
prints.
"""
