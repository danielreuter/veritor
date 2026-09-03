"""Scale benchmarks for ``veritor``: how each component behaves as its size parameter grows.

Run ``python -m benchmarks.run [--quick] [--only NAME ...] [--out PATH]`` to
measure and ``python -m benchmarks.report PATH -o docs/benchmarks.md`` to
render.  Every benchmark sweeps one size parameter over several decades,
records the median wall time of a few repeats, the ``tracemalloc`` peak and
the relevant output sizes, and fits a power law ``t = a * x**b`` to report the
observed exponent ``b``.  Nothing here touches ``src/veritor``: the package
drives the public API and, where a phase is not exposed, times around it.
"""
