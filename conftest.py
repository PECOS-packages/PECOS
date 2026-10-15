"""Repository-wide pytest setup, loaded before any test package's conftest."""

import os

# NumPy's bundled OpenBLAS starts one thread per core on import, and those
# threads busy-wait between the small matrix operations the tests issue. Under
# ``pytest -n auto`` every xdist worker gets its own pool, so a 32-core machine
# ran ~1000 spinning threads: the selene-plugins suite took 945 s instead of
# 30 s. The variable is read once, when numpy is first imported, so it must be
# set here, before workers are spawned (they inherit this environment) and
# before any test module imports numpy. ``setdefault`` keeps an explicit
# caller override.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
