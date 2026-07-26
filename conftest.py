"""Pytest runner configuration for the unittest suite."""

import os

# Heavy CI-only tests are invoked explicitly by their dedicated jobs.
collect_ignore = [
  "openpilot/selfdrive/test/process_replay/test_processes.py",
  "openpilot/selfdrive/test/process_replay/test_regen.py",
  "openpilot/tools/sim/",

  # tinygrad JIT has process-global state. Other test files import modeld → tinygrad,
  # which corrupts JIT captures for test_warp.py in the same process. Run separately in CI.
  "openpilot/sunnypilot/modeld_v2/tests/test_warp.py",
]


def pytest_collection_modifyitems(items):
  if os.environ.get("SKIP_SLOW"):
    items[:] = [item for item in items if not getattr(getattr(item, "cls", None), "SLOW_TEST", False)]
