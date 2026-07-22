#!/usr/bin/env python3
import ctypes
import errno
import os
import signal
import subprocess
import sys

# NOTE: Do NOT import anything here that needs be built (e.g. params)
from openpilot.common.basedir import BASEDIR
from openpilot.common.spinner import Spinner
from openpilot.common.text_window import TextWindow
from openpilot.common.hardware import HARDWARE, AGNOS

USBGPU_BOOTSTRAP_POWER_ON_BOOT = "USBGPU_BOOTSTRAP_POWER_ON_BOOT"
PR_SET_PDEATHSIG = 1


def _arm_scons_parent_death_signal() -> None:
  """Make SCons terminate if its build.py parent disappears, including SIGKILL."""
  if sys.platform != "linux":
    return

  parent_pid = os.getppid()
  if parent_pid == 1:
    raise RuntimeError("refusing to start SCons after build parent exit")

  libc = ctypes.CDLL(None, use_errno=True)
  prctl = libc.prctl
  prctl.argtypes = (ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong)
  prctl.restype = ctypes.c_int
  if prctl(PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0) != 0:
    err = ctypes.get_errno() or errno.EINVAL
    raise OSError(err, f"PR_SET_PDEATHSIG failed: {os.strerror(err)}")

  current_parent_pid = os.getppid()
  if current_parent_pid == 1 or current_parent_pid != parent_pid:
    raise RuntimeError("build parent changed while arming SCons cancellation")


def _scons_env(first_attempt: bool) -> dict[str, str]:
  build_env = {**os.environ, "PWD": BASEDIR}
  if not first_attempt:
    build_env.pop(USBGPU_BOOTSTRAP_POWER_ON_BOOT, None)
  return build_env


def build() -> None:
  spinner = Spinner()
  spinner.update_progress(0, 100)

  HARDWARE.set_power_save(False)
  if AGNOS:
    os.sched_setaffinity(0, range(8))  # ensure we can use the isolcpus cores

  # building with all cores can result in using too much memory, so retry serially
  compile_output: list[bytes] = []
  for attempt, parallelism in enumerate(([], ["-j4"], ["-j1"])):
    compile_output.clear()
    popen_kwargs = {"cwd": BASEDIR, "env": _scons_env(attempt == 0), "stderr": subprocess.PIPE}
    if sys.platform == "linux":
      popen_kwargs["preexec_fn"] = _arm_scons_parent_death_signal
    with subprocess.Popen(["scons", *parallelism], **popen_kwargs) as scons:
      assert scons.stderr is not None

      # Read progress from stderr and update spinner
      while scons.poll() is None:
        try:
          line = scons.stderr.readline()
          if line is None:
            continue
          line = line.rstrip()

          prefix = b'progress: '
          if line.startswith(prefix):
            progress = float(line[len(prefix):])
            spinner.update_progress(100 * min(1., progress / 100.), 100.)
          elif len(line):
            compile_output.append(line)
            print(line.decode('utf8', 'replace'))
        except Exception:
          pass

      # Drain and close the pipe before retrying or returning.
      for line in scons.stderr.read().split(b'\n'):
        line = line.rstrip()
        if len(line):
          compile_output.append(line)

    if scons.returncode == 0:
      break

  if scons.returncode != 0:
    # Build failed log errors
    error_s = b"\n".join(compile_output).decode('utf8', 'replace')

    # Show TextWindow
    spinner.close()
    if not os.getenv("CI"):
      with TextWindow("openpilot failed to build\n \n" + error_s) as t:
        t.wait_for_exit()
    exit(1)

if __name__ == "__main__":
  build()
