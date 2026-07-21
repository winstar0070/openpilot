#!/usr/bin/env python3
import os
from pathlib import Path
import subprocess

# NOTE: Do NOT import anything here that needs be built (e.g. params)
from openpilot.common.basedir import BASEDIR
from openpilot.common.spinner import Spinner
from openpilot.common.text_window import TextWindow
from openpilot.common.hardware import HARDWARE, AGNOS


USBGPU_BUILD_FAILURE_MARKER = os.getenv("USBGPU_BUILD_FAILURE_MARKER", "/tmp/openpilot_usbgpu_build_failure")
BUILD_PARALLELISM = ([], ["-j4"], ["-j1"])


def _clear_failure_marker(marker_path: str) -> None:
  try:
    os.unlink(marker_path)
  except FileNotFoundError:
    pass


def _read_failure_marker(marker_path: str) -> str | None:
  try:
    reason = Path(marker_path).read_text().strip()
  except (FileNotFoundError, OSError):
    return None
  return reason or "USB GPU session lost"


def run_build_attempts(run_attempt, marker_path: str = USBGPU_BUILD_FAILURE_MARKER) -> tuple[int, list[bytes], str | None]:
  """Run normal memory-pressure retries, plus one same-process-boundary USB GPU retry."""
  parallelism_index = 0
  hardware_retry_used = False
  while parallelism_index < len(BUILD_PARALLELISM):
    _clear_failure_marker(marker_path)
    returncode, compile_output = run_attempt(BUILD_PARALLELISM[parallelism_index])
    if returncode == 0:
      _clear_failure_marker(marker_path)
      return returncode, compile_output, None

    hardware_reason = _read_failure_marker(marker_path)
    if hardware_reason is not None:
      if hardware_retry_used:
        return returncode, compile_output, hardware_reason
      hardware_retry_used = True
      continue

    parallelism_index += 1

  return returncode, compile_output, None


def format_build_error(compile_output: list[bytes], hardware_reason: str | None,
                       marker_path: str = USBGPU_BUILD_FAILURE_MARKER) -> str:
  error_s = b"\n".join(compile_output).decode('utf8', 'replace')
  if hardware_reason is not None:
    error_s += ("\n \nUSB GPU session recovery failed: " + hardware_reason +
                "\nDisconnect eGPU power for at least 10 seconds, reconnect it, then restart openpilot." +
                f"\nDiagnostic marker: {marker_path}")
  return error_s


def build() -> None:
  spinner = Spinner()
  spinner.update_progress(0, 100)

  HARDWARE.set_power_save(False)
  if AGNOS:
    os.sched_setaffinity(0, range(8))  # ensure we can use the isolcpus cores

  def run_attempt(parallelism: list[str]) -> tuple[int, list[bytes]]:
    compile_output: list[bytes] = []
    build_env = {**os.environ, "PWD": BASEDIR, "USBGPU_BUILD_FAILURE_MARKER": USBGPU_BUILD_FAILURE_MARKER}
    with subprocess.Popen(["scons", *parallelism], cwd=BASEDIR, env=build_env, stderr=subprocess.PIPE) as scons:
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
    return scons.returncode, compile_output

  # Building with all cores can use too much memory. Hardware session loss gets one
  # fresh SCons/Python process at the same parallelism instead of memory retries.
  returncode, compile_output, hardware_reason = run_build_attempts(run_attempt)

  if returncode != 0:
    # Build failed log errors
    error_s = format_build_error(compile_output, hardware_reason)

    # Show TextWindow
    spinner.close()
    if not os.getenv("CI"):
      with TextWindow("openpilot failed to build\n \n" + error_s) as t:
        t.wait_for_exit()
    exit(1)

if __name__ == "__main__":
  build()
