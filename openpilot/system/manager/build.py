#!/usr/bin/env python3
import fcntl
import os
from collections.abc import Callable
from pathlib import Path
import subprocess

# NOTE: Do NOT import anything here that needs be built (e.g. params)
from openpilot.common.basedir import BASEDIR
from openpilot.common.file_chunker import (
  cleanup_incomplete_file_chunks, cleanup_stale_file_chunk_temps, is_file_chunked_valid,
)
from openpilot.common.spinner import Spinner
from openpilot.common.text_window import TextWindow
from openpilot.common.hardware import HARDWARE, AGNOS


USBGPU_BUILD_FAILURE_MARKER = os.getenv("USBGPU_BUILD_FAILURE_MARKER", "/tmp/openpilot_usbgpu_build_failure")
USBGPU_MODEL_PATH = Path(BASEDIR) / "openpilot/selfdrive/modeld/models/big_driving_tinygrad.pkl"
USBGPU_BUILD_LOCK = Path(BASEDIR) / "openpilot/selfdrive/modeld/models/.usb_gpu.lock"
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


def _attempt_failure_marker(marker_base: str, attempt: int) -> str:
  return f"{marker_base}.{os.getpid()}.{attempt}"


def _write_failure_marker_best_effort(marker_path: str, reason: str) -> None:
  try:
    Path(marker_path).write_text(reason.rstrip() + "\n")
  except OSError:
    pass


def cleanup_usbgpu_build_temps(output: Path = USBGPU_MODEL_PATH, lock_path: Path = USBGPU_BUILD_LOCK) -> None:
  lock_path.parent.mkdir(parents=True, exist_ok=True)
  with lock_path.open("a+") as lock_file:
    fcntl.flock(lock_file, fcntl.LOCK_EX)
    cleanup_stale_file_chunk_temps(output)
    cleanup_incomplete_file_chunks(output)


def run_build_attempts(run_attempt, marker_path: str = USBGPU_BUILD_FAILURE_MARKER,
                       artifact_valid: Callable[[], bool] | None = None) -> tuple[int, list[bytes], str | None, str]:
  """Run normal memory-pressure retries, plus one same-process-boundary USB GPU retry."""
  parallelism_index = 0
  hardware_retry_used = False
  attempt = 0
  while parallelism_index < len(BUILD_PARALLELISM):
    attempt += 1
    attempt_marker = _attempt_failure_marker(marker_path, attempt)
    _clear_failure_marker(attempt_marker)
    returncode, compile_output = run_attempt(BUILD_PARALLELISM[parallelism_index], hardware_retry_used, attempt_marker)
    if returncode == 0:
      if hardware_retry_used and artifact_valid is not None:
        try:
          valid = artifact_valid()
          validation_error = None
        except Exception as e:
          valid = False
          validation_error = f": {type(e).__name__}: {e}"
        if not valid:
          hardware_reason = "USB GPU retry exited successfully without a complete compiled model" + (validation_error or "")
          _write_failure_marker_best_effort(attempt_marker, hardware_reason)
          return 1, compile_output, hardware_reason, attempt_marker
      _clear_failure_marker(attempt_marker)
      return returncode, compile_output, None, attempt_marker

    hardware_reason = _read_failure_marker(attempt_marker)
    if hardware_reason is not None:
      if hardware_retry_used:
        return returncode, compile_output, hardware_reason, attempt_marker
      hardware_retry_used = True
      _clear_failure_marker(attempt_marker)
      continue

    parallelism_index += 1

  return returncode, compile_output, None, attempt_marker


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

  def run_attempt(parallelism: list[str], force_usbgpu: bool, marker_path: str) -> tuple[int, list[bytes]]:
    compile_output: list[bytes] = []
    cleanup_usbgpu_build_temps()
    build_env = {
      **os.environ,
      "PWD": BASEDIR,
      "USBGPU_BUILD_FAILURE_MARKER": marker_path,
      "USBGPU_BUILD_LOCK": str(USBGPU_BUILD_LOCK),
      "USBGPU_FORCE_BUILD": "1" if force_usbgpu else "0",
    }
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
  returncode, compile_output, hardware_reason, marker_path = run_build_attempts(
    run_attempt,
    artifact_valid=lambda: is_file_chunked_valid(USBGPU_MODEL_PATH, require_manifest=True),
  )

  if returncode != 0:
    # Build failed log errors
    error_s = format_build_error(compile_output, hardware_reason, marker_path)

    # Show TextWindow
    spinner.close()
    if not os.getenv("CI"):
      with TextWindow("openpilot failed to build\n \n" + error_s) as t:
        t.wait_for_exit()
    exit(1)

if __name__ == "__main__":
  build()
