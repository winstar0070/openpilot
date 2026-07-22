#!/usr/bin/env python3
"""Manually exercise ASM2464 F3 PCIe off/on while recording a durable diagnostic log.

This is a bench-only tool. It is not imported by the launcher or modeld and fails
closed unless openpilot reports offroad, deviceState is stopped, panda ignition is
off, and the caller explicitly accepts that external power may be required.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
import contextlib
import json
import os
from pathlib import Path
import re
import signal
import sys
import tempfile
import time
import uuid
from typing import Any


USBGPU_VID, USBGPU_PID = 0xADD1, 0x0001
USBGPU_BOOTSTRAP_MBIT, USBGPU_SUPERSPEED_MBIT = 12, 5000
DEFAULT_LOG_DIR = Path("/data/community/egpu_logs/power_cycle")
DEFAULT_SYSFS_ROOT = Path("/sys/bus/usb/devices")
DEFAULT_DWELL_MS = 500
MIN_DWELL_MS = 300
DEFAULT_REENUM_TIMEOUT_SEC = 15.0
STABLE_DURATION_SEC = 2.0
MAX_ATTEMPT_LOGS = 20


class PowerCycleSafetyError(RuntimeError):
  pass


class PowerCycleInterrupted(RuntimeError):
  pass


def _now_iso() -> str:
  from datetime import datetime
  return datetime.now().astimezone().isoformat()


def _error_details(exc: BaseException) -> dict[str, Any]:
  details: dict[str, Any] = {"type": type(exc).__name__, "message": str(exc)}
  code = getattr(exc, "code", None)
  if isinstance(code, int):
    details["code"] = code
  return details


class AttemptLog:
  def __init__(self, log_dir: Path, device: str, arguments: dict[str, Any]):
    self.log_dir = log_dir
    self.attempt_id = uuid.uuid4().hex
    timestamp = time.strftime("%Y%m%dT%H%M%S", time.localtime())
    self.path = log_dir / f"{timestamp}_{self.attempt_id}.json"
    self.latest_path = log_dir / "latest.json"
    self.data: dict[str, Any] = {
      "schemaVersion": 1,
      "attemptId": self.attempt_id,
      "startedAt": _now_iso(),
      "updatedAt": _now_iso(),
      "device": device,
      "logPath": str(self.path),
      "arguments": arguments,
      "status": "started",
      "events": [],
    }
    self._write()

  @staticmethod
  def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
      with os.fdopen(fd, "w") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
      os.replace(temporary_name, path)
    except BaseException:
      with contextlib.suppress(OSError):
        os.close(fd)
      with contextlib.suppress(OSError):
        os.unlink(temporary_name)
      raise

  def _write(self) -> None:
    self.data["updatedAt"] = _now_iso()
    payload = json.dumps(self.data, indent=2, sort_keys=True) + "\n"
    self._atomic_write(self.path, payload)
    self._atomic_write(self.latest_path, payload)

  def event(self, name: str, status: str, **details: Any) -> None:
    event: dict[str, Any] = {
      "name": name,
      "status": status,
      "timestamp": _now_iso(),
      "monotonicNs": time.monotonic_ns(),
    }
    event.update(details)
    self.data["events"].append(event)
    self._write()

  def finish(self, status: str, **details: Any) -> None:
    self.data.update(details)
    self.data["status"] = status
    self.data["finishedAt"] = _now_iso()
    self._write()
    self._prune()

  def _prune(self) -> None:
    attempts = [path for path in self.log_dir.glob("*.json") if path.name != self.latest_path.name]
    attempts.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
    for old_path in attempts[MAX_ATTEMPT_LOGS:]:
      with contextlib.suppress(OSError):
        old_path.unlink()


def _parse_device(device: str) -> tuple[int, int]:
  match = re.fullmatch(r"usb:([0-9]+)-([0-9]+)", device)
  if match is None:
    raise PowerCycleSafetyError("device must match usb:<busnum>-<devnum>")
  busnum, devnum = (int(value) for value in match.groups())
  if busnum <= 0 or devnum <= 0:
    raise PowerCycleSafetyError("USB busnum and devnum must be positive")
  return busnum, devnum


def _read_text(path: Path) -> str | None:
  try:
    return path.read_text().strip() or None
  except OSError:
    return None


def _read_int(path: Path, *, base: int = 10, allow_float: bool = False) -> int | None:
  value = _read_text(path)
  if value is None:
    return None
  try:
    return int(float(value)) if allow_float else int(value, base)
  except ValueError:
    return None


def _read_sysfs_state(path: Path) -> dict[str, Any]:
  return {
    "sysfsPath": str(path),
    "vendorId": _read_text(path / "idVendor"),
    "productId": _read_text(path / "idProduct"),
    "busnum": _read_int(path / "busnum"),
    "devnum": _read_int(path / "devnum"),
    "speedMbps": _read_int(path / "speed", allow_float=True),
    "configuration": _read_int(path / "bConfigurationValue"),
  }


def _find_sysfs_path(device: str, sysfs_root: Path = DEFAULT_SYSFS_ROOT) -> Path:
  busnum, devnum = _parse_device(device)
  matches = []
  try:
    candidates = list(sysfs_root.iterdir())
  except OSError as exc:
    raise PowerCycleSafetyError(f"unable to scan USB sysfs: {exc}") from exc
  for path in candidates:
    if _read_int(path / "idVendor", base=16) == USBGPU_VID and _read_int(path / "idProduct", base=16) == USBGPU_PID and \
       _read_int(path / "busnum") == busnum and _read_int(path / "devnum") == devnum:
      matches.append(path)
  if len(matches) != 1:
    raise PowerCycleSafetyError(f"expected one sysfs device for {device}, found {len(matches)}")
  return matches[0]


def _ready_identity(path: Path) -> tuple[str, int, int, int] | None:
  state = _read_sysfs_state(path)
  if state["vendorId"] != f"{USBGPU_VID:04x}" or state["productId"] != f"{USBGPU_PID:04x}":
    return None
  speed, configuration, devnum = state["speedMbps"], state["configuration"], state["devnum"]
  if speed != USBGPU_SUPERSPEED_MBIT or not isinstance(configuration, int) or configuration <= 0 or not isinstance(devnum, int):
    return None
  return str(path), devnum, speed, configuration


def _wait_same_path_ready(path: Path, timeout: float, stable_duration: float = STABLE_DURATION_SEC,
                          poll_interval: float = 0.1) -> tuple[str, int, int, int] | None:
  deadline = time.monotonic() + timeout
  stable_identity = None
  stable_since = None
  while True:
    now = time.monotonic()
    identity = _ready_identity(path)
    if identity is None:
      stable_identity = None
      stable_since = None
    elif identity != stable_identity:
      stable_identity = identity
      stable_since = now
    elif stable_since is not None and now - stable_since >= stable_duration:
      return stable_identity
    if now >= deadline:
      return None
    time.sleep(poll_interval)


def check_offroad_ignition_off(timeout: float = 3.0) -> dict[str, Any]:
  from openpilot.cereal import log, messaging
  from openpilot.common.params import Params

  params = Params()
  if not params.get_bool("IsOffroad"):
    raise PowerCycleSafetyError("IsOffroad is false")

  sm = messaging.SubMaster(["deviceState", "pandaStates"])
  deadline = time.monotonic() + timeout
  while time.monotonic() < deadline:
    sm.update(200)
    checks = all(sm.alive[service] and sm.valid[service] for service in ("deviceState", "pandaStates"))
    panda_states = list(sm["pandaStates"])
    known_pandas = [state for state in panda_states if state.pandaType != log.PandaState.PandaType.unknown]
    if not checks or not known_pandas:
      continue
    ignition = any(state.ignitionLine or state.ignitionCan for state in known_pandas)
    started = bool(sm["deviceState"].started)
    if ignition:
      raise PowerCycleSafetyError("panda reports ignition on")
    if started:
      raise PowerCycleSafetyError("deviceState.started is true")
    if not params.get_bool("IsOffroad"):
      raise PowerCycleSafetyError("IsOffroad changed while checking lifecycle")
    return {
      "isOffroad": True,
      "deviceStarted": False,
      "ignition": False,
      "knownPandaCount": len(known_pandas),
    }
  raise PowerCycleSafetyError("fresh valid deviceState and pandaStates were not available")


@contextlib.contextmanager
def _open_asm24_session(device: str) -> Iterator[Any]:
  from tinygrad.runtime.autogen import libusb
  from tinygrad.runtime.support.system import System
  from tinygrad.runtime.support.usb import USB3

  devices = USB3.list_devices(USBGPU_VID, USBGPU_PID)
  selected = None
  usb, lock_fd = None, None
  try:
    matches = [index for index, (_, identity) in enumerate(devices) if identity == device]
    if len(matches) != 1:
      raise PowerCycleSafetyError(f"expected one libusb device for {device}, found {len(matches)}")
    selected = devices.pop(matches[0])
    speed_code = USB3.device_speed(selected[0])
    allowed_codes = (libusb.LIBUSB_SPEED_FULL, libusb.LIBUSB_SPEED_SUPER)
    if speed_code not in allowed_codes:
      raise PowerCycleSafetyError("USB device must be at exact 12Mbps or 5000Mbps")
    lock_fd = System.flock_acquire(f"am_{device.lower()}.lock")
    selected_dev, selected = selected[0], None
    usb = USB3(selected_dev, 0x81, 0x83, 0x02, 0x04, use_bot=True)
    if not usb.is_custom:
      raise PowerCycleSafetyError("USB device is not running custom firmware")
    yield usb
  finally:
    try:
      if usb is not None:
        usb.close()
    finally:
      try:
        if selected is not None:
          libusb.libusb_unref_device(selected[0])
        for dev, _ in devices:
          libusb.libusb_unref_device(dev)
      finally:
        if lock_fd is not None:
          os.close(lock_fd)


def _send_power(usb: Any, enabled: bool) -> int:
  from tinygrad.runtime.support.usb import asm24_set_pcie_power
  return asm24_set_pcie_power(usb, enabled)


def _is_no_device_error(exc: BaseException) -> bool:
  from tinygrad.runtime.autogen import libusb
  from tinygrad.runtime.support.usb import USBError
  return isinstance(exc, USBError) and exc.code == libusb.LIBUSB_ERROR_NO_DEVICE


@contextlib.contextmanager
def _recover_on_termination() -> Iterator[None]:
  handled_signals = tuple(getattr(signal, name) for name in ("SIGTERM", "SIGHUP", "SIGQUIT") if hasattr(signal, name))
  previous = {signum: signal.getsignal(signum) for signum in handled_signals}
  def terminate(signum, frame):
    raise PowerCycleInterrupted(f"received signal {signum}")
  for signum in handled_signals:
    signal.signal(signum, terminate)
  try:
    yield
  finally:
    for signum, handler in previous.items():
      signal.signal(signum, handler)


def run_power_cycle(device: str, logger: AttemptLog, *, dwell_ms: int = DEFAULT_DWELL_MS,
                    reenum_timeout_sec: float = DEFAULT_REENUM_TIMEOUT_SEC,
                    sysfs_root: Path = DEFAULT_SYSFS_ROOT, dry_run: bool = False) -> dict[str, Any]:
  try:
    if dwell_ms < MIN_DWELL_MS:
      raise PowerCycleSafetyError(f"dwell must be at least {MIN_DWELL_MS}ms")
    if reenum_timeout_sec <= STABLE_DURATION_SEC:
      raise PowerCycleSafetyError("re-enumeration timeout must exceed the 2s stable duration")

    lifecycle = check_offroad_ignition_off()
    logger.event("lifecycle", "passed", **lifecycle)
    sysfs_path = _find_sysfs_path(device, sysfs_root)
    initial_state = _read_sysfs_state(sysfs_path)
    if initial_state["speedMbps"] not in (USBGPU_BOOTSTRAP_MBIT, USBGPU_SUPERSPEED_MBIT):
      raise PowerCycleSafetyError("initial USB speed must be exact 12Mbps or 5000Mbps")
    if not isinstance(initial_state["configuration"], int) or initial_state["configuration"] <= 0:
      raise PowerCycleSafetyError("initial USB device is not configured")
    logger.event("initial_usb_state", "passed", state=initial_state)

    if dry_run:
      logger.finish("dry_run_passed", lifecycle=lifecycle, initialState=initial_state)
      return logger.data

    with _recover_on_termination(), _open_asm24_session(device) as usb:
      logger.event("usb_session", "opened", product=getattr(usb, "product", None), identity=usb.configuration_identity())
      off_attempted, on_attempted = False, False
      try:
        off_started = time.monotonic()
        logger.event("f3_off", "started")
        off_attempted = True
        off_return_code = _send_power(usb, False)
        logger.event("f3_off", "acknowledged", returnCode=off_return_code,
                     durationMs=round((time.monotonic() - off_started) * 1000, 3))

        try:
          post_off_identity = usb.configuration_identity()
        except Exception as exc:
          logger.event("post_off_control", "failed", error=_error_details(exc))
        else:
          logger.event("post_off_control", "alive", identity=post_off_identity, state=_read_sysfs_state(sysfs_path))

        logger.event("off_dwell", "started", requestedMs=dwell_ms)
        time.sleep(dwell_ms / 1000.0)
        logger.event("off_dwell", "completed", requestedMs=dwell_ms)

        on_started = time.monotonic()
        logger.event("f3_on", "started")
        on_attempted = True
        try:
          on_return_code = _send_power(usb, True)
        except Exception as exc:
          if not _is_no_device_error(exc):
            logger.event("f3_on", "failed", durationMs=round((time.monotonic() - on_started) * 1000, 3), error=_error_details(exc))
            raise
          logger.event("f3_on", "detached_ambiguous", durationMs=round((time.monotonic() - on_started) * 1000, 3),
                       error=_error_details(exc))
        else:
          logger.event("f3_on", "acknowledged", returnCode=on_return_code,
                       durationMs=round((time.monotonic() - on_started) * 1000, 3))
      finally:
        if off_attempted and not on_attempted:
          with contextlib.suppress(BaseException):
            logger.event("recovery_f3_on", "started", reason="interrupted after F3 off attempt")
          try:
            recovery_return_code = _send_power(usb, True)
          except BaseException as exc:
            with contextlib.suppress(BaseException):
              logger.event("recovery_f3_on", "failed", error=_error_details(exc))
          else:
            with contextlib.suppress(BaseException):
              logger.event("recovery_f3_on", "acknowledged", returnCode=recovery_return_code)

    logger.event("reenumeration", "started", requiredSpeedMbps=USBGPU_SUPERSPEED_MBIT,
                 requiredStableSec=STABLE_DURATION_SEC, timeoutSec=reenum_timeout_sec)
    stable_identity = _wait_same_path_ready(sysfs_path, reenum_timeout_sec)
    final_identity = _ready_identity(sysfs_path)
    if stable_identity is None or final_identity != stable_identity:
      raise PowerCycleSafetyError("USB device did not reach stable configured 5000Mbps on the same sysfs path")
    final_state = _read_sysfs_state(sysfs_path)
    logger.event("reenumeration", "stable", identity=stable_identity, state=final_state)
    logger.finish("passed", lifecycle=lifecycle, initialState=initial_state, finalState=final_state)
    return logger.data
  except BaseException as exc:
    logger.finish("failed", error=_error_details(exc))
    raise


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description="Bench-only ASM2464 F3 PCIe off/on test with durable JSON logging")
  parser.add_argument("--device", required=True, help="Current libusb identity, for example usb:4-10")
  parser.add_argument("--confirm-risk", action="store_true",
                      help="Confirm that a failed ON may leave the GPU off until external power is cycled")
  parser.add_argument("--dry-run", action="store_true", help="Check lifecycle and device eligibility without sending F3")
  parser.add_argument("--dwell-ms", type=int, default=DEFAULT_DWELL_MS)
  parser.add_argument("--reenum-timeout-sec", type=float, default=DEFAULT_REENUM_TIMEOUT_SEC)
  parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
  args = parser.parse_args(argv)

  if not args.dry_run and not args.confirm_risk:
    parser.error("--confirm-risk is required because failed recovery may require external power")

  arguments = {
    "dryRun": args.dry_run,
    "dwellMs": args.dwell_ms,
    "reenumTimeoutSec": args.reenum_timeout_sec,
    "requiredStableSec": STABLE_DURATION_SEC,
  }
  try:
    logger = AttemptLog(args.log_dir, args.device, arguments)
    result = run_power_cycle(args.device, logger, dwell_ms=args.dwell_ms,
                             reenum_timeout_sec=args.reenum_timeout_sec, dry_run=args.dry_run)
  except BaseException as exc:
    log_path = str(logger.path) if "logger" in locals() else None
    print(json.dumps({"status": "failed", "error": _error_details(exc), "logPath": log_path}, sort_keys=True), file=sys.stderr)
    return 1
  print(json.dumps({"status": result["status"], "logPath": str(logger.path)}, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
