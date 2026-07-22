import io
import json
import os
import pickle
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / 'models'
TG_INPUT_DEVICES_PATH = MODELS_DIR / 'tg_input_devices.json'
USBGPU_VID = 0xADD1
USBGPU_PID = 0x0001
USBGPU_SYSFS_ROOT = Path("/sys/bus/usb/devices")
USBGPU_BOOTSTRAP_MBIT = 12
USBGPU_SUPERSPEED_MBIT = 5000
USBGPU_REENUM_TIMEOUT_MS = 15000
USBGPU_READY_TIMEOUT_SEC = 5.0
USBGPU_STABLE_DURATION_MS = 2000
USBGPU_PCIE_POWER_TIMEOUT_SEC = 12.0
USBGPU_BUILD_DIAGNOSTIC_PATH = Path("/data/community/egpu_logs/build/latest.json")
TINYGRAD_COMPILER_PROBE_BACKENDS = ("CUDA", "QCOM")
TINYGRAD_PCIE_POWER_MODULE = "tinygrad.runtime.support.asm24_pcie_power"

UsbGpuReadyIdentity = tuple[str, int, int, int]


def tinygrad_compiler_probe_script() -> str:
  return "\n".join((
    "from tinygrad import Device",
    f"for candidate in {TINYGRAD_COMPILER_PROBE_BACKENDS!r}:",
    "  try:",
    "    print(Device[candidate].device)",
    "    break",
    "  except Exception:",
    "    pass",
  )) + "\n"


def get_tg_input_devices(process_name: str, usbgpu: bool):
  with open(TG_INPUT_DEVICES_PATH) as f:
    return json.load(f)[process_name]['default' if not usbgpu else 'usbgpu']

def modeld_pkl_path(usbgpu: bool):
  prefix = 'big_' if usbgpu else ''
  return MODELS_DIR / f'{prefix}driving_tinygrad.pkl'

def dump_oob(obj, f):
  with tempfile.TemporaryFile(dir=".") as tmp:
    def buffer_callback(pb: pickle.PickleBuffer):
      m = pb.raw()
      tmp.write(struct.pack('<q', m.nbytes))
      tmp.write(m)
      pb.release() # keep peak ram at ~1 buffer
    stream = io.BytesIO()
    pickle.Pickler(stream, protocol=5, buffer_callback=buffer_callback).dump(obj)
    opcodes = stream.getvalue()
    f.write(struct.pack('<q', len(opcodes)))
    f.write(opcodes)
    tmp.seek(0)
    shutil.copyfileobj(tmp, f)

def load_oob(f):
  opcodes = f.read(struct.unpack('<q', f.read(8))[0])
  def buffers():
    prev = None
    while (h := f.read(8)):
      if prev is not None:
        prev.release()
      buf = bytearray(struct.unpack('<q', h)[0])
      f.readinto(buf)
      prev = pickle.PickleBuffer(buf)
      yield prev
  return pickle.load(io.BytesIO(opcodes), buffers=buffers())

def _usbgpu_devices(sysfs_root: Path = USBGPU_SYSFS_ROOT) -> list[Path]:
  devices = []
  try:
    candidates = sorted(sysfs_root.glob("*"))
  except OSError:
    return devices
  for d in candidates:
    try:
      if int((d / "idVendor").read_text(), 16) == USBGPU_VID and \
          int((d / "idProduct").read_text(), 16) == USBGPU_PID:
        devices.append(d)
    except (OSError, ValueError):
      pass
  return devices


def usbgpu_present(sysfs_root: Path = USBGPU_SYSFS_ROOT) -> bool:
  return bool(_usbgpu_devices(sysfs_root))


def wait_for_usbgpu_present(timeout: float, poll_interval: float = 0.1,
                            sysfs_root: Path = USBGPU_SYSFS_ROOT) -> bool:
  deadline = time.monotonic() + timeout
  while not usbgpu_present(sysfs_root):
    if time.monotonic() >= deadline:
      return False
    time.sleep(poll_interval)
  return True


def usbgpu_speed(sysfs_root: Path = USBGPU_SYSFS_ROOT) -> int | None:
  speeds = []
  for d in _usbgpu_devices(sysfs_root):
    try:
      speeds.append(int(float((d / "speed").read_text().strip())))
    except (OSError, ValueError):
      pass
  return max(speeds, default=None)


def usbgpu_speed_eligible(speed: int | None) -> bool:
  return speed in (USBGPU_BOOTSTRAP_MBIT, USBGPU_SUPERSPEED_MBIT)


def usbgpu_build_speed_eligible(speed: int | None, configuration: int | None) -> bool:
  return speed == USBGPU_SUPERSPEED_MBIT and configuration is not None and configuration > 0


def usbgpu_ready_identity(sysfs_root: Path = USBGPU_SYSFS_ROOT) -> UsbGpuReadyIdentity | None:
  identities = []
  for d in _usbgpu_devices(sysfs_root):
    try:
      speed = int(float((d / "speed").read_text().strip()))
      configuration = int((d / "bConfigurationValue").read_text().strip())
      devnum = int((d / "devnum").read_text().strip())
      if speed == USBGPU_SUPERSPEED_MBIT and configuration > 0:
        identities.append((str(d), devnum, speed, configuration))
    except (OSError, ValueError):
      pass
  return min(identities, default=None)


def usbgpu_superspeed_ready(sysfs_root: Path = USBGPU_SYSFS_ROOT) -> bool:
  return usbgpu_ready_identity(sysfs_root) is not None


def wait_for_usbgpu_ready_identity(timeout: float, stable_duration: float = 0.0, poll_interval: float = 0.1,
                                   sysfs_root: Path = USBGPU_SYSFS_ROOT) -> UsbGpuReadyIdentity | None:
  deadline = time.monotonic() + timeout
  stable_identity = None
  stable_since = None
  while True:
    now = time.monotonic()
    identity = usbgpu_ready_identity(sysfs_root)
    if identity is None:
      stable_identity = None
      stable_since = None
    elif identity != stable_identity:
      stable_identity = identity
      stable_since = now
    elif stable_since is not None and now - stable_since >= stable_duration:
      return stable_identity

    if identity is not None and stable_duration <= 0.0:
      return identity
    if now >= deadline:
      return None
    time.sleep(poll_interval)


def wait_for_usbgpu_ready(timeout: float, stable_duration: float = 0.0, poll_interval: float = 0.1,
                          sysfs_root: Path = USBGPU_SYSFS_ROOT) -> bool:
  return wait_for_usbgpu_ready_identity(timeout, stable_duration, poll_interval, sysfs_root) is not None


def _read_usbgpu_field(device: Path, field: str) -> str | None:
  try:
    return (device / field).read_text().strip() or None
  except OSError:
    return None


def _parse_usbgpu_int(value: str | None, *, allow_float: bool = False) -> int | None:
  try:
    return int(float(value)) if allow_float and value is not None else int(value) if value is not None else None
  except ValueError:
    return None


def get_usbgpu_build_diagnostic(artifact_path: str | Path, sysfs_root: Path = USBGPU_SYSFS_ROOT) -> dict[str, object]:
  devices = _usbgpu_devices(sysfs_root)
  states = []
  for device in devices:
    states.append({
      "sysfsPath": str(device),
      "vendorId": _read_usbgpu_field(device, "idVendor"),
      "productId": _read_usbgpu_field(device, "idProduct"),
      "speedMbps": _parse_usbgpu_int(_read_usbgpu_field(device, "speed"), allow_float=True),
      "configuration": _parse_usbgpu_int(_read_usbgpu_field(device, "bConfigurationValue")),
      "busnum": _parse_usbgpu_int(_read_usbgpu_field(device, "busnum")),
      "devnum": _parse_usbgpu_int(_read_usbgpu_field(device, "devnum")),
    })

  readable_speeds = [state["speedMbps"] for state in states if state["speedMbps"] is not None]
  selected_speed = max(readable_speeds, default=None)
  candidates = [state for state in states if state["speedMbps"] == selected_speed] if selected_speed is not None else states
  state = min(candidates, key=lambda item: item["sysfsPath"], default=None)
  speed = state["speedMbps"] if state is not None else None
  if state is None:
    reason_code = "device_not_found"
    reason = "USB GPU device was not detected"
  elif speed is None:
    reason_code = "speed_unavailable"
    reason = "USB GPU link speed could not be read"
  elif speed == USBGPU_BOOTSTRAP_MBIT:
    reason_code = "bootstrap"
    reason = "USB GPU bootstrap link detected; 5000Mbps is required to build the AMD target"
  elif speed == USBGPU_SUPERSPEED_MBIT:
    reason_code = "superspeed"
    reason = "USB GPU SuperSpeed link detected"
  else:
    reason_code = "unsupported_speed"
    reason = f"USB GPU link speed {speed}Mbps is not eligible"

  configuration = state["configuration"] if state is not None else None
  enabled = usbgpu_build_speed_eligible(speed, configuration)
  if speed == USBGPU_SUPERSPEED_MBIT and not enabled:
    reason = "USB GPU SuperSpeed link detected, but the device is not configured"
  try:
    artifact_present = Path(artifact_path).is_file()
  except OSError:
    artifact_present = False
  return {
    "timestamp": datetime.now().astimezone().isoformat(),
    "decision": "enable_target" if enabled else "skip_target",
    "reasonCode": reason_code,
    "reason": reason,
    "devicePresent": state is not None,
    "sysfsPath": state["sysfsPath"] if state is not None else None,
    "vendorId": state["vendorId"] if state is not None else None,
    "productId": state["productId"] if state is not None else None,
    "speedMbps": speed,
    "configuration": configuration,
    "busnum": state["busnum"] if state is not None else None,
    "devnum": state["devnum"] if state is not None else None,
    "artifactPresent": artifact_present,
    "allowedBootstrapMbps": USBGPU_BOOTSTRAP_MBIT,
    "allowedSuperSpeedMbps": USBGPU_SUPERSPEED_MBIT,
    "incrementalBuild": True,
  }


def _env_int(name: str, default: int) -> int:
  try:
    return max(0, int(os.getenv(name, str(default))))
  except ValueError:
    return default


def _env_float(name: str, default: float) -> float:
  try:
    return max(0.0, float(os.getenv(name, str(default))))
  except ValueError:
    return default


def _default_usbgpu_pcie_power_result() -> dict[str, object]:
  return {
    "pciePowerAction": None,
    "pciePowerAttempted": False,
    "pciePowerStatus": "not_applicable",
    "pciePowerDevice": None,
    "pciePowerSpeedMbps": None,
    "pciePowerReturnCode": None,
    "pciePowerDurationMs": 0,
    "pciePowerError": None,
  }


def _short_subprocess_text(value: object, limit: int = 512) -> str | None:
  if isinstance(value, bytes):
    value = value.decode(errors="replace")
  if not isinstance(value, str):
    return None
  text = " ".join(value.strip().split())
  return text[:limit] or None


def request_usbgpu_pcie_power_on(initial_diagnostic: dict[str, object], tinygrad_repo_root: str | Path, *,
                                 allow_power: bool = False, timeout_sec: float | None = None) -> dict[str, object]:
  """Request one PCIe power-on only for the exact 12Mbps bootstrap device.

  The helper runs the tinygrad USB-only command in a subprocess. An accepted command
  only permits the caller to wait for SuperSpeed; it never makes the AMD target eligible.
  """
  result = _default_usbgpu_pcie_power_result()
  if initial_diagnostic.get("reasonCode") != "bootstrap" or initial_diagnostic.get("speedMbps") != USBGPU_BOOTSTRAP_MBIT:
    return result

  result["pciePowerAction"] = "on"
  if not allow_power:
    result.update({
      "pciePowerStatus": "lifecycle_blocked",
      "pciePowerError": "PCIe power-on request requires launcher authorization for the initial build attempt",
    })
    return result

  busnum, devnum = initial_diagnostic.get("busnum"), initial_diagnostic.get("devnum")
  if not isinstance(busnum, int) or busnum <= 0 or not isinstance(devnum, int) or devnum <= 0:
    result.update({
      "pciePowerStatus": "identity_unavailable",
      "pciePowerError": "bootstrap USB busnum/devnum is unavailable",
    })
    return result

  device = f"usb:{busnum}-{devnum}"
  timeout_sec = _env_float("USBGPU_PCIE_POWER_TIMEOUT_SEC", USBGPU_PCIE_POWER_TIMEOUT_SEC) \
    if timeout_sec is None else max(0.0, timeout_sec)
  command = [sys.executable, "-m", TINYGRAD_PCIE_POWER_MODULE, "on", "--device", device]
  result.update({
    "pciePowerAttempted": True,
    "pciePowerDevice": device,
  })
  started = time.monotonic()
  try:
    completed = subprocess.run(command, cwd=Path(tinygrad_repo_root), capture_output=True, text=True,
                               check=False, timeout=timeout_sec)
  except subprocess.TimeoutExpired as exc:
    result.update({
      "pciePowerStatus": "timeout",
      "pciePowerError": _short_subprocess_text(exc.stderr) or f"PCIe power command timed out after {timeout_sec:g}s",
    })
  except OSError as exc:
    result.update({
      "pciePowerStatus": "error",
      "pciePowerError": _short_subprocess_text(str(exc)),
    })
  else:
    result["pciePowerReturnCode"] = completed.returncode
    stdout_lines = [line for line in completed.stdout.splitlines() if line.strip()]
    payload = None
    if len(stdout_lines) == 1:
      try:
        candidate = json.loads(stdout_lines[0])
        payload = candidate if isinstance(candidate, dict) else None
      except json.JSONDecodeError:
        pass

    if payload is None:
      result.update({
        "pciePowerStatus": "invalid_response",
        "pciePowerError": _short_subprocess_text(completed.stderr) or "PCIe power command returned invalid JSON",
      })
    else:
      child_status = payload.get("status")
      result.update({
        "pciePowerStatus": child_status if isinstance(child_status, str) else "invalid_response",
        "pciePowerSpeedMbps": payload.get("speedMbps") if isinstance(payload.get("speedMbps"), int) else None,
        "pciePowerError": _short_subprocess_text(payload.get("error")) or _short_subprocess_text(completed.stderr),
      })
      response_valid = payload.get("action") == "on" and payload.get("device") == device and \
        payload.get("speedMbps") == USBGPU_BOOTSTRAP_MBIT and child_status in ("sent", "detached")
      if completed.returncode != 0 or not response_valid:
        if completed.returncode == 0 and not response_valid:
          result["pciePowerStatus"] = "invalid_response"
          result["pciePowerError"] = result["pciePowerError"] or "PCIe power command response failed validation"
        elif result["pciePowerError"] is None:
          result["pciePowerError"] = f"PCIe power command exited with status {completed.returncode}"
  finally:
    result["pciePowerDurationMs"] = max(0, int(round((time.monotonic() - started) * 1000)))
  return result


def resolve_usbgpu_build_diagnostic(artifact_path: str | Path, sysfs_root: Path = USBGPU_SYSFS_ROOT, *,
                                    initial_diagnostic: dict[str, object] | None = None,
                                    pcie_power_result: dict[str, object] | None = None,
                                    reenum_timeout_ms: int | None = None, ready_timeout_sec: float | None = None,
                                    stable_duration_ms: int | None = None) -> dict[str, object]:
  """Wait for a stable, configured 5000Mbps device before enabling the AMD build target.

  A 12Mbps device may enter the long re-enumeration wait only after an accepted PCIe
  power request. This function never opens the USB device or starts AMD initialization;
  exact configured 5000Mbps stability is still required before enabling the target.
  """
  initial = dict(initial_diagnostic or get_usbgpu_build_diagnostic(artifact_path, sysfs_root))
  initial_reason = initial["reasonCode"]
  initial_speed = initial["speedMbps"]
  reenum_timeout_ms = _env_int("USB_REENUM_TIMEOUT_MS", USBGPU_REENUM_TIMEOUT_MS) if reenum_timeout_ms is None else max(0, reenum_timeout_ms)
  ready_timeout_sec = _env_float("USBGPU_READY_TIMEOUT_SEC", USBGPU_READY_TIMEOUT_SEC) if ready_timeout_sec is None else max(0.0, ready_timeout_sec)
  stable_duration_ms = _env_int("USB_SUPERSPEED_STABLE_MS", USBGPU_STABLE_DURATION_MS) \
    if stable_duration_ms is None else max(0, stable_duration_ms)

  diagnostic = dict(initial)
  power_result = _default_usbgpu_pcie_power_result()
  if pcie_power_result is not None:
    power_result.update({key: pcie_power_result.get(key, value) for key, value in power_result.items()})
  diagnostic.update({
    "initialSpeedMbps": initial_speed,
    "initialSysfsPath": initial["sysfsPath"],
    "initialBusnum": initial["busnum"],
    "initialDevnum": initial["devnum"],
    "pciePowerRecheckAttempted": False,
    "pciePowerRecheckSpeedMbps": None,
    "pciePowerRecheckConfiguration": None,
    "pciePowerRecheckEligibleForWait": False,
    "readinessWaitAttempted": False,
    "readinessStatus": "not_applicable",
    "readinessWaitMs": 0,
    "stableSysfsPath": None,
    "stableDevnum": None,
    "stableSpeedMbps": None,
    "stableConfiguration": None,
    "stableIdentityMatchedFinal": False,
    "requiredBuildSpeedMbps": USBGPU_SUPERSPEED_MBIT,
    "requiredConfiguration": True,
    "requiredStableMs": stable_duration_ms,
  })
  diagnostic.update(power_result)

  if initial_reason not in ("bootstrap", "superspeed"):
    return diagnostic

  power_accepted = (
    power_result["pciePowerAttempted"] is True and
    power_result["pciePowerReturnCode"] == 0 and
    power_result["pciePowerStatus"] in ("sent", "detached")
  )
  power_race_recheck_allowed = (
    power_result["pciePowerAttempted"] is True and
    power_result["pciePowerReturnCode"] == 2 and
    power_result["pciePowerStatus"] == "not_applicable"
  )

  bootstrap_power_race = False
  if initial_reason == "bootstrap" and not power_accepted:
    if power_result["pciePowerStatus"] == "lifecycle_blocked":
      diagnostic.update({
        "decision": "skip_target",
        "readinessStatus": "lifecycle_blocked",
        "reason": "USB GPU PCIe power-on request blocked without launcher authorization; " +
                  "AMD target skipped and QCOM build continues",
      })
      return diagnostic

    if not power_race_recheck_allowed:
      diagnostic.update({
        "decision": "skip_target",
        "readinessStatus": "power_failed",
        "reason": "USB GPU PCIe power-on request failed without an eligible re-enumeration race; " +
                  "AMD target skipped and QCOM build continues",
      })
      return diagnostic

    recheck = get_usbgpu_build_diagnostic(artifact_path, sysfs_root)
    recheck_superspeed = recheck["speedMbps"] == USBGPU_SUPERSPEED_MBIT
    diagnostic.update({
      "pciePowerRecheckAttempted": True,
      "pciePowerRecheckSpeedMbps": recheck["speedMbps"],
      "pciePowerRecheckConfiguration": recheck["configuration"],
      "pciePowerRecheckEligibleForWait": recheck_superspeed,
    })
    if not recheck_superspeed:
      diagnostic.update({
        "decision": "skip_target",
        "readinessStatus": "power_failed",
        "reason": "USB GPU PCIe power-on request failed and immediate recheck was not 5000Mbps; " +
                  "AMD target skipped and QCOM build continues",
      })
      return diagnostic

    # The device may have re-enumerated between the sysfs observation and the child
    # command. Validate that already-present 5000Mbps link with the short stable wait.
    diagnostic.update(recheck)
    diagnostic.update(power_result)
    bootstrap_power_race = True

  timeout = reenum_timeout_ms / 1000.0 if initial_reason == "bootstrap" and power_accepted else ready_timeout_sec
  started = time.monotonic()
  stable_identity = wait_for_usbgpu_ready_identity(timeout, stable_duration_ms / 1000.0, sysfs_root=sysfs_root)
  elapsed_ms = max(0, int(round((time.monotonic() - started) * 1000)))
  final = get_usbgpu_build_diagnostic(artifact_path, sysfs_root)
  final_identity = (final["sysfsPath"], final["devnum"], final["speedMbps"], final["configuration"])
  stable_identity_matched = stable_identity is not None and final_identity == stable_identity
  final_eligible = stable_identity_matched and usbgpu_build_speed_eligible(final["speedMbps"], final["configuration"])

  diagnostic.update(final)
  diagnostic.update({
    "initialSpeedMbps": initial_speed,
    "initialSysfsPath": initial["sysfsPath"],
    "initialBusnum": initial["busnum"],
    "initialDevnum": initial["devnum"],
    "readinessWaitAttempted": True,
    "readinessStatus": "stable" if final_eligible else "unstable" if stable_identity is not None else "timeout",
    "readinessWaitMs": elapsed_ms,
    "stableSysfsPath": stable_identity[0] if stable_identity is not None else None,
    "stableDevnum": stable_identity[1] if stable_identity is not None else None,
    "stableSpeedMbps": stable_identity[2] if stable_identity is not None else None,
    "stableConfiguration": stable_identity[3] if stable_identity is not None else None,
    "stableIdentityMatchedFinal": stable_identity_matched,
    "requiredBuildSpeedMbps": USBGPU_SUPERSPEED_MBIT,
    "requiredConfiguration": True,
    "requiredStableMs": stable_duration_ms,
    "decision": "enable_target" if final_eligible else "skip_target",
    "reasonCode": initial_reason,
  })
  if initial_reason == "bootstrap":
    if bootstrap_power_race:
      diagnostic["reason"] = (
        "USB GPU reached stable configured 5000Mbps before the PCIe power command completed"
        if final_eligible else
        f"USB GPU reached 5000Mbps during the PCIe power command race but did not remain configured and stable for {stable_duration_ms}ms"
      )
    else:
      diagnostic["reason"] = (
        "USB GPU bootstrap link transitioned to stable configured 5000Mbps after the PCIe power-on request"
        if final_eligible else
        f"USB GPU bootstrap link did not reach stable configured 5000Mbps within {reenum_timeout_ms}ms " +
        "following the PCIe power-on request"
      )
  elif not final_eligible:
    diagnostic["reason"] = f"USB GPU SuperSpeed link did not remain configured and stable for {stable_duration_ms}ms"
  return diagnostic


def format_usbgpu_build_decision(diagnostic: dict[str, object]) -> str:
  speed = diagnostic["speedMbps"]
  speed_text = f"{speed}Mbps" if speed is not None else "unavailable"
  artifact = "present" if diagnostic["artifactPresent"] else "missing"
  if diagnostic["decision"] == "enable_target":
    suffix = "SCons will rebuild only if stale"
    action = "enabled"
    artifact_text = f" artifact={artifact}"
  else:
    suffix = "QCOM build continues"
    action = "skipped"
    artifact_text = ""
  initial_speed = diagnostic.get("initialSpeedMbps")
  initial_text = f" initial_speed={initial_speed}Mbps" if initial_speed is not None and initial_speed != speed else ""
  readiness = diagnostic.get("readinessStatus")
  readiness_text = f" readiness={readiness}" if readiness not in (None, "not_applicable") else ""
  return " ".join((
    f'[USBGPU] target {action}: reason={diagnostic["reasonCode"]} speed={speed_text}{initial_text}{readiness_text}{artifact_text}',
    f'rule="stable configured 5000 only"; {suffix}',
  ))


def format_usbgpu_pcie_power(diagnostic: dict[str, object]) -> str:
  error = diagnostic.get("pciePowerError")
  error_text = f" error={json.dumps(error)}" if error is not None else ""
  speed = diagnostic.get("pciePowerSpeedMbps")
  speed_text = f"{speed}Mbps" if speed is not None else "unavailable"
  return " ".join((
    f'[USBGPU] PCIe power: action={diagnostic.get("pciePowerAction") or "none"}',
    f'attempted={str(diagnostic.get("pciePowerAttempted") is True).lower()}',
    f'status={diagnostic.get("pciePowerStatus") or "unknown"}',
    f'device={diagnostic.get("pciePowerDevice") or "unavailable"}',
    f'speed={speed_text}',
    f'returncode={diagnostic.get("pciePowerReturnCode")}',
    f'duration={diagnostic.get("pciePowerDurationMs", 0)}ms{error_text}',
  ))


def format_usbgpu_build_device(diagnostic: dict[str, object]) -> str:
  return " ".join((
    f'[USBGPU] device: path={diagnostic["sysfsPath"] or "unavailable"} vendor={diagnostic["vendorId"] or "unavailable"}',
    f'product={diagnostic["productId"] or "unavailable"} configuration={diagnostic["configuration"]}',
    f'busnum={diagnostic["busnum"]} devnum={diagnostic["devnum"]}',
  ))


def write_usbgpu_build_diagnostic(diagnostic: dict[str, object], path: Path = USBGPU_BUILD_DIAGNOSTIC_PATH) -> bool:
  tmp_path = None
  try:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(diagnostic, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as tmp:
      tmp_path = Path(tmp.name)
      tmp.write(content)
      tmp.flush()
      os.fsync(tmp.fileno())
    tmp_path.replace(path)
    return True
  except Exception as e:
    if tmp_path is not None:
      try:
        tmp_path.unlink(missing_ok=True)
      except OSError:
        pass
    print(f"[USBGPU] warning: unable to write build diagnostic {path}: {type(e).__name__}: {e}", file=sys.stderr)
    return False
