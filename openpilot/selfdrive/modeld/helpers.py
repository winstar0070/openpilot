import io
import json
import os
import pickle
import shutil
import struct
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
USBGPU_BUILD_DIAGNOSTIC_PATH = Path("/data/community/egpu_logs/build/latest.json")
TINYGRAD_COMPILER_PROBE_BACKENDS = ("CUDA", "QCOM")

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


def wait_for_usbgpu_ready(timeout: float, stable_duration: float = 0.0, poll_interval: float = 0.1,
                          sysfs_root: Path = USBGPU_SYSFS_ROOT) -> bool:
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
      return True

    if identity is not None and stable_duration <= 0.0:
      return True
    if now >= deadline:
      return False
    time.sleep(poll_interval)


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


def resolve_usbgpu_build_diagnostic(artifact_path: str | Path, sysfs_root: Path = USBGPU_SYSFS_ROOT, *,
                                    initial_diagnostic: dict[str, object] | None = None,
                                    reenum_timeout_ms: int | None = None, ready_timeout_sec: float | None = None,
                                    stable_duration_ms: int | None = None) -> dict[str, object]:
  """Wait for a stable, configured 5000Mbps device before enabling the AMD build target.

  A 12Mbps device is only a passive bootstrap observation. This function never opens the
  USB device or starts AMD initialization; firmware-driven re-enumeration must complete
  before the build target can be enabled.
  """
  initial = dict(initial_diagnostic or get_usbgpu_build_diagnostic(artifact_path, sysfs_root))
  initial_reason = initial["reasonCode"]
  initial_speed = initial["speedMbps"]
  reenum_timeout_ms = _env_int("USB_REENUM_TIMEOUT_MS", USBGPU_REENUM_TIMEOUT_MS) if reenum_timeout_ms is None else max(0, reenum_timeout_ms)
  ready_timeout_sec = _env_float("USBGPU_READY_TIMEOUT_SEC", USBGPU_READY_TIMEOUT_SEC) if ready_timeout_sec is None else max(0.0, ready_timeout_sec)
  stable_duration_ms = _env_int("USB_SUPERSPEED_STABLE_MS", USBGPU_STABLE_DURATION_MS) \
    if stable_duration_ms is None else max(0, stable_duration_ms)

  diagnostic = dict(initial)
  diagnostic.update({
    "initialSpeedMbps": initial_speed,
    "initialSysfsPath": initial["sysfsPath"],
    "initialDevnum": initial["devnum"],
    "readinessWaitAttempted": False,
    "readinessStatus": "not_applicable",
    "readinessWaitMs": 0,
    "requiredBuildSpeedMbps": USBGPU_SUPERSPEED_MBIT,
    "requiredConfiguration": True,
    "requiredStableMs": stable_duration_ms,
  })

  if initial_reason not in ("bootstrap", "superspeed"):
    return diagnostic

  timeout = reenum_timeout_ms / 1000.0 if initial_reason == "bootstrap" else ready_timeout_sec
  started = time.monotonic()
  ready = wait_for_usbgpu_ready(timeout, stable_duration_ms / 1000.0, sysfs_root=sysfs_root)
  elapsed_ms = max(0, int(round((time.monotonic() - started) * 1000)))
  final = get_usbgpu_build_diagnostic(artifact_path, sysfs_root)
  final_eligible = ready and usbgpu_build_speed_eligible(final["speedMbps"], final["configuration"])

  diagnostic.update(final)
  diagnostic.update({
    "initialSpeedMbps": initial_speed,
    "initialSysfsPath": initial["sysfsPath"],
    "initialDevnum": initial["devnum"],
    "readinessWaitAttempted": True,
    "readinessStatus": "stable" if final_eligible else "unstable" if ready else "timeout",
    "readinessWaitMs": elapsed_ms,
    "requiredBuildSpeedMbps": USBGPU_SUPERSPEED_MBIT,
    "requiredConfiguration": True,
    "requiredStableMs": stable_duration_ms,
    "decision": "enable_target" if final_eligible else "skip_target",
    "reasonCode": initial_reason,
  })
  if initial_reason == "bootstrap":
    diagnostic["reason"] = (
      "USB GPU bootstrap link transitioned to stable configured 5000Mbps"
      if final_eligible else
      f"USB GPU bootstrap link did not reach stable configured 5000Mbps within {reenum_timeout_ms}ms"
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


def format_usbgpu_build_device(diagnostic: dict[str, object]) -> str:
  return " ".join((
    f'[USBGPU] device: path={diagnostic["sysfsPath"] or "unavailable"} vendor={diagnostic["vendorId"] or "unavailable"}',
    f'product={diagnostic["productId"] or "unavailable"} configuration={diagnostic["configuration"]} devnum={diagnostic["devnum"]}',
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
