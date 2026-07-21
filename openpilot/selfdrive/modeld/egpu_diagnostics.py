import json
import os
import subprocess
import traceback
from datetime import datetime
from pathlib import Path

from openpilot.common.time_helpers import system_time_valid


EGPU_LOG_ROOT = Path("/data/community/egpu_logs")
USB_SYSFS_ROOT = Path("/sys/bus/usb/devices")
XHCI_SYSFS_PATH = Path("/sys/bus/platform/devices/xhci-hcd.1.auto")
SYSTEM_SYSFS_ROOT = Path("/sys")
EGPU_VENDOR_ID = "add1"
EGPU_PRODUCT_ID = "0001"
MAX_LOG_FILES = 20
MAX_KERNEL_LINES = 200

USB_FIELDS = (
  "authorized", "busnum", "devnum", "speed", "version", "bConfigurationValue",
  "manufacturer", "product", "serial", "power/control", "power/runtime_status",
  "power/runtime_active_time", "power/runtime_suspended_time", "power/autosuspend_delay_ms", "power/wakeup",
)
XHCI_FIELDS = (
  "power/control", "power/runtime_status", "power/runtime_active_time",
  "power/runtime_suspended_time", "power/autosuspend_delay_ms", "power/wakeup",
)
KERNEL_MARKERS = ("usb", "xhci", "asm24", "amdgpu", "pcie", "pci ")
POWER_FIELDS = {
  "batteryVoltageUv": "class/power_supply/bms/voltage_now",
  "batteryCurrentUa": "class/power_supply/bms/current_now",
  "batteryCapacityPercent": "class/power_supply/bms/capacity",
  "batteryStatus": "class/power_supply/bms/status",
  "systemPowerUw": "class/hwmon/hwmon1/power1_input",
}


def _read(path: Path) -> str | None:
  try:
    return path.read_text().strip()
  except OSError:
    return None


def _device_state(device: Path) -> dict:
  state = {"sysfsPath": str(device)}
  for field in USB_FIELDS:
    state[field.replace("/", "_")] = _read(device / field)
  try:
    state["resolvedPath"] = str(device.resolve())
  except OSError:
    state["resolvedPath"] = None
  return state


def _egpu_devices(sysfs_root: Path) -> list[dict]:
  devices = []
  try:
    candidates = sorted(sysfs_root.glob("*"))
  except OSError:
    return devices

  for device in candidates:
    vendor = (_read(device / "idVendor") or "").lower()
    product = (_read(device / "idProduct") or "").lower()
    if vendor == EGPU_VENDOR_ID and product == EGPU_PRODUCT_ID:
      devices.append(_device_state(device))
  return devices


def _xhci_state(xhci_path: Path) -> dict:
  state = {"sysfsPath": str(xhci_path), "present": xhci_path.exists()}
  try:
    state["driver"] = str((xhci_path / "driver").resolve()) if (xhci_path / "driver").exists() else None
  except OSError:
    state["driver"] = None
  for field in XHCI_FIELDS:
    state[field.replace("/", "_")] = _read(xhci_path / field)
  return state


def _system_power_state(sysfs_root: Path) -> dict:
  return {name: _read(sysfs_root / relative_path) for name, relative_path in POWER_FIELDS.items()}


def _kernel_log() -> dict:
  try:
    result = subprocess.run(["dmesg"], capture_output=True, text=True, timeout=2, check=False)
    lines = [line for line in result.stdout.splitlines() if any(marker in line.lower() for marker in KERNEL_MARKERS)]
    return {
      "returnCode": result.returncode,
      "stderr": result.stderr.strip() or None,
      "lines": lines[-MAX_KERNEL_LINES:],
    }
  except Exception as e:
    return {"error": f"{type(e).__name__}: {e}", "lines": []}


def collect_egpu_diagnostics(error: BaseException, log_root: Path = EGPU_LOG_ROOT,
                             usb_sysfs_root: Path = USB_SYSFS_ROOT,
                             xhci_sysfs_path: Path = XHCI_SYSFS_PATH,
                             system_sysfs_root: Path = SYSTEM_SYSFS_ROOT) -> tuple[Path | None, dict]:
  now = datetime.now().astimezone()
  diagnostics = {
    "timestamp": now.isoformat(),
    "systemTimeValid": system_time_valid(),
    "pid": os.getpid(),
    "error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()},
    "egpuDevices": _egpu_devices(usb_sysfs_root),
    "xhci": _xhci_state(xhci_sysfs_path),
    "systemPower": _system_power_state(system_sysfs_root),
    "kernelLog": _kernel_log(),
  }

  try:
    log_root.mkdir(parents=True, exist_ok=True)
    path = log_root / now.strftime("%Y-%m-%d--%H-%M-%S-%f.json")
    content = json.dumps(diagnostics, indent=2, sort_keys=True) + "\n"
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(content)
    tmp_path.replace(path)

    latest_tmp = log_root / "latest.tmp"
    latest_tmp.write_text(content)
    latest_tmp.replace(log_root / "latest.json")

    historical_logs = sorted(p for p in log_root.glob("*.json") if p.name != "latest.json")
    for old_path in historical_logs[:-MAX_LOG_FILES]:
      old_path.unlink(missing_ok=True)
    return path, diagnostics
  except OSError as e:
    diagnostics["saveError"] = f"{type(e).__name__}: {e}"
    return None, diagnostics
