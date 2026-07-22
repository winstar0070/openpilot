import json
import subprocess
from pathlib import Path

from openpilot.selfdrive.modeld import egpu_diagnostics
from openpilot.selfdrive.modeld.egpu_diagnostics import MAX_LOG_FILES, collect_egpu_diagnostics


def add_egpu(root: Path, speed: int = 5000) -> Path:
  device = root / "4-1"
  (device / "power").mkdir(parents=True)
  (device / "idVendor").write_text("add1\n")
  (device / "idProduct").write_text("0001\n")
  (device / "speed").write_text(f"{speed}\n")
  (device / "authorized").write_text("1\n")
  (device / "power" / "runtime_status").write_text("active\n")
  return device


def test_collects_usb_state_and_filtered_kernel_log(monkeypatch, tmp_path):
  usb_root, xhci_path, log_root = tmp_path / "usb", tmp_path / "xhci", tmp_path / "logs"
  usb_root.mkdir()
  (xhci_path / "power").mkdir(parents=True)
  add_egpu(usb_root)
  (xhci_path / "power" / "runtime_status").write_text("active\n")
  dmesg = subprocess.CompletedProcess([], 0, "noise\nxhci controller reset\nusb 4-1 disconnect\n", "")
  monkeypatch.setattr(egpu_diagnostics.subprocess, "run", lambda *args, **kwargs: dmesg)

  path, diagnostics = collect_egpu_diagnostics(RuntimeError("read failed"), log_root, usb_root, xhci_path)

  assert path is not None and path.exists()
  assert diagnostics["egpuDevices"][0]["speed"] == "5000"
  assert diagnostics["kernelLog"]["lines"] == ["xhci controller reset", "usb 4-1 disconnect"]
  saved_error = json.loads((log_root / "latest.json").read_text())["error"]
  assert saved_error["message"] == "read failed"
  assert "RuntimeError: read failed" in saved_error["traceback"]


def test_kernel_log_failure_does_not_block_save(monkeypatch, tmp_path):
  usb_root, xhci_path, log_root = tmp_path / "usb", tmp_path / "xhci", tmp_path / "logs"
  usb_root.mkdir()

  def raise_timeout(*args, **kwargs):
    raise subprocess.TimeoutExpired("dmesg", 2)

  monkeypatch.setattr(egpu_diagnostics.subprocess, "run", raise_timeout)
  path, diagnostics = collect_egpu_diagnostics(RuntimeError("timeout"), log_root, usb_root, xhci_path)

  assert path is not None and path.exists()
  assert diagnostics["kernelLog"]["lines"] == []
  assert "TimeoutExpired" in diagnostics["kernelLog"]["error"]


def test_retains_twenty_historical_logs(monkeypatch, tmp_path):
  usb_root, xhci_path, log_root = tmp_path / "usb", tmp_path / "xhci", tmp_path / "logs"
  usb_root.mkdir()
  log_root.mkdir()
  for index in range(MAX_LOG_FILES + 3):
    (log_root / f"2000-01-01--00-00-{index:02d}-000000.json").write_text("{}\n")
  monkeypatch.setattr(
    egpu_diagnostics.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess([], 0, "", ""),
  )

  path, _ = collect_egpu_diagnostics(RuntimeError("failure"), log_root, usb_root, xhci_path)

  assert path is not None
  historical_logs = [p for p in log_root.glob("*.json") if p.name != "latest.json"]
  assert len(historical_logs) == MAX_LOG_FILES


def test_preserves_traceback_captured_before_async_collection(monkeypatch, tmp_path):
  usb_root, xhci_path, log_root = tmp_path / "usb", tmp_path / "xhci", tmp_path / "logs"
  usb_root.mkdir()
  monkeypatch.setattr(
    egpu_diagnostics.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess([], 0, "", ""),
  )

  _, diagnostics = collect_egpu_diagnostics(
    RuntimeError("synthetic"), log_root, usb_root, xhci_path,
    traceback_text="Traceback (most recent call last):\nRuntimeError: original failure\n",
  )

  assert diagnostics["error"]["traceback"].endswith("RuntimeError: original failure\n")
