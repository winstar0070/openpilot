import json
import subprocess
import tempfile
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


def test_collects_egpu_state_and_filtered_kernel_log(monkeypatch):
  with tempfile.TemporaryDirectory() as tempdir:
    root = Path(tempdir)
    usb_root, xhci_path, log_root = root / "usb", root / "xhci", root / "logs"
    usb_root.mkdir()
    (xhci_path / "power").mkdir(parents=True)
    add_egpu(usb_root)
    (xhci_path / "power" / "runtime_status").write_text("active\n")
    dmesg = subprocess.CompletedProcess([], 0, "noise\nxhci controller reset\nusb 4-1 disconnect\n", "")

    monkeypatch.setattr(egpu_diagnostics.subprocess, "run", lambda *args, **kwargs: dmesg)
    path, diagnostics = collect_egpu_diagnostics(RuntimeError("read failed"), log_root, usb_root, xhci_path)

    assert path is not None and path.exists()
    assert diagnostics["egpuDevices"][0]["speed"] == "5000"
    assert diagnostics["egpuDevices"][0]["power_runtime_status"] == "active"
    assert diagnostics["kernelLog"]["lines"] == ["xhci controller reset", "usb 4-1 disconnect"]
    assert json.loads((log_root / "latest.json").read_text())["error"]["message"] == "read failed"


def test_kernel_log_failure_does_not_block_diagnostic_file(monkeypatch):
  with tempfile.TemporaryDirectory() as tempdir:
    root = Path(tempdir)
    usb_root, xhci_path, log_root = root / "usb", root / "xhci", root / "logs"
    usb_root.mkdir()
    def raise_timeout(*args, **kwargs):
      raise subprocess.TimeoutExpired("dmesg", 2)

    monkeypatch.setattr(egpu_diagnostics.subprocess, "run", raise_timeout)
    path, diagnostics = collect_egpu_diagnostics(RuntimeError("timeout"), log_root, usb_root, xhci_path)

    assert path is not None and path.exists()
    assert diagnostics["kernelLog"]["lines"] == []
    assert "TimeoutExpired" in diagnostics["kernelLog"]["error"]


def test_retains_only_recent_historical_logs(monkeypatch):
  with tempfile.TemporaryDirectory() as tempdir:
    root = Path(tempdir)
    usb_root, xhci_path, log_root = root / "usb", root / "xhci", root / "logs"
    usb_root.mkdir()
    log_root.mkdir()
    for index in range(MAX_LOG_FILES + 3):
      (log_root / f"2000-01-01--00-00-{index:02d}-000000.json").write_text("{}\n")

    dmesg = subprocess.CompletedProcess([], 0, "", "")
    monkeypatch.setattr(egpu_diagnostics.subprocess, "run", lambda *args, **kwargs: dmesg)
    path, _ = collect_egpu_diagnostics(RuntimeError("failure"), log_root, usb_root, xhci_path)

    assert path is not None
    historical_logs = [p for p in log_root.glob("*.json") if p.name != "latest.json"]
    assert len(historical_logs) == MAX_LOG_FILES
