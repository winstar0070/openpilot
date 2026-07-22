from pathlib import Path

from openpilot.selfdrive.modeld import helpers
from openpilot.selfdrive.modeld.helpers import usbgpu_present, usbgpu_ready_identity, usbgpu_speed
from openpilot.selfdrive.modeld.helpers import usbgpu_superspeed_ready, wait_for_usbgpu_present, wait_for_usbgpu_ready


def add_usb_device(root: Path, name: str, speed: int, configuration: str = "1", devnum: int = 1) -> Path:
  device = root / name
  device.mkdir()
  (device / "idVendor").write_text("add1\n")
  (device / "idProduct").write_text("0001\n")
  (device / "speed").write_text(f"{speed}\n")
  (device / "bConfigurationValue").write_text(f"{configuration}\n")
  (device / "devnum").write_text(f"{devnum}\n")
  return device


def test_bootstrap_is_present_but_not_ready(tmp_path):
  add_usb_device(tmp_path, "3-1", 12)
  assert usbgpu_present(tmp_path)
  assert usbgpu_speed(tmp_path) == 12
  assert usbgpu_ready_identity(tmp_path) is None
  assert not usbgpu_superspeed_ready(tmp_path)


def test_configured_superspeed_reports_raw_identity(tmp_path):
  device = add_usb_device(tmp_path, "4-1", 5000, devnum=7)
  assert usbgpu_ready_identity(tmp_path) == (str(device), 7, 5000, 1)
  assert usbgpu_superspeed_ready(tmp_path)
  assert wait_for_usbgpu_ready(0, sysfs_root=tmp_path)


def test_unconfigured_superspeed_is_not_ready(tmp_path):
  add_usb_device(tmp_path, "4-1", 5000, configuration="")
  assert usbgpu_present(tmp_path)
  assert usbgpu_speed(tmp_path) == 5000
  assert not usbgpu_superspeed_ready(tmp_path)


def test_waits_for_delayed_raw_presence(monkeypatch, tmp_path):
  elapsed = 0.0

  def sleep(interval: float):
    nonlocal elapsed
    elapsed += interval
    if elapsed == 2.0:
      add_usb_device(tmp_path, "3-1", 12)

  monkeypatch.setattr(helpers.time, "sleep", sleep)
  monkeypatch.setattr(helpers.time, "monotonic", lambda: elapsed)
  assert wait_for_usbgpu_present(2, poll_interval=1, sysfs_root=tmp_path)
  assert elapsed == 2.0


def test_wait_for_presence_times_out(monkeypatch, tmp_path):
  elapsed = 0.0

  def sleep(interval: float):
    nonlocal elapsed
    elapsed += interval

  monkeypatch.setattr(helpers.time, "sleep", sleep)
  monkeypatch.setattr(helpers.time, "monotonic", lambda: elapsed)
  assert not wait_for_usbgpu_present(2, poll_interval=1, sysfs_root=tmp_path)
  assert elapsed == 2.0


def test_ready_stability_resets_when_devnum_changes(monkeypatch, tmp_path):
  device = add_usb_device(tmp_path, "4-1", 5000, devnum=1)
  elapsed = 0.0

  def sleep(interval: float):
    nonlocal elapsed
    elapsed += interval
    if elapsed == 1.0:
      (device / "devnum").write_text("2\n")

  monkeypatch.setattr(helpers.time, "sleep", sleep)
  monkeypatch.setattr(helpers.time, "monotonic", lambda: elapsed)
  assert wait_for_usbgpu_ready(5, stable_duration=2, poll_interval=1, sysfs_root=tmp_path)
  assert elapsed == 3.0


def test_ready_stability_resets_when_configuration_drops(monkeypatch, tmp_path):
  device = add_usb_device(tmp_path, "4-1", 5000)
  elapsed = 0.0

  def sleep(interval: float):
    nonlocal elapsed
    elapsed += interval
    if elapsed == 1.0:
      (device / "bConfigurationValue").write_text("\n")
    elif elapsed == 2.0:
      (device / "bConfigurationValue").write_text("1\n")

  monkeypatch.setattr(helpers.time, "sleep", sleep)
  monkeypatch.setattr(helpers.time, "monotonic", lambda: elapsed)
  assert wait_for_usbgpu_ready(5, stable_duration=2, poll_interval=1, sysfs_root=tmp_path)
  assert elapsed == 4.0
