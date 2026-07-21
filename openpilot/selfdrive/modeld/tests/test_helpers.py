import tempfile
from pathlib import Path

from openpilot.selfdrive.modeld import helpers
from openpilot.selfdrive.modeld.helpers import usbgpu_present, usbgpu_speed, usbgpu_superspeed_ready, wait_for_usbgpu_present, wait_for_usbgpu_ready


def add_usb_device(root: Path, name: str, speed: int) -> None:
  device = root / name
  device.mkdir()
  (device / "idVendor").write_text("add1\n")
  (device / "idProduct").write_text("0001\n")
  (device / "speed").write_text(f"{speed}\n")


def test_bootstrap_is_present_but_not_ready():
  with tempfile.TemporaryDirectory() as tempdir:
    sysfs_root = Path(tempdir)
    add_usb_device(sysfs_root, "3-1", 12)
    assert usbgpu_present(sysfs_root)
    assert usbgpu_speed(sysfs_root) == 12
    assert not usbgpu_superspeed_ready(sysfs_root)


def test_superspeed_is_ready():
  with tempfile.TemporaryDirectory() as tempdir:
    sysfs_root = Path(tempdir)
    add_usb_device(sysfs_root, "4-1", 5000)
    assert usbgpu_present(sysfs_root)
    assert usbgpu_speed(sysfs_root) == 5000
    assert usbgpu_superspeed_ready(sysfs_root)
    assert wait_for_usbgpu_ready(0, sysfs_root=sysfs_root)


def test_absent():
  with tempfile.TemporaryDirectory() as tempdir:
    sysfs_root = Path(tempdir)
    assert not usbgpu_present(sysfs_root)
    assert usbgpu_speed(sysfs_root) is None
    assert not wait_for_usbgpu_ready(0, sysfs_root=sysfs_root)


def test_waits_for_delayed_presence(monkeypatch):
  with tempfile.TemporaryDirectory() as tempdir:
    sysfs_root = Path(tempdir)
    elapsed = 0.0

    def sleep(interval: float):
      nonlocal elapsed
      elapsed += interval
      if elapsed >= 3.0 and not usbgpu_present(sysfs_root):
        add_usb_device(sysfs_root, "4-1", 5000)

    monkeypatch.setattr(helpers.time, "sleep", sleep)
    monkeypatch.setattr(helpers.time, "monotonic", lambda: elapsed)
    assert wait_for_usbgpu_present(30, poll_interval=1, sysfs_root=sysfs_root)
    assert elapsed == 3.0


def test_wait_for_presence_times_out(monkeypatch):
  with tempfile.TemporaryDirectory() as tempdir:
    sysfs_root = Path(tempdir)
    elapsed = 0.0

    def sleep(interval: float):
      nonlocal elapsed
      elapsed += interval

    monkeypatch.setattr(helpers.time, "sleep", sleep)
    monkeypatch.setattr(helpers.time, "monotonic", lambda: elapsed)
    assert not wait_for_usbgpu_present(3, poll_interval=1, sysfs_root=sysfs_root)
    assert elapsed == 3.0
