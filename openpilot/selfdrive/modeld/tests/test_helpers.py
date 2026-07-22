import tempfile
import time as real_time
from pathlib import Path

from openpilot.selfdrive.modeld import helpers
from openpilot.selfdrive.modeld.helpers import UsbGpuLinkMonitor
from openpilot.selfdrive.modeld.helpers import usbgpu_present, usbgpu_ready_identity, usbgpu_speed, usbgpu_superspeed_ready
from openpilot.selfdrive.modeld.helpers import wait_for_usbgpu_present, wait_for_usbgpu_ready


def add_usb_device(root: Path, name: str, speed: int, configuration: str = "1", devnum: int = 1) -> Path:
  device = root / name
  device.mkdir()
  (device / "idVendor").write_text("add1\n")
  (device / "idProduct").write_text("0001\n")
  (device / "speed").write_text(f"{speed}\n")
  (device / "bConfigurationValue").write_text(f"{configuration}\n")
  (device / "devnum").write_text(f"{devnum}\n")
  return device


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
    assert usbgpu_ready_identity(sysfs_root) == (str(sysfs_root / "4-1"), 1, 5000, 1)
    assert wait_for_usbgpu_ready(0, sysfs_root=sysfs_root)


def test_unconfigured_superspeed_is_not_ready():
  with tempfile.TemporaryDirectory() as tempdir:
    sysfs_root = Path(tempdir)
    add_usb_device(sysfs_root, "4-1", 5000, configuration="")
    assert usbgpu_present(sysfs_root)
    assert usbgpu_speed(sysfs_root) == 5000
    assert not usbgpu_superspeed_ready(sysfs_root)
    assert not wait_for_usbgpu_ready(0, sysfs_root=sysfs_root)


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


def test_wait_for_ready_requires_continuous_stability(monkeypatch):
  with tempfile.TemporaryDirectory() as tempdir:
    sysfs_root = Path(tempdir)
    device = add_usb_device(sysfs_root, "4-1", 5000, configuration="")
    elapsed = 0.0

    def sleep(interval: float):
      nonlocal elapsed
      elapsed += interval
      if elapsed == 1.0:
        (device / "bConfigurationValue").write_text("1\n")
      elif elapsed == 2.0:
        (device / "bConfigurationValue").write_text("\n")
      elif elapsed == 3.0:
        (device / "bConfigurationValue").write_text("1\n")

    monkeypatch.setattr(helpers.time, "sleep", sleep)
    monkeypatch.setattr(helpers.time, "monotonic", lambda: elapsed)
    assert wait_for_usbgpu_ready(10, stable_duration=3, poll_interval=1, sysfs_root=sysfs_root)
    assert elapsed == 6.0


def test_wait_for_ready_restarts_stability_when_devnum_changes(monkeypatch):
  with tempfile.TemporaryDirectory() as tempdir:
    sysfs_root = Path(tempdir)
    device = add_usb_device(sysfs_root, "4-1", 5000, devnum=2)
    elapsed = 0.0

    def sleep(interval: float):
      nonlocal elapsed
      elapsed += interval
      if elapsed == 2.0:
        (device / "devnum").write_text("3\n")

    monkeypatch.setattr(helpers.time, "sleep", sleep)
    monkeypatch.setattr(helpers.time, "monotonic", lambda: elapsed)
    assert wait_for_usbgpu_ready(10, stable_duration=3, poll_interval=1, sysfs_root=sysfs_root)
    assert elapsed == 5.0


def test_link_monitor_handles_counter_wrap_at_threshold(monkeypatch):
  identity = ("/sys/devices/platform/11200000.ssusb/usb4/4-1", 1, 5000, 1)
  counts = iter((65530, 65535, 4))
  elapsed = 10.0

  monkeypatch.setattr(helpers, "usbgpu_ready_identity", lambda sysfs_root=helpers.USBGPU_SYSFS_ROOT: identity)
  monkeypatch.setattr(helpers, "_usbgpu_portli", lambda current_identity: Path("/portli"))
  monkeypatch.setattr(helpers, "_usbgpu_link_error_count", lambda portli: next(counts))
  monkeypatch.setattr(helpers.time, "monotonic", lambda: elapsed)

  monitor = UsbGpuLinkMonitor(poll_interval=1.0)
  monitor.sample()
  elapsed = 11.0
  monitor.sample()
  elapsed = 12.0
  monitor.sample()

  ready, rate = monitor.status(stable_duration=2.0)
  assert ready
  assert rate == helpers.USBGPU_LINK_ERROR_THRESHOLD


def test_link_monitor_rejects_sparse_samples(monkeypatch):
  identity = ("/sys/devices/platform/11200000.ssusb/usb4/4-1", 1, 5000, 1)
  counts = iter((0, 40))
  elapsed = 0.0
  monkeypatch.setattr(helpers, "usbgpu_ready_identity", lambda sysfs_root=helpers.USBGPU_SYSFS_ROOT: identity)
  monkeypatch.setattr(helpers, "_usbgpu_portli", lambda current_identity: Path("/portli"))
  monkeypatch.setattr(helpers, "_usbgpu_link_error_count", lambda portli: next(counts))
  monkeypatch.setattr(helpers.time, "monotonic", lambda: elapsed)

  monitor = UsbGpuLinkMonitor()
  monitor.sample()
  elapsed = 10.0
  monitor.sample()

  assert monitor.status(stable_duration=2.0) == (False, None)


def test_link_monitor_sums_multiple_counter_wraps(monkeypatch):
  identity = ("/sys/devices/platform/11200000.ssusb/usb4/4-1", 1, 5000, 1)
  counts = iter((0, 65535, 2))
  elapsed = 0.0
  monkeypatch.setattr(helpers, "usbgpu_ready_identity", lambda sysfs_root=helpers.USBGPU_SYSFS_ROOT: identity)
  monkeypatch.setattr(helpers, "_usbgpu_portli", lambda current_identity: Path("/portli"))
  monkeypatch.setattr(helpers, "_usbgpu_link_error_count", lambda portli: next(counts))
  monkeypatch.setattr(helpers.time, "monotonic", lambda: elapsed)

  monitor = UsbGpuLinkMonitor(poll_interval=1.0)
  monitor.sample()
  elapsed = 1.0
  monitor.sample()
  elapsed = 2.0
  monitor.sample()

  ready, rate = monitor.status(stable_duration=2.0)
  assert not ready
  assert rate == 32769.0


def test_link_monitor_reads_controller_portli(monkeypatch, tmp_path):
  controller_root = tmp_path / "11200000.ssusb"
  usb_root = controller_root / "usb4"
  usb_root.mkdir(parents=True)
  device = add_usb_device(usb_root, "4-1", 5000)
  sysfs_root = tmp_path / "sysfs"
  sysfs_root.mkdir()
  (sysfs_root / "4-1").symlink_to(device)
  (controller_root / "portli").write_text("0x0012\n")
  elapsed = 3.0
  monkeypatch.setattr(helpers.time, "monotonic", lambda: elapsed)

  monitor = UsbGpuLinkMonitor(sysfs_root, poll_interval=1.0)
  monitor.sample()
  elapsed = 4.0
  (controller_root / "portli").write_text("0x0017\n")
  monitor.sample()
  elapsed = 5.0
  (controller_root / "portli").write_text("0x001c\n")
  monitor.sample()

  assert monitor.status(stable_duration=2.0) == (True, helpers.USBGPU_LINK_ERROR_THRESHOLD)


def test_link_monitor_rejects_reenumeration_during_sample(monkeypatch):
  original = ("/sys/devices/platform/11200000.ssusb/usb4/4-1", 1, 5000, 1)
  replacement = ("/sys/devices/platform/11200000.ssusb/usb4/4-1", 2, 5000, 1)
  identities = iter((original, replacement))
  monkeypatch.setattr(helpers, "usbgpu_ready_identity", lambda sysfs_root=helpers.USBGPU_SYSFS_ROOT: next(identities))
  monkeypatch.setattr(helpers, "_usbgpu_portli", lambda current_identity: Path("/portli"))
  monkeypatch.setattr(helpers, "_usbgpu_link_error_count", lambda portli: 0)
  monkeypatch.setattr(helpers.time, "monotonic", lambda: 10.0)

  monitor = UsbGpuLinkMonitor()
  monitor.sample()

  assert monitor.status(stable_duration=0.0) == (False, None)


def test_wait_for_ready_restarts_after_high_link_error_rate(monkeypatch):
  with tempfile.TemporaryDirectory() as tempdir:
    sysfs_root = Path(tempdir)
    add_usb_device(sysfs_root, "4-1", 5000)
    elapsed = 0.0

    def sleep(interval: float):
      nonlocal elapsed
      elapsed += interval

    def link_errors(portli):
      return 30 if elapsed >= 3.0 else int(elapsed * 10)

    monkeypatch.setattr(helpers.time, "sleep", sleep)
    monkeypatch.setattr(helpers.time, "monotonic", lambda: elapsed)
    monkeypatch.setattr(helpers, "_usbgpu_portli", lambda identity: Path("/portli"))
    monkeypatch.setattr(helpers, "_usbgpu_link_error_count", link_errors)

    assert wait_for_usbgpu_ready(10, stable_duration=3, poll_interval=1, sysfs_root=sysfs_root)
    assert elapsed == 4.0


def test_wait_for_ready_recovers_with_production_timeout_after_high_error_rate(monkeypatch):
  with tempfile.TemporaryDirectory() as tempdir:
    sysfs_root = Path(tempdir)
    add_usb_device(sysfs_root, "4-1", 5000)
    elapsed = 0.0

    def sleep(interval: float):
      nonlocal elapsed
      elapsed += interval

    monkeypatch.setattr(helpers.time, "sleep", sleep)
    monkeypatch.setattr(helpers.time, "monotonic", lambda: elapsed)
    monkeypatch.setattr(helpers, "_usbgpu_portli", lambda identity: Path("/portli"))
    monkeypatch.setattr(helpers, "_usbgpu_link_error_count", lambda portli: 80 if elapsed >= 8.0 else int(elapsed * 10))

    assert wait_for_usbgpu_ready(15, stable_duration=8, poll_interval=1, sysfs_root=sysfs_root)
    assert elapsed == 9.0


def test_wait_for_ready_fails_if_supported_link_counter_disappears(monkeypatch):
  with tempfile.TemporaryDirectory() as tempdir:
    sysfs_root = Path(tempdir)
    add_usb_device(sysfs_root, "4-1", 5000)
    elapsed = 0.0

    def sleep(interval: float):
      nonlocal elapsed
      elapsed += interval

    monkeypatch.setattr(helpers.time, "sleep", sleep)
    monkeypatch.setattr(helpers.time, "monotonic", lambda: elapsed)
    monkeypatch.setattr(helpers, "_usbgpu_portli", lambda identity: Path("/portli"))
    monkeypatch.setattr(helpers, "_usbgpu_link_error_count", lambda portli: None)

    assert not wait_for_usbgpu_ready(4, stable_duration=2, poll_interval=1, sysfs_root=sysfs_root)
    assert elapsed == 4.0


def test_link_monitor_reuses_bootstrap_stability_observed_during_load(monkeypatch):
  with tempfile.TemporaryDirectory() as tempdir:
    sysfs_root = Path(tempdir)
    device = add_usb_device(sysfs_root, "4-1", 12)
    elapsed = 0.0
    monkeypatch.setattr(helpers.time, "monotonic", lambda: elapsed)

    monitor = UsbGpuLinkMonitor(sysfs_root)
    monitor.sample()
    (device / "speed").write_text("5000\n")
    for current_time in range(5, 14):
      elapsed = current_time
      monitor.sample()

    assert monitor.status(stable_duration=8.0) == (True, None)


def test_background_monitor_observes_bootstrap_reenumeration(tmp_path):
  device = add_usb_device(tmp_path, "3-1", 12)
  monitor = UsbGpuLinkMonitor(tmp_path, poll_interval=0.005)
  monitor.start()
  try:
    real_time.sleep(0.02)
    for child in device.iterdir():
      child.unlink()
    device.rmdir()
    real_time.sleep(0.02)
    add_usb_device(tmp_path, "4-1", 5000, devnum=2)

    deadline = real_time.monotonic() + 1.0
    ready = False
    while real_time.monotonic() < deadline:
      ready, _ = monitor.status(stable_duration=0.03)
      if ready:
        break
      real_time.sleep(0.01)
  finally:
    monitor.stop()

  assert ready


def test_ignition_lockout_is_atomically_replaced_and_cleared(tmp_path):
  marker = tmp_path / "usbgpu-lockout"

  helpers.set_usbgpu_ignition_lockout("first failure", marker)
  assert helpers.usbgpu_ignition_lockout_reason(marker) == "first failure"
  helpers.set_usbgpu_ignition_lockout("second failure", marker)
  assert helpers.usbgpu_ignition_lockout_reason(marker) == "second failure"
  assert list(tmp_path.iterdir()) == [marker]

  helpers.clear_usbgpu_ignition_lockout(marker)
  assert helpers.usbgpu_ignition_lockout_reason(marker) is None
