import subprocess
import tempfile
from pathlib import Path


LAUNCH_SCRIPT = Path(__file__).resolve().parents[4] / "launch_chffrplus.sh"


def make_fake_xhci(root: Path) -> tuple[Path, Path, Path]:
  usb_root, platform_root, driver_root = root / "usb", root / "platform", root / "driver"
  xhci_root = platform_root / "xhci-hcd.1.auto"
  usb_root.mkdir()
  xhci_root.mkdir(parents=True)
  driver_root.mkdir()
  (xhci_root / "driver").symlink_to(driver_root, target_is_directory=True)
  return usb_root, platform_root, driver_root


def test_prepare_usbgpu_waits_for_delayed_stable_superspeed_device():
  with tempfile.TemporaryDirectory() as tempdir:
    root = Path(tempdir)
    usb_root, platform_root, driver_root = make_fake_xhci(root)
    (driver_root / "bind").write_text("")
    (driver_root / "unbind").write_text("")
    speeds = root / "speeds"
    speeds.write_text("0\n0\n5000\n5000\n5000\n")

    script = f'''
      export USBGPU_USB_SYSFS_ROOT="{usb_root}"
      export USBGPU_PLATFORM_SYSFS_ROOT="{platform_root}"
      export USBGPU_ATTACH_TIMEOUT_SEC=5
      export USBGPU_INITIAL_STABILITY_TIMEOUT_SEC=3
      export USBGPU_STABLE_SECONDS=2
      source "{LAUNCH_SCRIPT}"
      usbgpu_speed() {{
        local speed
        speed="$(head -n 1 "{speeds}")"
        tail -n +2 "{speeds}" > "{speeds}.tmp"
        [ -s "{speeds}.tmp" ] && mv "{speeds}.tmp" "{speeds}"
        echo "${{speed:-0}}"
      }}
      sleep() {{ :; }}
      prepare_usbgpu
    '''
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True)

    assert "USB GPU ready at 5000M" in result.stdout
    assert (driver_root / "unbind").read_text() == ""


def test_prepare_usbgpu_reports_missing_device_after_grace_period():
  with tempfile.TemporaryDirectory() as tempdir:
    root = Path(tempdir)
    usb_root, platform_root, _ = make_fake_xhci(root)

    script = f'''
      export USBGPU_USB_SYSFS_ROOT="{usb_root}"
      export USBGPU_PLATFORM_SYSFS_ROOT="{platform_root}"
      export USBGPU_ATTACH_TIMEOUT_SEC=2
      source "{LAUNCH_SCRIPT}"
      sleep() {{ :; }}
      prepare_usbgpu
    '''
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True)

    assert "USB GPU not detected within 2s" in result.stdout


def test_prepare_usbgpu_resets_an_unstable_superspeed_link():
  with tempfile.TemporaryDirectory() as tempdir:
    root = Path(tempdir)
    usb_root, platform_root, driver_root = make_fake_xhci(root)
    (driver_root / "bind").write_text("")
    (driver_root / "unbind").write_text("")
    speeds = root / "speeds"
    speeds.write_text("5000\n5000\n0\n5000\n5000\n5000\n")

    script = f'''
      export USBGPU_USB_SYSFS_ROOT="{usb_root}"
      export USBGPU_PLATFORM_SYSFS_ROOT="{platform_root}"
      export USBGPU_ATTACH_TIMEOUT_SEC=1
      export USBGPU_INITIAL_STABILITY_TIMEOUT_SEC=2
      export USBGPU_RECOVERY_TIMEOUT_SEC=3
      export USBGPU_STABLE_SECONDS=2
      export USBGPU_RECOVERY_ATTEMPTS=1
      export USBGPU_REBIND_DELAY_SEC=0
      source "{LAUNCH_SCRIPT}"
      usbgpu_speed() {{
        local speed
        speed="$(head -n 1 "{speeds}")"
        tail -n +2 "{speeds}" > "{speeds}.tmp"
        [ -s "{speeds}.tmp" ] && mv "{speeds}.tmp" "{speeds}"
        echo "${{speed:-0}}"
      }}
      sleep() {{ :; }}
      sudo() {{ "$@"; }}
      prepare_usbgpu
    '''
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True)

    assert "USB GPU SuperSpeed link was not stable; resetting controller" in result.stdout
    assert "USB GPU ready at 5000M" in result.stdout
    assert (driver_root / "unbind").read_text() == "xhci-hcd.1.auto\n"


def test_prepare_usbgpu_recovers_a_low_speed_device():
  with tempfile.TemporaryDirectory() as tempdir:
    root = Path(tempdir)
    usb_root, platform_root, driver_root = make_fake_xhci(root)
    (driver_root / "bind").write_text("")
    (driver_root / "unbind").write_text("")
    speeds = root / "speeds"
    speeds.write_text("480\n5000\n5000\n5000\n")

    script = f'''
      export USBGPU_USB_SYSFS_ROOT="{usb_root}"
      export USBGPU_PLATFORM_SYSFS_ROOT="{platform_root}"
      export USBGPU_ATTACH_TIMEOUT_SEC=1
      export USBGPU_RECOVERY_TIMEOUT_SEC=3
      export USBGPU_STABLE_SECONDS=2
      export USBGPU_RECOVERY_ATTEMPTS=1
      export USBGPU_REBIND_DELAY_SEC=0
      source "{LAUNCH_SCRIPT}"
      usbgpu_speed() {{
        local speed
        speed="$(head -n 1 "{speeds}")"
        tail -n +2 "{speeds}" > "{speeds}.tmp"
        [ -s "{speeds}.tmp" ] && mv "{speeds}.tmp" "{speeds}"
        echo "${{speed:-0}}"
      }}
      sleep() {{ :; }}
      sudo() {{ "$@"; }}
      prepare_usbgpu
    '''
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True)

    assert "USB GPU detected at 480M" in result.stdout
    assert "USB GPU ready at 5000M" in result.stdout
    assert (driver_root / "unbind").read_text() == "xhci-hcd.1.auto\n"
