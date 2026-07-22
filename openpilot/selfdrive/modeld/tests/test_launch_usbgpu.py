import subprocess
import tempfile
from pathlib import Path
import pytest


LAUNCH_SCRIPT = Path(__file__).resolve().parents[4] / "launch_chffrplus.sh"


def make_fake_xhci(root: Path) -> tuple[Path, Path, Path]:
  usb_root, platform_root, driver_root = root / "usb", root / "platform", root / "driver"
  xhci_root = platform_root / "xhci-hcd.1.auto"
  usb_root.mkdir()
  xhci_root.mkdir(parents=True)
  driver_root.mkdir()
  (xhci_root / "driver").symlink_to(driver_root, target_is_directory=True)
  return usb_root, platform_root, driver_root


@pytest.mark.parametrize(("devnum", "valid"), (("", False), ("abc", False), ("0", False), ("7", True)))
def test_usbgpu_identity_requires_positive_numeric_devnum(devnum, valid):
  with tempfile.TemporaryDirectory() as tempdir:
    root = Path(tempdir)
    usb_root, platform_root, _ = make_fake_xhci(root)
    device = usb_root / "4-1"
    device.mkdir()
    (device / "idVendor").write_text("add1\n")
    (device / "idProduct").write_text("0001\n")
    (device / "speed").write_text("5000\n")
    (device / "bConfigurationValue").write_text("1\n")
    (device / "devnum").write_text(f"{devnum}\n")

    script = f'''
      export USBGPU_USB_SYSFS_ROOT="{usb_root}"
      export USBGPU_PLATFORM_SYSFS_ROOT="{platform_root}"
      source "{LAUNCH_SCRIPT}"
      usbgpu_identity 5000
    '''
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=False)

    assert (result.returncode == 0) is valid
    if valid:
      assert result.stdout.strip() == f"{device}|7|5000|1"


def test_prepare_usbgpu_waits_for_delayed_stable_superspeed_device():
  with tempfile.TemporaryDirectory() as tempdir:
    root = Path(tempdir)
    usb_root, platform_root, driver_root = make_fake_xhci(root)
    (driver_root / "bind").write_text("")
    (driver_root / "unbind").write_text("")
    speeds = root / "speeds"
    speeds.write_text("0\n0\n5000\n5000\n5000\n5000\n")

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
      usbgpu_identity() {{
        [ "$1" -ge 5000 ] 2>/dev/null && echo "mock-device|1|$1|1"
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
      usbgpu_identity() {{
        [ "$1" -ge 5000 ] 2>/dev/null && echo "mock-device|1|$1|1"
      }}
      sleep() {{ :; }}
      sudo() {{ "$@"; }}
      prepare_usbgpu
    '''
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True)

    assert "USB GPU SuperSpeed link was not stable; resetting controller" in result.stdout
    assert "USB GPU ready at 5000M" in result.stdout
    assert (driver_root / "unbind").read_text() == "xhci-hcd.1.auto\n"


def test_prepare_usbgpu_hands_low_speed_bootstrap_device_to_tinygrad():
  with tempfile.TemporaryDirectory() as tempdir:
    root = Path(tempdir)
    usb_root, platform_root, driver_root = make_fake_xhci(root)
    (driver_root / "bind").write_text("")
    (driver_root / "unbind").write_text("")
    speeds = root / "speeds"
    speeds.write_text("12\n")

    script = f'''
      export USBGPU_USB_SYSFS_ROOT="{usb_root}"
      export USBGPU_PLATFORM_SYSFS_ROOT="{platform_root}"
      export USBGPU_ATTACH_TIMEOUT_SEC=1
      source "{LAUNCH_SCRIPT}"
      usbgpu_speed() {{
        local speed
        speed="$(head -n 1 "{speeds}")"
        tail -n +2 "{speeds}" > "{speeds}.tmp"
        [ -s "{speeds}.tmp" ] && mv "{speeds}.tmp" "{speeds}"
        echo "${{speed:-0}}"
      }}
      usbgpu_identity() {{
        [ "$1" -ge 5000 ] 2>/dev/null && echo "mock-device|1|$1|1"
      }}
      sleep() {{ :; }}
      sudo() {{ "$@"; }}
      prepare_usbgpu
    '''
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True)

    assert "USB GPU detected at 12M" in result.stdout
    assert "USB GPU bootstrap device detected; handing off to tinygrad" in result.stdout
    assert (driver_root / "unbind").read_text() == ""
    assert (driver_root / "bind").read_text() == ""


def test_prepare_usbgpu_recovers_usb2_device_instead_of_handing_off():
  with tempfile.TemporaryDirectory() as tempdir:
    root = Path(tempdir)
    usb_root, platform_root, driver_root = make_fake_xhci(root)
    (driver_root / "bind").write_text("")
    (driver_root / "unbind").write_text("")
    speeds = root / "speeds"
    speeds.write_text("480\n5000\n5000\n")

    script = f'''
      export USBGPU_USB_SYSFS_ROOT="{usb_root}"
      export USBGPU_PLATFORM_SYSFS_ROOT="{platform_root}"
      export USBGPU_ATTACH_TIMEOUT_SEC=1
      export USBGPU_RECOVERY_TIMEOUT_SEC=2
      export USBGPU_STABLE_SECONDS=1
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
      usbgpu_identity() {{
        [ "$1" -ge 5000 ] 2>/dev/null && echo "mock-device|1|$1|1"
      }}
      sleep() {{ :; }}
      sudo() {{ "$@"; }}
      prepare_usbgpu
    '''
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True)

    assert "USB GPU detected at 480M" in result.stdout
    assert "handing off to tinygrad" not in result.stdout
    assert "Resetting xhci-hcd.1.auto" in result.stdout
    assert "USB GPU ready at 5000M" in result.stdout
    assert (driver_root / "unbind").read_text() == "xhci-hcd.1.auto\n"


def test_prepare_usbgpu_hands_bootstrap_to_tinygrad_after_one_recovery():
  with tempfile.TemporaryDirectory() as tempdir:
    root = Path(tempdir)
    usb_root, platform_root, driver_root = make_fake_xhci(root)
    (driver_root / "bind").write_text("")
    (driver_root / "unbind").write_text("")
    speeds = root / "speeds"
    speeds.write_text("480\n12\n")

    script = f'''
      export USBGPU_USB_SYSFS_ROOT="{usb_root}"
      export USBGPU_PLATFORM_SYSFS_ROOT="{platform_root}"
      export USBGPU_ATTACH_TIMEOUT_SEC=1
      export USBGPU_RECOVERY_ATTEMPTS=3
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

    assert "USB GPU bootstrap device detected after controller recovery; handing off to tinygrad" in result.stdout
    assert result.stdout.count("Resetting xhci-hcd.1.auto") == 1
    assert (driver_root / "unbind").read_text() == "xhci-hcd.1.auto\n"


def test_prepare_usbgpu_resets_when_superspeed_identity_changes():
  with tempfile.TemporaryDirectory() as tempdir:
    root = Path(tempdir)
    usb_root, platform_root, driver_root = make_fake_xhci(root)
    (driver_root / "bind").write_text("")
    (driver_root / "unbind").write_text("")
    speeds = root / "speeds"
    speeds.write_text("5000\n5000\n5000\n5000\n5000\n5000\n")
    identities = root / "identities"
    identities.write_text(
      "device-a|1|5000|1\ndevice-b|2|5000|1\ndevice-b|2|5000|1\ndevice-b|2|5000|1\ndevice-b|2|5000|1\n",
    )

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
      usbgpu_identity() {{
        local identity
        identity="$(head -n 1 "{identities}")"
        tail -n +2 "{identities}" > "{identities}.tmp"
        [ -s "{identities}.tmp" ] && mv "{identities}.tmp" "{identities}"
        echo "$identity"
      }}
      sleep() {{ :; }}
      sudo() {{ "$@"; }}
      prepare_usbgpu
    '''
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True)

    assert "USB GPU SuperSpeed link was not stable; resetting controller" in result.stdout
    assert "USB GPU ready at 5000M" in result.stdout
    assert (driver_root / "unbind").read_text() == "xhci-hcd.1.auto\n"
