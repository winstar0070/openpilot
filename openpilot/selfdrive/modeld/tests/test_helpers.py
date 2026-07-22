import json
from pathlib import Path

from openpilot.selfdrive.modeld import helpers
from openpilot.selfdrive.modeld.helpers import format_usbgpu_build_decision, format_usbgpu_build_device, get_usbgpu_build_diagnostic
from openpilot.selfdrive.modeld.helpers import resolve_usbgpu_build_diagnostic
from openpilot.selfdrive.modeld.helpers import usbgpu_present, usbgpu_ready_identity, usbgpu_speed
from openpilot.selfdrive.modeld.helpers import usbgpu_superspeed_ready, wait_for_usbgpu_present, wait_for_usbgpu_ready
from openpilot.selfdrive.modeld.helpers import usbgpu_build_speed_eligible, usbgpu_speed_eligible, write_usbgpu_build_diagnostic


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


def test_speed_eligibility_accepts_only_bootstrap_and_superspeed():
  assert usbgpu_speed_eligible(12)
  assert usbgpu_speed_eligible(5000)
  for speed in (None, 0, 480, 4999, 5001):
    assert not usbgpu_speed_eligible(speed)


def test_build_speed_eligibility_accepts_only_superspeed():
  assert usbgpu_build_speed_eligible(5000, 1)
  assert not usbgpu_build_speed_eligible(5000, 0)
  assert not usbgpu_build_speed_eligible(5000, None)
  for speed in (None, 12, 480, 4999, 5001):
    assert not usbgpu_build_speed_eligible(speed, 1)


def test_tinygrad_compiler_probe_never_enumerates_amd():
  script = helpers.tinygrad_compiler_probe_script()
  assert helpers.TINYGRAD_COMPILER_PROBE_BACKENDS == ("CUDA", "QCOM")
  assert "AMD" not in script
  assert "get_available_devices" not in script


def test_build_diagnostic_records_enabled_target_and_artifact(tmp_path):
  artifact = tmp_path / "big_driving_tinygrad.pkl.chunkmanifest"
  artifact.write_text("3\n")
  device = add_usb_device(tmp_path, "4-1", 5000, devnum=7)

  diagnostic = get_usbgpu_build_diagnostic(artifact, tmp_path)

  assert diagnostic == {
    "timestamp": diagnostic["timestamp"],
    "decision": "enable_target",
    "reasonCode": "superspeed",
    "reason": "USB GPU SuperSpeed link detected",
    "devicePresent": True,
    "sysfsPath": str(device),
    "vendorId": "add1",
    "productId": "0001",
    "speedMbps": 5000,
    "configuration": 1,
    "devnum": 7,
    "artifactPresent": True,
    "allowedBootstrapMbps": 12,
    "allowedSuperSpeedMbps": 5000,
    "incrementalBuild": True,
  }
  assert format_usbgpu_build_decision(diagnostic) == \
    '[USBGPU] target enabled: reason=superspeed speed=5000Mbps artifact=present rule="stable configured 5000 only"; SCons will rebuild only if stale'
  assert format_usbgpu_build_device(diagnostic) == f"[USBGPU] device: path={device} vendor=add1 product=0001 configuration=1 devnum=7"


def test_build_diagnostic_reason_codes(tmp_path):
  artifact = tmp_path / "missing.chunkmanifest"
  diagnostic = get_usbgpu_build_diagnostic(artifact, tmp_path)
  assert diagnostic["decision"] == "skip_target"
  assert diagnostic["reasonCode"] == "device_not_found"

  device = add_usb_device(tmp_path, "3-1", 12)
  diagnostic = get_usbgpu_build_diagnostic(artifact, tmp_path)
  assert diagnostic["decision"] == "skip_target"
  assert diagnostic["reasonCode"] == "bootstrap"
  assert diagnostic["reason"] == "USB GPU bootstrap link detected; 5000Mbps is required to build the AMD target"

  (device / "speed").write_text("unknown\n")
  diagnostic = get_usbgpu_build_diagnostic(artifact, tmp_path)
  assert diagnostic["decision"] == "skip_target"
  assert diagnostic["reasonCode"] == "speed_unavailable"

  (device / "speed").write_text("480\n")
  diagnostic = get_usbgpu_build_diagnostic(artifact, tmp_path)
  assert diagnostic["decision"] == "skip_target"
  assert diagnostic["reasonCode"] == "unsupported_speed"
  assert "artifact=" not in format_usbgpu_build_decision(diagnostic)
  assert format_usbgpu_build_decision(diagnostic).endswith('rule="stable configured 5000 only"; QCOM build continues')


def test_build_diagnostic_requires_configured_superspeed(tmp_path):
  add_usb_device(tmp_path, "4-1", 5000, configuration="0")

  diagnostic = get_usbgpu_build_diagnostic(tmp_path / "missing", tmp_path)

  assert diagnostic["decision"] == "skip_target"
  assert diagnostic["reasonCode"] == "superspeed"
  assert diagnostic["configuration"] == 0
  assert "not configured" in diagnostic["reason"]


def test_build_readiness_promotes_bootstrap_only_after_stable_superspeed(monkeypatch, tmp_path):
  device = add_usb_device(tmp_path, "4-1", 12, devnum=2)

  def transition(timeout, stable_duration, poll_interval=0.1, sysfs_root=helpers.USBGPU_SYSFS_ROOT):
    assert timeout == 15.0
    assert stable_duration == 2.0
    assert sysfs_root == tmp_path
    (device / "speed").write_text("5000\n")
    (device / "devnum").write_text("3\n")
    return True

  monkeypatch.setattr(helpers, "wait_for_usbgpu_ready", transition)
  diagnostic = resolve_usbgpu_build_diagnostic(
    tmp_path / "missing", tmp_path, reenum_timeout_ms=15000, stable_duration_ms=2000)

  assert diagnostic["decision"] == "enable_target"
  assert diagnostic["reasonCode"] == "bootstrap"
  assert diagnostic["initialSpeedMbps"] == 12
  assert diagnostic["speedMbps"] == 5000
  assert diagnostic["initialDevnum"] == 2
  assert diagnostic["devnum"] == 3
  assert diagnostic["readinessStatus"] == "stable"
  assert diagnostic["requiredBuildSpeedMbps"] == 5000
  assert diagnostic["requiredConfiguration"] is True
  assert diagnostic["requiredStableMs"] == 2000
  assert "initial_speed=12Mbps" in format_usbgpu_build_decision(diagnostic)


def test_build_readiness_keeps_bootstrap_disabled_on_timeout(monkeypatch, tmp_path):
  add_usb_device(tmp_path, "4-1", 12)
  monkeypatch.setattr(helpers, "wait_for_usbgpu_ready", lambda *args, **kwargs: False)

  diagnostic = resolve_usbgpu_build_diagnostic(
    tmp_path / "missing", tmp_path, reenum_timeout_ms=1, stable_duration_ms=0)

  assert diagnostic["decision"] == "skip_target"
  assert diagnostic["reasonCode"] == "bootstrap"
  assert diagnostic["speedMbps"] == 12
  assert diagnostic["readinessStatus"] == "timeout"
  assert "QCOM build continues" in format_usbgpu_build_decision(diagnostic)


def test_build_readiness_rechecks_final_state_after_successful_wait(monkeypatch, tmp_path):
  add_usb_device(tmp_path, "4-1", 12)
  monkeypatch.setattr(helpers, "wait_for_usbgpu_ready", lambda *args, **kwargs: True)

  diagnostic = resolve_usbgpu_build_diagnostic(
    tmp_path / "missing", tmp_path, reenum_timeout_ms=1, stable_duration_ms=0)

  assert diagnostic["decision"] == "skip_target"
  assert diagnostic["reasonCode"] == "bootstrap"
  assert diagnostic["readinessStatus"] == "unstable"


def test_build_readiness_validates_initial_superspeed(monkeypatch, tmp_path):
  add_usb_device(tmp_path, "4-1", 5000)

  def validate(timeout, stable_duration, poll_interval=0.1, sysfs_root=helpers.USBGPU_SYSFS_ROOT):
    assert timeout == 5.0
    assert stable_duration == 2.0
    assert sysfs_root == tmp_path
    return True

  monkeypatch.setattr(helpers, "wait_for_usbgpu_ready", validate)
  diagnostic = resolve_usbgpu_build_diagnostic(
    tmp_path / "missing", tmp_path, ready_timeout_sec=5.0, stable_duration_ms=2000)

  assert diagnostic["decision"] == "enable_target"
  assert diagnostic["reasonCode"] == "superspeed"
  assert diagnostic["readinessStatus"] == "stable"


def test_build_readiness_allows_superspeed_to_become_configured(monkeypatch, tmp_path):
  device = add_usb_device(tmp_path, "4-1", 5000, configuration="0")

  def configure(*args, **kwargs):
    (device / "bConfigurationValue").write_text("1\n")
    return True

  monkeypatch.setattr(helpers, "wait_for_usbgpu_ready", configure)
  diagnostic = resolve_usbgpu_build_diagnostic(
    tmp_path / "missing", tmp_path, ready_timeout_sec=5.0, stable_duration_ms=2000)

  assert diagnostic["decision"] == "enable_target"
  assert diagnostic["reasonCode"] == "superspeed"
  assert diagnostic["configuration"] == 1
  assert diagnostic["readinessStatus"] == "stable"


def test_build_readiness_does_not_wait_for_ineligible_speed(monkeypatch, tmp_path):
  add_usb_device(tmp_path, "4-1", 480)

  def unexpected_wait(*args, **kwargs):
    raise AssertionError("ineligible links must not enter readiness wait")

  monkeypatch.setattr(helpers, "wait_for_usbgpu_ready", unexpected_wait)
  diagnostic = resolve_usbgpu_build_diagnostic(tmp_path / "missing", tmp_path)

  assert diagnostic["decision"] == "skip_target"
  assert diagnostic["reasonCode"] == "unsupported_speed"
  assert diagnostic["readinessWaitAttempted"] is False


def test_build_diagnostic_selects_fastest_device_then_stable_path(tmp_path):
  add_usb_device(tmp_path, "3-2", 12, devnum=2)
  expected = add_usb_device(tmp_path, "4-1", 5000, devnum=3)
  add_usb_device(tmp_path, "4-2", 5000, devnum=4)

  diagnostic = get_usbgpu_build_diagnostic(tmp_path / "missing", tmp_path)

  assert diagnostic["decision"] == "enable_target"
  assert diagnostic["reasonCode"] == "superspeed"
  assert diagnostic["speedMbps"] == 5000
  assert diagnostic["sysfsPath"] == str(expected)
  assert diagnostic["devnum"] == 3


def test_build_diagnostic_write_is_atomic_and_best_effort(tmp_path, capsys):
  diagnostic = get_usbgpu_build_diagnostic(tmp_path / "missing", tmp_path)
  destination = tmp_path / "logs" / "build" / "latest.json"
  assert write_usbgpu_build_diagnostic(diagnostic, destination)
  assert json.loads(destination.read_text()) == diagnostic
  assert not list(destination.parent.glob("*.tmp"))

  blocker = tmp_path / "not-a-directory"
  blocker.write_text("block")
  assert not write_usbgpu_build_diagnostic(diagnostic, blocker / "latest.json")
  assert "[USBGPU] warning: unable to write build diagnostic" in capsys.readouterr().err


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
