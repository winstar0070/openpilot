import json
from pathlib import Path
import subprocess
import sys

from openpilot.selfdrive.modeld import helpers
from openpilot.selfdrive.modeld.helpers import format_usbgpu_build_decision, format_usbgpu_build_device, format_usbgpu_pcie_power
from openpilot.selfdrive.modeld.helpers import get_usbgpu_build_diagnostic, request_usbgpu_pcie_power_on, resolve_usbgpu_build_diagnostic
from openpilot.selfdrive.modeld.helpers import usbgpu_present, usbgpu_ready_identity, usbgpu_speed
from openpilot.selfdrive.modeld.helpers import usbgpu_superspeed_ready, wait_for_usbgpu_present, wait_for_usbgpu_ready
from openpilot.selfdrive.modeld.helpers import wait_for_usbgpu_ready_identity
from openpilot.selfdrive.modeld.helpers import usbgpu_build_speed_eligible, usbgpu_speed_eligible, write_usbgpu_build_diagnostic


def add_usb_device(root: Path, name: str, speed: int, configuration: str = "1", devnum: int = 1, busnum: int = 4) -> Path:
  device = root / name
  device.mkdir()
  (device / "idVendor").write_text("add1\n")
  (device / "idProduct").write_text("0001\n")
  (device / "speed").write_text(f"{speed}\n")
  (device / "bConfigurationValue").write_text(f"{configuration}\n")
  (device / "busnum").write_text(f"{busnum}\n")
  (device / "devnum").write_text(f"{devnum}\n")
  return device


def accepted_pcie_power_result(status: str = "sent", device: str = "usb:4-1") -> dict[str, object]:
  return {
    "pciePowerAction": "on",
    "pciePowerAttempted": True,
    "pciePowerStatus": status,
    "pciePowerDevice": device,
    "pciePowerSpeedMbps": 12,
    "pciePowerReturnCode": 0,
    "pciePowerDurationMs": 1,
    "pciePowerError": None,
  }


def pcie_power_stdout(status: str = "sent", device: str = "usb:4-1", error: str | None = None) -> str:
  return json.dumps({
    "action": "on",
    "device": device,
    "error": error,
    "speedMbps": 12,
    "status": status,
  }) + "\n"


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
  assert wait_for_usbgpu_ready_identity(0, sysfs_root=tmp_path) == (str(device), 7, 5000, 1)
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
    "busnum": 4,
    "devnum": 7,
    "artifactPresent": True,
    "allowedBootstrapMbps": 12,
    "allowedSuperSpeedMbps": 5000,
    "incrementalBuild": True,
  }
  assert format_usbgpu_build_decision(diagnostic) == \
    '[USBGPU] target enabled: reason=superspeed speed=5000Mbps artifact=present rule="stable configured 5000 only"; SCons will rebuild only if stale'
  assert format_usbgpu_build_device(diagnostic) == f"[USBGPU] device: path={device} vendor=add1 product=0001 configuration=1 busnum=4 devnum=7"


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


def test_pcie_power_on_runs_once_for_exact_bootstrap(monkeypatch, tmp_path):
  add_usb_device(tmp_path, "4-1", 12, devnum=10, busnum=4)
  initial = get_usbgpu_build_diagnostic(tmp_path / "missing", tmp_path)
  calls = []

  def run(command, **kwargs):
    calls.append((command, kwargs))
    return subprocess.CompletedProcess(command, 0, pcie_power_stdout(device="usb:4-10"), "")

  monkeypatch.setattr(helpers.subprocess, "run", run)
  result = request_usbgpu_pcie_power_on(initial, tmp_path / "tinygrad_repo", allow_power=True, timeout_sec=3.0)

  assert len(calls) == 1
  command, kwargs = calls[0]
  assert command == [sys.executable, "-m", helpers.TINYGRAD_PCIE_POWER_MODULE, "on", "--device", "usb:4-10"]
  assert kwargs == {
    "cwd": tmp_path / "tinygrad_repo",
    "capture_output": True,
    "text": True,
    "check": False,
    "timeout": 3.0,
  }
  assert "AMD" not in " ".join(command)
  assert result["pciePowerAttempted"] is True
  assert result["pciePowerStatus"] == "sent"
  assert result["pciePowerReturnCode"] == 0
  assert result["pciePowerError"] is None
  assert "status=sent" in format_usbgpu_pcie_power(result)


def test_pcie_power_is_lifecycle_blocked_by_default(monkeypatch, tmp_path):
  add_usb_device(tmp_path, "4-1", 12, devnum=10, busnum=4)
  initial = get_usbgpu_build_diagnostic(tmp_path / "missing", tmp_path)

  monkeypatch.setattr(helpers.subprocess, "run",
                      lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("blocked power must not run")))
  result = request_usbgpu_pcie_power_on(initial, tmp_path)
  diagnostic = resolve_usbgpu_build_diagnostic(
    tmp_path / "missing", tmp_path, initial_diagnostic=initial, pcie_power_result=result)

  assert result["pciePowerAction"] == "on"
  assert result["pciePowerAttempted"] is False
  assert result["pciePowerStatus"] == "lifecycle_blocked"
  assert diagnostic["decision"] == "skip_target"
  assert diagnostic["readinessStatus"] == "lifecycle_blocked"
  assert "without launcher authorization" in diagnostic["reason"]


def test_pcie_power_never_runs_without_initial_exact_bootstrap(monkeypatch, tmp_path):
  def unexpected_run(*args, **kwargs):
    raise AssertionError("PCIe power command must only run for the initial 12Mbps device")

  monkeypatch.setattr(helpers.subprocess, "run", unexpected_run)
  for reason, speed in (("device_not_found", None), ("superspeed", 5000), ("unsupported_speed", 480)):
    result = request_usbgpu_pcie_power_on({"reasonCode": reason, "speedMbps": speed}, tmp_path)
    assert result["pciePowerAttempted"] is False
    assert result["pciePowerStatus"] == "not_applicable"


def test_pcie_power_requires_addressable_bootstrap_identity(monkeypatch, tmp_path):
  def unexpected_run(*args, **kwargs):
    raise AssertionError("PCIe power command cannot run without busnum and devnum")

  monkeypatch.setattr(helpers.subprocess, "run", unexpected_run)
  result = request_usbgpu_pcie_power_on({
    "reasonCode": "bootstrap",
    "speedMbps": 12,
    "busnum": None,
    "devnum": 1,
  }, tmp_path, allow_power=True)
  assert result["pciePowerAttempted"] is False
  assert result["pciePowerStatus"] == "identity_unavailable"


def test_pcie_power_errors_fail_closed_before_readiness_wait(monkeypatch, tmp_path):
  add_usb_device(tmp_path, "4-1", 12)
  initial = get_usbgpu_build_diagnostic(tmp_path / "missing", tmp_path)

  for returncode, status in ((1, "error"), (2, "not_applicable")):
    def run(command, _returncode=returncode, _status=status, **kwargs):
      return subprocess.CompletedProcess(command, _returncode, pcie_power_stdout(_status, error="rejected"), "")

    monkeypatch.setattr(helpers.subprocess, "run", run)
    power_result = request_usbgpu_pcie_power_on(initial, tmp_path, allow_power=True)
    monkeypatch.setattr(helpers, "wait_for_usbgpu_ready_identity",
                        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("failed power must not wait")))
    diagnostic = resolve_usbgpu_build_diagnostic(
      tmp_path / "missing", tmp_path, initial_diagnostic=initial, pcie_power_result=power_result)
    assert diagnostic["decision"] == "skip_target"
    assert diagnostic["readinessWaitAttempted"] is False
    assert diagnostic["readinessStatus"] == "power_failed"
    assert diagnostic["pciePowerReturnCode"] == returncode
    assert "QCOM build continues" in diagnostic["reason"]


def test_only_exit2_not_applicable_can_enter_power_race_recheck(monkeypatch, tmp_path):
  device = add_usb_device(tmp_path, "4-1", 12)
  initial = get_usbgpu_build_diagnostic(tmp_path / "missing", tmp_path)
  (device / "speed").write_text("5000\n")
  (device / "devnum").write_text("2\n")

  monkeypatch.setattr(helpers, "wait_for_usbgpu_ready_identity",
                      lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("blocked failures must not wait")))
  blocked_results = []
  for status, returncode in (("error", 1), ("timeout", None), ("invalid_response", 0)):
    result = accepted_pcie_power_result()
    result.update({
      "pciePowerStatus": status,
      "pciePowerReturnCode": returncode,
      "pciePowerError": status,
    })
    blocked_results.append(result)

  for power_result in blocked_results:
    diagnostic = resolve_usbgpu_build_diagnostic(
      tmp_path / "missing", tmp_path, initial_diagnostic=initial, pcie_power_result=power_result)
    assert diagnostic["decision"] == "skip_target"
    assert diagnostic["readinessWaitAttempted"] is False
    assert diagnostic["readinessStatus"] == "power_failed"
    assert diagnostic["pciePowerRecheckAttempted"] is False


def test_pcie_power_race_rechecks_immediate_5000_then_requires_stability(monkeypatch, tmp_path):
  device = add_usb_device(tmp_path, "4-1", 12, devnum=1)
  initial = get_usbgpu_build_diagnostic(tmp_path / "missing", tmp_path)

  def raced(command, **kwargs):
    (device / "speed").write_text("5000\n")
    (device / "devnum").write_text("2\n")
    return subprocess.CompletedProcess(command, 2, pcie_power_stdout("not_applicable", error="no 12Mbps device"), "")

  def stable(timeout, stable_duration, poll_interval=0.1, sysfs_root=helpers.USBGPU_SYSFS_ROOT):
    assert timeout == 5.0
    assert stable_duration == 2.0
    assert sysfs_root == tmp_path
    return (str(device), 2, 5000, 1)

  monkeypatch.setattr(helpers.subprocess, "run", raced)
  monkeypatch.setattr(helpers, "wait_for_usbgpu_ready_identity", stable)
  power_result = request_usbgpu_pcie_power_on(initial, tmp_path, allow_power=True)
  diagnostic = resolve_usbgpu_build_diagnostic(
    tmp_path / "missing", tmp_path, initial_diagnostic=initial, pcie_power_result=power_result,
    ready_timeout_sec=5.0, stable_duration_ms=2000)

  assert power_result["pciePowerReturnCode"] == 2
  assert diagnostic["pciePowerStatus"] == "not_applicable"
  assert diagnostic["pciePowerRecheckAttempted"] is True
  assert diagnostic["pciePowerRecheckSpeedMbps"] == 5000
  assert diagnostic["pciePowerRecheckEligibleForWait"] is True
  assert diagnostic["readinessWaitAttempted"] is True
  assert diagnostic["readinessStatus"] == "stable"
  assert diagnostic["stableIdentityMatchedFinal"] is True
  assert diagnostic["decision"] == "enable_target"
  assert diagnostic["speedMbps"] == 5000

  monkeypatch.setattr(helpers, "wait_for_usbgpu_ready_identity", lambda *args, **kwargs: None)
  unstable = resolve_usbgpu_build_diagnostic(
    tmp_path / "missing", tmp_path, initial_diagnostic=initial, pcie_power_result=power_result,
    ready_timeout_sec=5.0, stable_duration_ms=2000)
  assert unstable["readinessStatus"] == "timeout"
  assert unstable["decision"] == "skip_target"


def test_pcie_power_timeout_oserror_and_invalid_json_fail_closed(monkeypatch, tmp_path):
  initial = {"reasonCode": "bootstrap", "speedMbps": 12, "busnum": 4, "devnum": 1}

  def timeout(*args, **kwargs):
    raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

  monkeypatch.setattr(helpers.subprocess, "run", timeout)
  assert request_usbgpu_pcie_power_on(initial, tmp_path, allow_power=True, timeout_sec=0.1)["pciePowerStatus"] == "timeout"

  monkeypatch.setattr(helpers.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("open failed")))
  assert request_usbgpu_pcie_power_on(initial, tmp_path, allow_power=True)["pciePowerStatus"] == "error"

  monkeypatch.setattr(helpers.subprocess, "run",
                      lambda command, **kwargs: subprocess.CompletedProcess(command, 0, "not json\n", ""))
  invalid = request_usbgpu_pcie_power_on(initial, tmp_path, allow_power=True)
  assert invalid["pciePowerStatus"] == "invalid_response"
  assert invalid["pciePowerReturnCode"] == 0


def test_build_readiness_promotes_bootstrap_only_after_stable_superspeed(monkeypatch, tmp_path):
  device = add_usb_device(tmp_path, "4-1", 12, devnum=2)

  def transition(timeout, stable_duration, poll_interval=0.1, sysfs_root=helpers.USBGPU_SYSFS_ROOT):
    assert timeout == 15.0
    assert stable_duration == 2.0
    assert sysfs_root == tmp_path
    (device / "speed").write_text("5000\n")
    (device / "devnum").write_text("3\n")
    return (str(device), 3, 5000, 1)

  monkeypatch.setattr(helpers, "wait_for_usbgpu_ready_identity", transition)
  diagnostic = resolve_usbgpu_build_diagnostic(
    tmp_path / "missing", tmp_path, pcie_power_result=accepted_pcie_power_result("detached"),
    reenum_timeout_ms=15000, stable_duration_ms=2000)

  assert diagnostic["decision"] == "enable_target"
  assert diagnostic["reasonCode"] == "bootstrap"
  assert diagnostic["initialSpeedMbps"] == 12
  assert diagnostic["speedMbps"] == 5000
  assert diagnostic["initialDevnum"] == 2
  assert diagnostic["devnum"] == 3
  assert diagnostic["readinessStatus"] == "stable"
  assert diagnostic["stableIdentityMatchedFinal"] is True
  assert diagnostic["requiredBuildSpeedMbps"] == 5000
  assert diagnostic["requiredConfiguration"] is True
  assert diagnostic["requiredStableMs"] == 2000
  assert diagnostic["reason"] == \
    "USB GPU bootstrap link transitioned to stable configured 5000Mbps after the PCIe power-on request"
  assert "initial_speed=12Mbps" in format_usbgpu_build_decision(diagnostic)


def test_build_readiness_keeps_bootstrap_disabled_on_timeout(monkeypatch, tmp_path):
  add_usb_device(tmp_path, "4-1", 12)
  monkeypatch.setattr(helpers, "wait_for_usbgpu_ready_identity", lambda *args, **kwargs: None)

  diagnostic = resolve_usbgpu_build_diagnostic(
    tmp_path / "missing", tmp_path, pcie_power_result=accepted_pcie_power_result(),
    reenum_timeout_ms=1, stable_duration_ms=0)

  assert diagnostic["decision"] == "skip_target"
  assert diagnostic["reasonCode"] == "bootstrap"
  assert diagnostic["speedMbps"] == 12
  assert diagnostic["readinessStatus"] == "timeout"
  assert "following the PCIe power-on request" in diagnostic["reason"]
  assert "QCOM build continues" in format_usbgpu_build_decision(diagnostic)


def test_build_readiness_rechecks_final_state_after_successful_wait(monkeypatch, tmp_path):
  device = add_usb_device(tmp_path, "4-1", 12)
  monkeypatch.setattr(helpers, "wait_for_usbgpu_ready_identity", lambda *args, **kwargs: (str(device), 1, 5000, 1))

  diagnostic = resolve_usbgpu_build_diagnostic(
    tmp_path / "missing", tmp_path, pcie_power_result=accepted_pcie_power_result(),
    reenum_timeout_ms=1, stable_duration_ms=0)

  assert diagnostic["decision"] == "skip_target"
  assert diagnostic["reasonCode"] == "bootstrap"
  assert diagnostic["readinessStatus"] == "unstable"


def test_build_readiness_validates_initial_superspeed(monkeypatch, tmp_path):
  device = add_usb_device(tmp_path, "4-1", 5000)

  def validate(timeout, stable_duration, poll_interval=0.1, sysfs_root=helpers.USBGPU_SYSFS_ROOT):
    assert timeout == 5.0
    assert stable_duration == 2.0
    assert sysfs_root == tmp_path
    return (str(device), 1, 5000, 1)

  monkeypatch.setattr(helpers, "wait_for_usbgpu_ready_identity", validate)
  diagnostic = resolve_usbgpu_build_diagnostic(
    tmp_path / "missing", tmp_path, ready_timeout_sec=5.0, stable_duration_ms=2000)

  assert diagnostic["decision"] == "enable_target"
  assert diagnostic["reasonCode"] == "superspeed"
  assert diagnostic["readinessStatus"] == "stable"


def test_build_readiness_allows_superspeed_to_become_configured(monkeypatch, tmp_path):
  device = add_usb_device(tmp_path, "4-1", 5000, configuration="0")

  def configure(*args, **kwargs):
    (device / "bConfigurationValue").write_text("1\n")
    return (str(device), 1, 5000, 1)

  monkeypatch.setattr(helpers, "wait_for_usbgpu_ready_identity", configure)
  diagnostic = resolve_usbgpu_build_diagnostic(
    tmp_path / "missing", tmp_path, ready_timeout_sec=5.0, stable_duration_ms=2000)

  assert diagnostic["decision"] == "enable_target"
  assert diagnostic["reasonCode"] == "superspeed"
  assert diagnostic["configuration"] == 1
  assert diagnostic["readinessStatus"] == "stable"


def test_build_readiness_rejects_device_replacement_after_stable_wait(monkeypatch, tmp_path):
  device = add_usb_device(tmp_path, "4-1", 5000, devnum=1)

  def replace_after_stable(*args, **kwargs):
    stable_identity = (str(device), 1, 5000, 1)
    (device / "devnum").write_text("2\n")
    return stable_identity

  monkeypatch.setattr(helpers, "wait_for_usbgpu_ready_identity", replace_after_stable)
  diagnostic = resolve_usbgpu_build_diagnostic(
    tmp_path / "missing", tmp_path, ready_timeout_sec=5.0, stable_duration_ms=2000)

  assert diagnostic["stableDevnum"] == 1
  assert diagnostic["devnum"] == 2
  assert diagnostic["stableIdentityMatchedFinal"] is False
  assert diagnostic["readinessStatus"] == "unstable"
  assert diagnostic["decision"] == "skip_target"


def test_build_readiness_does_not_wait_for_ineligible_speed(monkeypatch, tmp_path):
  add_usb_device(tmp_path, "4-1", 480)

  def unexpected_wait(*args, **kwargs):
    raise AssertionError("ineligible links must not enter readiness wait")

  monkeypatch.setattr(helpers, "wait_for_usbgpu_ready_identity", unexpected_wait)
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
  diagnostic = resolve_usbgpu_build_diagnostic(tmp_path / "missing", tmp_path)
  destination = tmp_path / "logs" / "build" / "latest.json"
  assert write_usbgpu_build_diagnostic(diagnostic, destination)
  assert json.loads(destination.read_text()) == diagnostic
  assert diagnostic["pciePowerStatus"] == "not_applicable"
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
  assert wait_for_usbgpu_ready_identity(5, stable_duration=2, poll_interval=1, sysfs_root=tmp_path) == \
    (str(device), 2, 5000, 1)
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
