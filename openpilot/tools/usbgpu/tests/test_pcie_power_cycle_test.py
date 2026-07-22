import contextlib
import json
import sys
from types import SimpleNamespace

import pytest

from openpilot.tools.usbgpu import pcie_power_cycle_test as power_cycle


DEVICE = "usb:4-10"


def write_usb_state(root, *, name="4-1", busnum=4, speed=5000, configuration=1, devnum=10):
  path = root / name
  path.mkdir(parents=True, exist_ok=True)
  fields = {
    "idVendor": "add1",
    "idProduct": "0001",
    "busnum": str(busnum),
    "devnum": str(devnum),
    "speed": str(speed),
    "bConfigurationValue": str(configuration),
  }
  for field_name, value in fields.items():
    (path / field_name).write_text(f"{value}\n")
  return path


def make_logger(tmp_path, *, dry_run=False):
  return power_cycle.AttemptLog(tmp_path / "logs", DEVICE, {
    "dryRun": dry_run,
    "dwellMs": 500,
    "reenumTimeoutSec": 15.0,
    "requiredStableSec": 2.0,
  })


def prepare_safe_run(monkeypatch, sysfs_path):
  usb = SimpleNamespace(product="custom v0.1", configuration_identity=lambda: (4, 10, 3, 1))
  monkeypatch.setattr(power_cycle, "check_offroad_ignition_off", lambda: {
    "isOffroad": True,
    "deviceStarted": False,
    "ignition": False,
    "knownPandaCount": 1,
  })
  monkeypatch.setattr(power_cycle, "_open_asm24_session", lambda device: contextlib.nullcontext(usb))
  monkeypatch.setattr(power_cycle, "_wait_same_path_ready", lambda path, timeout: (str(sysfs_path), 10, 5000, 1))
  monkeypatch.setattr(power_cycle.time, "sleep", lambda duration: None)
  return usb


def install_lifecycle_modules(monkeypatch, *, ignition=False, started=False, offroad=True):
  panda_state = SimpleNamespace(pandaType=1, ignitionLine=ignition, ignitionCan=False)
  services = {
    "deviceState": SimpleNamespace(started=started),
    "pandaStates": [panda_state],
  }

  class FakeSubMaster:
    alive = {"deviceState": True, "pandaStates": True}
    valid = {"deviceState": True, "pandaStates": True}

    def __init__(self, requested_services):
      assert requested_services == ["deviceState", "pandaStates"]

    def update(self, timeout):
      assert timeout == 200

    def __getitem__(self, service):
      return services[service]

  fake_cereal = SimpleNamespace(
    log=SimpleNamespace(PandaState=SimpleNamespace(PandaType=SimpleNamespace(unknown=0))),
    messaging=SimpleNamespace(SubMaster=FakeSubMaster),
  )
  fake_params = SimpleNamespace(Params=lambda: SimpleNamespace(get_bool=lambda key: offroad))
  monkeypatch.setitem(sys.modules, "openpilot.cereal", fake_cereal)
  monkeypatch.setitem(sys.modules, "openpilot.common.params", fake_params)


def test_lifecycle_requires_fresh_offroad_stopped_and_ignition_off(monkeypatch):
  install_lifecycle_modules(monkeypatch)
  assert power_cycle.check_offroad_ignition_off() == {
    "isOffroad": True, "deviceStarted": False, "ignition": False, "knownPandaCount": 1,
  }

  install_lifecycle_modules(monkeypatch, ignition=True)
  with pytest.raises(power_cycle.PowerCycleSafetyError, match="ignition on"):
    power_cycle.check_offroad_ignition_off()


def test_device_is_auto_detected_only_when_unambiguous(tmp_path):
  sysfs_root = tmp_path / "sysfs"
  write_usb_state(sysfs_root, devnum=10)
  assert power_cycle._detect_device(sysfs_root) == "usb:4-10"

  write_usb_state(sysfs_root, name="4-2", devnum=11)
  with pytest.raises(power_cycle.PowerCycleSafetyError, match="found 2"):
    power_cycle._detect_device(sysfs_root)


def test_power_cycle_logs_off_on_and_stable_same_path(monkeypatch, tmp_path):
  sysfs_path = write_usb_state(tmp_path / "sysfs")
  prepare_safe_run(monkeypatch, sysfs_path)
  transfers = []
  monkeypatch.setattr(power_cycle, "_send_power", lambda usb, enabled: transfers.append(enabled) or 0)

  logger = make_logger(tmp_path)
  result = power_cycle.run_power_cycle(DEVICE, logger, sysfs_root=sysfs_path.parent)

  assert transfers == [False, True]
  assert result["status"] == "passed"
  assert result["finalState"]["speedMbps"] == 5000
  events = [(event["name"], event["status"]) for event in result["events"]]
  assert ("f3_off", "acknowledged") in events
  assert ("post_off_control", "alive") in events
  assert ("f3_on", "acknowledged") in events
  assert ("reenumeration", "stable") in events
  assert json.loads(logger.latest_path.read_text())["status"] == "passed"


def test_off_failure_attempts_one_recovery_on_and_records_failure(monkeypatch, tmp_path):
  sysfs_path = write_usb_state(tmp_path / "sysfs")
  prepare_safe_run(monkeypatch, sysfs_path)
  transfers = []

  def transfer(usb, enabled):
    transfers.append(enabled)
    if enabled is False:
      raise RuntimeError("off transfer failed")
    return 0

  monkeypatch.setattr(power_cycle, "_send_power", transfer)
  logger = make_logger(tmp_path)
  with pytest.raises(RuntimeError, match="off transfer failed"):
    power_cycle.run_power_cycle(DEVICE, logger, sysfs_root=sysfs_path.parent)

  assert transfers == [False, True]
  logged = json.loads(logger.path.read_text())
  assert logged["status"] == "failed"
  assert any(event["name"] == "recovery_f3_on" and event["status"] == "acknowledged" for event in logged["events"])


def test_logging_failure_after_off_does_not_block_recovery_on(monkeypatch, tmp_path):
  sysfs_path = write_usb_state(tmp_path / "sysfs")
  prepare_safe_run(monkeypatch, sysfs_path)
  transfers = []
  monkeypatch.setattr(power_cycle, "_send_power", lambda usb, enabled: transfers.append(enabled) or 0)
  logger = make_logger(tmp_path)
  original_event = logger.event

  def fail_during_dwell(name, status, **details):
    if name == "off_dwell" and status == "started":
      raise OSError("log write failed")
    original_event(name, status, **details)

  logger.event = fail_during_dwell
  with pytest.raises(OSError, match="log write failed"):
    power_cycle.run_power_cycle(DEVICE, logger, sysfs_root=sysfs_path.parent)
  assert transfers == [False, True]


def test_on_no_device_is_ambiguous_until_stable_reenumeration(monkeypatch, tmp_path):
  class DetachedError(RuntimeError):
    pass

  sysfs_path = write_usb_state(tmp_path / "sysfs")
  prepare_safe_run(monkeypatch, sysfs_path)
  transfers = []

  def transfer(usb, enabled):
    transfers.append(enabled)
    if enabled is True:
      raise DetachedError("device detached")
    return 0

  monkeypatch.setattr(power_cycle, "_send_power", transfer)
  monkeypatch.setattr(power_cycle, "_is_no_device_error", lambda exc: isinstance(exc, DetachedError))
  logger = make_logger(tmp_path)
  result = power_cycle.run_power_cycle(DEVICE, logger, sysfs_root=sysfs_path.parent)

  assert transfers == [False, True]
  assert result["status"] == "passed"
  assert any(event["name"] == "f3_on" and event["status"] == "detached_ambiguous" for event in result["events"])


def test_final_identity_must_match_stable_identity(monkeypatch, tmp_path):
  sysfs_path = write_usb_state(tmp_path / "sysfs", devnum=10)
  prepare_safe_run(monkeypatch, sysfs_path)
  monkeypatch.setattr(power_cycle, "_send_power", lambda usb, enabled: 0)
  monkeypatch.setattr(power_cycle, "_wait_same_path_ready", lambda path, timeout: (str(sysfs_path), 11, 5000, 1))

  logger = make_logger(tmp_path)
  with pytest.raises(power_cycle.PowerCycleSafetyError, match="same sysfs path"):
    power_cycle.run_power_cycle(DEVICE, logger, sysfs_root=sysfs_path.parent)
  assert json.loads(logger.path.read_text())["status"] == "failed"


def test_lifecycle_failure_and_dry_run_never_send_power(monkeypatch, tmp_path):
  sysfs_path = write_usb_state(tmp_path / "sysfs")
  transfers = []
  monkeypatch.setattr(power_cycle, "_send_power", lambda usb, enabled: transfers.append(enabled))
  monkeypatch.setattr(power_cycle, "check_offroad_ignition_off",
                      lambda: (_ for _ in ()).throw(power_cycle.PowerCycleSafetyError("ignition on")))
  blocked_logger = make_logger(tmp_path)
  with pytest.raises(power_cycle.PowerCycleSafetyError, match="ignition on"):
    power_cycle.run_power_cycle(DEVICE, blocked_logger, sysfs_root=sysfs_path.parent)

  monkeypatch.setattr(power_cycle, "check_offroad_ignition_off", lambda: {
    "isOffroad": True, "deviceStarted": False, "ignition": False, "knownPandaCount": 1,
  })
  dry_logger = make_logger(tmp_path, dry_run=True)
  result = power_cycle.run_power_cycle(DEVICE, dry_logger, sysfs_root=sysfs_path.parent, dry_run=True)
  assert result["status"] == "dry_run_passed"
  assert transfers == []


def test_rejects_short_dwell_before_lifecycle_or_usb(monkeypatch, tmp_path):
  monkeypatch.setattr(power_cycle, "check_offroad_ignition_off",
                      lambda: (_ for _ in ()).throw(AssertionError("must not check lifecycle")))
  logger = make_logger(tmp_path)
  with pytest.raises(power_cycle.PowerCycleSafetyError, match="at least 300ms"):
    power_cycle.run_power_cycle(DEVICE, logger, dwell_ms=299)
  assert json.loads(logger.path.read_text())["status"] == "failed"
