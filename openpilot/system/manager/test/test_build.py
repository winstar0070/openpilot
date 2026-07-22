import io
import importlib
import signal
import sys
from types import SimpleNamespace

import pytest

from openpilot.common.basedir import BASEDIR


def import_manager_build(monkeypatch):
  fake_hardware = SimpleNamespace(HARDWARE=SimpleNamespace(set_power_save=lambda enabled: None), AGNOS=False)
  monkeypatch.setitem(sys.modules, "openpilot.common.hardware", fake_hardware)
  monkeypatch.delitem(sys.modules, "openpilot.system.manager.build", raising=False)
  return importlib.import_module("openpilot.system.manager.build")


def test_bootstrap_power_flag_is_forwarded_only_to_first_scons_attempt(monkeypatch):
  manager_build = import_manager_build(monkeypatch)
  monkeypatch.setattr(manager_build.sys, "platform", "linux")

  popen_calls = []

  class FakeProcess:
    def __init__(self, returncode):
      self.returncode = returncode
      self.stderr = io.BytesIO()

    def __enter__(self):
      return self

    def __exit__(self, exc_type, exc_value, traceback):
      return False

    def poll(self):
      return self.returncode

  def popen(command, **kwargs):
    popen_calls.append((command, kwargs))
    return FakeProcess(1 if len(popen_calls) < 3 else 0)

  monkeypatch.setenv(manager_build.USBGPU_BOOTSTRAP_POWER_ON_BOOT, "1")
  monkeypatch.setenv("USBGPU_TEST_PRESERVED", "yes")
  monkeypatch.setattr(manager_build.subprocess, "Popen", popen)
  monkeypatch.setattr(manager_build, "Spinner", lambda: SimpleNamespace(update_progress=lambda *args: None, close=lambda: None))
  manager_build.build()

  assert [call[0] for call in popen_calls] == [["scons"], ["scons", "-j4"], ["scons", "-j1"]]
  first_env = popen_calls[0][1]["env"]
  assert first_env[manager_build.USBGPU_BOOTSTRAP_POWER_ON_BOOT] == "1"
  for _, kwargs in popen_calls[1:]:
    assert manager_build.USBGPU_BOOTSTRAP_POWER_ON_BOOT not in kwargs["env"]
  for _, kwargs in popen_calls:
    build_env = kwargs["env"]
    assert build_env["PWD"] == BASEDIR
    assert build_env["USBGPU_TEST_PRESERVED"] == "yes"
    assert kwargs["preexec_fn"] is manager_build._arm_scons_parent_death_signal


def test_linux_scons_parent_death_signal_is_armed(monkeypatch):
  manager_build = import_manager_build(monkeypatch)
  calls = []
  def prctl(*args):
    calls.append(args)
    return 0
  libc = SimpleNamespace(prctl=prctl)
  parents = iter((42, 42))
  monkeypatch.setattr(manager_build.sys, "platform", "linux")
  monkeypatch.setattr(manager_build.os, "getppid", lambda: next(parents))
  monkeypatch.setattr(manager_build.ctypes, "CDLL", lambda *args, **kwargs: libc)

  manager_build._arm_scons_parent_death_signal()

  assert calls == [(manager_build.PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)]


def test_linux_scons_parent_death_signal_failure_is_fail_closed(monkeypatch):
  manager_build = import_manager_build(monkeypatch)
  def prctl(*args):
    return -1
  libc = SimpleNamespace(prctl=prctl)
  monkeypatch.setattr(manager_build.sys, "platform", "linux")
  monkeypatch.setattr(manager_build.os, "getppid", lambda: 42)
  monkeypatch.setattr(manager_build.ctypes, "CDLL", lambda *args, **kwargs: libc)
  monkeypatch.setattr(manager_build.ctypes, "get_errno", lambda: 1)

  with pytest.raises(OSError, match="PR_SET_PDEATHSIG failed"):
    manager_build._arm_scons_parent_death_signal()
