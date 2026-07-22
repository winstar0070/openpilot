import os
import pytest
import signal
import time
from types import SimpleNamespace

from openpilot.cereal import log
from opendbc.car.structs import car
from openpilot.common.params import Params, ParamKeyFlag
import openpilot.system.manager.manager as manager
from openpilot.system.manager.process import ensure_running
from openpilot.system.manager.process_config import managed_processes, procs
from openpilot.common.hardware import HARDWARE

os.environ['FAKEUPLOAD'] = "1"

MAX_STARTUP_TIME = 3
BLACKLIST_PROCS = ['manage_athenad', 'pandad', 'pigeond']


def panda_state(*, ignition_line=False, ignition_can=False, known=True):
  return SimpleNamespace(
    pandaType=log.PandaState.PandaType.uno if known else log.PandaState.PandaType.unknown,
    ignitionLine=ignition_line,
    ignitionCan=ignition_can,
  )


class TestManager:
  def setup_method(self):
    HARDWARE.set_power_save(False)

    # ensure clean CarParams
    params = Params()
    params.clear_all()

  def teardown_method(self):
    manager.manager_cleanup()

  def test_duplicate_procs(self):
    assert len(procs) == len(managed_processes), "Duplicate process names"

  def test_blacklisted_procs(self):
    # TODO: ensure there are blacklisted procs until we have a dedicated test
    assert len(BLACKLIST_PROCS), "No blacklisted procs to test not_run"

  def test_set_params_with_default_value(self):
    params = Params()
    params.clear_all()

    os.environ['PREPAREONLY'] = '1'
    manager.main()
    for k in params.all_keys():
      default_value = params.get_default_value(k)
      if default_value is not None:
        assert params.get(k) == default_value
    assert params.get("OpenpilotEnabledToggle")
    assert params.get("RouteCount") == 0

  def test_usbgpu_init_error_is_persistent(self):
    params = Params()
    error = "USBDeviceSessionLost: disconnected"
    params.put("UsbGpuInitError", error, block=True)

    for flag in (ParamKeyFlag.CLEAR_ON_MANAGER_START, ParamKeyFlag.CLEAR_ON_ONROAD_TRANSITION,
                 ParamKeyFlag.CLEAR_ON_OFFROAD_TRANSITION, ParamKeyFlag.CLEAR_ON_IGNITION_ON):
      params.clear_all(flag)
      assert params.get("UsbGpuInitError") == error

  def test_usbgpu_init_error_clears_after_confirmed_ignition_off(self):
    assert manager.should_clear_usbgpu_init_error(
      False, True, [panda_state(), panda_state()], True, True,
    )

  @pytest.mark.parametrize("started,device_valid,pandas,pandas_valid,pandas_alive", [
    (True, True, [panda_state()], True, True),
    (False, False, [panda_state()], True, True),
    (False, True, [panda_state()], False, True),
    (False, True, [panda_state()], True, False),
    (False, True, [], True, True),
    (False, True, [panda_state(known=False)], True, True),
    (False, True, [panda_state(), panda_state(known=False)], True, True),
    (False, True, [panda_state(ignition_line=True)], True, True),
    (False, True, [panda_state(ignition_can=True)], True, True),
  ])
  def test_usbgpu_init_error_stays_locked_without_confirmed_ignition_off(self, started, device_valid, pandas,
                                                                         pandas_valid, pandas_alive):
    assert not manager.should_clear_usbgpu_init_error(started, device_valid, pandas, pandas_valid, pandas_alive)

  @pytest.mark.skip("this test is flaky the way it's currently written, should be moved to test_onroad")
  def test_clean_exit(self, subtests):
    """
      Ensure all processes exit cleanly when stopped.
    """
    HARDWARE.set_power_save(False)
    manager.manager_init()

    CP = car.CarParams.new_message()
    procs = ensure_running(managed_processes.values(), True, Params(), CP, not_run=BLACKLIST_PROCS)

    time.sleep(10)

    for p in procs:
      with subtests.test(proc=p.name):
        state = p.get_process_state_msg()
        assert state.running, f"{p.name} not running"
        exit_code = p.stop(retry=False)

        assert p.name not in BLACKLIST_PROCS, f"{p.name} was started"

        assert exit_code is not None, f"{p.name} failed to exit"

        # TODO: interrupted blocking read exits with 1 in cereal. use a more unique return code
        exit_codes = [0, 1]
        if p.sigkill:
          exit_codes = [-signal.SIGKILL]
        assert exit_code in exit_codes, f"{p.name} died with {exit_code}"
