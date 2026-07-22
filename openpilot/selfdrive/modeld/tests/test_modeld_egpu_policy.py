import inspect

import numpy as np
import pytest

from tinygrad.runtime.support.usb import USBDeviceSessionLost
from openpilot.selfdrive.modeld import modeld


class FakeParams:
  def __init__(self, values=None):
    self.values = dict(values or {})

  def get(self, key):
    return self.values.get(key)

  def put_bool(self, key, value):
    self.values[key] = value

  def put(self, key, value):
    self.values[key] = value

  def remove(self, key):
    self.values.pop(key, None)


class FakeModel:
  QUEUE_DEV = "AMD"

  def __init__(self):
    self.lat_delay = 0.25
    self.prev_desire = np.arange(8, dtype=np.float32)
    self.npy = {"prev_feat": np.arange(4, dtype=np.float32)}


def test_minimal_timeout_defaults():
  assert modeld.USBGPU_ATTACH_TIMEOUT == 2.0
  assert modeld.USBGPU_READY_TIMEOUT == 5.0
  assert modeld.USBGPU_STABLE_DURATION == 2.0
  assert modeld.USBGPU_RUNTIME_HCQ_TIMEOUT_MS == 2000


def test_recognizes_session_lost_through_exception_chain():
  session_error = USBDeviceSessionLost("AMD USB device session lost")
  try:
    raise RuntimeError("outer") from session_error
  except RuntimeError as error:
    assert modeld.is_recognized_usbgpu_error(error)


def test_recognizes_only_exact_legacy_reenumeration_timeouts():
  for message in modeld.USBGPU_REENUMERATION_TIMEOUT_MESSAGES:
    assert modeld.is_recognized_usbgpu_error(RuntimeError(message))
  assert not modeld.is_recognized_usbgpu_error(RuntimeError("ASM24 initialization failed"))
  assert not modeld.is_recognized_usbgpu_error(RuntimeError("Wait timeout: 2000 ms"))


def test_recognizes_libusb_no_device_code_only():
  USBError = type("USBError", (RuntimeError,), {})
  no_device = USBError("No such device")
  no_device.errno = -4
  io_error = USBError("Input/output error")
  io_error.errno = -1

  assert modeld.is_recognized_usbgpu_error(no_device)
  assert not modeld.is_recognized_usbgpu_error(io_error)
  assert not modeld.is_recognized_usbgpu_error(RuntimeError("No such device"))


def test_existing_init_error_skips_amd_for_same_onroad_manager_cycle(monkeypatch):
  params = FakeParams({"UsbGpuInitError": "USBDeviceSessionLost: disconnected"})
  monkeypatch.setattr(modeld, "usbgpu_model_compiled", lambda: True)
  monkeypatch.setattr(modeld, "usbgpu_present", lambda: True)
  monkeypatch.setattr(modeld, "usbgpu_speed", lambda: 5000)
  monkeypatch.setattr(modeld, "wait_for_usbgpu_present", lambda timeout: (_ for _ in ()).throw(AssertionError("must not wait")))

  use_usbgpu, new_failure = modeld.select_usbgpu(params)
  assert not use_usbgpu
  assert new_failure is None
  assert params.values["UsbGpuInitError"] == "USBDeviceSessionLost: disconnected"
  assert params.values["UsbGpuReady"] is False


def test_attach_timeout_records_lockout_and_uses_two_second_default(monkeypatch):
  params = FakeParams()
  waits = []
  monkeypatch.setattr(modeld, "usbgpu_model_compiled", lambda: True)
  monkeypatch.setattr(modeld, "usbgpu_present", lambda: False)
  monkeypatch.setattr(modeld, "wait_for_usbgpu_present", lambda timeout: waits.append(timeout) or False)

  use_usbgpu, new_failure = modeld.select_usbgpu(params)
  assert not use_usbgpu
  assert waits == [2.0]
  assert new_failure == params.values["UsbGpuInitError"]
  assert "not detected within 2.0s" in params.values["UsbGpuInitError"]
  assert params.values["UsbGpuReady"] is False


def test_unsupported_speed_requests_one_diagnostic(monkeypatch):
  params = FakeParams()
  monkeypatch.setattr(modeld, "usbgpu_model_compiled", lambda: True)
  monkeypatch.setattr(modeld, "usbgpu_present", lambda: True)
  monkeypatch.setattr(modeld, "usbgpu_speed", lambda: 480)

  use_usbgpu, new_failure = modeld.select_usbgpu(params)

  assert not use_usbgpu
  assert new_failure is not None and "unsupported 480M" in new_failure
  assert new_failure == params.values["UsbGpuInitError"]


def test_ready_success_sets_runtime_timeout_and_ready(monkeypatch):
  params = FakeParams()
  amd_model = FakeModel()
  configured = []
  monkeypatch.setattr(modeld, "wait_for_usbgpu_ready", lambda *args, **kwargs: True)
  monkeypatch.setattr(modeld, "configure_usbgpu_runtime", lambda model: configured.append(model))
  monkeypatch.setattr(modeld, "usbgpu_present", lambda: True)

  usbgpu, state, reason, traceback_text = modeld.finish_usbgpu_startup(amd_model, params)

  assert usbgpu
  assert state is None and reason is None and traceback_text is None
  assert configured == [amd_model]
  assert params.values["UsbGpuReady"] is True
  assert "UsbGpuInitError" not in params.values


def test_ready_failure_marks_device_lost_before_qcom_load(monkeypatch):
  params = FakeParams()
  amd_model = FakeModel()
  events = []
  monkeypatch.setattr(modeld, "wait_for_usbgpu_ready", lambda *args, **kwargs: False)
  monkeypatch.setattr(modeld, "usbgpu_present", lambda: True)
  monkeypatch.setattr(modeld, "mark_usbgpu_device_lost", lambda model=None, error=None: events.append((model, error)))
  monkeypatch.setattr(modeld, "ModelState", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("QCOM loaded too early")))

  usbgpu, state, reason, traceback_text = modeld.finish_usbgpu_startup(amd_model, params)

  assert not usbgpu
  assert state is not None and reason is not None and traceback_text is None
  assert events[0][0] is amd_model
  assert params.values["UsbGpuReady"] is False


def test_runtime_configuration_generic_error_is_not_hidden(monkeypatch):
  params = FakeParams()
  monkeypatch.setattr(modeld, "wait_for_usbgpu_ready", lambda *args, **kwargs: True)
  monkeypatch.setattr(
    modeld, "configure_usbgpu_runtime", lambda model: (_ for _ in ()).throw(RuntimeError("programming bug")),
  )
  monkeypatch.setattr(
    modeld, "mark_usbgpu_device_lost", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not mark")),
  )

  with pytest.raises(RuntimeError, match="programming bug"):
    modeld.finish_usbgpu_startup(FakeModel(), params)


def test_runtime_configuration_session_loss_falls_back(monkeypatch):
  params = FakeParams()
  marked = []
  monkeypatch.setattr(modeld, "wait_for_usbgpu_ready", lambda *args, **kwargs: True)
  monkeypatch.setattr(
    modeld, "configure_usbgpu_runtime",
    lambda model: (_ for _ in ()).throw(USBDeviceSessionLost("session lost while configuring")),
  )
  monkeypatch.setattr(modeld, "usbgpu_present", lambda: True)
  monkeypatch.setattr(modeld, "mark_usbgpu_device_lost", lambda model=None, error=None: marked.append(model))

  usbgpu, state, reason, traceback_text = modeld.finish_usbgpu_startup(FakeModel(), params)

  assert not usbgpu
  assert state is not None and "USBDeviceSessionLost" in reason
  assert traceback_text is not None and "session lost while configuring" in traceback_text
  assert len(marked) == 1


def test_fallback_restores_last_successful_host_state(monkeypatch):
  amd_model = FakeModel()
  previous_state = modeld._capture_qcom_fallback_state(amd_model)
  amd_model.prev_desire[:] = -1
  amd_model.npy["prev_feat"][:] = -2
  qcom_model = FakeModel()
  qcom_model.lat_delay = 0.0
  qcom_model.prev_desire[:] = 0
  qcom_model.npy["prev_feat"][:] = 0
  monkeypatch.setattr(modeld, "ModelState", lambda cam_w, cam_h, usbgpu: qcom_model)

  replacement = modeld.switch_to_qcom(1928, 1208, previous_state)

  assert replacement is qcom_model
  assert replacement.lat_delay == 0.25
  assert np.array_equal(replacement.prev_desire, np.arange(8, dtype=np.float32))
  assert np.array_equal(replacement.npy["prev_feat"], np.arange(4, dtype=np.float32))


def test_runtime_fallback_rejects_generic_errors_and_discards_failed_frame():
  source = inspect.getsource(modeld.main)
  run = source.index("model_output = model.run(bufs")
  catch = source.index("except Exception as e:", run)
  classify = source.index("is_recognized_usbgpu_error(e)", catch)
  immediate_raise = source.index("raise", classify)
  prepare = source.index("prepare_usbgpu_fallback", immediate_raise)
  release = source.index("model = None", prepare)
  collect = source.index("gc.collect()", release)
  qcom = source.index("switch_to_qcom", collect)
  discard = source.index("continue", qcom)
  timing = source.index("mt2 =", discard)

  assert catch < classify < immediate_raise < prepare < release < collect < qcom < discard < timing
  assert "model.run(bufs" not in source[prepare:timing]


def test_initialization_fallback_rejects_generic_errors():
  source = inspect.getsource(modeld.main)
  load = source.index("model = ModelState")
  catch = source.index("except Exception as e:", load)
  classify = source.index("is_recognized_usbgpu_error(e)", catch)
  immediate_raise = source.index("raise", classify)
  fallback = source.index("prepare_usbgpu_fallback", immediate_raise)
  assert catch < classify < immediate_raise < fallback


def test_diagnostics_start_only_after_fresh_qcom_output_is_published():
  source = inspect.getsource(modeld.main)
  selection = source.index("pending_egpu_diagnostic: tuple")
  runtime = source.index("model_output = model.run(bufs")
  qcom = source.index("switch_to_qcom", runtime)
  discard = source.index("continue", qcom)
  publish = source.index("pm.send('modelDataV2SP'", discard)
  diagnostics = source.index("schedule_egpu_diagnostics", publish)
  assert selection < runtime
  assert qcom < discard < publish < diagnostics


def test_async_diagnostics_uses_captured_traceback(monkeypatch):
  captured = {}

  class ImmediateThread:
    def __init__(self, target, **kwargs):
      self.target = target

    def start(self):
      self.target()

  def log_diagnostics(error, traceback_text=None):
    captured["error"] = error
    captured["traceback"] = traceback_text

  monkeypatch.setattr(modeld.threading, "Thread", ImmediateThread)
  monkeypatch.setattr(modeld, "log_egpu_diagnostics", log_diagnostics)
  modeld.schedule_egpu_diagnostics("USB failure", "Traceback: original\n")

  assert type(captured["error"]) is RuntimeError
  assert captured["error"].__traceback__ is None
  assert captured["traceback"] == "Traceback: original\n"
