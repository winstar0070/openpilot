import inspect
import gc
import numpy as np
import pytest
import weakref

from openpilot.selfdrive.modeld import modeld


@pytest.fixture(autouse=True)
def avoid_host_ignition_lockout(monkeypatch):
  monkeypatch.setattr(modeld, "set_usbgpu_ignition_lockout", lambda reason: None)


class FakeParams:
  def __init__(self):
    self.values = {}

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


class FakeLinkMonitor:
  def __init__(self, ready: bool, rate: float | None, identity=("/sys/4-1", 1, 5000, 1)):
    self.ready = ready
    self.rate = rate
    self.identity = identity
    self.resets = 0

  def sample(self, blocking: bool = True):
    pass

  def status(self, stable_duration: float):
    return self.ready, self.rate

  def current_identity(self):
    return self.identity

  def reset_stability(self):
    self.resets += 1


def test_ready_failure_switches_to_qcom_before_runtime(monkeypatch):
  params = FakeParams()
  amd_model = FakeModel()
  marked = []

  monkeypatch.setattr(modeld, "wait_for_usbgpu_ready", lambda *args, **kwargs: False)
  monkeypatch.setattr(modeld, "usbgpu_present", lambda: True)
  monkeypatch.setattr(modeld, "mark_usbgpu_device_lost", lambda model=None, error=None: marked.append(model))
  monkeypatch.setattr(modeld, "ModelState", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("QCOM loaded too early")))

  usbgpu, fallback_state, reason, traceback_text = modeld.finish_usbgpu_startup(amd_model, 1928, 1208, params)

  assert not usbgpu
  assert fallback_state is not None
  assert reason is not None
  assert traceback_text is None
  assert marked == [amd_model]
  assert params.values["UsbGpuPresent"] is True
  assert params.values["UsbGpuReady"] is False
  assert "did not remain configured" in params.values["UsbGpuInitError"]


def test_ready_success_keeps_amd_and_sets_runtime_timeout(monkeypatch):
  params = FakeParams()
  amd_model = FakeModel()
  configured = []

  monkeypatch.setattr(modeld, "wait_for_usbgpu_ready", lambda *args, **kwargs: True)
  monkeypatch.setattr(modeld, "usbgpu_present", lambda: True)
  monkeypatch.setattr(modeld, "configure_usbgpu_runtime", lambda model: configured.append(model))

  usbgpu, fallback_state, reason, traceback_text = modeld.finish_usbgpu_startup(amd_model, 1928, 1208, params)

  assert usbgpu
  assert fallback_state is None
  assert reason is None
  assert traceback_text is None
  assert configured == [amd_model]
  assert params.values["UsbGpuPresent"] is True
  assert params.values["UsbGpuReady"] is True
  assert "UsbGpuInitError" not in params.values


def test_model_load_link_sample_starts_fresh_post_init_window(monkeypatch):
  params = FakeParams()
  amd_model = FakeModel()
  configured = []
  wait_calls = []
  monitor = FakeLinkMonitor(True, 1.0)

  def wait_for_ready(*args, **kwargs):
    assert monitor.resets == 1
    wait_calls.append((args, kwargs))
    return True

  monkeypatch.setattr(modeld, "wait_for_usbgpu_ready", wait_for_ready)
  monkeypatch.setattr(modeld, "usbgpu_present", lambda: True)
  monkeypatch.setattr(modeld, "configure_usbgpu_runtime", lambda model: configured.append(model))

  usbgpu, fallback_state, reason, traceback_text = modeld.finish_usbgpu_startup(amd_model, 1928, 1208, params, monitor)

  assert usbgpu
  assert fallback_state is None
  assert reason is None
  assert traceback_text is None
  assert configured == [amd_model]
  assert monitor.resets == 1
  assert wait_calls[0][1]["monitor"] is monitor
  assert params.values["UsbGpuReady"] is True


def test_unstable_model_load_link_sample_requires_link_validation(monkeypatch):
  params = FakeParams()
  amd_model = FakeModel()
  wait_calls = []
  monitor = FakeLinkMonitor(False, 6.0)

  monkeypatch.setattr(modeld, "wait_for_usbgpu_ready", lambda *args, **kwargs: wait_calls.append((args, kwargs)) or True)
  monkeypatch.setattr(modeld, "usbgpu_present", lambda: True)
  monkeypatch.setattr(modeld, "configure_usbgpu_runtime", lambda model: None)

  usbgpu, fallback_state, reason, traceback_text = modeld.finish_usbgpu_startup(amd_model, 1928, 1208, params, monitor)

  assert usbgpu
  assert fallback_state is None
  assert reason is None
  assert traceback_text is None
  assert wait_calls == [((modeld.USBGPU_READY_TIMEOUT,), {
    "stable_duration": modeld.USBGPU_STABLE_DURATION,
    "monitor": monitor,
  })]


def test_runtime_timeout_configuration_failure_switches_to_qcom(monkeypatch):
  params = FakeParams()
  amd_model = FakeModel()
  marked = []

  monkeypatch.setattr(modeld, "wait_for_usbgpu_ready", lambda *args, **kwargs: True)
  monkeypatch.setattr(modeld, "usbgpu_present", lambda: True)
  monkeypatch.setattr(modeld, "configure_usbgpu_runtime", lambda model: (_ for _ in ()).throw(RuntimeError("failed")))
  monkeypatch.setattr(modeld, "mark_usbgpu_device_lost", lambda model=None, error=None: marked.append(model))
  monkeypatch.setattr(modeld, "ModelState", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("QCOM loaded too early")))

  usbgpu, fallback_state, reason, traceback_text = modeld.finish_usbgpu_startup(amd_model, 1928, 1208, params)

  assert not usbgpu
  assert fallback_state is not None
  assert reason is not None
  assert traceback_text is not None and "RuntimeError: failed" in traceback_text
  assert marked == [amd_model]
  assert params.values["UsbGpuReady"] is False
  assert "Failed to configure AMD runtime" in params.values["UsbGpuInitError"]


def test_switch_to_qcom_preserves_delay_and_marks_amd_lost(monkeypatch):
  amd_model = FakeModel()
  qcom_model = FakeModel()
  qcom_model.lat_delay = 0.0
  marked = []

  monkeypatch.setattr(modeld, "mark_usbgpu_device_lost", lambda model=None, error=None: marked.append(model))
  monkeypatch.setattr(modeld, "ModelState", lambda cam_w, cam_h, usbgpu: qcom_model)

  assert modeld.switch_to_qcom(amd_model, 1928, 1208) is qcom_model
  assert marked == [amd_model]
  assert qcom_model.lat_delay == 0.25
  assert np.array_equal(qcom_model.prev_desire, amd_model.prev_desire)
  assert np.array_equal(qcom_model.npy["prev_feat"], amd_model.npy["prev_feat"])


def test_prepare_fallback_marks_and_clears_ready_without_loading_qcom(monkeypatch):
  params = FakeParams()
  amd_model = FakeModel()
  events = []

  monkeypatch.setattr(modeld, "mark_usbgpu_device_lost", lambda model=None, error=None: events.append("mark"))
  monkeypatch.setattr(modeld, "record_usbgpu_failure", lambda params, reason: events.append("record"))
  monkeypatch.setattr(modeld, "ModelState", lambda *args, **kwargs: events.append("qcom"))

  fallback_state, reason = modeld.prepare_usbgpu_fallback(amd_model, params, RuntimeError("lost"))

  assert fallback_state is not None
  assert reason == "unknown: RuntimeError: lost"
  assert events == ["mark", "record"]


def test_runtime_catch_discards_failed_frame_before_diagnostics():
  source = inspect.getsource(modeld.main)
  runtime_catch = source[source.index("except Exception as e:", source.index("model.run(bufs")):]

  disable = runtime_catch.index("USBGPU = False")
  fallback = runtime_catch.index("prepare_usbgpu_fallback")
  discard = runtime_catch.index("continue", fallback)
  next_timing = runtime_catch.index("mt2 =", fallback)
  diagnostics = runtime_catch.index("schedule_egpu_diagnostics")
  published = runtime_catch.index("pm.send('modelDataV2SP'")

  assert disable < fallback < discard < next_timing < published < diagnostics
  assert "model.run(bufs" not in runtime_catch[fallback:next_timing]


def test_post_init_always_starts_fresh_stability_window(monkeypatch):
  params = FakeParams()
  amd_model = FakeModel()
  monitor = FakeLinkMonitor(True, 1.0, identity=("/sys/4-2", 2, 5000, 1))

  monkeypatch.setattr(modeld, "wait_for_usbgpu_ready", lambda *args, **kwargs: True)
  monkeypatch.setattr(modeld, "usbgpu_present", lambda: True)
  monkeypatch.setattr(modeld, "configure_usbgpu_runtime", lambda model: None)

  usbgpu, fallback_state, reason, traceback_text = modeld.finish_usbgpu_startup(amd_model, 1928, 1208, params, monitor)

  assert usbgpu
  assert fallback_state is None
  assert reason is None
  assert traceback_text is None
  assert monitor.resets == 1


def test_startup_qcom_constructor_runs_after_amd_reference_is_released(monkeypatch):
  params = FakeParams()
  amd_model = FakeModel()
  amd_ref = weakref.ref(amd_model)
  qcom_model = FakeModel()

  monkeypatch.setattr(modeld, "wait_for_usbgpu_ready", lambda *args, **kwargs: False)
  monkeypatch.setattr(modeld, "usbgpu_present", lambda: True)
  monkeypatch.setattr(modeld, "mark_usbgpu_device_lost", lambda model=None, error=None: None)

  usbgpu, fallback_state, _, _ = modeld.finish_usbgpu_startup(amd_model, 1928, 1208, params)
  assert not usbgpu
  amd_model = None
  gc.collect()
  assert amd_ref() is None

  def load_qcom(cam_w, cam_h, usbgpu):
    assert amd_ref() is None
    return qcom_model

  monkeypatch.setattr(modeld, "ModelState", load_qcom)
  assert modeld.switch_to_qcom(None, 1928, 1208, mark_lost=False, fallback_state=fallback_state) is qcom_model


def test_main_releases_startup_amd_before_loading_qcom():
  source = inspect.getsource(modeld.main)
  startup = source.index("USBGPU, startup_fallback_state")
  release = source.index("model = None", startup)
  collect = source.index("gc.collect()", release)
  load = source.index("switch_to_qcom", collect)

  assert startup < release < collect < load


def test_async_diagnostics_uses_captured_traceback_without_original_exception(monkeypatch):
  captured = {}

  class ImmediateThread:
    def __init__(self, target, **kwargs):
      self.target = target

    def start(self):
      self.target()

  def log_diagnostics(error, traceback_text=None):
    captured["error"] = error
    captured["traceback"] = traceback_text
    return "/tmp/diagnostic.json"

  monkeypatch.setattr(modeld.threading, "Thread", ImmediateThread)
  monkeypatch.setattr(modeld, "log_egpu_diagnostics", log_diagnostics)

  modeld.schedule_egpu_diagnostics("hardware: failure", "Traceback: original\n")

  assert type(captured["error"]) is RuntimeError
  assert captured["error"].__traceback__ is None
  assert captured["traceback"] == "Traceback: original\n"
  source = inspect.getsource(modeld.schedule_egpu_diagnostics)
  assert "Params" not in source
  assert "record_usbgpu_failure" not in source


def test_incomplete_usbgpu_manifest_is_not_compiled(monkeypatch, tmp_path):
  model_path = tmp_path / "big_driving_tinygrad.pkl"
  (tmp_path / "big_driving_tinygrad.pkl.chunkmanifest").write_text("2\n")
  (tmp_path / "big_driving_tinygrad.pkl.chunk01of02").write_bytes(b"partial")
  monkeypatch.setattr(modeld, "modeld_pkl_path", lambda usbgpu: model_path)

  assert not modeld.usbgpu_model_compiled()


@pytest.mark.parametrize(("speed", "eligible"), ((None, False), (0, False), (12, True), (480, False), (5000, True), (10000, True)))
def test_usbgpu_speed_eligibility(speed, eligible):
  assert modeld.usbgpu_speed_eligible(speed) is eligible


def test_main_requires_eligible_link_speed_for_amd():
  source = inspect.getsource(modeld.main)
  assert "USBGPU = _present and _compiled and _speed_eligible" in source


def test_runtime_fallback_uses_snapshot_from_before_failed_run(monkeypatch):
  params = FakeParams()
  amd_model = FakeModel()
  previous_desire = amd_model.prev_desire.copy()
  previous_feat = amd_model.npy["prev_feat"].copy()
  fallback_state = modeld._capture_qcom_fallback_state(amd_model)

  amd_model.prev_desire[:] = -1
  amd_model.npy["prev_feat"][:] = -2
  monkeypatch.setattr(modeld, "mark_usbgpu_device_lost", lambda model=None, error=None: None)
  monkeypatch.setattr(modeld, "usbgpu_present", lambda: True)

  state, _ = modeld.prepare_usbgpu_fallback(
    amd_model, params, RuntimeError("failed after mutation"), fallback_state=fallback_state,
  )

  assert np.array_equal(state["prev_desire"], previous_desire)
  assert np.array_equal(state["prev_feat"], previous_feat)
  source = inspect.getsource(modeld.main)
  run = source.index("model_output = model.run(bufs")
  snapshot = source.rindex("_capture_qcom_fallback_state", 0, run)
  fallback = source.index("fallback_state=usbgpu_fallback_state", run)
  assert snapshot < run < fallback
