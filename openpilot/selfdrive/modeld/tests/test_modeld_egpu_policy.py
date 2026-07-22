import inspect

from openpilot.selfdrive.modeld import modeld


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
  lat_delay = 0.25


class FakeLinkMonitor:
  def __init__(self, ready: bool, rate: float | None):
    self.ready = ready
    self.rate = rate

  def sample(self, blocking: bool = True):
    pass

  def status(self, stable_duration: float):
    return self.ready, self.rate


def test_ready_failure_switches_to_qcom_before_runtime(monkeypatch):
  params = FakeParams()
  amd_model = FakeModel()
  qcom_model = FakeModel()
  marked = []

  monkeypatch.setattr(modeld, "wait_for_usbgpu_ready", lambda *args, **kwargs: False)
  monkeypatch.setattr(modeld, "usbgpu_present", lambda: True)
  monkeypatch.setattr(modeld, "mark_usbgpu_device_lost", lambda model=None, error=None: marked.append(model))
  monkeypatch.setattr(modeld, "ModelState", lambda cam_w, cam_h, usbgpu: qcom_model if not usbgpu else None)

  model, usbgpu = modeld.finish_usbgpu_startup(amd_model, 1928, 1208, params)

  assert model is qcom_model
  assert not usbgpu
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

  model, usbgpu = modeld.finish_usbgpu_startup(amd_model, 1928, 1208, params)

  assert model is amd_model
  assert usbgpu
  assert configured == [amd_model]
  assert params.values["UsbGpuPresent"] is True
  assert params.values["UsbGpuReady"] is True
  assert "UsbGpuInitError" not in params.values


def test_model_load_link_sample_skips_duplicate_stability_wait(monkeypatch):
  params = FakeParams()
  amd_model = FakeModel()
  configured = []
  monitor = FakeLinkMonitor(True, 1.0)

  monkeypatch.setattr(modeld, "wait_for_usbgpu_ready", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("duplicate wait")))
  monkeypatch.setattr(modeld, "usbgpu_present", lambda: True)
  monkeypatch.setattr(modeld, "configure_usbgpu_runtime", lambda model: configured.append(model))

  model, usbgpu = modeld.finish_usbgpu_startup(amd_model, 1928, 1208, params, monitor)

  assert model is amd_model
  assert usbgpu
  assert configured == [amd_model]
  assert params.values["UsbGpuReady"] is True


def test_unstable_model_load_link_sample_requires_link_validation(monkeypatch):
  params = FakeParams()
  amd_model = FakeModel()
  wait_calls = []
  monitor = FakeLinkMonitor(False, 6.0)

  monkeypatch.setattr(modeld, "wait_for_usbgpu_ready", lambda *args, **kwargs: wait_calls.append((args, kwargs)) or True)
  monkeypatch.setattr(modeld, "usbgpu_present", lambda: True)
  monkeypatch.setattr(modeld, "configure_usbgpu_runtime", lambda model: None)

  model, usbgpu = modeld.finish_usbgpu_startup(amd_model, 1928, 1208, params, monitor)

  assert model is amd_model
  assert usbgpu
  assert wait_calls == [((modeld.USBGPU_READY_TIMEOUT,), {
    "stable_duration": modeld.USBGPU_STABLE_DURATION,
    "monitor": monitor,
  })]


def test_runtime_timeout_configuration_failure_switches_to_qcom(monkeypatch):
  params = FakeParams()
  amd_model = FakeModel()
  qcom_model = FakeModel()
  marked = []

  monkeypatch.setattr(modeld, "wait_for_usbgpu_ready", lambda *args, **kwargs: True)
  monkeypatch.setattr(modeld, "usbgpu_present", lambda: True)
  monkeypatch.setattr(modeld, "configure_usbgpu_runtime", lambda model: (_ for _ in ()).throw(RuntimeError("failed")))
  monkeypatch.setattr(modeld, "mark_usbgpu_device_lost", lambda model=None, error=None: marked.append(model))
  monkeypatch.setattr(modeld, "ModelState", lambda cam_w, cam_h, usbgpu: qcom_model)

  model, usbgpu = modeld.finish_usbgpu_startup(amd_model, 1928, 1208, params)

  assert model is qcom_model
  assert not usbgpu
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


def test_fail_over_marks_and_clears_ready_before_loading_qcom(monkeypatch):
  params = FakeParams()
  amd_model = FakeModel()
  qcom_model = FakeModel()
  events = []

  monkeypatch.setattr(modeld, "mark_usbgpu_device_lost", lambda model=None, error=None: events.append("mark"))
  monkeypatch.setattr(modeld, "record_usbgpu_failure", lambda params, reason: events.append("record"))
  monkeypatch.setattr(modeld, "ModelState", lambda cam_w, cam_h, usbgpu: events.append("qcom") or qcom_model)

  replacement, reason = modeld.fail_over_usbgpu(amd_model, 1928, 1208, params, RuntimeError("lost"))

  assert replacement is qcom_model
  assert reason == "RuntimeError: lost"
  assert events == ["mark", "record", "qcom"]


def test_runtime_catch_disables_amd_before_fallback_and_diagnostics():
  source = inspect.getsource(modeld.main)
  runtime_catch = source[source.index("except Exception as e:", source.index("model.run(bufs")):]

  disable = runtime_catch.index("USBGPU = False")
  fallback = runtime_catch.index("fail_over_usbgpu")
  diagnostics = runtime_catch.index("log_egpu_diagnostics")
  rerun = runtime_catch.index("model.run(bufs", fallback)

  assert disable < fallback < diagnostics < rerun
