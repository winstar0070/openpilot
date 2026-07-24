#!/usr/bin/env python3
import gc
import json
import os
os.environ['GMMU'] = '0' # for usbgpu fast loading, noop for qcom
from tinygrad.tensor import Tensor
from tinygrad.runtime.support.usb import USBDeviceSessionLost
import threading
import time
import traceback
import numpy as np
import openpilot.cereal.messaging as messaging
from openpilot.cereal import log
from opendbc.car.structs import car
from openpilot.cereal.messaging import PubMaster, SubMaster
from msgq.visionipc import VisionIpcClient, VisionStreamType, VisionBuf
from opendbc.car.car_helpers import get_demo_car_params
from openpilot.common.swaglog import cloudlog
from openpilot.common.params import Params
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import config_realtime_process, DT_MDL
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
from openpilot.common.transformations.model import get_warp_matrix
from openpilot.selfdrive.controls.lib.desire_helper import DesireHelper
from openpilot.selfdrive.controls.lib.drive_helpers import get_accel_from_plan, smooth_value, get_curvature_from_plan
from openpilot.selfdrive.modeld.parse_model_outputs import Parser
from openpilot.selfdrive.modeld.compile_modeld import make_input_queues, WARP_INPUTS, POLICY_INPUTS
from openpilot.selfdrive.modeld.fill_model_msg import fill_model_msg, fill_driving_model_data, fill_pose_msg, PublishState
from openpilot.common.file_chunker import open_file_chunked, get_manifest_path
from openpilot.selfdrive.modeld.constants import ModelConstants, Plan
from openpilot.selfdrive.modeld.egpu_diagnostics import collect_egpu_diagnostics
from openpilot.selfdrive.modeld.helpers import USBGPU_SUPERSPEED_MBIT, usbgpu_present, usbgpu_ready_identity, usbgpu_speed, usbgpu_speed_eligible
from openpilot.selfdrive.modeld.helpers import wait_for_usbgpu_present, wait_for_usbgpu_ready
from openpilot.selfdrive.modeld.helpers import modeld_pkl_path, get_tg_input_devices, load_oob
from openpilot.selfdrive.modeld.usbgpu_link import wait_usbgpu_link

from openpilot.sunnypilot.livedelay.helpers import get_lat_delay
from openpilot.sunnypilot.modeld_v2.modeld_base import ModelStateBase

PROCESS_NAME = "openpilot.selfdrive.modeld.modeld"
SEND_RAW_PRED = os.getenv('SEND_RAW_PRED')
USBGPU_ATTACH_TIMEOUT = float(os.getenv("USBGPU_ATTACH_TIMEOUT_SEC", "2"))
USBGPU_READY_TIMEOUT = float(os.getenv("USBGPU_READY_TIMEOUT_SEC", "5"))
USBGPU_STABLE_DURATION = float(os.getenv("USBGPU_STABLE_DURATION_SEC", "2"))
USBGPU_RUNTIME_HCQ_TIMEOUT_MS = int(os.getenv("USBGPU_RUNTIME_HCQ_TIMEOUT_MS", "2000"))
USBGPU_RUNTIME_WATCHDOG_INTERVAL = float(os.getenv("USBGPU_RUNTIME_WATCHDOG_INTERVAL_SEC", "0.1"))
USBGPU_OUTPUT_VALIDATION_KEYS = ("plan", "pose", "meta", "hidden_state")

USBGPU_REENUMERATION_TIMEOUT_MESSAGES = {
  "ASM24 USB device did not re-enumerate before timeout",
  "ASM24 USB device did not remain stable at SuperSpeed",
}

LAT_SMOOTH_SECONDS = 0.0
LONG_SMOOTH_SECONDS = 0.3
MIN_LAT_CONTROL_SPEED = 0.3


def log_egpu_diagnostics(error: BaseException, traceback_text: str | None = None):
  diagnostic_path = None
  try:
    diagnostic_path, diagnostics = collect_egpu_diagnostics(error, traceback_text=traceback_text)
    diagnostic_summary = {
      "path": str(diagnostic_path) if diagnostic_path is not None else None,
      "egpuDevices": diagnostics.get("egpuDevices", []),
      "xhci": diagnostics.get("xhci", {}),
      "systemPower": diagnostics.get("systemPower", {}),
      "kernelLogError": diagnostics.get("kernelLog", {}).get("error"),
      "saveError": diagnostics.get("saveError"),
    }
    cloudlog.error(f"USB GPU diagnostics: {json.dumps(diagnostic_summary, separators=(',', ':'))}")
  except Exception:
    cloudlog.exception("USB GPU diagnostic collection failed")
  return diagnostic_path


def usbgpu_model_compiled() -> bool:
  return os.path.isfile(get_manifest_path(modeld_pkl_path(usbgpu=True)))


def record_modeld_runtime(params: Params, model, usbgpu: bool, usbgpu_state: str) -> None:
  backend = "USB+AMD" if usbgpu else str(getattr(model, "QUEUE_DEV", "QCOM"))
  params.put("ModeldBackend", backend)
  params.put("ModeldModel", modeld_pkl_path(usbgpu).name)
  params.put("UsbGpuState", usbgpu_state)


def validate_usbgpu_model_output(model_output: dict[str, np.ndarray] | None) -> None:
  if not isinstance(model_output, dict):
    raise RuntimeError("USB GPU first inference returned no model output")

  for key in USBGPU_OUTPUT_VALIDATION_KEYS:
    value = model_output.get(key)
    if not isinstance(value, np.ndarray) or value.size == 0:
      raise RuntimeError(f"USB GPU first inference output {key!r} is missing or empty")
    if not np.isfinite(value).all():
      raise RuntimeError(f"USB GPU first inference output {key!r} contains non-finite values")


def _exception_chain(error: BaseException):
  seen = set()
  current: BaseException | None = error
  while current is not None and id(current) not in seen:
    seen.add(id(current))
    yield current
    current = current.__cause__ if current.__cause__ is not None else current.__context__


def _is_libusb_no_device(error: BaseException) -> bool:
  if type(error).__name__ != "USBError":
    return False
  return any(getattr(error, attr, None) == -4 for attr in ("errno", "code", "value"))


def is_recognized_usbgpu_error(error: BaseException) -> bool:
  for cause in _exception_chain(error):
    if isinstance(cause, USBDeviceSessionLost):
      return True
    if str(cause) in USBGPU_REENUMERATION_TIMEOUT_MESSAGES:
      return True
    if _is_libusb_no_device(cause):
      return True
  return False


def _opened_usbgpu_device(model=None):
  from tinygrad.device import Device
  device_name = Device.canonicalize(getattr(model, "QUEUE_DEV", "AMD"))
  return Device[device_name] if device_name in Device._opened_devices else None


def mark_usbgpu_device_lost(model=None, error: Exception | None = None) -> None:
  device = _opened_usbgpu_device(model)
  if device is None:
    return
  marker = getattr(device, "mark_device_lost", None)
  if not callable(marker):
    raise RuntimeError("opened AMD device cannot be marked lost")
  marker(error or RuntimeError("USB GPU session abandoned by modeld"))


def configure_usbgpu_runtime(model) -> None:
  device = _opened_usbgpu_device(model)
  if device is None:
    raise RuntimeError("initialized AMD device is not registered")
  device.wait_timeout_ms = USBGPU_RUNTIME_HCQ_TIMEOUT_MS


def record_usbgpu_failure(params: Params, error: str, present: bool | None = None) -> None:
  params.put_bool("UsbGpuPresent", usbgpu_present() if present is None else present)
  params.put_bool("UsbGpuReady", False)
  params.put("UsbGpuState", "lost")
  params.put("UsbGpuInitError", error)


def select_usbgpu(params: Params) -> tuple[bool, str | None]:
  compiled = usbgpu_model_compiled()
  prior_error = params.get("UsbGpuInitError")
  new_failure = None
  detection_started = time.monotonic()
  present = usbgpu_present()
  if compiled and not present and prior_error is None:
    cloudlog.warning(f"USB GPU model compiled; waiting up to {USBGPU_ATTACH_TIMEOUT:.1f}s for attachment")
    present = wait_for_usbgpu_present(USBGPU_ATTACH_TIMEOUT)
  detection_time = time.monotonic() - detection_started
  speed = usbgpu_speed() if present else None
  use_usbgpu = compiled and present and usbgpu_speed_eligible(speed) and prior_error is None

  params.put_bool("UsbGpuPresent", present)
  params.put_bool("UsbGpuCompiled", compiled)
  params.put_bool("UsbGpuReady", False)
  if prior_error is not None:
    params.put("UsbGpuState", "locked_out")
    cloudlog.error(f"USB GPU disabled for current onroad manager cycle after prior failure: {prior_error}")
  elif use_usbgpu:
    params.put("UsbGpuState", "initializing")
    params.remove("UsbGpuInitError")
    device_kind = "SuperSpeed" if speed == USBGPU_SUPERSPEED_MBIT else "bootstrap"
    cloudlog.warning(f"USB GPU detected after {detection_time:.1f}s at {speed}M; initializing {device_kind} device")
  elif compiled and present:
    error = f"USB GPU detected at unsupported {speed}M link speed; expected bootstrap 12M or SuperSpeed"
    record_usbgpu_failure(params, error, present=present)
    params.put("UsbGpuState", "unsupported")
    cloudlog.error(f"{error}; falling back to QCOM")
    new_failure = error
  elif compiled:
    error = f"USB GPU not detected within {USBGPU_ATTACH_TIMEOUT:.1f}s"
    record_usbgpu_failure(params, error, present=present)
    params.put("UsbGpuState", "absent")
    cloudlog.error(f"{error}; falling back to QCOM")
    new_failure = error
  else:
    params.put("UsbGpuState", "unavailable")
    params.remove("UsbGpuInitError")
  return use_usbgpu, new_failure


def _capture_qcom_fallback_state(model) -> dict[str, object]:
  state: dict[str, object] = {"lat_delay": getattr(model, "lat_delay", None)}
  if isinstance(prev_desire := getattr(model, "prev_desire", None), np.ndarray):
    state["prev_desire"] = prev_desire.copy()
  npy = getattr(model, "npy", None)
  if isinstance(npy, dict) and isinstance(prev_feat := npy.get("prev_feat"), np.ndarray):
    state["prev_feat"] = prev_feat.copy()
  return state


def _restore_qcom_fallback_state(model, state: dict[str, object]) -> None:
  if (lat_delay := state.get("lat_delay")) is not None:
    model.lat_delay = lat_delay
  npy = getattr(model, "npy", None)
  for name in ("prev_desire", "prev_feat"):
    destination = getattr(model, name, None) if name == "prev_desire" else npy.get(name) if isinstance(npy, dict) else None
    source = state.get(name)
    if isinstance(destination, np.ndarray) and isinstance(source, np.ndarray) and destination.shape == source.shape:
      destination[:] = source


def _format_usbgpu_failure(error: BaseException) -> str:
  return f"{type(error).__name__}: {error}"


def _clear_exception_tracebacks(error: BaseException) -> None:
  for cause in _exception_chain(error):
    cause.__traceback__ = None


def prepare_usbgpu_fallback(model, params: Params, error: Exception,
                            fallback_state: dict[str, object] | None = None) -> tuple[dict[str, object], str]:
  state = fallback_state if fallback_state is not None else _capture_qcom_fallback_state(model)
  reason = _format_usbgpu_failure(error)
  record_usbgpu_failure(params, reason)
  mark_usbgpu_device_lost(model, error)
  return state, reason


def switch_to_qcom(cam_w: int, cam_h: int, fallback_state: dict[str, object]):
  replacement = ModelState(cam_w, cam_h, usbgpu=False)
  _restore_qcom_fallback_state(replacement, fallback_state)
  return replacement


def schedule_egpu_diagnostics(failure_reason: str, traceback_text: str | None = None) -> None:
  def collect() -> None:
    log_egpu_diagnostics(RuntimeError(failure_reason), traceback_text)

  threading.Thread(target=collect, name="egpu-diagnostics", daemon=True).start()


class UsbGpuRuntimeWatchdog:
  def __init__(self, params: Params, poll_interval: float = USBGPU_RUNTIME_WATCHDOG_INTERVAL):
    identity = usbgpu_ready_identity()
    if identity is None:
      raise USBDeviceSessionLost("USB GPU has no validated runtime identity")

    self.params = params
    self.poll_interval = poll_interval
    self.expected_identity = identity
    self._failure_reason = None
    self._lost_event = threading.Event()
    self._stop_event = threading.Event()
    self._thread = None

  @property
  def failure_reason(self) -> str | None:
    return self._failure_reason

  def check(self) -> bool:
    identity = usbgpu_ready_identity()
    if identity == self.expected_identity:
      return True

    if not self._lost_event.is_set():
      self._failure_reason = (
        f"USB GPU runtime identity changed: expected {self.expected_identity!r}, got {identity!r}"
      )
      self.params.put_bool("UsbGpuPresent", usbgpu_present())
      self.params.put_bool("UsbGpuReady", False)
      self.params.put("UsbGpuState", "lost")
      self.params.put("UsbGpuInitError", self._failure_reason)
      self._lost_event.set()
    return False

  def _run(self) -> None:
    while not self._stop_event.wait(self.poll_interval):
      if not self.check():
        return

  def start(self) -> None:
    self._thread = threading.Thread(target=self._run, name="usbgpu-runtime-watchdog", daemon=True)
    self._thread.start()

  def stop(self) -> None:
    self._stop_event.set()
    if self._thread is not None:
      self._thread.join(timeout=max(1.0, self.poll_interval * 2))

  def raise_if_lost(self) -> None:
    if self._lost_event.is_set():
      raise USBDeviceSessionLost(self._failure_reason or "USB GPU runtime identity lost")


def finish_usbgpu_startup(model, params: Params):
  if not wait_for_usbgpu_ready(USBGPU_READY_TIMEOUT, stable_duration=USBGPU_STABLE_DURATION):
    error = RuntimeError("AMD probe succeeded but USB device did not remain configured at SuperSpeed")
    state, reason = prepare_usbgpu_fallback(model, params, error)
    return False, state, reason, None

  try:
    configure_usbgpu_runtime(model)
  except Exception as e:
    if not is_recognized_usbgpu_error(e):
      raise
    failure_traceback = "".join(traceback.format_exception(e))
    state, reason = prepare_usbgpu_fallback(model, params, e)
    _clear_exception_tracebacks(e)
    return False, state, reason, failure_traceback

  params.put_bool("UsbGpuPresent", usbgpu_present())
  params.put_bool("UsbGpuReady", False)
  record_modeld_runtime(params, model, usbgpu=True, usbgpu_state="validating")
  params.remove("UsbGpuInitError")
  return True, None, None, None


def get_action_from_model(model_output: dict[str, np.ndarray], prev_action: log.ModelDataV2.Action,
                          lat_action_t: float, long_action_t: float, v_ego: float) -> log.ModelDataV2.Action:
  if 'action' not in model_output:
    plan = model_output['plan'][0]
    desired_accel, should_stop = get_accel_from_plan(plan[:,Plan.VELOCITY][:,0],
                                                     plan[:,Plan.ACCELERATION][:,0],
                                                     ModelConstants.T_IDXS,
                                                     action_t=long_action_t)
    desired_curvature = get_curvature_from_plan(plan[:,Plan.T_FROM_CURRENT_EULER][:,2],
                                                plan[:,Plan.ORIENTATION_RATE][:,2],
                                                ModelConstants.T_IDXS,
                                                v_ego,
                                                lat_action_t)
  else:
    desired_accel = model_output['action'][0,1]
    desired_curvature = model_output['action'][0,0] / (max(1.0, v_ego))**2
    should_stop = (v_ego < 0.3 and desired_accel < 0.1)
  desired_accel = smooth_value(desired_accel, prev_action.desiredAcceleration, LONG_SMOOTH_SECONDS)
  if v_ego > MIN_LAT_CONTROL_SPEED:
    desired_curvature = smooth_value(desired_curvature, prev_action.desiredCurvature, LAT_SMOOTH_SECONDS)
  else:
    desired_curvature = prev_action.desiredCurvature

  return log.ModelDataV2.Action(desiredCurvature=float(desired_curvature),
                                desiredAcceleration=float(desired_accel),
                                shouldStop=bool(should_stop))


class FrameMeta:
  frame_id: int = 0
  timestamp_sof: int = 0
  timestamp_eof: int = 0

  def __init__(self, vipc=None):
    if vipc is not None:
      self.frame_id, self.timestamp_sof, self.timestamp_eof = vipc.frame_id, vipc.timestamp_sof, vipc.timestamp_eof


class ModelState(ModelStateBase):
  prev_desire: np.ndarray  # for tracking the rising edge of the pulse

  def __init__(self, cam_w: int, cam_h: int, usbgpu: bool):
    ModelStateBase.__init__(self)
    self.LAT_SMOOTH_SECONDS = LAT_SMOOTH_SECONDS
    input_devices = get_tg_input_devices(PROCESS_NAME, usbgpu)
    self.WARP_DEV, self.QUEUE_DEV = input_devices['WARP_DEV'], input_devices['QUEUE_DEV']
    jits = load_oob(open_file_chunked(modeld_pkl_path(usbgpu)))
    metadata = jits['metadata']
    self.input_shapes = metadata['input_shapes']
    self.vision_input_names = [k for k in self.input_shapes if 'img' in k]
    self.output_slices = metadata['output_slices']

    self.prev_desire = np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32)

    self.frame_skip = ModelConstants.MODEL_RUN_FREQ // ModelConstants.MODEL_CONTEXT_FREQ
    self.input_queues, self.npy = make_input_queues(self.input_shapes, self.frame_skip, device=self.QUEUE_DEV)
    self.full_frames: dict[str, Tensor] = {}
    self._blob_cache: dict[tuple[str, int], Tensor] = {}
    self.parser = Parser()
    self.frame_buf_params = {k: get_nv12_info(cam_w, cam_h) for k in ('img', 'big_img')}
    self.run_policy = jits['run_policy']
    self.warp = jits[(cam_w,cam_h)]

  def slice_outputs(self, model_outputs: np.ndarray, output_slices: dict[str, slice]) -> dict[str, np.ndarray]:
    parsed_model_outputs = {k: model_outputs[np.newaxis, v] for k,v in output_slices.items()}
    return parsed_model_outputs

  def run(self, bufs: dict[str, VisionBuf], transforms: dict[str, np.ndarray],
          inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray] | None:
    for key in bufs.keys():
      ptr = np.frombuffer(bufs[key].data, dtype=np.uint8).ctypes.data
      yuv_size = self.frame_buf_params[key][3]
      # There is a ringbuffer of imgs, just cache tensors pointing to all of them
      cache_key = (key, ptr)
      if cache_key not in self._blob_cache:
        self._blob_cache[cache_key] = Tensor.from_blob(ptr, (yuv_size,), dtype='uint8', device=self.WARP_DEV)
      self.full_frames[key] = self._blob_cache[cache_key]

    # Model decides when action is completed, so desire input is just a pulse triggered on rising edge
    inputs['desire_pulse'][0] = 0
    self.npy['desire'][:] = np.where(inputs['desire_pulse'] - self.prev_desire > .99, inputs['desire_pulse'], 0)
    self.prev_desire[:] = inputs['desire_pulse']
    self.npy['traffic_convention'][:] = inputs['traffic_convention']
    self.npy['action_t'][:] = inputs['action_t']
    self.npy['tfm'][:,:] = transforms['img'][:,:]
    self.npy['big_tfm'][:,:] = transforms['big_img'][:,:]

    warped = self.warp(**{k: self.input_queues[k] for k in WARP_INPUTS}, frame=self.full_frames['img'], big_frame=self.full_frames['big_img'])

    outs, = self.run_policy(
      **{k: self.input_queues[k] for k in POLICY_INPUTS if k in self.input_queues}, warped=warped
    )
    model_output = outs.numpy()[0]
    outputs_dict = self.parser.parse_outputs(self.slice_outputs(model_output, self.output_slices))
    self.npy['prev_feat'][:] = model_output[self.output_slices['hidden_state']]

    if SEND_RAW_PRED:
      outputs_dict['raw_pred'] = model_output.copy()
    return outputs_dict


def main(demo=False):
  cloudlog.warning("modeld init")

  params = Params()
  USBGPU, selection_failure = select_usbgpu(params)
  usbgpu_state = "initializing" if USBGPU else (params.get("UsbGpuState") or "unavailable")

  config_realtime_process(7, 54)

  # visionipc clients
  while True:
    available_streams = VisionIpcClient.available_streams("camerad", block=False)
    if available_streams:
      use_extra_client = VisionStreamType.VISION_STREAM_WIDE_ROAD in available_streams and VisionStreamType.VISION_STREAM_ROAD in available_streams
      main_wide_camera = VisionStreamType.VISION_STREAM_ROAD not in available_streams
      break
    time.sleep(.1)

  vipc_client_main_stream = VisionStreamType.VISION_STREAM_WIDE_ROAD if main_wide_camera else VisionStreamType.VISION_STREAM_ROAD
  vipc_client_main = VisionIpcClient("camerad", vipc_client_main_stream, True)
  vipc_client_extra = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_WIDE_ROAD, False)
  cloudlog.warning(f"vision stream set up, main_wide_camera: {main_wide_camera}, use_extra_client: {use_extra_client}")

  while not vipc_client_main.connect(False):
    time.sleep(0.1)
  while use_extra_client and not vipc_client_extra.connect(False):
    time.sleep(0.1)

  cloudlog.warning(f"connected main cam with buffer size: {vipc_client_main.buffer_len} ({vipc_client_main.width} x {vipc_client_main.height})")
  if use_extra_client:
    cloudlog.warning(f"connected extra cam with buffer size: {vipc_client_extra.buffer_len} ({vipc_client_extra.width} x {vipc_client_extra.height})")

  if USBGPU:
    wait_usbgpu_link()
  st = time.monotonic()
  cloudlog.warning("loading model")
  pending_egpu_diagnostic: tuple[str, str | None] | None = (selection_failure, None) if selection_failure is not None else None
  runtime_watchdog: UsbGpuRuntimeWatchdog | None = None
  try:
    model = ModelState(vipc_client_main.width, vipc_client_main.height, USBGPU)
  except Exception as e:
    if not USBGPU or not is_recognized_usbgpu_error(e):
      raise

    USBGPU = False
    usbgpu_state = "fallback"
    failure_traceback = "".join(traceback.format_exception(e))
    cloudlog.exception("USB GPU model initialization failed, falling back to QCOM")
    fallback_state, failure_reason = prepare_usbgpu_fallback(None, params, e)
    _clear_exception_tracebacks(e)
    gc.collect()
    model = switch_to_qcom(vipc_client_main.width, vipc_client_main.height, fallback_state)
    record_modeld_runtime(params, model, usbgpu=False, usbgpu_state=usbgpu_state)
    pending_egpu_diagnostic = (failure_reason, failure_traceback)

  model_load_time = time.monotonic() - st
  link_validation_started = time.monotonic()
  link_validation_status = "skipped"
  if USBGPU:
    USBGPU, startup_fallback_state, startup_failure, startup_traceback = finish_usbgpu_startup(model, params)
    if not USBGPU:
      usbgpu_state = "fallback"
      link_validation_status = "failed"
      assert startup_fallback_state is not None and startup_failure is not None
      model = None
      gc.collect()
      model = switch_to_qcom(vipc_client_main.width, vipc_client_main.height, startup_fallback_state)
      record_modeld_runtime(params, model, usbgpu=False, usbgpu_state=usbgpu_state)
      pending_egpu_diagnostic = (startup_failure, startup_traceback)
    else:
      link_validation_status = "passed; first inference pending"
      try:
        runtime_watchdog = UsbGpuRuntimeWatchdog(params)
        runtime_watchdog.start()
      except USBDeviceSessionLost as e:
        USBGPU = False
        usbgpu_state = "fallback"
        link_validation_status = "failed after validation"
        failure_traceback = "".join(traceback.format_exception(e))
        fallback_state, failure_reason = prepare_usbgpu_fallback(model, params, e)
        _clear_exception_tracebacks(e)
        model = None
        gc.collect()
        model = switch_to_qcom(vipc_client_main.width, vipc_client_main.height, fallback_state)
        record_modeld_runtime(params, model, usbgpu=False, usbgpu_state=usbgpu_state)
        pending_egpu_diagnostic = (failure_reason, failure_traceback)
  else:
    record_modeld_runtime(params, model, usbgpu=False, usbgpu_state=usbgpu_state)
  link_validation_time = time.monotonic() - link_validation_started
  active_backend = "USB+AMD" if USBGPU else str(getattr(model, "QUEUE_DEV", "QCOM"))
  active_model = modeld_pkl_path(USBGPU).name
  cloudlog.warning("; ".join((
    f"models loaded in {model_load_time:.1f}s",
    f"USB GPU link validation={link_validation_status} in {link_validation_time:.1f}s",
    f"backend={active_backend}",
    f"model={active_model}",
    "modeld starting",
  )))

  # messaging
  pm = PubMaster(["modelV2", "drivingModelData", "cameraOdometry", "modelDataV2SP"])
  sm = SubMaster(["deviceState", "carState", "roadCameraState", "liveCalibration", "driverMonitoringState", "carControl", "liveDelay"])

  publish_state = PublishState()
  params = Params()

  # setup filter to track dropped frames
  frame_dropped_filter = FirstOrderFilter(0., 10., 1. / ModelConstants.MODEL_RUN_FREQ)
  frame_id = 0
  last_vipc_frame_id = 0
  run_count = 0
  usbgpu_first_inference_pending = USBGPU

  model_transform_main = np.zeros((3, 3), dtype=np.float32)
  model_transform_extra = np.zeros((3, 3), dtype=np.float32)
  live_calib_seen = False
  buf_main, buf_extra = None, None
  meta_main = FrameMeta()
  meta_extra = FrameMeta()

  if demo:
    CP = get_demo_car_params()
  else:
    CP = messaging.log_from_bytes(params.get("CarParams", block=True), car.CarParams)
  cloudlog.info("modeld got CarParams: %s", CP.brand)

  # TODO this needs more thought, use .2s extra for now to estimate other delays
  # TODO Move smooth seconds to action function
  long_delay = CP.longitudinalActuatorDelay + LONG_SMOOTH_SECONDS
  prev_action = log.ModelDataV2.Action()

  DH = DesireHelper()

  while True:
    # Keep receiving frames until we are at least 1 frame ahead of previous extra frame
    while meta_main.timestamp_sof < meta_extra.timestamp_sof + 25000000:
      buf_main = vipc_client_main.recv()
      meta_main = FrameMeta(vipc_client_main)
      if buf_main is None:
        break

    if buf_main is None:
      cloudlog.debug("vipc_client_main no frame")
      continue

    if use_extra_client:
      # Keep receiving extra frames until frame id matches main camera
      while True:
        buf_extra = vipc_client_extra.recv()
        meta_extra = FrameMeta(vipc_client_extra)
        if buf_extra is None or meta_main.timestamp_sof < meta_extra.timestamp_sof + 25000000:
          break

      if buf_extra is None:
        cloudlog.debug("vipc_client_extra no frame")
        continue

      if abs(meta_main.timestamp_sof - meta_extra.timestamp_sof) > 10000000:
        cloudlog.error(f"frames out of sync! main: {meta_main.frame_id} ({meta_main.timestamp_sof / 1e9:.5f}),\
                         extra: {meta_extra.frame_id} ({meta_extra.timestamp_sof / 1e9:.5f})")

    else:
      # Use single camera
      buf_extra = buf_main
      meta_extra = meta_main

    sm.update(0)
    desire = DH.desire
    is_rhd = sm["driverMonitoringState"].isRHD
    frame_id = sm["roadCameraState"].frameId
    v_ego = max(sm["carState"].vEgo, 0.)
    if sm.frame % 60 == 0:
      model.lat_delay = get_lat_delay(params, sm["liveDelay"].lateralDelay)
    lat_delay = model.lat_delay + LAT_SMOOTH_SECONDS
    if sm.updated["liveCalibration"] and sm.seen['roadCameraState'] and sm.seen['deviceState']:
      device_from_calib_euler = np.array(sm["liveCalibration"].rpyCalib, dtype=np.float32)
      dc = DEVICE_CAMERAS[(str(sm['deviceState'].deviceType), str(sm['roadCameraState'].sensor))]
      model_transform_main = get_warp_matrix(device_from_calib_euler, dc.ecam.intrinsics if main_wide_camera else dc.fcam.intrinsics, False).astype(np.float32)
      has_wide_camera = use_extra_client or main_wide_camera
      model_transform_extra = get_warp_matrix(device_from_calib_euler, dc.ecam.intrinsics if has_wide_camera else dc.fcam.intrinsics, True).astype(np.float32)
      live_calib_seen = True

    traffic_convention = np.zeros(2)
    traffic_convention[int(is_rhd)] = 1

    vec_desire = np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32)
    if desire >= 0 and desire < ModelConstants.DESIRE_LEN:
      vec_desire[desire] = 1

    # tracked dropped frames
    vipc_dropped_frames = max(0, meta_main.frame_id - last_vipc_frame_id - 1)
    frames_dropped = frame_dropped_filter.update(min(vipc_dropped_frames, 10))
    if run_count < 10: # let frame drops warm up
      frame_dropped_filter.x = 0.
      frames_dropped = 0.
    run_count = run_count + 1

    frame_drop_ratio = frames_dropped / (1 + frames_dropped)

    bufs = {name: buf_extra if 'big' in name else buf_main for name in model.vision_input_names}
    transforms = {name: model_transform_extra if 'big' in name else model_transform_main for name in model.vision_input_names}
    frame_delay = DT_MDL # compensate for time passed since the frame was captured: current_time - timestamp_eof is 50ms on average
    action_delay = DT_MDL / 2 # middle of the interval between model output (current state) and next frame (expected state)
    lat_action_t = lat_delay + frame_delay + action_delay
    long_action_t = long_delay + frame_delay + action_delay
    inputs: dict[str, np.ndarray] = {
      'desire_pulse': vec_desire,
      'traffic_convention': traffic_convention,
      'action_t': np.array([lat_action_t, long_action_t], dtype=np.float32),
    }

    # Keep host state from the last successful frame. A failed AMD call may
    # mutate its mirrors before the USB session is marked lost.
    usbgpu_fallback_state = _capture_qcom_fallback_state(model) if USBGPU else None
    mt1 = time.perf_counter()
    try:
      if runtime_watchdog is not None:
        runtime_watchdog.raise_if_lost()
      model_output = model.run(bufs, transforms, inputs)
      if runtime_watchdog is not None and not runtime_watchdog.check():
        runtime_watchdog.raise_if_lost()
      if USBGPU and usbgpu_first_inference_pending:
        validate_usbgpu_model_output(model_output)
    except Exception as e:
      first_inference_failure = USBGPU and usbgpu_first_inference_pending
      if not USBGPU or (not first_inference_failure and not is_recognized_usbgpu_error(e)):
        raise

      USBGPU = False
      usbgpu_first_inference_pending = False
      usbgpu_state = "fallback"
      failure_traceback = "".join(traceback.format_exception(e))
      cloudlog.exception("USB GPU model execution failed, falling back to QCOM")
      if runtime_watchdog is not None:
        runtime_watchdog.stop()
        runtime_watchdog = None
      fallback_state, failure_reason = prepare_usbgpu_fallback(
        model, params, e, fallback_state=usbgpu_fallback_state,
      )
      _clear_exception_tracebacks(e)
      model = None
      gc.collect()
      model = switch_to_qcom(vipc_client_main.width, vipc_client_main.height, fallback_state)
      record_modeld_runtime(params, model, usbgpu=False, usbgpu_state=usbgpu_state)
      pending_egpu_diagnostic = (failure_reason, failure_traceback)
      # Never replay the failed AMD frame or publish its partially-mutated state.
      continue
    mt2 = time.perf_counter()
    model_execution_time = mt2 - mt1

    if USBGPU and usbgpu_first_inference_pending:
      usbgpu_first_inference_pending = False
      params.put_bool("UsbGpuPresent", True)
      params.put_bool("UsbGpuReady", True)
      params.remove("UsbGpuInitError")
      record_modeld_runtime(params, model, usbgpu=True, usbgpu_state="active")
      cloudlog.warning("; ".join((
        f"USB GPU first inference validated in {model_execution_time * 1e3:.1f}ms",
        "backend=USB+AMD",
        f"model={modeld_pkl_path(usbgpu=True).name}",
      )))

    if model_output is not None:
      modelv2_send = messaging.new_message('modelV2')
      drivingdata_send = messaging.new_message('drivingModelData')
      posenet_send = messaging.new_message('cameraOdometry')
      mdv2sp_send = messaging.new_message('modelDataV2SP')

      action = get_action_from_model(model_output, prev_action, lat_action_t, long_action_t, v_ego)
      prev_action = action
      fill_model_msg(modelv2_send, model_output, action,
                     publish_state, meta_main.frame_id, meta_extra.frame_id, frame_id,
                     frame_drop_ratio, meta_main.timestamp_eof, model_execution_time, live_calib_seen)

      desire_state = modelv2_send.modelV2.meta.desireState
      l_lane_change_prob = desire_state[log.Desire.laneChangeLeft]
      r_lane_change_prob = desire_state[log.Desire.laneChangeRight]
      lane_change_prob = l_lane_change_prob + r_lane_change_prob
      DH.update(sm['carState'], sm['carControl'].latActive, lane_change_prob)
      modelv2_send.modelV2.meta.laneChangeState = DH.lane_change_state
      modelv2_send.modelV2.meta.laneChangeDirection = DH.lane_change_direction
      mdv2sp_send.modelDataV2SP.laneTurnDirection = DH.lane_turn_direction

      fill_driving_model_data(drivingdata_send, modelv2_send)
      fill_pose_msg(posenet_send, model_output, meta_main.frame_id, vipc_dropped_frames, meta_main.timestamp_eof, live_calib_seen)
      pm.send('modelV2', modelv2_send)
      pm.send('drivingModelData', drivingdata_send)
      pm.send('cameraOdometry', posenet_send)
      pm.send('modelDataV2SP', mdv2sp_send)
      if pending_egpu_diagnostic is not None:
        schedule_egpu_diagnostics(*pending_egpu_diagnostic)
        pending_egpu_diagnostic = None
    last_vipc_frame_id = meta_main.frame_id


if __name__ == "__main__":
  try:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', help='A boolean for demo mode.')
    args = parser.parse_args()
    main(demo=args.demo)
  except KeyboardInterrupt:
    cloudlog.warning("got SIGINT")
