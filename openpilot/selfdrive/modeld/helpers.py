import io
import json
import os
import pickle
import shutil
import struct
import tempfile
import threading
import time
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / 'models'
TG_INPUT_DEVICES_PATH = MODELS_DIR / 'tg_input_devices.json'
USBGPU_VID = 0xADD1
USBGPU_PID = 0x0001
USBGPU_SYSFS_ROOT = Path("/sys/bus/usb/devices")
USBGPU_SUPERSPEED_MBIT = 5000
USBGPU_LINK_ERROR_THRESHOLD = 5.0
USBGPU_LINK_ERROR_COUNTER_MASK = 0xFFFF
USBGPU_LINK_SAMPLE_SECONDS = 2.0
USBGPU_LINK_MIN_GAP_TOLERANCE = 0.5
USBGPU_IGNITION_LOCKOUT_PATH = Path("/tmp/openpilot_usbgpu_ignition_lockout")

UsbGpuReadyIdentity = tuple[str, int, int, int]


def usbgpu_ignition_lockout_reason(path: Path | None = None) -> str | None:
  marker = path or USBGPU_IGNITION_LOCKOUT_PATH
  try:
    return marker.read_text().strip() or "USB GPU session failed earlier in this ignition"
  except FileNotFoundError:
    return None
  except OSError as e:
    # An unreadable existing marker must fail closed for the current ignition.
    return f"USB GPU ignition lockout marker unreadable: {e}"


def set_usbgpu_ignition_lockout(reason: str, path: Path | None = None) -> None:
  marker = path or USBGPU_IGNITION_LOCKOUT_PATH
  marker.parent.mkdir(parents=True, exist_ok=True)
  tmp_path = None
  try:
    with tempfile.NamedTemporaryFile("w", dir=marker.parent, prefix=f".{marker.name}.", delete=False) as f:
      tmp_path = Path(f.name)
      f.write(reason.rstrip() + "\n")
    os.replace(tmp_path, marker)
  finally:
    if tmp_path is not None:
      tmp_path.unlink(missing_ok=True)


def clear_usbgpu_ignition_lockout(path: Path | None = None) -> None:
  try:
    (path or USBGPU_IGNITION_LOCKOUT_PATH).unlink()
  except FileNotFoundError:
    pass


def get_tg_input_devices(process_name: str, usbgpu: bool):
  with open(TG_INPUT_DEVICES_PATH) as f:
    return json.load(f)[process_name]['default' if not usbgpu else 'usbgpu']

def modeld_pkl_path(usbgpu: bool):
  prefix = 'big_' if usbgpu else ''
  return MODELS_DIR / f'{prefix}driving_tinygrad.pkl'

def dump_oob(obj, f):
  with tempfile.TemporaryFile(dir=".") as tmp:
    def buffer_callback(pb: pickle.PickleBuffer):
      m = pb.raw()
      tmp.write(struct.pack('<q', m.nbytes))
      tmp.write(m)
      pb.release() # keep peak ram at ~1 buffer
    stream = io.BytesIO()
    pickle.Pickler(stream, protocol=5, buffer_callback=buffer_callback).dump(obj)
    opcodes = stream.getvalue()
    f.write(struct.pack('<q', len(opcodes)))
    f.write(opcodes)
    tmp.seek(0)
    shutil.copyfileobj(tmp, f)

def load_oob(f):
  opcodes = f.read(struct.unpack('<q', f.read(8))[0])
  def buffers():
    prev = None
    while (h := f.read(8)):
      if prev is not None:
        prev.release()
      buf = bytearray(struct.unpack('<q', h)[0])
      f.readinto(buf)
      prev = pickle.PickleBuffer(buf)
      yield prev
  return pickle.load(io.BytesIO(opcodes), buffers=buffers())

def _usbgpu_devices(sysfs_root: Path = USBGPU_SYSFS_ROOT) -> list[Path]:
  devices = []
  for d in sysfs_root.glob("*"):
    try:
      if int((d / "idVendor").read_text(), 16) == USBGPU_VID and \
          int((d / "idProduct").read_text(), 16) == USBGPU_PID:
        devices.append(d)
    except (OSError, ValueError):
      pass
  return devices

def usbgpu_present(sysfs_root: Path = USBGPU_SYSFS_ROOT) -> bool:
  return bool(_usbgpu_devices(sysfs_root))

def wait_for_usbgpu_present(timeout: float, poll_interval: float = 0.1, sysfs_root: Path = USBGPU_SYSFS_ROOT) -> bool:
  deadline = time.monotonic() + timeout
  while not usbgpu_present(sysfs_root):
    if time.monotonic() >= deadline:
      return False
    time.sleep(poll_interval)
  return True

def usbgpu_speed(sysfs_root: Path = USBGPU_SYSFS_ROOT) -> int | None:
  speeds = []
  for d in _usbgpu_devices(sysfs_root):
    try:
      speeds.append(int(float((d / "speed").read_text().strip())))
    except (OSError, ValueError):
      pass
  return max(speeds, default=None)


def usbgpu_ready_identity(sysfs_root: Path = USBGPU_SYSFS_ROOT) -> UsbGpuReadyIdentity | None:
  identities = []
  for d in _usbgpu_devices(sysfs_root):
    try:
      speed = int(float((d / "speed").read_text().strip()))
      configuration = int((d / "bConfigurationValue").read_text().strip())
      devnum = int((d / "devnum").read_text().strip())
      if speed >= USBGPU_SUPERSPEED_MBIT and configuration > 0:
        identities.append((str(d), devnum, speed, configuration))
    except (OSError, ValueError):
      pass
  return min(identities, default=None)


def usbgpu_superspeed_ready(sysfs_root: Path = USBGPU_SYSFS_ROOT) -> bool:
  return usbgpu_ready_identity(sysfs_root) is not None


def _usbgpu_controller(device: Path) -> Path | None:
  try:
    return next((parent for parent in device.resolve().parents if parent.name.endswith(".ssusb")), None)
  except OSError:
    return None


def _usbgpu_portli(identity: UsbGpuReadyIdentity) -> Path | None:
  ctrl = _usbgpu_controller(Path(identity[0]))
  if ctrl is None:
    return None
  portli = ctrl / "portli"
  return portli if portli.exists() else None


def _usbgpu_link_error_count(portli: Path) -> int | None:
  try:
    return int(portli.read_text().strip(), 0) & USBGPU_LINK_ERROR_COUNTER_MASK
  except (OSError, ValueError):
    return None


class UsbGpuLinkMonitor:
  def __init__(self, sysfs_root: Path = USBGPU_SYSFS_ROOT, poll_interval: float = 0.1):
    self.sysfs_root = sysfs_root
    self.poll_interval = poll_interval
    self._lock = threading.Lock()
    self._sample_lock = threading.Lock()
    self._stop_event = threading.Event()
    self._thread = None
    self._identity = None
    self._stable_since = None
    self._link_errors_required = False
    self._link_samples = []

  def _reset_locked(self):
    self._identity = None
    self._stable_since = None
    self._link_errors_required = False
    self._link_samples = []

  def sample(self, blocking: bool = True) -> bool:
    if not self._sample_lock.acquire(blocking=blocking):
      return False
    try:
      identity = usbgpu_ready_identity(self.sysfs_root)
      portli = _usbgpu_portli(identity) if identity is not None else None
      link_errors = _usbgpu_link_error_count(portli) if portli is not None else None

      # Do not accept a stale controller counter after detach or re-enumeration.
      if identity is not None and usbgpu_ready_identity(self.sysfs_root) != identity:
        identity = None
        portli = None
        link_errors = None
      now = time.monotonic()

      with self._lock:
        if identity is None:
          self._reset_locked()
          return True

        if identity != self._identity:
          self._identity = identity
          self._stable_since = now
          self._link_errors_required = portli is not None
          self._link_samples = []
        elif portli is not None:
          self._link_errors_required = True

        if self._link_errors_required:
          if link_errors is None:
            self._link_samples = []
          else:
            self._link_samples.append((now, link_errors))
            cutoff = now - USBGPU_LINK_SAMPLE_SECONDS
            while len(self._link_samples) > 2 and self._link_samples[1][0] <= cutoff:
              self._link_samples.pop(0)
      return True
    finally:
      self._sample_lock.release()

  def _link_error_rate_locked(self, now: float) -> float | None:
    if len(self._link_samples) < 2:
      return None
    started, _ = self._link_samples[0]
    ended, _ = self._link_samples[-1]
    elapsed = ended - started
    max_gap = max(USBGPU_LINK_MIN_GAP_TOLERANCE, self.poll_interval * 1.5)
    if elapsed < USBGPU_LINK_SAMPLE_SECONDS or elapsed > USBGPU_LINK_SAMPLE_SECONDS + max_gap or now - ended > max_gap:
      return None
    total_delta = 0
    for index in range(1, len(self._link_samples)):
      previous_time, previous_count = self._link_samples[index - 1]
      current_time, current_count = self._link_samples[index]
      if current_time - previous_time > max_gap:
        return None
      total_delta += (current_count - previous_count) & USBGPU_LINK_ERROR_COUNTER_MASK
    return total_delta / elapsed

  def status(self, stable_duration: float) -> tuple[bool, float | None]:
    now = time.monotonic()
    with self._lock:
      if self._identity is None or self._stable_since is None or now - self._stable_since < stable_duration:
        return False, None
      if stable_duration <= 0.0 or not self._link_errors_required:
        return True, None
      rate = self._link_error_rate_locked(now)
      return rate is not None and rate <= USBGPU_LINK_ERROR_THRESHOLD, rate

  def current_identity(self) -> UsbGpuReadyIdentity | None:
    with self._lock:
      return self._identity

  def reset_stability(self) -> None:
    """Start a fresh post-init window without touching the USB device."""
    with self._sample_lock, self._lock:
      if self._identity is None:
        self._reset_locked()
        return
      now = time.monotonic()
      self._stable_since = now
      self._link_samples = [(now, self._link_samples[-1][1])] if self._link_samples else []

  def _run(self):
    while not self._stop_event.wait(self.poll_interval):
      self.sample()

  def start(self):
    self.sample()
    self._thread = threading.Thread(target=self._run, name="usbgpu-link-monitor", daemon=True)
    self._thread.start()

  def stop(self):
    self._stop_event.set()
    if self._thread is not None:
      self._thread.join(timeout=max(1.0, self.poll_interval * 2))

def wait_for_usbgpu_ready(timeout: float, stable_duration: float = 0.0, poll_interval: float = 0.1,
                          sysfs_root: Path = USBGPU_SYSFS_ROOT, monitor: UsbGpuLinkMonitor | None = None) -> bool:
  deadline = time.monotonic() + timeout
  link_monitor = monitor or UsbGpuLinkMonitor(sysfs_root, poll_interval)
  while True:
    link_monitor.sample(blocking=False)
    ready, _ = link_monitor.status(stable_duration)
    if ready:
      return True
    if time.monotonic() >= deadline:
      return False
    time.sleep(poll_interval)
