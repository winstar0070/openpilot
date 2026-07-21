import io
import json
import pickle
import shutil
import struct
import tempfile
import time
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / 'models'
TG_INPUT_DEVICES_PATH = MODELS_DIR / 'tg_input_devices.json'
USBGPU_VID = 0xADD1
USBGPU_PID = 0x0001
USBGPU_SYSFS_ROOT = Path("/sys/bus/usb/devices")
USBGPU_SUPERSPEED_MBIT = 5000


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


def usbgpu_ready_identity(sysfs_root: Path = USBGPU_SYSFS_ROOT) -> tuple[str, int, int, int] | None:
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

def wait_for_usbgpu_ready(timeout: float, stable_duration: float = 0.0, poll_interval: float = 0.1,
                          sysfs_root: Path = USBGPU_SYSFS_ROOT) -> bool:
  deadline = time.monotonic() + timeout
  stable_since = None
  stable_identity = None
  while True:
    now = time.monotonic()
    identity = usbgpu_ready_identity(sysfs_root)
    if identity is not None:
      if stable_since is None or identity != stable_identity:
        stable_since = now
        stable_identity = identity
      if now - stable_since >= stable_duration:
        return True
    else:
      stable_since = None
      stable_identity = None
    if time.monotonic() >= deadline:
      return False
    time.sleep(poll_interval)
