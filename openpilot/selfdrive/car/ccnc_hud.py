"""Fresh model snapshots for the Hyundai ccNC display; no control inputs."""
import json
import os
import stat

from opendbc.car.hyundai.ccnc_model import read_model_lanes
from opendbc.car.hyundai.ccnc_objects import read_object_lanes, read_radar_objects

CCNC_SLOT_PROBE_PATH = '/dev/shm/ccnc-slot-probe.json'


def _unique_object(pairs):
  result = {}
  for key, value in pairs:
    if key in result:
      raise ValueError('duplicate request key')
    result[key] = value
  return result


def _read_probe_request(path):
  fd = None
  try:
    # Never block card on a FIFO or follow a substituted symlink.
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    info = os.fstat(fd)
    if (not stat.S_ISREG(info.st_mode) or info.st_size > 1024 or
        info.st_uid not in (0, os.getuid()) or info.st_mode & 0o022):
      return None
    raw = os.read(fd, 1025)
    if len(raw) > 1024:
      return None
    request = json.loads(raw.decode('utf-8'), object_pairs_hook=_unique_object)
    return request if type(request) is dict else None
  except (OSError, UnicodeError, ValueError):
    return None
  finally:
    if fd is not None:
      os.close(fd)


def update_ccnc_probe(controller, enabled=True, path=CCNC_SLOT_PROBE_PATH):
  if hasattr(controller, 'ccnc_probe_request'):
    # Replay must never consume a live operator request. The controller checks
    # gear, speed, control state and the camera frame age on every apply call.
    controller.ccnc_probe_request = _read_probe_request(path) if enabled else None
    if not enabled and hasattr(controller, 'ccnc_probe'):
      controller.ccnc_probe.cancel()


def update_ccnc_model(controller, model, valid, model_time_ns, now_ns):
  if not hasattr(controller, "ccnc_model"):
    return
  # Use the message's monotonic timestamp, including in replay. A held SubMaster
  # value must not keep a stale lane-change state or green highlight alive.
  fresh = valid and 0 <= now_ns - model_time_ns <= 250_000_000
  controller.ccnc_model = read_model_lanes(model, model_time_ns * 1e-9) if fresh else None
  if hasattr(controller, "ccnc_object_lanes"):
    controller.ccnc_object_lanes = read_object_lanes(model, model_time_ns * 1e-9) if fresh else None


def update_ccnc_radar(controller, radar, valid, now_ns, can_packets=()):
  if not hasattr(controller, "ccnc_radar"):
    return
  raw_reader = getattr(controller, "ccnc_raw_radar", None)
  if not valid:
    controller.ccnc_radar = None
    if raw_reader is not None:
      raw_reader.reset()
  elif raw_reader is not None:
    controller.ccnc_radar = raw_reader.update(can_packets, now_ns * 1e-9)
  elif radar is not None:
    # None means no new radar cycle, whereas an empty points list means all
    # objects disappeared. The display independently expires held snapshots.
    controller.ccnc_radar = read_radar_objects(radar, now_ns * 1e-9)
