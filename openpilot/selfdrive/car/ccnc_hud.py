"""Fresh model snapshots for the Hyundai ccNC display; no control inputs."""
from opendbc.car.hyundai.ccnc_model import read_model_lanes
from opendbc.car.hyundai.ccnc_objects import read_object_lanes, read_radar_objects


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
