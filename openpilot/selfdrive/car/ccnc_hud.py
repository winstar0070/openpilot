"""Fresh model snapshots for the Hyundai ccNC display; no control inputs."""
from opendbc.car.hyundai.ccnc_model import read_model_lanes


def update_ccnc_model(controller, model, valid, model_time_ns, now_ns):
  if not hasattr(controller, "ccnc_model"):
    return
  # Use the message's monotonic timestamp, including in replay. A held SubMaster
  # value must not keep a stale lane-change state or green highlight alive.
  fresh = valid and 0 <= now_ns - model_time_ns <= 250_000_000
  controller.ccnc_model = read_model_lanes(model, model_time_ns * 1e-9) if fresh else None
