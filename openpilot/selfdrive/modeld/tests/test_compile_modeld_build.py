from pathlib import Path

import pytest

import openpilot.selfdrive.modeld.compile_modeld as compile_modeld


class USBDeviceSessionLost(RuntimeError):
  pass


@pytest.mark.parametrize("exc", [
  USBDeviceSessionLost("device disconnected"),
  RuntimeError("TLP completion status: Unsupported Request: 0x10004488"),
  RuntimeError("TLP completion status: Completer Abort"),
  RuntimeError("LIBUSB_ERROR_NO_DEVICE"),
  RuntimeError("LIBUSB_ERROR_IO"),
])
def test_usbgpu_session_failure_reason(exc):
  assert compile_modeld.usbgpu_session_failure_reason(exc) is not None


def test_usbgpu_session_failure_reason_checks_exception_chain():
  cause = USBDeviceSessionLost("device disconnected")
  wrapper = RuntimeError("compile failed")
  wrapper.__cause__ = cause
  assert compile_modeld.usbgpu_session_failure_reason(wrapper) == "USBDeviceSessionLost: device disconnected"


def test_non_hardware_failure_has_no_reason():
  assert compile_modeld.usbgpu_session_failure_reason(ValueError("bad model shape")) is None


def test_unattributed_hcq_timeout_has_no_hardware_reason():
  assert compile_modeld.usbgpu_session_failure_reason(RuntimeError("Wait timeout: 10000 ms!")) is None


def test_dump_oob_atomic_replaces_only_after_success(tmp_path, monkeypatch):
  output = tmp_path / "model.pkl"
  output.write_bytes(b"old")
  monkeypatch.setattr(compile_modeld, "dump_oob", lambda obj, f: f.write(obj))

  compile_modeld.dump_oob_atomic(b"new", str(output))

  assert output.read_bytes() == b"new"
  assert list(tmp_path.glob(".model.pkl.*")) == []


def test_dump_oob_atomic_preserves_output_on_dump_failure(tmp_path, monkeypatch):
  output = tmp_path / "model.pkl"
  output.write_bytes(b"old")

  def fail_dump(obj, f):
    f.write(b"partial")
    raise RuntimeError("dump failed")

  monkeypatch.setattr(compile_modeld, "dump_oob", fail_dump)
  with pytest.raises(RuntimeError, match="dump failed"):
    compile_modeld.dump_oob_atomic({}, str(output))

  assert output.read_bytes() == b"old"
  assert list(tmp_path.glob(".model.pkl.*")) == []


def test_hardware_failure_cleans_outputs_and_writes_marker(tmp_path):
  output = tmp_path / "model.pkl"
  marker = tmp_path / "failure.marker"
  for path in (output, Path(f"{output}.chunkmanifest"), Path(f"{output}.chunk01of02"), Path(f"{output}.chunk02of02")):
    path.write_bytes(b"stale")

  compile_modeld.handle_compile_failure(str(output), str(marker), USBDeviceSessionLost("link lost"))

  assert not output.exists()
  assert not Path(f"{output}.chunkmanifest").exists()
  assert list(tmp_path.glob("model.pkl.chunk*of*")) == []
  assert marker.read_text() == "USBDeviceSessionLost: link lost\n"


def test_general_failure_cleans_outputs_without_marker(tmp_path):
  output = tmp_path / "model.pkl"
  marker = tmp_path / "failure.marker"
  output.write_bytes(b"partial")
  Path(f"{output}.chunkmanifest").write_text("2")

  compile_modeld.handle_compile_failure(str(output), str(marker), ValueError("bad model"))

  assert not output.exists()
  assert not Path(f"{output}.chunkmanifest").exists()
  assert not marker.exists()


def test_marker_failure_does_not_replace_original_or_prevent_cleanup(tmp_path, monkeypatch):
  output = tmp_path / "model.pkl"
  output.write_bytes(b"partial")
  original = USBDeviceSessionLost("link lost")

  def fail_marker(*args):
    raise RuntimeError("marker transport failed")

  monkeypatch.setattr(compile_modeld, "write_failure_marker", fail_marker)
  with pytest.raises(USBDeviceSessionLost) as exc_info:
    try:
      raise original
    except USBDeviceSessionLost as exc:
      compile_modeld.handle_compile_failure(str(output), str(tmp_path / "failure.marker"), exc)
      raise

  assert exc_info.value is original
  assert not output.exists()


def test_write_failure_marker_cleans_temp_when_replace_fails(tmp_path, monkeypatch):
  marker = tmp_path / "failure.marker"
  marker.write_text("old\n")

  def fail_replace(*args):
    raise PermissionError("read only")

  monkeypatch.setattr(compile_modeld.os, "replace", fail_replace)
  with pytest.raises(PermissionError, match="read only"):
    compile_modeld.write_failure_marker(str(marker), "new")

  assert marker.read_text() == "old\n"
  assert list(tmp_path.glob(".failure.marker.*")) == []
