import ast
from pathlib import Path

import openpilot.system.manager.build as manager_build


def make_runner(tmp_path, outcomes, calls):
  marker = tmp_path / "failure.marker"
  outcomes = iter(outcomes)

  def run(parallelism, force_usbgpu, attempt_marker):
    calls.append((list(parallelism), force_usbgpu, Path(attempt_marker)))
    returncode, hardware_reason = next(outcomes)
    if hardware_reason is not None:
      Path(attempt_marker).write_text(hardware_reason)
    return returncode, [f"attempt {len(calls)}".encode()]

  return run, marker


def test_general_failures_keep_memory_retry_sequence(tmp_path):
  calls = []
  run, marker = make_runner(tmp_path, [(1, None), (1, None), (1, None)], calls)

  returncode, output, reason, _ = manager_build.run_build_attempts(run, str(marker))

  assert returncode == 1
  assert output == [b"attempt 3"]
  assert reason is None
  assert [(parallelism, forced) for parallelism, forced, _ in calls] == [([], False), (["-j4"], False), (["-j1"], False)]
  assert len({attempt_marker for _, _, attempt_marker in calls}) == 3


def test_hardware_failure_gets_one_fresh_process_retry(tmp_path):
  calls = []
  run, marker = make_runner(tmp_path, [(1, "link lost"), (0, None)], calls)

  returncode, _, reason, final_marker = manager_build.run_build_attempts(run, str(marker), artifact_valid=lambda: True)

  assert returncode == 0
  assert reason is None
  assert [(parallelism, forced) for parallelism, forced, _ in calls] == [([], False), ([], True)]
  assert final_marker == str(calls[-1][2])
  assert not marker.exists()
  assert not calls[0][2].exists()
  assert not calls[1][2].exists()


def test_second_hardware_failure_stops_without_memory_retries(tmp_path):
  calls = []
  run, marker = make_runner(tmp_path, [(1, "first loss"), (1, "second loss")], calls)

  returncode, output, reason, final_marker = manager_build.run_build_attempts(run, str(marker))

  assert returncode == 1
  assert output == [b"attempt 2"]
  assert reason == "second loss"
  assert [(parallelism, forced) for parallelism, forced, _ in calls] == [([], False), ([], True)]
  assert final_marker == str(calls[-1][2])
  assert calls[-1][2].read_text() == "second loss"


def test_hardware_retry_uses_current_parallelism(tmp_path):
  calls = []
  run, marker = make_runner(tmp_path, [(1, None), (1, "link lost"), (1, "link lost again")], calls)

  _, _, reason, _ = manager_build.run_build_attempts(run, str(marker))

  assert reason == "link lost again"
  assert [(parallelism, forced) for parallelism, forced, _ in calls] == [([], False), (["-j4"], False), (["-j4"], True)]


def test_successful_hardware_retry_requires_complete_artifact(tmp_path):
  calls = []
  run, marker = make_runner(tmp_path, [(1, "link lost"), (0, None)], calls)

  returncode, output, reason, final_marker = manager_build.run_build_attempts(
    run, str(marker), artifact_valid=lambda: False,
  )

  assert returncode == 1
  assert output == [b"attempt 2"]
  assert reason == "USB GPU retry exited successfully without a complete compiled model"
  assert Path(final_marker).read_text().strip() == reason
  assert [(parallelism, forced) for parallelism, forced, _ in calls] == [([], False), ([], True)]


def test_cleanup_usbgpu_build_temps_preserves_published_artifacts(tmp_path):
  output = tmp_path / "model.pkl"
  lock = tmp_path / ".usb_gpu.lock"
  stale_paths = [
    tmp_path / ".model.pkl.dead",
    tmp_path / "model.pkl.chunkmanifest.tmp.1",
    tmp_path / "model.pkl.chunk01of02.tmp.1",
  ]
  published_paths = [
    tmp_path / "model.pkl.chunkmanifest",
    tmp_path / "model.pkl.chunk01of01",
  ]
  for path in stale_paths:
    path.write_bytes(b"data")
  published_paths[0].write_text("1")
  published_paths[1].write_bytes(b"data")

  manager_build.cleanup_usbgpu_build_temps(output, lock)

  assert all(not path.exists() for path in stale_paths)
  assert all(path.exists() for path in published_paths)


def test_cleanup_usbgpu_build_temps_removes_published_orphans_without_manifest(tmp_path):
  output = tmp_path / "model.pkl"
  lock = tmp_path / ".usb_gpu.lock"
  orphan = tmp_path / "model.pkl.chunk01of02"
  orphan.write_bytes(b"orphan")

  manager_build.cleanup_usbgpu_build_temps(output, lock)

  assert not orphan.exists()


def test_hardware_failure_message_has_power_cycle_and_marker_path(tmp_path):
  marker = tmp_path / "failure.marker"
  message = manager_build.format_build_error([b"scons failed"], "TLP Unsupported Request", str(marker))

  assert "at least 10 seconds" in message
  assert "restart openpilot" in message
  assert str(marker) in message


def test_usbgpu_scons_signature_tracks_stable_compile_command():
  sconstruct = Path(manager_build.BASEDIR) / "openpilot/selfdrive/modeld/SConscript"
  source = sconstruct.read_text()

  assert "sources.append(Value(compile_cmd))" in source
  assert "base_command=compile_cmd" in source
  assert "os.getenv(\"USBGPU_BUILD_FAILURE_MARKER\"" in source
  assert "Value(cmd)" not in source


def test_usbgpu_sconscript_build_speed_eligibility():
  sconstruct = Path(manager_build.BASEDIR) / "openpilot/selfdrive/modeld/SConscript"
  tree = ast.parse(sconstruct.read_text(), filename=str(sconstruct))
  function, _ = _find_function_with_parent(tree, "usbgpu_build_speed_eligible")
  namespace = {"USBGPU_SUPERSPEED_MBIT": 5000}
  module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
  exec(compile(module, str(sconstruct), "exec"), namespace)

  eligible = namespace["usbgpu_build_speed_eligible"]
  assert not eligible(None)
  assert eligible(12)
  assert not eligible(480)
  assert eligible(5000)
  assert eligible(10000)


def _find_function_with_parent(tree, name):
  for parent in ast.walk(tree):
    for child in ast.iter_child_nodes(parent):
      if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef) and child.name == name:
        return child, parent
  raise AssertionError(f"missing function {name}")


def test_qcom_only_sconscript_does_not_evaluate_usbgpu_action_defaults():
  sconstruct = Path(manager_build.BASEDIR) / "openpilot/selfdrive/modeld/SConscript"
  tree = ast.parse(sconstruct.read_text(), filename=str(sconstruct))
  _, parent = _find_function_with_parent(tree, "do_usbgpu_build")

  assert isinstance(parent, ast.If)
  assert isinstance(parent.test, ast.Name) and parent.test.id == "usbgpu"
  usbgpu_action_only = ast.If(test=parent.test, body=parent.body, orelse=[])
  module = ast.fix_missing_locations(ast.Module(body=[usbgpu_action_only], type_ignores=[]))
  exec(compile(module, str(sconstruct), "exec"), {"usbgpu": False})


def test_usbgpu_action_signature_ignores_attempt_marker():
  from SCons.Script import Action, Environment

  sconstruct = Path(manager_build.BASEDIR) / "openpilot/selfdrive/modeld/SConscript"
  tree = ast.parse(sconstruct.read_text(), filename=str(sconstruct))
  function, _ = _find_function_with_parent(tree, "do_usbgpu_build")

  def action_signature(marker):
    def do_chunk(*args):
      return None

    namespace = {
      "compile_cmd": "stable compile command",
      "cmd": f"stable compile command --failure-marker {marker}",
      "target_pkl_path": "/model.pkl",
      "usbgpu_lock": "/lock",
      "do_chunk": do_chunk,
    }
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, str(sconstruct), "exec"), namespace)
    action = Action(namespace["do_usbgpu_build"])
    return bytes(action.get_contents([], [], Environment()))

  assert action_signature("/tmp/marker.first") == action_signature("/tmp/marker.second")
