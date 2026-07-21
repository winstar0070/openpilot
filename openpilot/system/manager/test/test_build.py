import openpilot.system.manager.build as manager_build


def make_runner(tmp_path, outcomes, calls):
  marker = tmp_path / "failure.marker"
  outcomes = iter(outcomes)

  def run(parallelism):
    calls.append(list(parallelism))
    returncode, hardware_reason = next(outcomes)
    if hardware_reason is not None:
      marker.write_text(hardware_reason)
    return returncode, [f"attempt {len(calls)}".encode()]

  return run, marker


def test_general_failures_keep_memory_retry_sequence(tmp_path):
  calls = []
  run, marker = make_runner(tmp_path, [(1, None), (1, None), (1, None)], calls)

  returncode, output, reason = manager_build.run_build_attempts(run, str(marker))

  assert returncode == 1
  assert output == [b"attempt 3"]
  assert reason is None
  assert calls == [[], ["-j4"], ["-j1"]]


def test_hardware_failure_gets_one_fresh_process_retry(tmp_path):
  calls = []
  run, marker = make_runner(tmp_path, [(1, "link lost"), (0, None)], calls)

  returncode, _, reason = manager_build.run_build_attempts(run, str(marker))

  assert returncode == 0
  assert reason is None
  assert calls == [[], []]
  assert not marker.exists()


def test_second_hardware_failure_stops_without_memory_retries(tmp_path):
  calls = []
  run, marker = make_runner(tmp_path, [(1, "first loss"), (1, "second loss")], calls)

  returncode, output, reason = manager_build.run_build_attempts(run, str(marker))

  assert returncode == 1
  assert output == [b"attempt 2"]
  assert reason == "second loss"
  assert calls == [[], []]


def test_hardware_retry_uses_current_parallelism(tmp_path):
  calls = []
  run, marker = make_runner(tmp_path, [(1, None), (1, "link lost"), (1, "link lost again")], calls)

  _, _, reason = manager_build.run_build_attempts(run, str(marker))

  assert reason == "link lost again"
  assert calls == [[], ["-j4"], ["-j4"]]


def test_hardware_failure_message_has_power_cycle_and_marker_path(tmp_path):
  marker = tmp_path / "failure.marker"
  message = manager_build.format_build_error([b"scons failed"], "TLP Unsupported Request", str(marker))

  assert "at least 10 seconds" in message
  assert "restart openpilot" in message
  assert str(marker) in message
