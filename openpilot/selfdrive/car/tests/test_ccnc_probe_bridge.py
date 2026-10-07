import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from openpilot.selfdrive.car.ccnc_hud import update_ccnc_probe


class TestCcncProbeBridge(unittest.TestCase):
  def setUp(self):
    self.tmp = tempfile.TemporaryDirectory()
    self.addCleanup(self.tmp.cleanup)
    self.path = Path(self.tmp.name) / 'probe.json'
    self.controller = SimpleNamespace(ccnc_probe_request=None)

  def read(self):
    update_ccnc_probe(self.controller, path=str(self.path))
    return self.controller.ccnc_probe_request

  def test_regular_request_and_removal(self):
    request = {'token': 'a' * 32, 'issued_ns': 1, 'expires_ns': 2, 'slot': 'LEFT',
               'status': 2, 'distance': 8., 'lateral': 3.6}
    self.path.write_text(json.dumps(request))
    self.path.chmod(0o600)
    self.assertEqual(self.read(), request)
    self.path.unlink()
    self.assertIsNone(self.read())

  def test_invalid_content(self):
    for content in (b'{', b'[]', b'null', b'\xff', b' ' * 1025, b'{"slot":"LEFT","slot":"RIGHT"}'):
      with self.subTest(content=content[:30]):
        self.path.write_bytes(content)
        self.path.chmod(0o600)
        self.assertIsNone(self.read())

  def test_unsafe_file_types_and_permissions(self):
    target = self.path.with_name('target')
    target.write_text('{}')
    self.path.symlink_to(target)
    self.assertIsNone(self.read())
    self.path.unlink()
    os.mkfifo(self.path)
    self.assertIsNone(self.read())
    self.path.unlink()
    self.path.mkdir()
    self.assertIsNone(self.read())
    self.path.rmdir()
    self.path.write_text('{}')
    for mode in (0o622, 0o620, 0o602):
      self.path.chmod(mode)
      self.assertIsNone(self.read())

  def test_disabled_replay_and_other_controllers_never_open(self):
    with patch('os.open', side_effect=AssertionError('must not read')):
      self.controller.ccnc_probe_request = {}
      update_ccnc_probe(self.controller, enabled=False, path=str(self.path))
      self.assertIsNone(self.controller.ccnc_probe_request)
      update_ccnc_probe(SimpleNamespace(), path=str(self.path))

  def test_invalid_control_cancels_even_without_apply(self):
    self.controller.ccnc_probe = Mock()
    with patch('os.open', side_effect=AssertionError('must not read')):
      update_ccnc_probe(self.controller, enabled=False)
    self.controller.ccnc_probe.cancel.assert_called_once_with()

  def test_wrong_owner_and_read_failure(self):
    self.path.write_text('{}')
    self.path.chmod(0o600)
    metadata = SimpleNamespace(st_mode=0o100600, st_size=2, st_uid=os.getuid() + 1000)
    with patch('os.fstat', return_value=metadata):
      self.assertIsNone(self.read())
    with patch('os.read', side_effect=OSError('failed')):
      self.assertIsNone(self.read())


if __name__ == '__main__':
  unittest.main()
