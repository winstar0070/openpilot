import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from openpilot.tools import ccnc_slot_probe as probe


def safe_sm(now=1_000_000_000):
  class SM(dict):
    valid = {'carState': True, 'carControl': True}
    alive = {'carState': True, 'carControl': True}
    logMonoTime = {'carState': now, 'carControl': now}
  return SM(carState=NS(canValid=True, gearShifter='park', vEgo=0.),
            carControl=NS(enabled=False, latActive=False, longActive=False))


class TestSlotProbe(unittest.TestCase):
  def test_demo_plan_is_multiple_slots_and_fixed_lease_motion(self):
    output = io.StringIO()
    with patch.object(probe, 'run', side_effect=AssertionError('live path')), contextlib.redirect_stdout(output):
      self.assertEqual(probe.main(['--demo', '--dry-run']), 0)
    phases = json.loads(output.getvalue())
    self.assertEqual(len(phases), 7)
    self.assertEqual([m['slot'] for m in phases[0]['markers']], ['FRONT', 'ALT'])
    self.assertEqual(len(phases[3]['markers']), 6)
    self.assertEqual([m['status'] for m in phases[4]['markers']], [2, 2, 2, 2, 4, 4])
    self.assertTrue(all(m['distance'] > m['end_distance'] for m in phases[5]['markers'][:4]))
    self.assertTrue(all(m['distance'] < m['end_distance'] for m in phases[5]['markers'][4:]))
    req = probe.request(phases[5], 1_000_000_000)
    self.assertEqual(set(req), {'token', 'issued_ns', 'expires_ns', 'markers'})
    self.assertEqual(req['expires_ns'] - req['issued_ns'], 8_000_000_000)
    self.assertLess(len(json.dumps(req, separators=(',', ':')).encode()), 1024)
    for args in (['--demo', '--seconds', '2'], ['--demo', '--status', '4'], ['--demo', '--lateral', '2']):
      with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
        probe.parse_args(args)

  def test_args_plan_and_no_io_dry_run(self):
    output = io.StringIO()
    with patch.object(probe, 'run', side_effect=AssertionError('live path')), contextlib.redirect_stdout(output):
      self.assertEqual(probe.main(['--scan', '--dry-run']), 0)
    phases = json.loads(output.getvalue())
    self.assertEqual([p['slot'] for p in phases], list(probe.SLOTS) + ['LEFT_REAR', 'RIGHT_REAR'])
    self.assertEqual([p['status'] for p in phases], [2] * 6 + [4, 4])
    self.assertEqual(phases[0]['lateral'], 0.)
    self.assertEqual(phases[2]['lateral'], 3.6)
    self.assertEqual(probe.plan(probe.parse_args(['--slot', 'alt']))[0]['status'], 2)
    for args in [[], ['--slot', 'alt', '--status', '4'], ['--slot', 'front', '--seconds', '11'],
                 ['--slot', 'left', '--distance', 'nan'], ['--slot', 'left', '--distance', '0'],
                 ['--slot', 'left', '--lateral', '13'], ['--scan', '--slot', 'left'], ['--scan', '--status', '4'],
                 ['--slot', 'front', '--status', '3'], ['--slot', 'left', '--status', '4'],
                 ['--slot', 'right', '--status', '3'], ['--slot', 'front', '--seconds', '1e-10']]:
      with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
        probe.parse_args(args)

  def test_preflight_all_gates(self):
    now = 1_000_000_000
    self.assertIsNone(probe.unsafe_reason(safe_sm(), now))
    mutations = [lambda s: s.valid.update(carState=False), lambda s: s.alive.update(carControl=False),
                 lambda s: s.logMonoTime.update(carState=now - 250_000_001),
                 lambda s: s.logMonoTime.update(carControl=now + 1),
                 lambda s: setattr(s['carState'], 'canValid', False),
                 lambda s: setattr(s['carState'], 'gearShifter', 'drive'),
                 lambda s: setattr(s['carState'], 'vEgo', .101),
                 lambda s: setattr(s['carState'], 'vEgo', math.nan)]
    for key in ('enabled', 'latActive', 'longActive'):
      mutations.append(lambda s, k=key: setattr(s['carControl'], k, True))
    for change in mutations:
      sm = safe_sm()
      sm.valid = dict(sm.valid)
      sm.alive = dict(sm.alive)
      sm.logMonoTime = dict(sm.logMonoTime)
      change(sm)
      self.assertIsNotNone(probe.unsafe_reason(sm, now))

  def test_atomic_request_cleanup_and_lock(self):
    phase = probe.plan(probe.parse_args(['--slot', 'left-rear']))[0]
    req = probe.request(phase, 123)
    self.assertEqual(req['expires_ns'] - req['issued_ns'], 8_000_000_000)
    self.assertEqual(len(req['token']), 32)
    with tempfile.TemporaryDirectory() as tmp:
      path, lock = Path(tmp) / 'request.json', Path(tmp) / 'lock'
      with probe.exclusive_lock(lock):
        with self.assertRaises(RuntimeError), probe.exclusive_lock(lock):
          pass
        probe.atomic_write(path, req)
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        probe.cleanup(path, 'other')
        self.assertEqual(json.loads(path.read_text()), req)
        probe.cleanup(path, req['token'])
        self.assertFalse(path.exists())
        probe.cleanup(path, req['token'])


class TestLiveProtocol(unittest.TestCase):
  def test_demo_model_packing_and_cli_ack_agree(self):
    from opendbc.can import CANPacker
    from opendbc.car.hyundai.ccnc_probe import CcncSlotProbe
    packer = CANPacker('hyundai_canfd_generated')
    for phase in probe.plan(probe.parse_args(['--demo'])):
      model = CcncSlotProbe()
      req = probe.request(phase, 1_000_000_000)
      for now in (1_000_000_000, 3_000_000_000, 5_000_000_000, 8_900_000_000):
        values = model.update(req, now_ns=now, parked=True, stationary=True, controls_inactive=True,
                              can_valid=True, display_fresh=True)
        self.assertIsNotNone(values)
        msg = NS(address=0x162, src=0, dat=packer.make_can_msg('CCNC_0x162', 0, values)[1])
        self.assertTrue(probe.marker_observed([msg], req, now + 5_000_000))

  def test_demo_ack_multiple_slots_and_time_varying_distances(self):
    from opendbc.can import CANPacker
    packer = CANPacker('hyundai_canfd_generated')
    phases = probe.plan(probe.parse_args(['--demo']))
    for phase in phases:
      req = probe.request(phase, 1_000_000_000)
      values = {}
      for marker in req['markers']:
        prefix = 'LEAD' if marker['slot'] == 'FRONT' else 'LEAD_' + marker['slot']
        status = prefix + '_STATUS' if marker['slot'].endswith('REAR') else prefix
        values.update({status: marker['status'], prefix + '_DISTANCE': (marker['distance'] + marker['end_distance']) / 2,
                       prefix + '_LATERAL': marker['lateral']})
      msg = NS(address=0x162, src=0, dat=packer.make_can_msg('CCNC_0x162', 0, values)[1])
      self.assertTrue(probe.marker_observed([msg], req, 5_000_000_000))
      if any(m['distance'] != m['end_distance'] for m in req['markers']):
        self.assertFalse(probe.marker_observed([msg], req, 1_000_000_000))
      first = req['markers'][0]
      field = 'LEAD' if first['slot'] == 'FRONT' else 'LEAD_' + first['slot']
      values[field] = 0
      msg.dat = packer.make_can_msg('CCNC_0x162', 0, values)[1]
      self.assertFalse(probe.marker_observed([msg], req, 5_000_000_000))

  def test_ack_crc_and_all_eighteen_fields(self):
    from opendbc.can import CANPacker
    packer = CANPacker('hyundai_canfd_generated')
    prefixes = ('LEAD', 'LEAD_ALT', 'LEAD_LEFT', 'LEAD_RIGHT', 'LEAD_LEFT_REAR', 'LEAD_RIGHT_REAR')
    for slot, prefix in zip(probe.SLOTS, prefixes, strict=True):
      req = {'slot': slot, 'status': 2, 'distance': 8., 'lateral': 0. if slot in ('FRONT', 'ALT') else 3.6}
      status = prefix + '_STATUS' if slot.endswith('REAR') else prefix
      values = {status: 2, prefix + '_DISTANCE': req['distance'], prefix + '_LATERAL': req['lateral']}
      data = packer.make_can_msg('CCNC_0x162', 0, values)[1]
      msg = NS(address=0x162, src=0, dat=data)
      self.assertTrue(probe.marker_observed([msg], req))
      self.assertFalse(probe.marker_observed([NS(address=0x162, src=2, dat=data)], req))
      corrupt = bytearray(data)
      corrupt[0] ^= 1
      self.assertFalse(probe.marker_observed([NS(address=0x162, src=0, dat=corrupt)], req))
      values['LEAD_RIGHT_DISTANCE' if slot != 'RIGHT' else 'LEAD_LEFT_DISTANCE'] = 1.
      msg.dat = packer.make_can_msg('CCNC_0x162', 0, values)[1]
      self.assertFalse(probe.marker_observed([msg], req))

  def test_run_ack_abort_cleanup_scan_and_preflight(self):
    for mode in ('ok', 'no_ack', 'unsafe', 'interrupt', 'preflight', 'write_error', 'scan', 'short_no_ack', 'demo', 'motion_stopped'):
      with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
        clock = [1_000_000_000]
        request_path = Path(tmp) / 'req'
        sm = safe_sm()
        sm.valid = dict(sm.valid, sendcan=True)
        sm.alive = dict(sm.alive)
        sm.logMonoTime = dict(sm.logMonoTime, sendcan=clock[0])
        sm.updated = {'sendcan': True}
        sm['sendcan'] = []
        ticks = [0]

        def update(timeout, ticks=ticks, clock=clock, sm=sm, mode=mode):
          ticks[0] += 1
          clock[0] += 50_000_000
          sm.logMonoTime = dict.fromkeys(('carState', 'carControl', 'sendcan'), clock[0])
          if mode == 'preflight' or mode == 'unsafe' and ticks[0] > 4:
            sm['carState'].gearShifter = 'drive'
          if mode == 'interrupt' and ticks[0] > 4:
            raise KeyboardInterrupt

        sm.update = update
        args = (['--demo', '--seconds', '3'] if mode in ('demo', 'motion_stopped') else
                ['--scan', '--seconds', '.2'] if mode == 'scan' else
                ['--slot', 'left', '--seconds', '.2' if mode == 'short_no_ack' else '3'])
        phases = probe.plan(probe.parse_args(args))
        if mode == 'motion_stopped':
          phases = phases[5:6]
        def ack(*unused, mode=mode, ticks=ticks):
          return mode not in ('no_ack', 'short_no_ack') and not (mode == 'motion_stopped' and ticks[0] > 5)
        with patch.object(probe, 'subscriber', return_value=sm), patch.object(probe, 'REQUEST_PATH', request_path), \
             patch.object(probe, 'LOCK_PATH', Path(tmp) / 'lock'), patch.object(probe.time, 'monotonic_ns', side_effect=lambda clock=clock: clock[0]), \
             patch.object(probe, 'marker_observed', side_effect=ack), contextlib.redirect_stdout(io.StringIO()):
          if mode == 'write_error':
            with patch.object(probe.os, 'replace', side_effect=OSError('write error')), self.assertRaises(OSError):
              probe.run(phases)
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ['lock'])
          elif mode in ('no_ack', 'short_no_ack', 'unsafe', 'preflight', 'motion_stopped'):
            with self.assertRaises(RuntimeError):
              probe.run(phases)
          elif mode == 'interrupt':
            with self.assertRaises(KeyboardInterrupt):
              probe.run(phases)
          else:
            probe.run(phases)
          self.assertFalse(request_path.exists())
          if mode == 'no_ack':
            self.assertLessEqual(clock[0], 3_150_000_000)

  def test_main_failure_and_short_no_ack(self):
    with patch.object(probe, 'run', side_effect=RuntimeError('unsafe')), contextlib.redirect_stdout(io.StringIO()):
      self.assertEqual(probe.main(['--slot', 'front']), 1)
    with patch.object(probe, 'run'), contextlib.redirect_stdout(io.StringIO()):
      self.assertEqual(probe.main(['--slot', 'front']), 0)


if __name__ == '__main__':
  unittest.main()
