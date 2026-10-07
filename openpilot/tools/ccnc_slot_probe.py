#!/usr/bin/env python3
"""Park-only ccNC slot markers. Writes requests; never publishes CAN messages."""
import argparse
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import tempfile
import time
import uuid

REQUEST_PATH = Path('/dev/shm/ccnc-slot-probe.json')
LOCK_PATH = Path('/dev/shm/ccnc-slot-probe.lock')
SLOTS = ('FRONT', 'ALT', 'LEFT', 'RIGHT', 'LEFT_REAR', 'RIGHT_REAR')
FRESH_NS = 250_000_000
# DBC bit position, width, little endian. Rear fields use mixed byte order.
FIELDS = {
  'FRONT': ((64, 5, True), (69, 11, True), (80, 7, True)),
  'ALT': ((88, 5, True), (93, 11, True), (104, 7, True)),
  'LEFT': ((112, 5, True), (117, 11, True), (128, 7, True)),
  'RIGHT': ((136, 5, True), (141, 11, True), (152, 7, True)),
  'LEFT_REAR': ((167, 5, False), (175, 8, False), (182, 7, False)),
  'RIGHT_REAR': ((196, 5, False), (197, 8, True), (205, 7, True)),
}


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument('--slot', choices=[s.lower().replace('_', '-') for s in SLOTS])
  group.add_argument('--scan', action='store_true', help='Six status-2 slots, then both rear slots with status 4; 2-second gaps')
  group.add_argument('--demo', action='store_true', help='Simultaneous slots and forward/rear distance sweeps; record all seven phases')
  parser.add_argument('--seconds', type=float, default=8., help='Seconds per marker (0 < seconds <= 10)')
  parser.add_argument('--status', type=int, choices=(1, 2, 3, 4), default=2)
  parser.add_argument('--distance', type=float, default=8., help='Marker distance in meters, 0.1..25.5')
  parser.add_argument('--lateral', type=float, help='Marker lateral magnitude, 0..12.7; default 0 front/ALT, 3.6 side')
  parser.add_argument('--dry-run', action='store_true', help='Print plan only; no messaging or request files')
  args = parser.parse_args(argv)
  for name, value, low, high in [('seconds', args.seconds, 0., 10.), ('distance', args.distance, .1, 25.5),
                                ('lateral', args.lateral if args.lateral is not None else 0., 0., 12.7)]:
    if not math.isfinite(value) or not low <= value <= high or name == 'seconds' and value == 0:
      parser.error(f'{name} is outside the permitted range')
  if int(args.seconds * 1e9) < 1:
    parser.error('seconds must produce a lease of at least 1 nanosecond')
  if args.slot in ('front', 'alt', 'left', 'right') and args.status not in (1, 2):
    parser.error('Non-rear slots support only status 1 or 2')
  if args.scan and args.status != 2:
    parser.error('--scan fixes status 2 for six slots, then 4 for the two rear slots')
  if args.demo and (args.seconds < 3 or args.status != 2 or args.distance != 8 or args.lateral is not None):
    parser.error('--demo uses fixed positions/statuses and requires 3..10 seconds per phase')
  return args


def plan(args):
  if args.demo:
    def marker(slot, distance, end=None, status=2):
      return {'slot': slot, 'status': status, 'distance': distance, 'end_distance': distance if end is None else end,
              'lateral': 0. if slot in ('FRONT', 'ALT') else 3.6}

    all_static = [marker(s, d) for s, d in zip(SLOTS, (8., 18., 12., 12., 6., 6.), strict=True)]
    profiles = [
      ('중앙 앞차 두 슬롯: 8m / 18m', all_static[:2]),
      ('왼쪽 앞·뒤 슬롯', [all_static[2], all_static[4]]),
      ('오른쪽 앞·뒤 슬롯', [all_static[3], all_static[5]]),
      ('6개 슬롯 동시 표시: 상태 2', all_static),
      ('6개 슬롯 동시 표시: 옆뒤 상태 4', [dict(m, status=4 if m['slot'].endswith('REAR') else 2) for m in all_static]),
      ('거리 이동: 앞차 접근 / 옆뒤 멀어짐',
       [marker(s, a, b) for s, a, b in zip(SLOTS, (20., 25., 20., 20., 5., 5.), (5., 10., 5., 5., 20., 20.), strict=True)]),
      ('거리 이동: 앞차 멀어짐 / 옆뒤 접근',
       [marker(s, a, b) for s, a, b in zip(SLOTS, (5., 10., 5., 5., 20., 20.), (20., 25., 20., 20., 5., 5.), strict=True)]),
    ]
    return [{'label': label, 'markers': markers, 'seconds': args.seconds} for label, markers in profiles]
  slots = [(s, 2) for s in SLOTS] + [('LEFT_REAR', 4), ('RIGHT_REAR', 4)] if args.scan else [
    (args.slot.upper().replace('-', '_'), args.status)]
  return [{'slot': slot, 'status': status, 'distance': round(args.distance, 1),
           'lateral': round(args.lateral if args.lateral is not None else 0. if slot in ('FRONT', 'ALT') else 3.6, 1),
           'seconds': args.seconds} for slot, status in slots]


def request(phase, now):
  if 'markers' in phase:
    return {'token': uuid.uuid4().hex, 'issued_ns': now, 'expires_ns': now + int(phase['seconds'] * 1e9), 'markers': phase['markers']}
  return dict(token=uuid.uuid4().hex, issued_ns=now, expires_ns=now + int(phase['seconds'] * 1e9),
              **{k: phase[k] for k in ('slot', 'status', 'distance', 'lateral')})


@contextmanager
def exclusive_lock(path=LOCK_PATH):
  fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
  try:
    try:
      fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as e:
      raise RuntimeError('Another slot probe owns the lock') from e
    yield
  finally:
    os.close(fd)


def atomic_write(path, req):
  fd, tmp = tempfile.mkstemp(prefix='.ccnc-slot-probe-', dir=path.parent)
  try:
    with os.fdopen(fd, 'w') as f:
      json.dump(req, f, separators=(',', ':'))
      f.flush()
      os.fsync(f.fileno())
    os.replace(tmp, path)
  finally:
    if os.path.exists(tmp):
      os.unlink(tmp)


def cleanup(path, token):
  try:
    if json.loads(path.read_text()).get('token') == token:
      path.unlink()
  except (OSError, ValueError, AttributeError):
    pass


def unsafe_reason(sm, now):
  for service in ('carState', 'carControl'):
    if not sm.valid[service] or not sm.alive[service] or not 0 <= now - sm.logMonoTime[service] <= FRESH_NS:
      return f'{service} is invalid, missing, or stale'
  cs, cc = sm['carState'], sm['carControl']
  if not cs.canValid or str(cs.gearShifter) != 'park' or not math.isfinite(cs.vEgo) or abs(cs.vEgo) > .1:
    return 'Requires valid CAN, gear P and speed <= 0.1 m/s'
  if cc.enabled or cc.latActive or cc.longActive:
    return 'Control must be disabled (enabled/latActive/longActive all false)'
  return None


def raw_field(data, start, width, little):
  if little:
    return (int.from_bytes(data, 'little') >> start) & ((1 << width) - 1)
  value = 0
  for _ in range(width):
    value = (value << 1) | ((data[start // 8] >> (start % 8)) & 1)
    start = start + 15 if start % 8 == 0 else start - 1
  return value


def marker_observed(messages, req, now_ns=None):
  # Import only in live/ack mode. Dry-run needs no native or vehicle libraries.
  from opendbc.car.hyundai.hyundaicanfd import hkg_can_fd_checksum
  for msg in messages:
    if msg.address != 0x162 or msg.src != 0 or len(msg.dat) != 32:
      continue
    data = bytes(msg.dat)
    if hkg_can_fd_checksum(0x162, None, data) != int.from_bytes(data[:2], 'little'):
      continue
    markers = req.get('markers', [req])
    fraction = 0.
    if 'markers' in req:
      elapsed = (now_ns if now_ns is not None else req['issued_ns']) - req['issued_ns']
      fraction = max(0., min(1., elapsed / (req['expires_ns'] - req['issued_ns'])))
    expected = {}
    moving = set()
    for marker in markers:
      distance = marker['distance'] + (marker.get('end_distance', marker['distance']) - marker['distance']) * fraction
      slot = marker['slot']
      expected[slot] = (marker['status'], round(distance * 10), round(marker['lateral'] * 10))
      if marker.get('end_distance', marker['distance']) != marker['distance']:
        moving.add(slot)
    matches = True
    for slot, fields in FIELDS.items():
      actual = tuple(raw_field(data, *field) for field in fields)
      status, distance, lateral = expected.get(slot, (0, 0, 0))
      # sendcan is stamped just after apply: allow one 10cm bin for motion,
      # while still requiring exact statuses, lateral and empty-slot fields.
      matches &= actual[0] == status and actual[2] == lateral and abs(actual[1] - distance) <= (1 if slot in moving else 0)
    if matches:
      return True
  return False


def checked_update(sm):
  sm.update(50)
  now = time.monotonic_ns()
  reason = unsafe_reason(sm, now)
  if reason:
    raise RuntimeError(reason)
  return now


def subscriber():
  # A read-only subscriber. The existing CarD publisher handles gated requests.
  import openpilot.cereal.messaging as messaging
  return messaging.SubMaster(['carState', 'carControl', 'sendcan'], ignore_alive=['sendcan'], ignore_avg_freq=['sendcan'])


def run(phases):
  sm = subscriber()
  with exclusive_lock(LOCK_PATH):
    deadline = time.monotonic_ns() + 3_000_000_000
    while True:
      sm.update(50)
      now = time.monotonic_ns()
      reason = unsafe_reason(sm, now)
      if reason is None:
        break
      if now >= deadline:
        raise RuntimeError(f'Preflight failed: {reason}')
    print('Keep gear P and record the cluster. Synthetic slot markers are not detected vehicles.', flush=True)
    for index, phase in enumerate(phases):
      now = checked_update(sm)
      req = request(phase, now)
      try:
        atomic_write(REQUEST_PATH, req)
        label = phase.get('label', '') or f"{phase['slot']} status={phase['status']}"
        print(f"{index + 1}/{len(phases)} {label} ({phase['seconds']:g}s)", flush=True)
        acknowledged = False
        last_ack = req['issued_ns']
        moving = any(m['distance'] != m['end_distance'] for m in req.get('markers', []))
        while True:
          now = checked_update(sm)
          if now >= req['expires_ns']:
            break
          if (sm.updated['sendcan'] and sm.valid['sendcan'] and req['issued_ns'] <= sm.logMonoTime['sendcan'] <= now and
              now - sm.logMonoTime['sendcan'] <= FRESH_NS and marker_observed(sm['sendcan'], req, sm.logMonoTime['sendcan'])):
            if not acknowledged:
              print('HUD packet observed; this does not confirm cluster pixels.', flush=True)
            acknowledged = True
            last_ack = now
          if not acknowledged and now - req['issued_ns'] >= 2_000_000_000:
            raise RuntimeError('No matching HUD packet within 2 seconds; marker request aborted')
          if moving and acknowledged and now - last_ack > 500_000_000:
            raise RuntimeError('Moving HUD packets stopped matching; marker request aborted')
        if not acknowledged:
          raise RuntimeError('Marker expired without a matching HUD packet')
      finally:
        cleanup(REQUEST_PATH, req['token'])
      if index + 1 < len(phases):
        print('Normal display gap (2 seconds).', flush=True)
        gap_end = time.monotonic_ns() + 2_000_000_000
        while time.monotonic_ns() < gap_end:
          checked_update(sm)


def main(argv=None):
  args = parse_args(argv)
  phases = plan(args)
  # Parse separately from live imports, filesystem access and IPC.
  if args.dry_run:
    print(json.dumps(phases, indent=2))
    return 0
  try:
    run(phases)
  except (RuntimeError, OSError, ImportError, KeyboardInterrupt) as e:
    print(f'Slot probe stopped: {e}')
    return 1
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
