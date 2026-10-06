import unittest
from types import SimpleNamespace

from openpilot.selfdrive.car.ccnc_hud import update_ccnc_model, update_ccnc_radar
from openpilot.cereal import log
from opendbc.car import structs
from opendbc.can import CANPacker, CANParser
from opendbc.car.hyundai.ccnc_objects import CcncObjectDisplay
from opendbc.car.hyundai.hyundaicanfd import CanBus, create_ccnc


class TestCcncHud(unittest.TestCase):
  def setUp(self):
    self.controller = SimpleNamespace(ccnc_model=None, ccnc_object_lanes=None, ccnc_radar=None)
    self.model = SimpleNamespace(laneLines=[SimpleNamespace(x=[0.0], y=[y]) for y in (-5.4, -1.8, 1.8, 5.4)],
                                 laneLineProbs=[0.9] * 4, laneLineStds=[0.1] * 4,
                                 roadEdges=[SimpleNamespace(x=[0.0], y=[y]) for y in (-9.0, 9.0)], roadEdgeStds=[0.1, 0.1],
                                 meta=SimpleNamespace(laneChangeState=SimpleNamespace(raw=2), laneChangeDirection=SimpleNamespace(raw=1)))

  def test_fresh_snapshot_and_expired_invalid_or_future_data(self):
    update_ccnc_model(self.controller, self.model, True, 1_000_000_000, 1_100_000_000)
    self.assertEqual(self.controller.ccnc_model.lanes, (-5.4, -1.8, 1.8, 5.4))
    self.model.laneLines[1].y[0] = -1.0
    self.assertEqual(self.controller.ccnc_model.lanes[1], -1.8)
    for valid, now in ((False, 1_100_000_000), (True, 1_250_000_001), (True, 999_999_999)):
      update_ccnc_model(self.controller, self.model, True, 1_000_000_000, 1_100_000_000)
      update_ccnc_model(self.controller, self.model, valid, 1_000_000_000, now)
      self.assertIsNone(self.controller.ccnc_model)

  def test_real_model_schema(self):
    model = log.ModelDataV2.new_message()
    model.init('laneLines', 4)
    for line, y in zip(model.laneLines, (-5.4, -1.8, 1.8, 5.4), strict=True):
      line.x = [0.0]
      line.y = [y]
    model.init('roadEdges', 2)
    for edge, y in zip(model.roadEdges, (-9.0, 9.0), strict=True):
      edge.x = [0.0]
      edge.y = [y]
    model.roadEdgeStds = [0.1, 0.1]
    model.laneLineProbs = [0.9] * 4
    model.laneLineStds = [0.1] * 4
    model.meta.laneChangeState = 'laneChangeStarting'
    model.meta.laneChangeDirection = 'left'
    update_ccnc_model(self.controller, model.as_reader(), True, 1_000_000_000, 1_050_000_000)
    self.assertEqual((self.controller.ccnc_model.state, self.controller.ccnc_model.direction), (2, 1))
    self.assertAlmostEqual(self.controller.ccnc_model.lanes[1], -1.8)
    self.assertEqual(self.controller.ccnc_model.edges, (-9.0, 9.0))

  def test_other_car_controller_untouched(self):
    controller = SimpleNamespace()
    update_ccnc_model(controller, None, False, 0, 1)
    self.assertEqual(vars(controller), {})

  def test_object_lane_snapshot_expires_with_model_and_supports_old_controller(self):
    for line in self.model.laneLines:
      line.x = [0., 80.]
      line.y = line.y * 2
    update_ccnc_model(self.controller, self.model, True, 1_000_000_000, 1_050_000_000)
    self.assertEqual(self.controller.ccnc_object_lanes.lines[0], ((0., -5.4), (80., -5.4)))
    update_ccnc_model(self.controller, self.model, False, 1_000_000_000, 1_050_000_000)
    self.assertIsNone(self.controller.ccnc_object_lanes)
    old = SimpleNamespace(ccnc_model=None)
    update_ccnc_model(old, self.model, True, 1_000_000_000, 1_050_000_000)
    self.assertFalse(hasattr(old, 'ccnc_object_lanes'))

  def test_radar_frames_copy_measurements_and_clear_on_error(self):
    radar = structs.RadarData()
    radar.points = [structs.RadarData.RadarPoint(trackId=7, dRel=25., yRel=3.5, vRel=0.)]
    update_ccnc_radar(self.controller, radar, True, 1_000_000_000)
    self.assertEqual(self.controller.ccnc_radar.objects[0].lateral, -3.5)
    radar.points[0].dRel = 90.
    self.assertEqual(self.controller.ccnc_radar.objects[0].distance, 25.)
    update_ccnc_radar(self.controller, None, True, 1_010_000_000)
    self.assertEqual(self.controller.ccnc_radar.timestamp, 1.)
    radar.errors.canError = True
    update_ccnc_radar(self.controller, radar, True, 1_050_000_000)
    self.assertIsNone(self.controller.ccnc_radar)
    radar.errors.canError = False
    update_ccnc_radar(self.controller, radar, True, 1_060_000_000)
    update_ccnc_radar(self.controller, None, False, 1_070_000_000)
    self.assertIsNone(self.controller.ccnc_radar)
    controller = SimpleNamespace()
    update_ccnc_radar(controller, radar, True, 1_000_000_000)
    self.assertEqual(vars(controller), {})

  def test_radar_and_model_snapshots_reach_can_and_expire(self):
    for line in self.model.laneLines:
      line.x = [0., 80.]
      line.y = line.y * 2
    radar = structs.RadarData()
    radar.points = [structs.RadarData.RadarPoint(trackId=7, dRel=25., yRel=3.5, vRel=0.)]
    update_ccnc_model(self.controller, self.model, True, 1_000_000_000, 1_050_000_000)
    update_ccnc_radar(self.controller, radar, True, 1_000_000_000)
    parser = CANParser('hyundai_canfd_generated', [('CCNC_0x161', 0), ('CCNC_0x162', 0)], 0)
    stock_161, stock_162 = dict(parser.vl['CCNC_0x161']), dict(parser.vl['CCNC_0x162'])
    stock_162['COUNTER'] = 23
    display = CcncObjectDisplay()
    for now, expected in ((1.05, 2), (1.251, 0)):
      values = display.update(self.controller.ccnc_radar, self.controller.ccnc_object_lanes, now)
      messages = create_ccnc(CANPacker('hyundai_canfd_generated'), CanBus(None, fingerprint={}), False, True,
                            SimpleNamespace(leftLaneDepart=False, rightLaneDepart=False), False, False,
                            stock_161, stock_162, {}, True, SimpleNamespace(vEgo=20., leftBlindspot=False, rightBlindspot=False),
                            False, True, send_161=False, object_values=values)
      self.assertEqual(len(messages), 1)
      self.assertEqual(messages[0][0], 0x162)
      parser.update([(int(now * 1e9), messages)])
      received = parser.vl['CCNC_0x162']
      self.assertEqual(received['LEAD_LEFT'], expected)
      self.assertEqual(received['COUNTER'], 23)
      if expected:
        self.assertEqual(received['LEAD_LEFT_DISTANCE'], 25.)
        self.assertEqual(received['LEAD_LEFT_LATERAL'], 3.5)
    self.assertEqual(stock_162['LEAD_LEFT'], 0)


if __name__ == '__main__':
  unittest.main()
