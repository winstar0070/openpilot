import unittest
from types import SimpleNamespace

from openpilot.selfdrive.car.ccnc_hud import update_ccnc_model
from openpilot.cereal import log


class TestCcncHud(unittest.TestCase):
  def setUp(self):
    self.controller = SimpleNamespace(ccnc_model=None)
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


if __name__ == '__main__':
  unittest.main()
