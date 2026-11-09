"""Regresiones del modelo y los nuevos operadores. Ejecutar con unittest."""

import math
import random
import unittest
from unittest.mock import patch

from customer import Customer, EXTRA_LARGE, GUARANTEED, LARGE, NON_GUARANTEED, VERY_LARGE
from energy_problem import (
  OPERATOR_SETS, all_actions, calibrate_schedule, close_plan, polish_transport,
  random_action, run_hill_climbing, run_simulated_annealing,
)
from energy_state import (
  CLOSE, INSERT_RELOCATE, MOVE, OPEN_BATCH, SWAP, UNSERVED, Instance, State,
)
from power_plant import PowerPlant, TYPE_A, TYPE_C


def make_state(plants, customers, assignments, penalty=None):
  state = State(Instance(plants, customers, penalty))
  for customer, plant in zip(customers, assignments):
    state.move(state.instance.customers.index(customer), plant)
  state.refresh()
  return state


class SearchImprovementTests(unittest.TestCase):
  def test_annealing_checks_sparse_neighborhood_before_stopping(self):
    plants = [PowerPlant(TYPE_A, 2500, 0, 0), PowerPlant(TYPE_C, 1.5, 0, 0)]
    customers = [Customer(LARGE, 1 if i == 0 else 2, GUARANTEED, 0, 0)
                 for i in range(1000)]
    state = make_state(plants, customers, [0] * len(customers))
    random.seed(0)
    self.assertEqual(random_action(state, (MOVE,)), [(MOVE, 0, 1)])
    empty = State(Instance([], []))
    metrics = {}
    self.assertIs(run_simulated_annealing(empty, (MOVE,), 0, 300, 0.001, 10, 1,
                                         metrics=metrics), empty)
    self.assertEqual(metrics["stop_reason"], "no_successors")
    self.assertEqual(random_action(state, ()), [])

  def test_annealing_fallback_returns_one_action(self):
    plants = [PowerPlant(TYPE_C, 10, 0, 0)]
    customers = [Customer(LARGE, 1.5, NON_GUARANTEED, 0, 0)]
    state = make_state(plants, customers, [UNSERVED])
    metrics = {"fallbacks": 0}
    with patch("energy_problem.MAX_RANDOM_TRIES", 0):
      self.assertEqual(random_action(state, (MOVE,), metrics), [(MOVE, 0, 0)])
    self.assertEqual(metrics["fallbacks"], 1)

  def test_swap_includes_asymmetric_pair_in_both_algorithms(self):
    plants = [PowerPlant(TYPE_C, 10, 0, 0), PowerPlant(TYPE_C, 10, 0, 0)]
    customers = [Customer(LARGE, 1, GUARANTEED, x, 0) for x in (0, 1, 10)]
    state = make_state(plants, customers, (UNSERVED, 0, 1))
    self.assertEqual(state.instance.neighbors(1), [[1], [0], [1]])
    self.assertEqual(state.instance.swap_pairs(1), ((0, 1), (1, 2)))
    self.assertEqual(state.instance.swap_pairs(10), ((0, 1), (0, 2), (1, 2)))
    with patch("energy_problem.SWAP_NEIGHBORS", 1):
      self.assertEqual(all_actions(state, (SWAP,)), [(SWAP, 1, 2)])
      self.assertEqual(random_action(state, (SWAP,)), [(SWAP, 1, 2)])

  def test_close_reserves_capacity_for_guaranteed_customer(self):
    plants = [PowerPlant(TYPE_C, cap, 0, 0) for cap in (20, 10)]
    customers = [
      Customer(EXTRA_LARGE, 7, NON_GUARANTEED, 0, 0),
      Customer(EXTRA_LARGE, 6, GUARANTEED, 0, 0),
      Customer(VERY_LARGE, 3, GUARANTEED, 0, 0),
    ]
    state = make_state(plants, customers, (0, 0, 1))
    plan = close_plan(state, 0)
    self.assertEqual(plan, ((1, 1), (0, UNSERVED)))
    closed = state.successor((CLOSE, 0, plan))
    self.assertTrue(closed.is_valid())
    self.assertEqual(closed.profit() - state.profit(), 4050)

  def test_open_batch_crosses_fixed_startup_cost(self):
    plants = [PowerPlant(TYPE_C, 10, 0, 0)]
    customers = [Customer(LARGE, 1.9, NON_GUARANTEED, 0, 0) for _ in range(5)]
    state = make_state(plants, customers, [UNSERVED] * 5)
    original = run_hill_climbing(state, OPERATOR_SETS["all"], 1)
    improved = run_hill_climbing(state, OPERATOR_SETS["all_open_batch"], 1)
    self.assertEqual(original.assignment, [UNSERVED] * 5)
    self.assertEqual(len(all_actions(state, (OPEN_BATCH,))), 1)
    self.assertEqual(improved.assignment, [0] * 5)
    self.assertTrue(improved.is_valid())
    self.assertEqual(improved.profit() - original.profit(), 225)
    self.assertEqual(improved.h, improved.heuristic())

  def test_relocation_inserts_guaranteed_without_dropping_service(self):
    plants = [PowerPlant(TYPE_C, 10, 0, 0), PowerPlant(TYPE_C, 10, 0, 0)]
    customers = [Customer(EXTRA_LARGE, demand, GUARANTEED, 0, 0) for demand in (5, 3, 6, 5)]
    state = make_state(plants, customers, (0, 0, 1, UNSERVED), penalty=3200)
    self.assertFalse(any(state.fits(3, p) for p in range(2)))
    self.assertEqual(len(all_actions(state, (INSERT_RELOCATE,))), 1)
    action = all_actions(state, (INSERT_RELOCATE,))[0]
    final = state.successor(action)
    self.assertTrue(final.is_valid())
    self.assertEqual(final.assignment, [0, 1, 1, 0])
    self.assertEqual(final.h, final.heuristic())

  def test_transport_polish_preserves_profit_and_active_plants(self):
    plants = [PowerPlant(TYPE_C, 10, x, 0) for x in (0, 50)]
    customers = [Customer(LARGE, 1.5, GUARANTEED, x, 0) for x in (0, 50, 50)]
    state = make_state(plants, customers, (0, 0, 1))
    final = polish_transport(state)
    self.assertTrue(final.is_valid())
    self.assertEqual(final.profit(), state.profit())
    self.assertTrue(all(final.n_clients))
    self.assertLess(sum(final.used_mw), sum(state.used_mw))
    self.assertEqual(state.assignment, [0, 0, 1])

  def test_calibration_uses_measured_loss_as_acceptance_reference(self):
    plants = [PowerPlant(TYPE_C, 10, 0, 0)]
    customers = [Customer(LARGE, 1.5, GUARANTEED, 0, 0)]
    state = make_state(plants, customers, (UNSERVED,), penalty=3200)
    params, loss = calibrate_schedule(state, (MOVE,), 1, limit=100, stiter=10,
                                     target_acceptance=0.25, samples=4)
    self.assertIsNotNone(loss)
    self.assertAlmostEqual(math.exp(-loss / params["k"]), 0.25)
    cold_temperature = params["k"] * math.exp(-params["lam"] * 60)
    self.assertAlmostEqual(math.exp(-loss / cold_temperature), 0.01)


if __name__ == "__main__":
  unittest.main()
