"""
PROJECT: PRACTICAL ASSIGNMENT OF ABIA (UPC)
AUTHORS: ...
DESCRIPTION:
  Comprueba los operadores y mide una ejecucion real de cada algoritmo.
  Run with: python check_problem.py.
"""

from math import isclose
import random
from time import perf_counter
from check_state import build_instance
from energy_problem import (
  OPERATOR_SETS,
  all_actions,
  random_action,
  run_hill_climbing,
  run_simulated_annealing,
)
from energy_state import UNSERVED, greedy_initial_state, random_initial_state


def check_random_walk(steps: int = 3000) -> None:
  """Aplica acciones al azar y compara el valor incremental con el recalculado."""
  random.seed(7)
  instance = build_instance(1)
  operators = OPERATOR_SETS["all"]
  state = random_initial_state(instance, 1)
  seen = set()

  for _ in range(steps):
    action = random_action(state, operators)[0]
    seen.add(action[0])
    state = state.successor(action).materialize()
    assert isclose(state.h, state.heuristic(), abs_tol=1e-6), action
    assert state.is_valid(), action

  assert seen == set(operators), seen
  print(f"Camino aleatorio de {steps} pasos: OK (operadores usados: {sorted(seen)})")


def check_all_successors() -> None:
  """Comprueba cada sucesor de Hill Climbing, sin muestrear."""
  instance = build_instance(3)
  state = greedy_initial_state(instance)
  actions = all_actions(state, OPERATOR_SETS["all"])
  for action in actions[::50]:
    successor = state.successor(action)
    expected = successor.h
    successor.materialize()
    assert isclose(expected, successor.heuristic(), abs_tol=1e-6), action
    assert successor.is_valid(), action
  print(f"Sucesores comprobados: {len(actions[::50])} de {len(actions)}: OK")


def main() -> None:
  check_random_walk()
  check_all_successors()

  print("\nHill Climbing, semilla 1")
  print("conjunto      inicial  beneficio ini  beneficio fin  pasos  servidos  "
        "centrales  segundos")
  instance = build_instance(1)
  starts = (("greedy", greedy_initial_state(instance)),
            ("random", random_initial_state(instance, 1)))
  for name, operators in OPERATOR_SETS.items():
    for start_name, initial in starts:
      begin = perf_counter()
      final = run_hill_climbing(initial, operators, 1)
      elapsed = perf_counter() - begin
      assert final.is_valid()
      served = sum(p != UNSERVED for p in final.assignment)
      running = sum(n > 0 for n in final.n_clients)
      print(f"{name:<12}  {start_name:<7}  {initial.profit():>13.2f}  "
            f"{final.profit():>13.2f}  {final.steps:>5}  {served:>8}  "
            f"{running:>9}  {elapsed:>8.1f}", flush=True)

  print("\nSimulated Annealing, semilla 1, 100000 iteraciones")
  initial = greedy_initial_state(instance)
  begin = perf_counter()
  final = run_simulated_annealing(initial, OPERATOR_SETS["all"], 1,
                                  k=2000, lam=0.0001, limit=100000, stiter=100)
  print(f"beneficio {final.profit():.2f}, valido {final.is_valid()}, "
        f"{perf_counter() - begin:.1f} s")


if __name__ == "__main__":
  main()
