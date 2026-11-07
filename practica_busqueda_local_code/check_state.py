"""
PROJECT: PRACTICAL ASSIGNMENT OF ABIA (UPC)
AUTHORS: ...
DESCRIPTION:
  Comprueba el estado y las soluciones iniciales en el escenario del enunciado.
  Run with: python check_state.py.
"""

from math import isclose
from time import perf_counter
from customers import generate_customers
from energy_state import (
  InfeasibleError,
  Instance,
  State,
  UNSERVED,
  greedy_initial_state,
  random_initial_state,
)
from power_plants import generate_power_plants

SEEDS = range(1, 11)


def build_instance(seed: int) -> Instance:
  """Crea el escenario de los experimentos 1 a 3."""
  plants = generate_power_plants((5, 10, 25), seed)
  customers = generate_customers(1000, (0.25, 0.30, 0.45), 0.75, seed)
  return Instance(plants, customers)


def check_small_example() -> None:
  """Compara la heuristica con un caso calculado a mano."""
  from customer import Customer, GUARANTEED, LARGE, NON_GUARANTEED
  from power_plant import PowerPlant, TYPE_C

  # Central C de 100 MW: en marcha cuesta 5000 + 150 * 100 = 20000.
  plant = PowerPlant(TYPE_C, 100.0, 0, 0)
  # A 50 km pierde un 20 %: 8 MW llegan si se envian 10 MW.
  near = Customer(LARGE, 2.0, GUARANTEED, 30, 40)
  # No garantizado sin servir: penaliza 50 * 1 = 50.
  far = Customer(LARGE, 1.0, NON_GUARANTEED, 0, 5)

  instance = Instance([plant], [near, far])
  state = State(instance)
  near_index = instance.customers.index(near)
  assert isclose(instance.sent_mw[near_index][0], 2.5)
  state.move(near_index, 0)

  # 20000 de central + 50 de penalizacion - 2 * 600 de ingreso = 18850.
  assert isclose(state.cost(), 18850.0)
  assert isclose(state.heuristic(), 18850.0 - 1500.0)
  assert isclose(state.used_mw[0], 2.5)

  # Al quitar el cliente la central se para y queda sin clientes.
  state.move(near_index, UNSERVED)
  assert state.n_clients[0] == 0 and isclose(state.used_mw[0], 0.0)
  assert not state.is_valid()


def main() -> None:
  check_small_example()
  print("Ejemplo calculado a mano: OK\n")
  print("seed  estrategia  beneficio    servidos  centrales  ms")

  for seed in SEEDS:
    instance = build_instance(seed)
    strategies = (("greedy", lambda: greedy_initial_state(instance)),
                  ("random", lambda: random_initial_state(instance, seed)))
    for name, build in strategies:
      start = perf_counter()
      try:
        state = build()
      except InfeasibleError as error:
        print(f"{seed:>4}  {name:<10}  NO FACTIBLE: {error}")
        continue
      elapsed = (perf_counter() - start) * 1000
      assert state.is_valid()
      served = sum(plant != UNSERVED for plant in state.assignment)
      running = sum(n > 0 for n in state.n_clients)
      print(f"{seed:>4}  {name:<10}  {state.profit():>10.2f}  {served:>8}  "
            f"{running:>9}  {elapsed:.1f}")

  # Tiempo de evaluar la heuristica: decide si hara falta optimizarla.
  state = greedy_initial_state(build_instance(1))
  start = perf_counter()
  for _ in range(2000):
    state.heuristic()
  heuristic_us = (perf_counter() - start) / 2000 * 1e6
  start = perf_counter()
  for _ in range(2000):
    state.copy()
  copy_us = (perf_counter() - start) / 2000 * 1e6
  print(f"\nHeuristica: {heuristic_us:.1f} us; copia del estado: {copy_us:.1f} us")


if __name__ == "__main__":
  main()
