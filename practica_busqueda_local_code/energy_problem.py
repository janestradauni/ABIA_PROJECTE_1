"""
PROJECT: PRACTICAL ASSIGNMENT OF ABIA (UPC)
AUTHORS: ...
DESCRIPTION:
  Operadores de busqueda, generacion de sucesores e integracion con AIMA.
  Hill Climbing recibe todos los sucesores; Simulated Annealing, uno al azar.
"""

from collections.abc import Callable
from math import exp
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "aima-python-master"))

from aima.search import Problem, hill_climbing, simulated_annealing
from energy_state import CLOSE, EPSILON, MOVE, REPLACE, SWAP, UNSERVED, State

# Clientes cercanos con los que se prueba SWAP en Hill Climbing. Probar todos los
# pares daria unos 500.000 sucesores por paso con 1000 clientes.
SWAP_NEIGHBORS = 10

# Intentos para encontrar una accion aplicable al azar.
MAX_RANDOM_TRIES = 1000

# Variante de MOVE del experimento 5: un cliente sin servicio solo puede pasar a
# la central mas cercana donde quepa. Sus acciones siguen siendo de tipo MOVE.
NEAREST_MOVE = "nearest_move"

# Conjuntos de operadores. El experimento 1 compara los cinco primeros.
OPERATOR_SETS = {
  "move": (MOVE,),
  "move_swap": (MOVE, SWAP),
  "move_replace": (MOVE, REPLACE),
  "move_close": (MOVE, CLOSE),
  "all": (MOVE, SWAP, REPLACE, CLOSE),
  "all_nearest": (NEAREST_MOVE, SWAP, REPLACE, CLOSE),
}


def close_plan(state: State, plant: int) -> tuple | None:
  """Reparte los clientes de una central entre las otras que estan en marcha.

  Los clientes que no caben quedan sin servicio si su contrato lo permite.

  Returns:
    tuple | None: Pares (cliente, destino), o None si algun cliente que debe
      servirse no cabe en ninguna central.
  """
  instance = state.instance
  customers = [c for c, p in enumerate(state.assignment) if p == plant]
  customers.sort(key=lambda c: (-instance.customers[c].demand_mw, c))
  used = state.used_mw[:]
  plan = []

  for customer in customers:
    sent_mw = instance.sent_mw[customer]
    target = next((p for p in instance.plant_order[customer]
                   if p != plant and state.n_clients[p] > 0
                   and used[p] + sent_mw[p] <= instance.capacity[p] + EPSILON),
                  UNSERVED)
    if target == UNSERVED and not instance.can_drop[customer]:
      return None
    if target != UNSERVED:
      used[target] += sent_mw[target]
    plan.append((customer, target))

  return tuple(plan)


def _swap_fits(state: State, first: int, second: int) -> bool:
  """Indica si dos clientes servidos pueden intercambiar sus centrales."""
  instance = state.instance
  first_plant = state.assignment[first]
  second_plant = state.assignment[second]
  sent_first = instance.sent_mw[first]
  sent_second = instance.sent_mw[second]
  return (state.used_mw[first_plant] - sent_first[first_plant]
          + sent_second[first_plant] <= instance.capacity[first_plant] + EPSILON
          and state.used_mw[second_plant] - sent_second[second_plant]
          + sent_first[second_plant] <= instance.capacity[second_plant] + EPSILON)


def _replace_fits(state: State, unserved: int, served: int) -> bool:
  """Indica si un cliente sin servicio cabe en el lugar de uno servido."""
  instance = state.instance
  plant = state.assignment[served]
  return (state.used_mw[plant] - instance.sent_mw[served][plant]
          + instance.sent_mw[unserved][plant]
          <= instance.capacity[plant] + EPSILON)


def _move_actions(state: State) -> list[tuple]:
  """Genera todos los MOVE aplicables."""
  instance = state.instance
  free = [instance.capacity[p] + EPSILON - state.used_mw[p]
          for p in range(instance.n_plants)]
  actions = []
  for customer, current in enumerate(state.assignment):
    for plant, sent in enumerate(instance.sent_mw[customer]):
      if plant != current and sent <= free[plant]:
        actions.append((MOVE, customer, plant))
    if current != UNSERVED and instance.can_drop[customer]:
      actions.append((MOVE, customer, UNSERVED))
  return actions


def _nearest_plant(state: State, customer: int) -> int | None:
  """Devuelve la central con menos perdidas donde cabe el cliente."""
  return next((p for p in state.instance.plant_order[customer]
               if state.fits(customer, p)), None)


def _nearest_move_actions(state: State) -> list[tuple]:
  """Genera los MOVE, con una sola opcion para cada cliente sin servicio."""
  instance = state.instance
  free = [instance.capacity[p] + EPSILON - state.used_mw[p]
          for p in range(instance.n_plants)]
  actions = []
  for customer, current in enumerate(state.assignment):
    if current == UNSERVED:
      plant = _nearest_plant(state, customer)
      if plant is not None:
        actions.append((MOVE, customer, plant))
      continue
    for plant, sent in enumerate(instance.sent_mw[customer]):
      if plant != current and sent <= free[plant]:
        actions.append((MOVE, customer, plant))
    if instance.can_drop[customer]:
      actions.append((MOVE, customer, UNSERVED))
  return actions


def _swap_actions(state: State) -> list[tuple]:
  """Genera los SWAP entre cada cliente y sus vecinos mas cercanos."""
  assignment = state.assignment
  neighbors = state.instance.neighbors(SWAP_NEIGHBORS)
  actions = []
  for first, plant in enumerate(assignment):
    if plant != UNSERVED:
      for second in neighbors[first]:
        if (first < second and assignment[second] not in (UNSERVED, plant)
            and _swap_fits(state, first, second)):
          actions.append((SWAP, first, second))
  return actions


def _replace_actions(state: State) -> list[tuple]:
  """Genera los REPLACE de clientes sin servicio por clientes prescindibles."""
  can_drop = state.instance.can_drop
  unserved = [c for c, p in enumerate(state.assignment) if p == UNSERVED]
  served = [c for c, p in enumerate(state.assignment)
            if p != UNSERVED and can_drop[c]]
  return [(REPLACE, u, s) for u in unserved for s in served
          if _replace_fits(state, u, s)]


def _close_actions(state: State) -> list[tuple]:
  """Genera un CLOSE por cada central en marcha que se pueda vaciar."""
  actions = []
  for plant, clients in enumerate(state.n_clients):
    if clients > 0:
      plan = close_plan(state, plant)
      if plan is not None:
        actions.append((CLOSE, plant, plan))
  return actions


GENERATORS = {MOVE: _move_actions, NEAREST_MOVE: _nearest_move_actions,
              SWAP: _swap_actions, REPLACE: _replace_actions,
              CLOSE: _close_actions}


def all_actions(state: State, operators: tuple) -> list[tuple]:
  """Genera todos los sucesores para Hill Climbing, en orden determinista."""
  actions = []
  for operator in operators:
    actions += GENERATORS[operator](state)
  return actions


def _random_move(state: State) -> tuple | None:
  """Prueba un MOVE con parametros al azar."""
  instance = state.instance
  customer = random.randrange(instance.n_customers)
  plant = random.randrange(-1, instance.n_plants)
  current = state.assignment[customer]
  if plant == current:
    return None
  if plant == UNSERVED:
    return (MOVE, customer, plant) if instance.can_drop[customer] else None
  return (MOVE, customer, plant) if state.fits(customer, plant) else None


def _random_nearest_move(state: State) -> tuple | None:
  """Prueba un MOVE; si el cliente no tiene servicio, va a su central cercana."""
  customer = random.randrange(state.instance.n_customers)
  if state.assignment[customer] != UNSERVED:
    plant = random.randrange(-1, state.instance.n_plants)
    if plant == state.assignment[customer]:
      return None
    if plant == UNSERVED:
      return (MOVE, customer, plant) if state.instance.can_drop[customer] else None
    return (MOVE, customer, plant) if state.fits(customer, plant) else None
  plant = _nearest_plant(state, customer)
  return None if plant is None else (MOVE, customer, plant)


def _random_swap(state: State) -> tuple | None:
  """Prueba un SWAP con dos clientes cualesquiera."""
  first = random.randrange(state.instance.n_customers)
  second = random.randrange(state.instance.n_customers)
  first_plant = state.assignment[first]
  second_plant = state.assignment[second]
  if (UNSERVED in (first_plant, second_plant) or first_plant == second_plant
      or not _swap_fits(state, first, second)):
    return None
  return (SWAP, first, second)


def _random_replace(state: State) -> tuple | None:
  """Prueba un REPLACE con dos clientes al azar."""
  unserved = random.randrange(state.instance.n_customers)
  served = random.randrange(state.instance.n_customers)
  if (state.assignment[unserved] != UNSERVED
      or state.assignment[served] == UNSERVED
      or not state.instance.can_drop[served]
      or not _replace_fits(state, unserved, served)):
    return None
  return (REPLACE, unserved, served)


def _random_close(state: State) -> tuple | None:
  """Prueba un CLOSE sobre una central al azar."""
  plant = random.randrange(state.instance.n_plants)
  if state.n_clients[plant] == 0:
    return None
  plan = close_plan(state, plant)
  return None if plan is None else (CLOSE, plant, plan)


RANDOM_GENERATORS = {MOVE: _random_move, NEAREST_MOVE: _random_nearest_move,
                     SWAP: _random_swap, REPLACE: _random_replace,
                     CLOSE: _random_close}


def random_action(state: State, operators: tuple) -> list[tuple]:
  """Elige un operador al azar y genera un solo sucesor para Simulated Annealing."""
  for _ in range(MAX_RANDOM_TRIES):
    action = RANDOM_GENERATORS[random.choice(operators)](state)
    if action is not None:
      return [action]
  return []


class EnergyProblem(Problem):
  """Problema de asignacion para los algoritmos de AIMA."""

  def __init__(self, initial: State, operators: tuple,
               stochastic: bool = False) -> None:
    super().__init__(initial)
    self.operators = operators
    self.generate = random_action if stochastic else all_actions

  def actions(self, state: State) -> list[tuple]:
    return self.generate(state.materialize(), self.operators)

  def result(self, state: State, action: tuple) -> State:
    return state.successor(action)

  def value(self, state: State) -> float:
    """AIMA maximiza y nuestra heuristica se minimiza: se cambia el signo."""
    return -state.h

  def goal_test(self, state: State) -> bool:
    """En busqueda local no se puede reconocer el estado final."""
    return False


def stepped_schedule(k: float, lam: float, limit: int,
                     stiter: int) -> Callable[[int], float]:
  """Temperatura k * e^(-lam * t), actualizada cada stiter iteraciones."""
  return lambda t: k * exp(-lam * (t - t % stiter)) if t < limit else 0


def run_hill_climbing(initial: State, operators: tuple, seed: int) -> State:
  """Ejecuta Hill Climbing de AIMA. La semilla fija el desempate entre sucesores."""
  random.seed(seed)
  return hill_climbing(EnergyProblem(initial, operators)).materialize()


def run_simulated_annealing(initial: State, operators: tuple, seed: int,
                            k: float, lam: float, limit: int,
                            stiter: int) -> State:
  """Ejecuta Simulated Annealing de AIMA."""
  random.seed(seed)
  problem = EnergyProblem(initial, operators, stochastic=True)
  schedule = stepped_schedule(k, lam, limit, stiter)
  return simulated_annealing(problem, schedule).materialize()
