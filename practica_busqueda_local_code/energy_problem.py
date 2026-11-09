"""
PROJECT: PRACTICAL ASSIGNMENT OF ABIA (UPC)
AUTHORS: ...
DESCRIPTION:
  Operadores de busqueda, generacion de sucesores e integracion con AIMA.
  Hill Climbing recibe todos los sucesores; Simulated Annealing, uno al azar.
"""

from collections.abc import Callable
from math import exp, log
from pathlib import Path
from statistics import median
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "aima-python-master"))

from aima.search import Problem, hill_climbing, simulated_annealing
from energy_state import (
  CLOSE, EPSILON, INSERT_RELOCATE, MOVE, OPEN_BATCH, REPLACE, SWAP, UNSERVED, State,
)

# Clientes cercanos con los que se prueba SWAP en Hill Climbing. Probar todos los
# pares daria unos 500.000 sucesores por paso con 1000 clientes.
SWAP_NEIGHBORS = 10

# Intentos para encontrar una accion aplicable al azar.
MAX_RANDOM_TRIES = 1000
RELOCATION_TARGETS = 5

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
  "all_open_batch": (MOVE, SWAP, REPLACE, CLOSE, OPEN_BATCH),
  "all_relocate": (MOVE, SWAP, REPLACE, CLOSE, INSERT_RELOCATE),
}
BASE_OPERATOR_NAMES = ("move", "move_swap", "move_replace", "move_close", "all")


def close_plan(state: State, plant: int) -> tuple | None:
  """Reparte los clientes de una central entre las otras que estan en marcha.

  Los clientes que no caben quedan sin servicio si su contrato lo permite.

  Returns:
    tuple | None: Pares (cliente, destino), o None si algun cliente que debe
      servirse no cabe en ninguna central.
  """
  instance = state.instance
  customers = [c for c, p in enumerate(state.assignment) if p == plant]
  customers.sort(key=lambda c: (instance.can_drop[c], -instance.customers[c].demand_mw, c))
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
  return [(SWAP, first, second)
          for first, second in state.instance.swap_pairs(SWAP_NEIGHBORS)
          if assignment[first] != UNSERVED
          and assignment[second] not in (UNSERVED, assignment[first])
          and _swap_fits(state, first, second)]


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


def _open_batch(state: State, plant: int) -> tuple | None:
  """Construye una apertura que compensa su sobrecoste de activacion."""
  if state.n_clients[plant]:
    return None
  instance = state.instance
  unserved = [c for c, current in enumerate(state.assignment) if current == UNSERVED]
  unserved.sort(key=lambda c: (-instance.gain[c] / instance.sent_mw[c][plant], c))
  load = 0.0
  plan = []
  for customer in unserved:
    sent = instance.sent_mw[customer][plant]
    if load + sent <= instance.capacity[plant] + EPSILON:
      plan.append((customer, plant))
      load += sent
  if len(plan) >= 2 and sum(instance.gain[c] for c, _ in plan) > instance.start_cost[plant]:
    return (OPEN_BATCH, plant, tuple(plan))
  return None


def _open_actions(state: State) -> list[tuple]:
  return [action for plant in range(state.instance.n_plants)
          if (action := _open_batch(state, plant)) is not None]


def _insert_relocate(state: State, customer: int) -> tuple | None:
  """Inserta un garantizado bloqueado recolocando un cliente ya servido."""
  instance = state.instance
  if (state.assignment[customer] != UNSERVED or not instance.guaranteed[customer]
      or any(state.fits(customer, p) for p in range(instance.n_plants))):
    return None
  targets = [p for p in instance.plant_order[customer]
             if instance.sent_mw[customer][p] <= instance.capacity[p] + EPSILON]
  for target in targets[:RELOCATION_TARGETS]:
    required = (state.used_mw[target] + instance.sent_mw[customer][target]
                - instance.capacity[target])
    for occupant, plant in enumerate(state.assignment):
      if plant != target or instance.sent_mw[occupant][target] + EPSILON < required:
        continue
      destination = next((p for p in instance.plant_order[occupant]
                          if p != target and state.n_clients[p] and state.fits(occupant, p)),
                         None)
      if destination is not None:
        return (INSERT_RELOCATE, target, ((occupant, destination), (customer, target)))
  return None


def _relocate_actions(state: State) -> list[tuple]:
  return [action for customer in range(state.instance.n_customers)
          if (action := _insert_relocate(state, customer)) is not None]


GENERATORS = {MOVE: _move_actions, NEAREST_MOVE: _nearest_move_actions,
              SWAP: _swap_actions, REPLACE: _replace_actions,
              CLOSE: _close_actions, OPEN_BATCH: _open_actions,
              INSERT_RELOCATE: _relocate_actions}


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
  """Muestrea el mismo vecindario de SWAP que Hill Climbing."""
  pairs = state.instance.swap_pairs(SWAP_NEIGHBORS)
  if not pairs:
    return None
  first, second = random.choice(pairs)
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


def _random_open(state: State) -> tuple | None:
  return _open_batch(state, random.randrange(state.instance.n_plants))


def _random_relocate(state: State) -> tuple | None:
  return _insert_relocate(state, random.randrange(state.instance.n_customers))


RANDOM_GENERATORS = {MOVE: _random_move, NEAREST_MOVE: _random_nearest_move,
                     SWAP: _random_swap, REPLACE: _random_replace,
                     CLOSE: _random_close, OPEN_BATCH: _random_open,
                     INSERT_RELOCATE: _random_relocate}


def random_action(state: State, operators: tuple, metrics: dict | None = None) -> list[tuple]:
  """Elige un operador al azar y genera un solo sucesor para Simulated Annealing."""
  if not operators or not state.instance.n_customers or not state.instance.n_plants:
    return []
  for _ in range(MAX_RANDOM_TRIES):
    action = RANDOM_GENERATORS[random.choice(operators)](state)
    if action is not None:
      return [action]
  # Un muestreo fallido no demuestra que el estado carezca de sucesores.
  if metrics is not None:
    metrics["fallbacks"] = metrics.get("fallbacks", 0) + 1
  for operator in random.sample(operators, len(operators)):
    actions = GENERATORS[operator](state)
    if actions:
      return [random.choice(actions)]
  return []


class EnergyProblem(Problem):
  """Problema de asignacion para los algoritmos de AIMA."""

  def __init__(self, initial: State, operators: tuple,
               stochastic: bool = False, metrics: dict | None = None) -> None:
    super().__init__(initial)
    self.operators = operators
    self.stochastic = stochastic
    self.metrics = metrics

  def actions(self, state: State) -> list[tuple]:
    state = state.materialize()
    if not self.stochastic:
      return all_actions(state, self.operators)
    actions = random_action(state, self.operators, self.metrics)
    if self.metrics is not None:
      self.metrics["iterations"] += 1
      if not actions:
        self.metrics["stop_reason"] = "no_successors"
    return actions

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
                            stiter: int, *, metrics: dict | None = None) -> State:
  """Ejecuta Simulated Annealing de AIMA."""
  random.seed(seed)
  if metrics is not None:
    metrics.update(iterations=0, fallbacks=0, stop_reason="schedule")
  problem = EnergyProblem(initial, operators, stochastic=True, metrics=metrics)
  schedule = stepped_schedule(k, lam, limit, stiter)
  return simulated_annealing(problem, schedule).materialize()


def calibrate_schedule(initial: State, operators: tuple, seed: int, *,
                       limit: int, stiter: int, target_acceptance: float,
                       samples: int = 128) -> tuple[dict, float | None]:
  """Estima una temperatura a partir de empeoramientos en un paseo piloto."""
  if not 0.01 < target_acceptance < 1:
    raise ValueError("target_acceptance must be between 0.01 and 1")
  if limit <= 0 or stiter <= 0 or limit % stiter or samples <= 0:
    raise ValueError("positive limit, samples and a step size dividing the limit are required")
  random.seed(seed)
  state = initial
  losses = []
  for _ in range(samples):
    actions = random_action(state, operators)
    if not actions:
      break
    successor = state.successor(actions[0]).materialize()
    loss = successor.h - state.h
    if loss > EPSILON:
      losses.append(loss)
    state = successor
  params = {"k": 300.0, "lam": 1e-3, "limit": limit, "stiter": stiter}
  if not losses:
    return params, None
  loss = median(losses)
  start_temperature = -loss / log(target_acceptance)
  cold_temperature = -loss / log(0.01)
  cold_step = max(stiter, int(0.6 * limit) // stiter * stiter)
  params.update(k=start_temperature, lam=log(start_temperature / cold_temperature) / cold_step)
  return params, loss


def polish_transport(initial: State, max_steps: int = 20) -> State:
  """Reduce potencia enviada con los mismos clientes servidos y costes."""
  state = initial
  sent = state.instance.sent_mw
  for _ in range(max_steps):
    best = None
    best_saving = EPSILON
    for customer, old in enumerate(state.assignment):
      if old == UNSERVED or state.n_clients[old] <= 1:
        continue
      for plant in state.instance.plant_order[customer]:
        saving = sent[customer][old] - sent[customer][plant]
        if saving <= best_saving:
          break
        if state.n_clients[plant] and state.fits(customer, plant):
          best, best_saving = (MOVE, customer, plant), saving
    for action in _swap_actions(state):
      _, first, second = action
      p, q = state.assignment[first], state.assignment[second]
      saving = sent[first][p] + sent[second][q] - sent[first][q] - sent[second][p]
      if saving > best_saving:
        best, best_saving = action, saving
    if best is None:
      break
    state = state.successor(best).materialize()
  return state
