"""
PROJECT: PRACTICAL ASSIGNMENT OF ABIA (UPC)
AUTHORS: ...
DESCRIPTION:
  Representacion del problema para la busqueda local:
    - Instance: datos fijos de un escenario, precalculados una sola vez.
    - State: que central sirve a cada cliente (lo unico que cambia al buscar).
  Incluye la funcion heuristica y las dos estrategias de solucion inicial.
"""

from collections.abc import Sequence
from math import fsum, hypot
from random import Random
import numpy as np
from customer import Customer, GUARANTEED
from energy import (
  get_guaranteed_tariff,
  get_loss_fraction,
  get_non_guaranteed_tariff,
  get_non_service_penalty,
  get_production_cost_per_mw,
  get_running_cost,
  get_stopped_cost,
)
from power_plant import PowerPlant

# Valor de assignment[c] para un cliente sin servicio.
UNSERVED = -1

# Margen para los errores de redondeo al sumar y restar potencias.
EPSILON = 1e-9

# Perdida maxima preferida en la solucion inicial aleatoria.
RANDOM_MAX_LOSS = 0.2

# Tipos de accion. Cada accion es una tupla cuyo primer elemento es uno de estos:
#   (MOVE, cliente, central): cambia la central del cliente, o UNSERVED.
#   (SWAP, cliente1, cliente2): dos clientes servidos intercambian central.
#   (REPLACE, sin_servicio, servido): el primero ocupa el lugar del segundo.
#   (CLOSE, central, plan): para la central; plan es ((cliente, destino), ...).
MOVE, SWAP, REPLACE, CLOSE = range(4)


class InfeasibleError(Exception):
  """No se ha podido servir a todos los clientes garantizados."""


class Instance:
  """Datos fijos de un escenario. Todos los estados comparten este objeto."""

  def __init__(self,
               plants: Sequence[PowerPlant],
               customers: Sequence[Customer],
               guaranteed_penalty: float | None = None) -> None:
    """Precalcula los datos del escenario.

    Args:
      plants (Sequence[PowerPlant]): Centrales de la instancia.
      customers (Sequence[Customer]): Clientes de la instancia.
      guaranteed_penalty (float | None): Con None, los garantizados no pueden
        quedarse sin servicio. Con un valor, pueden, y la heuristica suma esos
        euros por MW no servido (experimento 5).
    """
    self.plants = list(plants)

    # El enunciado obliga a recorrer los clientes por (x, y) en Hill Climbing.
    # Ordenarlos aqui una vez hace que el indice ya siga ese orden.
    self.customers = sorted(customers, key=lambda c: (c.x, c.y))

    self.n_plants = len(self.plants)
    self.n_customers = len(self.customers)
    self.capacity = [p.production_mw for p in self.plants]
    self.guaranteed_penalty = guaranteed_penalty

    # Coste diario de cada central segun este en marcha o parada.
    self.running_cost = [
      get_running_cost(p.plant_type)
      + get_production_cost_per_mw(p.plant_type) * p.production_mw
      for p in self.plants]
    self.stopped_cost = [get_stopped_cost(p.plant_type) for p in self.plants]
    self.start_cost = [running - stopped for running, stopped
                       in zip(self.running_cost, self.stopped_cost)]

    # sent_mw[c][p]: potencia que debe salir de p para que lleguen d MW a c.
    self.sent_mw = [
      [c.demand_mw / (1.0 - get_loss_fraction(hypot(c.x - p.x, c.y - p.y)))
       for p in self.plants]
      for c in self.customers]

    # Centrales de cada cliente, de menor a mayor potencia enviada.
    self.plant_order = [
      sorted(range(self.n_plants), key=lambda p, sent=sent: (sent[p], p))
      for sent in self.sent_mw]

    self.guaranteed = [c.contract_type == GUARANTEED for c in self.customers]
    self.can_drop = [guaranteed_penalty is not None or not guaranteed
                     for guaranteed in self.guaranteed]

    # Ingreso si se sirve al cliente e indemnizacion real si no se le sirve.
    self.revenue = [
      c.demand_mw * (get_guaranteed_tariff(c.customer_type)
                     if c.contract_type == GUARANTEED
                     else get_non_guaranteed_tariff(c.customer_type))
      for c in self.customers]
    self.real_penalty = [
      0.0 if c.contract_type == GUARANTEED
      else c.demand_mw * get_non_service_penalty(c.customer_type)
      for c in self.customers]

    # Penalizacion que ve la busqueda: la real mas la del experimento 5.
    self.penalty = [
      c.demand_mw * guaranteed_penalty
      if guaranteed_penalty is not None and c.contract_type == GUARANTEED
      else real
      for c, real in zip(self.customers, self.real_penalty)]

    # Cuanto mejora el coste al pasar a servir al cliente.
    self.gain = [revenue + penalty
                 for revenue, penalty in zip(self.revenue, self.penalty)]

    # Coste de referencia del enunciado: todas las centrales paradas.
    self.reference_cost = fsum(self.stopped_cost)

    self._neighbors: dict[int, list[list[int]]] = {}

  def neighbors(self, k: int) -> list[list[int]]:
    """Devuelve, para cada cliente, sus k clientes mas cercanos."""
    if k not in self._neighbors:
      xy = np.array([(c.x, c.y) for c in self.customers], dtype=float)
      distance = np.hypot(xy[:, None, 0] - xy[None, :, 0],
                          xy[:, None, 1] - xy[None, :, 1])
      np.fill_diagonal(distance, np.inf)
      order = np.argsort(distance, axis=1, kind="stable")[:, :k]
      self._neighbors[k] = [sorted(row) for row in order.tolist()]
    return self._neighbors[k]


class State:
  """Asignacion de clientes a centrales.

  Una central esta en marcha si y solo si tiene algun cliente: parada siempre
  cuesta menos, asi que no hace falta guardar su estado por separado.

  Hill Climbing evalua decenas de miles de sucesores y solo se queda con uno.
  Por eso successor() calcula el valor sin copiar nada: guarda el padre y la
  accion, y materialize() construye las listas solo si el sucesor se elige.
  """

  __slots__ = ("instance", "assignment", "used_mw", "n_clients", "h", "steps",
               "parent", "action")

  def __init__(self, instance: Instance) -> None:
    """Crea el estado vacio: ningun cliente servido."""
    self.instance = instance
    self.assignment = [UNSERVED] * instance.n_customers
    self.used_mw = [0.0] * instance.n_plants
    self.n_clients = [0] * instance.n_plants
    self.steps = 0
    self.parent = None
    self.action = None
    self.refresh()

  def copy(self) -> "State":
    """Copia lo que cambia y comparte la instancia, que es inmutable."""
    other = State.__new__(State)
    other.instance = self.instance
    other.assignment = self.assignment[:]
    other.used_mw = self.used_mw[:]
    other.n_clients = self.n_clients[:]
    other.h = self.h
    other.steps = self.steps
    other.parent = None
    other.action = None
    return other

  def fits(self, customer: int, plant: int) -> bool:
    """Indica si a la central le queda potencia para servir al cliente."""
    instance = self.instance
    return (self.used_mw[plant] + instance.sent_mw[customer][plant]
            <= instance.capacity[plant] + EPSILON)

  def move(self, customer: int, plant: int) -> None:
    """Asigna el cliente a la central, o lo deja sin servicio con UNSERVED."""
    sent_mw = self.instance.sent_mw[customer]
    previous = self.assignment[customer]
    if previous != UNSERVED:
      self.used_mw[previous] -= sent_mw[previous]
      self.n_clients[previous] -= 1
    if plant != UNSERVED:
      self.used_mw[plant] += sent_mw[plant]
      self.n_clients[plant] += 1
    self.assignment[customer] = plant

  def is_valid(self) -> bool:
    """Comprueba capacidades y garantizados recalculando desde cero."""
    instance = self.instance
    used = [[] for _ in range(instance.n_plants)]
    for customer, plant in enumerate(self.assignment):
      if plant != UNSERVED:
        used[plant].append(instance.sent_mw[customer][plant])
      elif instance.guaranteed[customer]:
        return False
    return all(fsum(mw) <= instance.capacity[plant] + EPSILON
               for plant, mw in enumerate(used))

  def _plant_terms(self) -> list[float]:
    """Devuelve el coste diario de cada central."""
    instance = self.instance
    return [instance.running_cost[p] if n > 0 else instance.stopped_cost[p]
            for p, n in enumerate(self.n_clients)]

  def cost(self) -> float:
    """Devuelve el coste que ve la busqueda: costes menos ingresos."""
    instance = self.instance
    terms = self._plant_terms()
    terms += [instance.penalty[c] if plant == UNSERVED else -instance.revenue[c]
              for c, plant in enumerate(self.assignment)]
    return fsum(terms)

  def profit(self) -> float:
    """Devuelve el beneficio real en euros, sin penalizaciones artificiales."""
    instance = self.instance
    terms = [-cost for cost in self._plant_terms()]
    terms += [-instance.real_penalty[c] if plant == UNSERVED
              else instance.revenue[c]
              for c, plant in enumerate(self.assignment)]
    return fsum(terms)

  def heuristic(self) -> float:
    """Devuelve el valor a minimizar, relativo al coste de referencia."""
    return fsum((self.cost(), -self.instance.reference_cost))

  def refresh(self) -> None:
    """Recalcula desde cero el valor heuristico guardado."""
    self.h = self.heuristic()

  def _delta_terms(self, action: tuple) -> list[float]:
    """Devuelve el valor actual y lo que la accion le suma o le resta."""
    instance = self.instance
    terms = [self.h]
    kind = action[0]

    if kind == MOVE:
      _, customer, new = action
      old = self.assignment[customer]
      if old == UNSERVED:
        terms.append(-instance.gain[customer])
      elif self.n_clients[old] == 1:
        terms.append(-instance.start_cost[old])
      if new == UNSERVED:
        terms.append(instance.gain[customer])
      elif self.n_clients[new] == 0:
        terms.append(instance.start_cost[new])

    elif kind == REPLACE:
      _, unserved, served = action
      terms.append(-instance.gain[unserved])
      terms.append(instance.gain[served])

    # SWAP no cambia ni quien se sirve ni que centrales estan en marcha.
    return terms

  def _apply(self, action: tuple) -> None:
    """Aplica la accion sobre este estado."""
    kind = action[0]
    if kind == MOVE:
      self.move(action[1], action[2])
    elif kind == SWAP:
      _, first, second = action
      first_plant = self.assignment[first]
      self.move(first, self.assignment[second])
      self.move(second, first_plant)
    elif kind == REPLACE:
      _, unserved, served = action
      plant = self.assignment[served]
      self.move(served, UNSERVED)
      self.move(unserved, plant)
    else:
      for customer, plant in action[2]:
        self.move(customer, plant)

  def successor(self, action: tuple) -> "State":
    """Devuelve el estado sucesor, con su valor pero aun sin construir."""
    if action[0] == CLOSE:
      state = self.copy()
      state._apply(action)
      state.refresh()
      state.steps = self.steps + 1
      return state

    state = State.__new__(State)
    state.instance = self.instance
    state.assignment = None
    state.used_mw = None
    state.n_clients = None
    state.h = fsum(self._delta_terms(action))
    state.steps = self.steps + 1
    state.parent = self
    state.action = action
    return state

  def materialize(self) -> "State":
    """Construye las listas de un sucesor pendiente y devuelve el estado."""
    if self.assignment is None:
      parent = self.parent
      self.assignment = parent.assignment[:]
      self.used_mw = parent.used_mw[:]
      self.n_clients = parent.n_clients[:]
      self._apply(self.action)
      self.parent = None
      self.action = None
    return self


def greedy_initial_state(instance: Instance) -> State:
  """Solucion buena: cada cliente va a la central que menos potencia pierde.

  Primero coloca a los garantizados, de mayor a menor demanda, porque los
  grandes son los mas dificiles de encajar. Despues anade a los no garantizados
  solo en centrales ya en marcha: esa produccion ya esta pagada, asi que
  servirlos siempre sale a cuenta.

  Raises:
    InfeasibleError: Algun cliente garantizado no cabe en ninguna central.
  """
  state = State(instance)
  by_demand = sorted(range(instance.n_customers),
                     key=lambda c: (-instance.customers[c].demand_mw, c))

  for customer in by_demand:
    if instance.guaranteed[customer]:
      plant = next((p for p in instance.plant_order[customer]
                    if state.fits(customer, p)), None)
      if plant is None:
        raise InfeasibleError(f"Garantizado {customer} sin central disponible.")
      state.move(customer, plant)

  for customer in by_demand:
    if not instance.guaranteed[customer]:
      plant = next((p for p in instance.plant_order[customer]
                    if state.n_clients[p] > 0 and state.fits(customer, p)), None)
      if plant is not None:
        state.move(customer, plant)

  state.refresh()
  return state


def random_initial_state(instance: Instance, seed: int) -> State:
  """Solucion mala: garantizados en centrales al azar y nadie mas servido.

  Sirve para medir cuanto influye partir de una solucion peor.

  Raises:
    InfeasibleError: Algun cliente garantizado no cabe en ninguna central.
  """
  rng = Random(seed)
  state = State(instance)
  customers = [c for c in range(instance.n_customers) if instance.guaranteed[c]]
  rng.shuffle(customers)

  for customer in customers:
    options = [p for p in range(instance.n_plants) if state.fits(customer, p)]
    if not options:
      raise InfeasibleError(f"Garantizado {customer} sin central disponible.")

    # Sin este limite, las perdidas de hasta el 60 % agotan la capacidad total.
    max_sent = instance.customers[customer].demand_mw / (1.0 - RANDOM_MAX_LOSS)
    acceptable = [p for p in options
                  if instance.sent_mw[customer][p] <= max_sent + EPSILON]
    state.move(customer, rng.choice(acceptable or options))

  state.refresh()
  return state
