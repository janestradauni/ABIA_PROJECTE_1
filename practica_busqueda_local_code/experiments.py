"""
PROJECT: PRACTICAL ASSIGNMENT OF ABIA (UPC)
AUTHORS: ...
DESCRIPTION:
  Ejecuta los experimentos del enunciado y guarda una fila por ejecucion en
  results/expN.csv. Todas las configuraciones usan las mismas semillas.
  Run with: python experiments.py 1 2 [--workers 14].
"""

from argparse import ArgumentParser
import csv
from multiprocessing import Pool
from pathlib import Path
from time import perf_counter
from customers import generate_customers
from energy_problem import (
  OPERATOR_SETS,
  run_hill_climbing,
  run_simulated_annealing,
)
from energy_state import (
  InfeasibleError,
  Instance,
  State,
  UNSERVED,
  greedy_initial_state,
  random_initial_state,
)
from power_plants import generate_power_plants

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Escenario base del enunciado.
BASE_PLANTS = (5, 10, 25)
BASE_CUSTOMERS = 1000
TYPE_PROPORTIONS = (0.25, 0.30, 0.45)
GUARANTEED_PROPORTION = 0.75

# Repeticiones: el enunciado pide un minimo de 10.
SEEDS = tuple(range(1, 11))

# Ejecuciones por semilla de la solucion inicial aleatoria, que varia entre ellas.
RANDOM_STARTS = 5

# Decisiones fijadas con los resultados de los experimentos anteriores.
CHOSEN_OPERATORS = "all"
CHOSEN_INITIAL = "greedy"
SA_PARAMS = {"k": 300.0, "lam": 1e-3, "limit": 100000, "stiter": 100}

# Iteraciones de la primera exploracion de k y lambda (experimento 3).
SA_EXPLORATION_LIMIT = 300000

# Primer experimento: exploracion de valores extremos de Simulated Annealing.
SA_K_VALUES = (100.0, 1000.0, 10000.0, 100000.0)
SA_LAMBDA_VALUES = (1e-3, 1e-4, 3e-5)

# Penalizaciones del experimento 5, en euros por MW no servido.
PENALTIES = (0, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600)
EXTENDED_PENALTIES = (51200, 102400, 204800)
NEAREST_PENALTIES = (800, 1600, 3200, 6400, 12800, 25600, 51200, 102400, 204800)

# Limites del experimento 4. Hay que parar al perder las instancias factibles.
CUSTOMER_STEPS = tuple(range(250, 2001, 250))
PLANT_MULTIPLIERS = (1, 2, 3, 4, 5, 6)

FIELDS = ("experiment", "algorithm", "operators", "initial", "seed", "repeat",
          "n_customers", "plants_a", "plants_b", "plants_c", "penalty",
          "k", "lam", "limit", "stiter", "initial_profit", "profit", "steps",
          "seconds", "served", "guaranteed_unserved", "valid",
          "running_a", "running_b", "running_c",
          "used_a", "used_b", "used_c", "demand_served")


def build_instance(seed: int, plants: tuple = BASE_PLANTS,
                   n_customers: int = BASE_CUSTOMERS,
                   penalty: float | None = None) -> Instance:
  """Crea una instancia del escenario con las proporciones del enunciado."""
  return Instance(
    generate_power_plants(plants, seed),
    generate_customers(n_customers, TYPE_PROPORTIONS, GUARANTEED_PROPORTION, seed),
    penalty)


def initial_state(instance: Instance, name: str, seed: int) -> State:
  """Construye la solucion inicial indicada."""
  if name == "greedy":
    return greedy_initial_state(instance)
  if name == "random":
    return random_initial_state(instance, seed)
  return State(instance)


def describe(state: State) -> dict:
  """Resume una solucion para guardarla en el CSV."""
  instance = state.instance
  running = [0, 0, 0]
  used = [0.0, 0.0, 0.0]
  for plant, clients in zip(instance.plants, state.n_clients):
    running[plant.plant_type] += clients > 0
  for index, plant in enumerate(instance.plants):
    used[plant.plant_type] += state.used_mw[index]

  served = [c for c, p in enumerate(state.assignment) if p != UNSERVED]
  return {
    "profit": round(state.profit(), 2),
    "steps": state.steps,
    "served": len(served),
    "guaranteed_unserved": sum(
      p == UNSERVED and instance.guaranteed[c]
      for c, p in enumerate(state.assignment)),
    "valid": state.is_valid(),
    "running_a": running[0], "running_b": running[1], "running_c": running[2],
    "used_a": round(used[0], 2), "used_b": round(used[1], 2),
    "used_c": round(used[2], 2),
    "demand_served": round(sum(instance.customers[c].demand_mw for c in served), 2),
  }


def run_task(task: dict) -> dict:
  """Ejecuta una busqueda completa. Devuelve una fila del CSV."""
  plants = task["plants"]
  row = {
    "experiment": task["experiment"], "algorithm": task["algorithm"],
    "operators": task["operators"], "initial": task["initial"],
    "seed": task["seed"], "repeat": task.get("repeat", 0),
    "n_customers": task["n_customers"], "plants_a": plants[0],
    "plants_b": plants[1], "plants_c": plants[2],
    "penalty": task.get("penalty", ""),
  }
  instance = build_instance(task["seed"], plants, task["n_customers"],
                            task.get("penalty"))
  # Cada repeticion aleatoria usa su propia semilla, sin cambiar la instancia.
  run_seed = task["seed"] * 1000 + task.get("repeat", 0)
  initial = initial_state(instance, task["initial"], run_seed)
  operators = OPERATOR_SETS[task["operators"]]

  start = perf_counter()
  if task["algorithm"] == "hc":
    final = run_hill_climbing(initial, operators, run_seed)
  else:
    sa = task["sa"]
    row.update(sa)
    final = run_simulated_annealing(initial, operators, run_seed, **sa)
  row["seconds"] = round(perf_counter() - start, 4)

  row["initial_profit"] = round(initial.profit(), 2)
  row.update(describe(final))
  return row


def task(experiment: int, algorithm: str, seed: int, **overrides) -> dict:
  """Crea una tarea con los valores del escenario base."""
  values = {"experiment": experiment, "algorithm": algorithm, "seed": seed,
            "operators": CHOSEN_OPERATORS, "initial": CHOSEN_INITIAL,
            "plants": BASE_PLANTS, "n_customers": BASE_CUSTOMERS}
  values.update(overrides)
  if algorithm == "sa" and "sa" not in values:
    values["sa"] = SA_PARAMS
  return values


def feasible_seeds(plants: tuple, n_customers: int, count: int = 10,
                   max_seed: int = 60) -> list[int]:
  """Busca semillas cuya instancia admite una solucion valida."""
  seeds = []
  for seed in range(1, max_seed + 1):
    try:
      greedy_initial_state(build_instance(seed, plants, n_customers))
    except InfeasibleError:
      continue
    seeds.append(seed)
    if len(seeds) == count:
      break
  return seeds


def experiment_1() -> list[dict]:
  """Compara los conjuntos de operadores con Hill Climbing."""
  return [task(1, "hc", seed, operators=name, initial=initial, repeat=repeat)
          for name in OPERATOR_SETS
          for initial, repeats in (("greedy", 1), ("random", RANDOM_STARTS))
          for seed in SEEDS
          for repeat in range(repeats)]


def experiment_2() -> list[dict]:
  """Compara las soluciones iniciales con los operadores elegidos."""
  return [task(2, "hc", seed, initial=initial, repeat=repeat)
          for initial, repeats in (("greedy", 1), ("random", RANDOM_STARTS))
          for seed in SEEDS
          for repeat in range(repeats)]


def experiment_3() -> list[dict]:
  """Explora k y lambda de Simulated Annealing, con su referencia en HC."""
  tasks = [task(3, "hc", seed) for seed in SEEDS]
  for k in SA_K_VALUES:
    for lam in SA_LAMBDA_VALUES:
      sa = {"k": k, "lam": lam, "limit": SA_EXPLORATION_LIMIT, "stiter": 100}
      tasks += [task(3, "sa", seed, sa=sa) for seed in SEEDS]
  return tasks


def experiment_3_refined() -> list[dict]:
  """Afina alrededor del mejor punto y estudia iteraciones y stiter."""
  settings = [{"k": k, "lam": lam, "limit": 100000, "stiter": 100}
              for k in (300.0, 1000.0, 3000.0)
              for lam in (3e-4, 1e-3, 3e-3)]
  settings += [{"k": 1000.0, "lam": 1e-3, "limit": 100000, "stiter": stiter}
               for stiter in (1, 10, 1000)]
  settings += [{"k": 1000.0, "lam": 1e-3, "limit": limit, "stiter": 100}
               for limit in (20000, 50000)]
  return [task(30, "sa", seed, sa=sa) for sa in settings for seed in SEEDS]


def experiment_4() -> list[dict]:
  """Mide el tiempo de Hill Climbing al crecer clientes y centrales."""
  tasks = []
  for n_customers in CUSTOMER_STEPS:
    seeds = feasible_seeds(BASE_PLANTS, n_customers)
    print(f"{n_customers} clientes: {len(seeds)} instancias factibles", flush=True)
    tasks += [task(4, "hc", seed, n_customers=n_customers) for seed in seeds]
  for multiplier in PLANT_MULTIPLIERS:
    plants = tuple(n * multiplier for n in BASE_PLANTS)
    seeds = feasible_seeds(plants, BASE_CUSTOMERS)
    print(f"{sum(plants)} centrales: {len(seeds)} instancias factibles", flush=True)
    tasks += [task(4, "hc", seed, plants=plants) for seed in seeds]
  return tasks


def experiment_5() -> list[dict]:
  """Sustituye la restriccion de garantizados por una penalizacion."""
  return [task(5, algorithm, seed, initial="empty", penalty=float(penalty))
          for penalty in PENALTIES
          for algorithm in ("hc", "sa")
          for seed in SEEDS]


def experiment_5_extended() -> list[dict]:
  """Prueba penalizaciones mayores para buscar el rango que siempre es valido."""
  return [task(5, algorithm, seed, initial="empty", penalty=float(penalty))
          for penalty in EXTENDED_PENALTIES
          for algorithm in ("hc", "sa")
          for seed in SEEDS]


def experiment_5_nearest() -> list[dict]:
  """Repite el experimento 5 sirviendo siempre desde la central mas cercana."""
  return [task(5, algorithm, seed, initial="empty", operators="all_nearest",
               penalty=float(penalty))
          for penalty in NEAREST_PENALTIES
          for algorithm in ("hc", "sa")
          for seed in SEEDS]


def experiment_6() -> list[dict]:
  """Estudia el efecto de duplicar y triplicar las centrales de tipo C."""
  return [task(6, algorithm, seed, plants=(5, 10, 25 * factor))
          for factor in (1, 2, 3)
          for algorithm in ("hc", "sa")
          for seed in SEEDS]


# El 30, el 50 y el 51 son rondas adicionales de los experimentos 3 y 5.
EXPERIMENTS = {1: experiment_1, 2: experiment_2, 3: experiment_3,
               30: experiment_3_refined, 4: experiment_4, 5: experiment_5,
               50: experiment_5_extended, 51: experiment_5_nearest,
               6: experiment_6}


def main() -> None:
  parser = ArgumentParser(description=__doc__)
  parser.add_argument("experiments", nargs="+", type=int, choices=EXPERIMENTS)
  parser.add_argument("--workers", type=int, default=14)
  args = parser.parse_args()
  RESULTS_DIR.mkdir(exist_ok=True)

  for number in args.experiments:
    tasks = EXPERIMENTS[number]()
    path = RESULTS_DIR / f"exp{number}.csv"
    start = perf_counter()
    with (path.open("w", newline="", encoding="utf-8") as file,
          Pool(args.workers) as pool):
      writer = csv.DictWriter(file, FIELDS)
      writer.writeheader()
      for done, row in enumerate(pool.imap_unordered(run_task, tasks), 1):
        writer.writerow(row)
        file.flush()
        if done % 20 == 0 or done == len(tasks):
          print(f"exp{number}: {done}/{len(tasks)} "
                f"({perf_counter() - start:.0f} s)", flush=True)


if __name__ == "__main__":
  main()
