"""
PROJECT: PRACTICAL ASSIGNMENT OF ABIA (UPC)
AUTHORS: ...
DESCRIPTION:
  Generate a list of random power plants with the original production ranges.
  Counts are exact for each type. Random generation does not guarantee feasibility.
TODO:
  Choose counts and seeds for experiments. Implement initial customer assignments
  and feasibility checks in your search modules.
"""

from collections.abc import Sequence
from math import floor
from random import Random
from power_plant import PowerPlant

# Production intervals for types A, B and C; upper bounds are excluded.
PRODUCTION_RANGES = (
  (250.0, 750.0),
  (100.0, 250.0),
  (10.0, 100.0),
)

def generate_power_plants(counts: Sequence[int], seed: int) -> list[PowerPlant]:
  """Generate plants ordered by type.

  Args:
    counts (Sequence[int]): Three non-negative counts to create power plants 
      of types A, B and C.
    seed (int): Seed for the local random number generator.

  Returns:
    list[PowerPlant]: Newly constructed power plants.

  Raises:
    ValueError: Counts do not contain exactly three non-negative integers.
  """
  if len(counts) != 3 or any(type(n) is not int or n < 0 for n in counts):
    raise ValueError("Plant counts must contain three non-negative integers.")

  # A local generator keeps experiments reproducible without global RNG changes.
  rng = Random(seed)
  plants = []

  for plant_type, count in enumerate(counts):
    lower, upper = PRODUCTION_RANGES[plant_type]
    for _ in range(count):
      # Sample once per plant; truncate to two decimals.
      production = floor((lower + rng.random() * (upper - lower)) * 100) / 100
      plants.append(PowerPlant(plant_type, production,
                               rng.randrange(100), rng.randrange(100)))

  return plants
