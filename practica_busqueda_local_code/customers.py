"""
PROJECT: PRACTICAL ASSIGNMENT OF ABIA (UPC)
AUTHORS: ...
DESCRIPTION:
  Generate a list of random customers with the original demands and contracts.
  Type and contract proportions are sampling probabilities, not exact counts.
TODO:
  Choose scenario parameters and construct initial assignments in your search
  modules. Check whether all guaranteed customers can be supplied.
"""

from collections.abc import Sequence
from math import floor, isclose, isfinite
from random import Random
from customer import (
  Customer,
  EXTRA_LARGE,
  GUARANTEED,
  LARGE,
  NON_GUARANTEED,
  VERY_LARGE,
)

# Demand intervals for extra-large, very large and large customers, respectively.
# Upper bounds are excluded.
DEMAND_RANGES = (
  (5.0, 20.0),
  (2.0, 5.0),
  (1.0, 2.0),
)

def generate_customers(count: int, 
                       type_proportions: Sequence[float],
                       guaranteed_proportion: float, 
                       seed: int) -> list[Customer]:
  """Generate customers.

  Args:
    count (int): Non-negative number of customers to generate.
    type_proportions (Sequence[float]): Three probabilities, summing to one,
      for extra-large, very large and large customers, respectively.
    guaranteed_proportion (float): Probability of a guaranteed contract, in [0, 1].
    seed (int): Base seed; the local generator uses seed + 1.

  Returns:
    list[Customer]: Newly constructed customers, without supply assignments.

  Raises:
    ValueError: The count or the supplied probabilities are invalid.
  """
  
  # Check count
  if type(count) is not int or count < 0:
    raise ValueError("Customer count must be a non-negative integer.")

  # Check each probability and allow small floating-point errors in the sum.
  if len(type_proportions) != 3 or any(
      not isfinite(p) or not 0 <= p <= 1 for p in type_proportions):
    raise ValueError("Type proportions must contain three probabilities.")
  total = sum(type_proportions)
  if not isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
    raise ValueError("Customer type proportions must sum to one.")
  if not isfinite(guaranteed_proportion) or not 0 <= guaranteed_proportion <= 1:
    raise ValueError("Guaranteed proportion must be between zero and one.")

  # Preserve the original seed offset and normalize the accepted proportions.
  rng = Random(seed + 1)
  first = type_proportions[0] / total
  second = (type_proportions[0] + type_proportions[1]) / total
  customers = []

  for _ in range(count):
    # Select the consumption category from its cumulative probabilities.
    draw = rng.random()
    if draw < first:
      customer_type = EXTRA_LARGE
    elif draw < second:
      customer_type = VERY_LARGE
    else:
      customer_type = LARGE

    contract = (GUARANTEED if rng.random() < guaranteed_proportion
                else NON_GUARANTEED)
    lower, upper = DEMAND_RANGES[customer_type]
    
    # Demand is sampled once and truncated, not rounded, to two decimals.
    demand = floor((lower + rng.random() * (upper - lower)) * 100) / 100
    customers.append(Customer(customer_type, demand, contract,
                              rng.randrange(100), rng.randrange(100)))

  return customers
