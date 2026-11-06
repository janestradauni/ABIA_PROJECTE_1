"""
PROJECT: PRACTICAL ASSIGNMENT OF ABIA (UPC)
AUTHORS: ...
DESCRIPTION:
  Provide the original daily tariffs, production costs and transport loss table.
  Fixed running costs, variable production costs and stopped costs are distinct.
TODO:
  Combine these values to calculate revenue, penalties, full plant costs and
  transport-adjusted demand when implementing your state evaluation functions.
"""
from math import isfinite

# Rows follow the customer types; columns are guaranteed, non-guaranteed, penalty.
# Values are daily rates per MW, including the penalty for an unserved customer.
CUSTOMER_TARIFFS = (
  (400.0, 300.0, 50.0),
  (500.0, 400.0, 50.0),
  (600.0, 500.0, 50.0),
)

# Rows follow plant types A, B, C: production rate, fixed running cost, stopped cost.
PLANT_COSTS = (
  (50.0, 20000.0, 15000.0),
  (80.0, 10000.0, 5000.0),
  (150.0, 5000.0, 1500.0),
)

# Each threshold is inclusive; distances above 75 have a loss fraction of 0.60.
LOSS_THRESHOLDS = (
  (10.0, 0.0),
  (25.0, 0.1),
  (50.0, 0.2),
  (75.0, 0.4),
)

def _check_type(entity_type: int) -> None:
  """Reject type codes outside the three defined categories."""
  if type(entity_type) is not int or not 0 <= entity_type <= 2:
    raise ValueError("Type must be an integer from zero to two.")

def get_guaranteed_tariff(customer_type: int) -> float:
  """Return the daily tariff per MW for a guaranteed customer."""
  _check_type(customer_type)
  return CUSTOMER_TARIFFS[customer_type][0]

def get_non_guaranteed_tariff(customer_type: int) -> float:
  """Return the daily tariff per MW for a non-guaranteed customer."""
  _check_type(customer_type)
  return CUSTOMER_TARIFFS[customer_type][1]

def get_non_service_penalty(customer_type: int) -> float:
  """Return the daily penalty per MW for an unserved non-guaranteed customer."""
  _check_type(customer_type)
  return CUSTOMER_TARIFFS[customer_type][2]

def get_production_cost_per_mw(plant_type: int) -> float:
  """Return the daily variable production cost per MW."""
  _check_type(plant_type)
  return PLANT_COSTS[plant_type][0]

def get_running_cost(plant_type: int) -> float:
  """Return the daily fixed running cost, excluding variable production costs."""
  _check_type(plant_type)
  return PLANT_COSTS[plant_type][1]

def get_stopped_cost(plant_type: int) -> float:
  """Return the daily fixed cost of a stopped plant."""
  _check_type(plant_type)
  return PLANT_COSTS[plant_type][2]

def get_loss_fraction(distance: float) -> float:
  """Return the fraction of sent power lost at a finite non-negative distance."""
  if not isfinite(distance) or distance < 0:
    raise ValueError("Distance must be finite and non-negative.")
  for threshold, loss in LOSS_THRESHOLDS:
    if distance <= threshold:
      return loss

  # No artificial distance limit or out-of-bounds table access is required.
  return 0.6
