"""
PROJECT: PRACTICAL ASSIGNMENT OF ABIA (UPC)
AUTHORS: ...
DESCRIPTION:
  Define customer characteristics: consumption type, demand, contract and location.
TODO:
  Use these objects to implement customer assignments and service constraints in
  your search state. Keep the problem instance unchanged during the search.
"""

from math import isfinite

# Extra-large customer: generated demand is in [5, 20) MW.
EXTRA_LARGE = 0

# Very large customer: generated demand is in [2, 5) MW.
VERY_LARGE = 1

# Large customer: generated demand is in [1, 3) MW.
LARGE = 2

# Guaranteed contract: the customer must receive the contracted supply.
GUARANTEED = 0

# Non-guaranteed contract: an unserved customer incurs the specified penalty.
NON_GUARANTEED = 1

class Customer:
  """Represent the characteristics of a customer."""

  def __init__(self, 
               customer_type: int, 
               demand_mw: float,
               contract_type: int, 
               x: int, 
               y: int) -> None:
    """Construct a customer after validating its input parameters.

    Args:
      customer_type (int): EXTRA_LARGE, VERY_LARGE or LARGE.
      demand_mw (float): Finite, strictly positive demand in MW. 
      contract_type (int): GUARANTEED or NON_GUARANTEED.
      x (int): Horizontal coordinate; generated instances use values 0 to 99.
      y (int): Vertical coordinate; generated instances use values 0 to 99.

    Raises:
      TypeError: A category or coordinate is not an integer, or the demand
        is not an int or float. Boolean values are rejected.
      ValueError: A category code is unknown, or the demand is non-positive
        or non-finite.
    """
    # Check category codes before storing any instance attributes.
    if type(customer_type) is not int or type(contract_type) is not int:
      raise TypeError("Customer and contract types must be integer codes.")
    if customer_type not in (EXTRA_LARGE, VERY_LARGE, LARGE):
      raise ValueError("Unknown customer type.")
    if contract_type not in (GUARANTEED, NON_GUARANTEED):
      raise ValueError("Unknown contract type.")

    # Demand must be a positive number; bool is not a valid numeric input here.
    if isinstance(demand_mw, bool) or not isinstance(demand_mw, (int, float)):
      raise TypeError("Customer demand must be an int or float.")
    if not isfinite(demand_mw) or demand_mw <= 0:
      raise ValueError("Customer demand must be finite and strictly positive.")

    # Check integer coordinates without imposing generator-only bounds.
    if type(x) is not int or type(y) is not int:
      raise TypeError("Customer coordinates must be integers.")

    self.customer_type = customer_type
    self.demand_mw = float(demand_mw)
    self.contract_type = contract_type
    self.x = x
    self.y = y
