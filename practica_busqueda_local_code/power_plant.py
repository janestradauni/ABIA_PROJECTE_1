"""
PROJECT: PRACTICAL ASSIGNMENT OF ABIA (UPC)
AUTHORS: ...
DESCRIPTION:
  Define power plant characteristics: type, specified production and location.
TODO:
  Represent plant activity and customer assignments in your search state.
  Keep the specified production and other instance data unchanged during search.
"""

from math import isfinite

# Type A plant: generated production is in [250, 750) MW.
TYPE_A = 0

# Type B plant: generated production is in [100, 250) MW.
TYPE_B = 1

# Type C plant: generated production is in [10, 100) MW.
TYPE_C = 2

class PowerPlant:
  """Represent the characteristics of a power plant."""

  def __init__(self, 
               plant_type: int, 
               production_mw: float,
               x: int, 
               y: int) -> None:
    """Construct a power plant after validating its input parameters.

    Args:
      plant_type (int): TYPE_A, TYPE_B or TYPE_C.
      production_mw (float): Finite, strictly positive production in MW when
        running. Integers are also accepted and stored as floats.
      x (int): Horizontal coordinate; generated instances use values 0 to 99.
      y (int): Vertical coordinate; generated instances use values 0 to 99.

    Raises:
      TypeError: The type or a coordinate is not an integer, or production
        is not an int or float. Boolean values are rejected.
      ValueError: The type code is unknown, or production is non-positive
        or non-finite.
    """
    # Check the plant category before storing any instance attributes.
    if type(plant_type) is not int:
      raise TypeError("Plant type must be an integer code.")
    if plant_type not in (TYPE_A, TYPE_B, TYPE_C):
      raise ValueError("Unknown plant type.")

    # Production is an instance parameter, not the current assigned demand.
    if isinstance(production_mw, bool) or not isinstance(production_mw, (int, float)):
      raise TypeError("Plant production must be an int or float.")
    if not isfinite(production_mw) or production_mw <= 0:
      raise ValueError("Plant production must be finite and strictly positive.")

    # Check integer coordinates without imposing generator-only bounds.
    if type(x) is not int or type(y) is not int:
      raise TypeError("Plant coordinates must be integers.")

    self.plant_type = plant_type
    self.production_mw = float(production_mw)
    self.x = x
    self.y = y
