"""
PROJECT: PRACTICAL ASSIGNMENT OF ABIA (UPC)
AUTHORS: ...
DESCRIPTION:
  Demonstrate instance generation and economic lookups. 
  Report hypothetical totals without supply assignments.
  Run with: python demo_energy.py.
TODO:
  Implement your search state, operators, evaluation and AIMA interface in separate
  modules. Build an experimental entry point using this demonstration as a guide.
"""

from math import hypot
from customer import GUARANTEED
from customers import generate_customers
from power_plants import generate_power_plants
from energy import (
  get_guaranteed_tariff, 
  get_non_guaranteed_tariff,
  get_production_cost_per_mw, 
  get_running_cost,
  get_stopped_cost, 
  get_loss_fraction)

def euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
  """Return the Euclidean distance between two coordinate pairs."""
  return hypot(x1 - x2, y1 - y2)

def run_demo() -> None:
  """Print instance summaries without assigning customers to power plants."""
  plants = generate_power_plants((5, 10, 25), seed=4)
  customers = generate_customers(1000, (0.2, 0.3, 0.5), 0.5, seed=1)

  print(f"Power plants: {len(plants)}; customers: {len(customers)}")
  plant = plants[0]
  print(f"Example plant: type={plant.plant_type}, production={plant.production_mw} MW, "
        f"coordinates=({plant.x}, {plant.y})")
  customer = customers[0]
  print(f"Example customer: type={customer.customer_type}, "
        f"demand={customer.demand_mw} MW, contract={customer.contract_type}, "
        f"coordinates=({customer.x}, {customer.y})")

  # Running and stopped costs describe alternative scenarios, not additive costs.
  production = production_cost = running_cost = stopped_cost = 0.0
  for plant in plants:
    production += plant.production_mw
    production_cost += (get_production_cost_per_mw(plant.plant_type) * plant.production_mw)
    running_cost += get_running_cost(plant.plant_type)
    stopped_cost += get_stopped_cost(plant.plant_type)

  # Assume every customer is served only to demonstrate tariff lookups.
  guaranteed_demand = non_guaranteed_demand = revenue = 0.0
  for customer in customers:
    if customer.contract_type == GUARANTEED:
      guaranteed_demand += customer.demand_mw
      tariff = get_guaranteed_tariff(customer.customer_type)
    else:
      non_guaranteed_demand += customer.demand_mw
      tariff = get_non_guaranteed_tariff(customer.customer_type)
    revenue += customer.demand_mw * tariff

  print(f"Production (if every plant runs): {production:.2f} MW")
  print(f"Guaranteed demand: {guaranteed_demand:.2f} MW")
  print(f"Non-guaranteed demand: {non_guaranteed_demand:.2f} MW")
  print(f"Total demand: {guaranteed_demand + non_guaranteed_demand:.2f} MW")
  print(f"Potential revenue if every customer is served: {revenue:.2f} EUR/day")
  print(f"Variable cost if every plant runs: {production_cost:.2f} EUR/day")
  print(f"Fixed cost if every plant runs: {running_cost:.2f} EUR/day")
  print(f"Total cost if every plant runs: {production_cost + running_cost:.2f} EUR/day")
  print(f"Total cost if every plant is stopped: {stopped_cost:.2f} EUR/day")

  # Nearest-plant losses are an unweighted summary; capacity is ignored here.
  loss_sum = 0.0
  for customer in customers:
    distance = min(euclidean_distance(customer.x, customer.y, plant.x, plant.y) for plant in plants)
    loss_sum += get_loss_fraction(distance)
  mean_loss = loss_sum / len(customers)
  print(f"Mean loss to nearest plants (capacity ignored): {mean_loss:.2%}")

if __name__ == "__main__":
  run_demo()
