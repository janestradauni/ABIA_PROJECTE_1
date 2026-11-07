"""
PROJECT: PRACTICAL ASSIGNMENT OF ABIA (UPC)
AUTHORS: ...
DESCRIPTION:
  Convierte los CSV de results/ en tablas, graficos y una pagina resumen:
    - results/tables/*.csv
    - results/figures/*.png
    - results/resumen.html (se abre con el navegador, sin ejecutar nada)
  Run with: python analysis.py.
"""

from base64 import b64encode
from pathlib import Path
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from scipy import stats

RESULTS = Path(__file__).resolve().parent / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"

# Paleta categorica validada, en orden fijo.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
ALGORITHM_COLORS = {"hc": BLUE, "sa": ORANGE}
ALGORITHM_NAMES = {"hc": "Hill Climbing", "sa": "Simulated Annealing"}
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"

OPERATOR_ORDER = ["move", "move_swap", "move_replace", "move_close", "all"]

# El portatil se suspendio durante estas ejecuciones de Hill Climbing del
# experimento 51: su reloj cuenta esas horas, asi que el tiempo no es valido.
SUSPENDED_PENALTIES = (102400, 204800)

plt.rcParams.update({
  "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
  "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
  "axes.titlelocation": "left", "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
  "text.color": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
  "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
  "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
  "lines.linewidth": 2, "lines.markersize": 7, "legend.frameon": False,
})

THOUSANDS = FuncFormatter(lambda value, _: f"{value:,.0f}".replace(",", "."))
MILLIONS = FuncFormatter(lambda value, _: f"{value / 1e6:.2f} M€".replace(".", ","))


def load(number: int) -> pd.DataFrame:
  """Lee el CSV de un experimento."""
  return pd.read_csv(RESULTS / f"exp{number}.csv")


def summarize(data: pd.DataFrame, by: list[str], **columns: str) -> pd.DataFrame:
  """Calcula n, media y desviacion de cada columna por grupo."""
  grouped = data.groupby(by)
  table = pd.DataFrame({"n": grouped.size()})
  for name, column in columns.items():
    table[f"{name}_media"] = grouped[column].mean()
    table[f"{name}_desv"] = grouped[column].std()
  return table.reset_index()


def save_table(table: pd.DataFrame, name: str) -> pd.DataFrame:
  """Guarda la tabla redondeada y la devuelve."""
  # Las probabilidades y lambda son muy pequenas: redondearlas las dejaria en cero.
  decimals = {column: 2 for column in table.columns
              if not column.startswith("p_") and column != "lam"}
  table = table.round(decimals)
  table.to_csv(TABLES / f"{name}.csv", index=False)
  return table


def save_figure(figure: plt.Figure, name: str) -> None:
  """Guarda el grafico y libera su memoria."""
  figure.tight_layout()
  figure.savefig(FIGURES / f"{name}.png", dpi=170)
  plt.close(figure)


def paired_test(first: pd.Series, second: pd.Series, label: str) -> dict:
  """Compara dos series emparejadas por semilla."""
  difference = first - second
  has_difference = bool(difference.abs().sum())
  wins = int((difference > 0).sum())
  losses = int((difference < 0).sum())
  sign = (stats.binomtest(wins, wins + losses).pvalue
          if wins + losses else float("nan"))
  return {"comparacion": label, "diferencia_media": difference.mean(),
          "gana": wins, "pierde": losses, "empata": int((difference == 0).sum()),
          "p_t_pareada": stats.ttest_rel(first, second).pvalue
          if has_difference else float("nan"),
          "p_signos": sign}


def boxes(axis: plt.Axes, groups: list, labels: list[str], color: str) -> None:
  """Dibuja un diagrama de cajas de un solo color."""
  parts = axis.boxplot(groups, tick_labels=labels, patch_artist=True, widths=0.5,
                       medianprops={"color": TEXT, "linewidth": 2},
                       whiskerprops={"color": MUTED}, capprops={"color": MUTED},
                       flierprops={"markeredgecolor": MUTED, "markersize": 5})
  for box in parts["boxes"]:
    box.set(facecolor=color, alpha=0.55, edgecolor=color)


def experiment_1() -> dict:
  """Operadores."""
  data = load(1)
  data["mejora"] = data["profit"] - data["initial_profit"]
  table = summarize(data, ["initial", "operators"], beneficio="profit",
                    mejora="mejora", pasos="steps", segundos="seconds",
                    servidos="served")

  greedy = data[data["initial"] == "greedy"]
  profit = greedy.pivot(index="seed", columns="operators", values="profit")
  tests = pd.DataFrame([paired_test(profit[a], profit[b], f"{a} - {b}")
                        for a, b in (("all", "move"), ("all", "move_close"),
                                     ("move_close", "move"),
                                     ("move_replace", "move"),
                                     ("move_swap", "move"))])

  labels = [name.replace("_", "\n+ ") for name in OPERATOR_ORDER]
  figure, axes = plt.subplots(1, 2, figsize=(10, 3.8))
  boxes(axes[0], [greedy.loc[greedy["operators"] == name, "mejora"]
                  for name in OPERATOR_ORDER], labels, BLUE)
  axes[0].set_title("Mejora sobre la solución inicial greedy")
  axes[0].set_ylabel("€ por día")
  axes[0].yaxis.set_major_formatter(THOUSANDS)
  boxes(axes[1], [greedy.loc[greedy["operators"] == name, "seconds"]
                  for name in OPERATOR_ORDER], labels, BLUE)
  axes[1].set_title("Tiempo de ejecución")
  axes[1].set_ylabel("segundos")
  save_figure(figure, "exp1_operadores")
  return {"tablas": [("Resultados por conjunto", save_table(table, "exp1_operadores")),
                     ("Comparación por semilla, desde greedy",
                      save_table(tests, "exp1_pruebas"))],
          "figura": "exp1_operadores"}


def experiment_2() -> dict:
  """Solucion inicial."""
  data = load(2)
  table = summarize(data, ["initial"], inicial="initial_profit", final="profit",
                    pasos="steps", segundos="seconds", servidos="served")
  means = data.groupby(["seed", "initial"])["profit"].mean().unstack()
  tests = pd.DataFrame([paired_test(means["greedy"], means["random"],
                                    "greedy - random")])

  figure, axis = plt.subplots(figsize=(6.2, 3.8))
  positions = np.arange(2)
  names = ["greedy", "random"]
  rows = table.set_index("initial").loc[names]
  axis.bar(positions - 0.17, rows["inicial_media"], 0.32, color=BLUE,
           yerr=rows["inicial_desv"], ecolor=MUTED, label="Solución inicial")
  axis.bar(positions + 0.17, rows["final_media"], 0.32, color=ORANGE,
           yerr=rows["final_desv"], ecolor=MUTED, label="Tras Hill Climbing")
  axis.set_xticks(positions, ["Greedy", "Aleatoria"])
  axis.set_title("Beneficio medio antes y después de la búsqueda")
  axis.yaxis.set_major_formatter(MILLIONS)
  axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
  save_figure(figure, "exp2_solucion_inicial")
  return {"tablas": [("Resultados por solución inicial",
                      save_table(table, "exp2_solucion_inicial")),
                     ("Comparación por semilla", save_table(tests, "exp2_pruebas"))],
          "figura": "exp2_solucion_inicial"}


def experiment_3() -> dict:
  """Parametros de Simulated Annealing."""
  first = load(3)
  second = load(30)
  sa = first[first["algorithm"] == "sa"]
  hc = first[first["algorithm"] == "hc"].set_index("seed")["profit"].sort_index()
  hc_mean = hc.mean()

  grid = summarize(sa, ["k", "lam"], beneficio="profit", segundos="seconds",
                   servidos="served")
  refined = summarize(second, ["limit", "stiter", "k", "lam"], beneficio="profit",
                      segundos="seconds", servidos="served")
  chosen = second[(second["k"] == 300) & (second["lam"] == 1e-3)
                  & (second["limit"] == 100000) & (second["stiter"] == 100)]
  tests = pd.DataFrame([paired_test(chosen.set_index("seed")["profit"].sort_index(),
                                    hc, "SA elegido - Hill Climbing")])

  matrix = grid.pivot(index="k", columns="lam", values="beneficio_media")
  figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
  axes[0].imshow(matrix.values, cmap="Blues", aspect="auto")
  axes[0].grid(False)
  axes[0].set_xticks(range(matrix.shape[1]), [f"{v:g}" for v in matrix.columns])
  axes[0].set_yticks(range(matrix.shape[0]), [f"{v:g}" for v in matrix.index])
  axes[0].set_xlabel("λ")
  axes[0].set_ylabel("k")
  axes[0].set_title("Beneficio medio, primera exploración (M€)")
  middle = (np.nanmax(matrix.values) + np.nanmin(matrix.values)) / 2
  for row in range(matrix.shape[0]):
    for column in range(matrix.shape[1]):
      value = matrix.values[row, column]
      axes[0].text(column, row, f"{value / 1e6:.3f}".replace(".", ","),
                   ha="center", va="center",
                   color="white" if value > middle else TEXT)

  # Evolucion con las iteraciones para k = 1000 y lambda = 0,001.
  selected = pd.concat([
    sa[(sa["k"] == 1000) & (sa["lam"] == 1e-3)],
    second[(second["k"] == 1000) & (second["lam"] == 1e-3)
           & (second["stiter"] == 100)]])
  by_limit = selected.groupby("limit")["profit"].agg(["mean", "std", "count"])
  error = by_limit["std"] / np.sqrt(by_limit["count"])
  axes[1].errorbar(by_limit.index, by_limit["mean"], yerr=error, color=ORANGE,
                   marker="o", capsize=3)
  axes[1].axhline(hc_mean, color=BLUE, linestyle="--")
  axes[1].annotate("Hill Climbing", (by_limit.index.max(), hc_mean),
                   xytext=(0, 5), textcoords="offset points", ha="right", color=MUTED)
  axes[1].set_title("Beneficio según iteraciones (k=1000, λ=0,001)")
  axes[1].set_xlabel("iteraciones totales (barras: error estándar)")
  axes[1].xaxis.set_major_formatter(THOUSANDS)
  axes[1].yaxis.set_major_formatter(MILLIONS)
  save_figure(figure, "exp3_parametros_sa")
  return {"tablas": [("Primera exploración de k y λ (300.000 iteraciones)",
                      save_table(grid, "exp3_exploracion")),
                     ("Segunda ronda", save_table(refined, "exp3_afinado")),
                     ("SA elegido frente a Hill Climbing",
                      save_table(tests, "exp3_pruebas"))],
          "figura": "exp3_parametros_sa"}


def experiment_4() -> dict:
  """Escalado del tiempo."""
  data = load(4)
  data["centrales"] = data["plants_a"] + data["plants_b"] + data["plants_c"]
  # El escenario de 1000 clientes y 40 centrales esta en las dos series.
  data = data.drop_duplicates(["seed", "n_customers", "centrales"])
  customers = summarize(data[data["centrales"] == 40], ["n_customers"],
                        segundos="seconds", pasos="steps", beneficio="profit")
  plants = summarize(data[data["n_customers"] == 1000], ["centrales"],
                     segundos="seconds", pasos="steps", beneficio="profit")

  # Exponente b de tiempo = a * centrales^b, ajustado en escala logaritmica.
  exponent = np.polyfit(np.log(plants["centrales"]),
                        np.log(plants["segundos_media"]), 1)[0]
  fits = pd.DataFrame([{"ajuste": "tiempo ~ centrales^b", "b": exponent}])

  figure, axes = plt.subplots(1, 3, figsize=(12, 3.7))
  axes[0].errorbar(customers["n_customers"], customers["segundos_media"],
                   yerr=customers["segundos_desv"], color=BLUE, marker="o", capsize=3)
  axes[0].set_title("Tiempo, 40 centrales")
  axes[0].set_xlabel("clientes (barras: desviación estándar)")
  axes[0].set_ylabel("segundos")
  axes[0].set_ylim(bottom=0)
  axes[1].errorbar(plants["centrales"], plants["segundos_media"],
                   yerr=plants["segundos_desv"], color=BLUE, marker="o", capsize=3)
  axes[1].set_title("Tiempo, 1000 clientes")
  axes[1].set_xlabel("centrales")
  axes[1].set_ylabel("segundos")
  axes[2].plot(plants["centrales"], plants["beneficio_media"], color=BLUE, marker="o")
  axes[2].set_title("Beneficio, 1000 clientes")
  axes[2].set_xlabel("centrales")
  axes[2].yaxis.set_major_formatter(MILLIONS)
  axes[2].set_ylim(bottom=0)
  save_figure(figure, "exp4_escalado")
  return {"tablas": [("Crecimiento de clientes", save_table(customers, "exp4_clientes")),
                     ("Crecimiento de centrales", save_table(plants, "exp4_centrales")),
                     ("Ajuste del exponente", save_table(fits, "exp4_ajuste"))],
          "figura": "exp4_escalado"}


def penalty_table(data: pd.DataFrame) -> pd.DataFrame:
  """Resume las ejecuciones por algoritmo y penalizacion."""
  table = summarize(data, ["algorithm", "penalty"], beneficio="profit",
                    garantizados_sin_servir="guaranteed_unserved",
                    servidos="served", segundos="seconds")
  valid = data.groupby(["algorithm", "penalty"])["valid"].sum().reset_index()
  return table.merge(valid.rename(columns={"valid": "validas"}))


def experiment_5() -> dict:
  """Penalizacion de los clientes garantizados."""
  data = load(5)
  if (RESULTS / "exp50.csv").exists():
    data = pd.concat([data, load(50)])
  table = penalty_table(data)

  figure, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))
  penalties = sorted(table["penalty"].unique())
  positions = {value: index for index, value in enumerate(penalties)}
  labels = [f"{value:,.0f}".replace(",", ".") for value in penalties]
  columns = (("garantizados_sin_servir_media", "Garantizados sin servir (media)"),
             ("validas", "Ejecuciones válidas de 10"),
             ("beneficio_media", "Beneficio real medio"))
  for axis, (column, title) in zip(axes, columns):
    for algorithm, color in ALGORITHM_COLORS.items():
      rows = table[table["algorithm"] == algorithm]
      axis.plot(rows["penalty"].map(positions), rows[column], color=color,
                marker="o", label=ALGORITHM_NAMES[algorithm])
    axis.set_title(title)
    axis.set_xticks(range(len(penalties)), labels, rotation=45, ha="right")
    axis.set_xlabel("penalización (€ por MW)")
  axes[1].set_ylim(-0.5, 10.5)
  axes[2].yaxis.set_major_formatter(MILLIONS)
  handles, names = axes[0].get_legend_handles_labels()
  figure.legend(handles, names, loc="lower center", ncol=2)
  figure.tight_layout(rect=(0, 0.07, 1, 1))
  figure.savefig(FIGURES / "exp5_penalizacion.png", dpi=170)
  plt.close(figure)
  return {"tablas": [("Resultados por penalización",
                      save_table(table, "exp5_penalizacion"))],
          "figura": "exp5_penalizacion"}


def experiment_5_nearest() -> dict:
  """Compara los operadores all y all_nearest con penalizacion."""
  data = load(51)
  suspended = (data["algorithm"] == "hc") & data["penalty"].isin(SUSPENDED_PENALTIES)
  data.loc[suspended, "seconds"] = np.nan
  table = penalty_table(data)
  only_valid = data[data["valid"]].groupby(["algorithm", "penalty"])["profit"].mean()
  table = table.merge(only_valid.rename("beneficio_validas_media").reset_index(),
                      how="left")

  reference = penalty_table(pd.concat([load(5), load(50)]))
  penalties = sorted(table["penalty"].unique())
  reference = reference[reference["penalty"].isin(penalties)]
  positions = {value: index for index, value in enumerate(penalties)}
  labels = [f"{value:,.0f}".replace(",", ".") for value in penalties]

  figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
  columns = (("validas", "Ejecuciones válidas de 10"),
             ("beneficio_media", "Beneficio real medio (incluye no válidas)"))
  for axis, (column, title) in zip(axes, columns):
    for algorithm, color in ALGORITHM_COLORS.items():
      for rows, style, marker, name in ((reference, "--", "s", "all"),
                                        (table, "-", "o", "all_nearest")):
        rows = rows[rows["algorithm"] == algorithm]
        axis.plot(rows["penalty"].map(positions), rows[column], color=color,
                  linestyle=style, marker=marker,
                  label=f"{ALGORITHM_NAMES[algorithm]}, {name}")
    axis.set_title(title)
    axis.set_xticks(range(len(penalties)), labels, rotation=45, ha="right")
    axis.set_xlabel("penalización (€ por MW)")
  axes[0].set_ylim(-0.5, 10.5)
  axes[1].yaxis.set_major_formatter(MILLIONS)
  handles, names = axes[0].get_legend_handles_labels()
  figure.legend(handles, names, loc="lower center", ncol=4)
  figure.tight_layout(rect=(0, 0.07, 1, 1))
  figure.savefig(FIGURES / "exp5_central_cercana.png", dpi=170)
  plt.close(figure)
  return {"tablas": [("Resultados con all_nearest (sin tiempo donde hubo suspensión)",
                      save_table(table, "exp5_central_cercana"))],
          "figura": "exp5_central_cercana"}


def experiment_6() -> dict:
  """Mas centrales de tipo C."""
  data = load(6)
  table = summarize(data, ["plants_c", "algorithm"], beneficio="profit",
                    segundos="seconds", servidos="served",
                    marcha_a="running_a", marcha_b="running_b", marcha_c="running_c",
                    mw_a="used_a", mw_b="used_b", mw_c="used_c")

  tests = []
  for count, rows in data.groupby("plants_c"):
    pivot = rows.pivot(index="seed", columns="algorithm", values="profit")
    tests.append(paired_test(pivot["sa"], pivot["hc"], f"SA - HC, {count} centrales C"))
  tests = pd.DataFrame(tests)

  figure, axes = plt.subplots(1, 3, figsize=(12.5, 4))
  counts = sorted(table["plants_c"].unique())
  labels = [f"{count} C\n{name}" for count in counts for name in ("HC", "SA")]
  order = table.sort_values(["plants_c", "algorithm"])
  positions = np.arange(len(order))
  bottom = np.zeros(len(order))
  for column, color, name in (("mw_a_media", BLUE, "Tipo A"),
                              ("mw_b_media", ORANGE, "Tipo B"),
                              ("mw_c_media", AQUA, "Tipo C")):
    values = order[column].to_numpy()
    axes[0].bar(positions, values, 0.7, bottom=bottom, color=color, label=name,
                edgecolor=SURFACE, linewidth=1.5)
    bottom += values
  axes[0].set_xticks(positions, labels)
  axes[0].set_title("MW asignados por tipo de central")
  axes[0].set_ylabel("MW")
  axes[0].yaxis.set_major_formatter(THOUSANDS)
  axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3)

  for algorithm, color in ALGORITHM_COLORS.items():
    rows = table[table["algorithm"] == algorithm]
    axes[1].errorbar(rows["plants_c"], rows["beneficio_media"],
                     yerr=rows["beneficio_desv"], color=color, marker="o", capsize=3,
                     label=ALGORITHM_NAMES[algorithm])
    axes[2].plot(rows["plants_c"], rows["segundos_media"], color=color, marker="o",
                 label=ALGORITHM_NAMES[algorithm])
  axes[1].set_title("Beneficio medio")
  axes[1].yaxis.set_major_formatter(MILLIONS)
  axes[1].set_xlabel("centrales de tipo C")
  axes[1].set_xticks(counts)
  axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
  axes[2].set_title("Tiempo medio")
  axes[2].set_ylabel("segundos")
  axes[2].set_xlabel("centrales de tipo C")
  axes[2].set_xticks(counts)
  axes[2].set_ylim(bottom=0)
  axes[2].legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
  save_figure(figure, "exp6_centrales_c")
  return {"tablas": [("Resultados por escenario", save_table(table, "exp6_centrales_c")),
                     ("SA frente a HC", save_table(tests, "exp6_pruebas"))],
          "figura": "exp6_centrales_c"}


def write_summary(sections: list[tuple[str, dict]]) -> None:
  """Escribe una pagina con todas las tablas y graficos incrustados."""
  body = []
  for title, section in sections:
    body.append(f"<h2>{title}</h2>")
    image = b64encode((FIGURES / f"{section['figura']}.png").read_bytes()).decode()
    body.append(f'<img alt="{title}" src="data:image/png;base64,{image}">')
    for name, table in section["tablas"]:
      body.append(f"<h3>{name}</h3><div class='scroll'>"
                  f"{table.to_html(index=False, border=0, na_rep='')}</div>")
  page = f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Resultados ABIA</title><style>
body {{ background: {SURFACE}; color: {TEXT}; font: 15px/1.5 system-ui, sans-serif;
  max-width: 1100px; margin: 0 auto; padding: 16px; }}
h2 {{ margin-top: 2.5em; }} h3 {{ color: {MUTED}; font-size: 1em; }}
img {{ max-width: 100%; height: auto; }}
.scroll {{ overflow-x: auto; }}
table {{ border-collapse: collapse; font-variant-numeric: tabular-nums; }}
th, td {{ padding: 4px 10px; text-align: right; border-bottom: 1px solid {GRID}; }}
th {{ color: {MUTED}; }}
</style></head><body><h1>Resultados de los experimentos</h1>
{''.join(body)}</body></html>"""
  (RESULTS / "resumen.html").write_text(page, encoding="utf-8")


def main() -> None:
  TABLES.mkdir(exist_ok=True)
  FIGURES.mkdir(exist_ok=True)
  write_summary([
    ("Experimento 1: operadores", experiment_1()),
    ("Experimento 2: solución inicial", experiment_2()),
    ("Experimento 3: parámetros de Simulated Annealing", experiment_3()),
    ("Experimento 4: tiempo según el tamaño", experiment_4()),
    ("Experimento 5: penalización de garantizados", experiment_5()),
    ("Experimento 5: variante con la central más cercana", experiment_5_nearest()),
    ("Experimento 6: más centrales de tipo C", experiment_6()),
  ])
  print(f"Hecho: {RESULTS / 'resumen.html'}")


if __name__ == "__main__":
  main()
