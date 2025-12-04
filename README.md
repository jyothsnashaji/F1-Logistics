# F1 Logistics Multi-Objective Optimization — Quick Start

This repo implements a multi-objective optimization pipeline for planning an F1 race season logistics schedule. It generates Pareto-optimal schedules using an Adaptive ε-Constraint method and ranks them via D-CRITIC + Modified TOPSIS.

---

## Overview of `solver.py`

At a high level, `solver.py` performs three phases:

1) Data Preparation
- Loads circuits and transport links from CSVs in `data/`.
- Reads season configuration (start/end circuits, number of races K, time window, rest days, sea lead time).
- Structures data into a format suitable for MILP modeling.

2) Optimization (Adaptive ε-Constraint)
- Defines four objectives: Z1 cost (min), Z2 emissions (min), Z3 revenue (max)
- Selects a primary objective to optimize directly.
- Adds ε-constraints for the other objectives using a grid built from utopia/nadir bounds.
- Solves a sequence of MILP problems, each yielding a feasible schedule with binary decisions x_ij^m (taking link (i→j) with mode m) and y_i (race hosted at i).
- Collects all non-dominated solutions into a Decision Matrix (the Pareto frontier).

3) Decision Making (D-CRITIC + Modified TOPSIS)
- Computes weights from D-CRITIC (contrast intensity + distance-correlation conflict).
- Normalizes and weights the Decision Matrix.
- Determines Positive/Negative Ideal Solutions per objective direction.
- Computes closeness scores and ranks schedules.

Outputs typically include:
- A Decision Matrix with objective values per schedule (and optionally decision variables)
- Final ranking table with closeness scores and ranks

---

## Configure `constants.py`

Set or verify the following constants before running:

- DATA FILE PATHS
  - `CIRCUITS_CSV`: path to circuits (e.g., `data/circuits.csv`, or the provided `circuits_toy.csv` / `circuits_mini.csv`).
  - `TRANSPORT_LINKS_CSV`: path to transport links (e.g., `data/transport_links.csv`, or the provided `transport_links_toy.csv` / `transport_links_mini.csv`).
  - If season configuration is read from CSV, set `SEASON_CONFIG_CSV`; otherwise configure via explicit constants below.

- SEASON PARAMETERS
  - `SEASON_START`: circuit_id where the season starts.
  - `SEASON_END`: circuit_id where the season ends.
  - `K_RACES`: total number of required races.
  - `T_MAX`: maximum total days available for the season.
  - `T_MIN_REST`: minimum rest/prep days between consecutive races.
  - `SEA_LEAD_TIME_MIN`: minimum required lead time for sea freight links.
  - `TIME_PER_LINK_DAYS`: time taken per selected transport link (if modeled as constant per link in your solver variant).

- OPTIMIZATION SETTINGS
  - `PRIMARY_OBJECTIVE_IDX`: index of the objective to optimize directly. Use 1-based or 0-based indexing consistently with your code. Typical mapping:
    - 1 (or 0): Z1 — Minimize cost
    - 2 (or 1): Z2 — Minimize emissions
    - 3 (or 2): Z3 — Maximize revenue
  - `NUM_GRID_POINTS`: resolution of ε-grid for constraint objectives (e.g., 10–50). Larger values explore more trade-offs but take longer.
  - Optional solver/backend toggles (e.g., `SOLVER_NAME`, `TIME_LIMIT_SEC`, `MIP_GAP`), if present in your `constants.py`.

- NORMALIZATION/TOPSIS (if configurable)
  - `NORMALIZATION_METHOD`: e.g., vector normalization.
  - `OBJECTIVE_DIRECTIONS`: ensure correct min/max flags for Z1, Z2, Z3.

---

## CSV Formats

Headers are case-sensitive; keep them exactly as shown.

### `CIRCUITS` CSV
- Columns (as in `data/circuits.csv`):
  - `circuit_id` (string, PK): unique circuit identifier
  - `Circuit` (string): human-readable circuit name
  - `Hosting Fees` (float): revenue/hosting fee for the circuit
  - `Source` (string): source URL for the fee figure

Example:
```
circuit_id,Circuit,Hosting Fees,Source
C1,Albert Park (Melbourne-Australia),41000000,https://f1chronicle.com/how-much-does-monaco-pay-to-host-f1/
```

### `TRANSPORT_LINKS` CSV
- Each row is a directed arc for a specific transport mode.
- Columns (as in `data/transport_links.csv`):
  - `link_id` (string, PK): unique identifier for the link/mode row
  - `from_id` (string): origin circuit_id (FK to `CIRCUITS.circuit_id`)
  - `to_id` (string): destination circuit_id (FK to `CIRCUITS.circuit_id`)
  - `from` (string): origin circuit name
  - `to` (string): destination circuit name
  - `mode` (string): e.g., `Air`, `Road`, `Sea`
  - `distance_km` (float): distance between circuits in kilometers
  - `cost` (float): logistical cost (Z1)
  - `emission` (float): CO2e emissions (Z2)
  - `data_sources` (string): sources/notes for the row values

Example:
```
link_id,from_id,to_id,from,to,mode,distance_km,cost,emission,data_sources
L1,C1,C2,Albert Park (Melbourne-Australia),Shanghai International Circuit (China),Air,8078.5,484710.0,242355.0,DEFRA conversion factors (GHG): https://www.gov.uk/government/publications/greenhouse-gas-reporting-conversion-factors-2023; F1 circuits list / calendar: https://en.wikipedia.org/wiki/List_of_Formula_One_circuits and https://www.formula1.com/en/racing/2025
```

## Setup

Use a virtual environment and install dependencies from `requirements.txt`.

```zsh
# From the project root
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## How to Run

Basic run (uses paths and settings from `constants.py`):

```zsh
source .venv/bin/activate
python solver.py
```

Options:
- Switch datasets by changing `CIRCUITS_CSV` and `TRANSPORT_LINKS_CSV` in `constants.py`:
- Tune `PRIMARY_OBJECTIVE_IDX` and `NUM_GRID_POINTS` to explore trade-offs.

Outputs:
- Console/log summary of optimization runs
- Decision Matrix and final rankings (printed or saved, depending on your implementation)

---