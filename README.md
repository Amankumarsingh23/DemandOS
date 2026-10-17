# DemandOS

DemandOS is a learning-driven retail demand intelligence project. Its goal is to
turn credible sales forecasts and uncertainty estimates into cost-aware inventory
decisions.

## Current milestone: Season 1

This repository contains the project foundation only. Forecasting models,
inventory optimization, APIs, and dashboards will be added in later sessions.
There are no model results or datasets included yet.

The initial dataset candidate is M5: daily product/store sales, calendar
information, and selling prices. Sales are observations, not necessarily true
demand: stockouts can hide purchases customers wanted to make. Any future demand
estimation or inventory simulation must state its assumptions explicitly.

## Local setup

Requires Python 3.11 or newer. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e .
.\.venv\Scripts\python -c "import demandos; print(demandos.__version__)"
```

On macOS/Linux, use `.venv/bin/python` in place of
`.\.venv\Scripts\python`.

Run the foundation smoke check after installation:

```powershell
.\.venv\Scripts\python -m unittest discover -s tests -v
```

## Repository layout

| Path | Purpose |
| --- | --- |
| `configs/` | Configuration files as components are introduced |
| `data/raw/` | Original input data; keep unchanged |
| `data/interim/` | Intermediate transformations |
| `data/processed/` | Prepared analysis and modeling data |
| `docs/architecture.md` | Scope, assumptions, and planned system flow |
| `docs/decisions/` | Architecture decision records |
| `notebooks/` | Exploratory analysis and learning experiments |
| `src/demandos/` | Installable Python package |
| `scripts/` | Repeatable project commands when needed |
| `tests/` | Automated checks |

Data contents, local environments, secrets, and generated build files are ignored
by Git. Obtain datasets separately under their applicable terms; no download is
automated at this milestone.

See [architecture](docs/architecture.md) and the
[foundation decision](docs/decisions/0001-project-foundation.md) for scope and
design rationale.
