# DemandOS architecture

## Season 1 scope

Establish an installable Python package, documented repository layout, and a
package installation smoke check. No data processing or models are implemented.

## Problem and observational unit

The initial forecasting unit is `(store, product, date)`, with observed daily
units sold as the initial target. M5 is the dataset candidate, with calendar and
price information as potential inputs.

In a simplified setting without replenishment or backorders:

`observed sales = min(customer demand, available stock)`

M5 sales do not supply a complete history of true demand or available inventory.
Do not label observed sales as uncensored demand. Any simulated inventory,
unfulfilled demand, or business costs must be distinguished from measured data.

## Planned flow

1. Preserve original inputs in `data/raw/`.
2. Validate and transform data through `data/interim/` to `data/processed/`.
3. Establish sales forecasting baselines and time-based evaluation.
4. Add probabilistic forecasts and evaluate uncertainty.
5. Evaluate inventory policies using explicit demand and cost assumptions.
6. Introduce serving and monitoring when those milestones are reached.

Steps 1–6 describe future work. Component directories and dependencies will be
introduced only when implementation requires them.

## Evaluation principles

- Use only information available at the forecast origin; avoid future leakage.
- Keep training observations earlier than evaluation periods.
- Evaluate forecast accuracy and uncertainty before relying on predictions.
- Assess inventory costs and service levels under documented assumptions;
  prediction accuracy alone does not establish business value.

## Engineering foundation

Use a `src/` layout so imports resolve through an installed package. Start with
no runtime dependencies. Use Python's standard-library `unittest` for the small
installation smoke check. Add libraries and focused behavioral tests alongside
the components that need them.
