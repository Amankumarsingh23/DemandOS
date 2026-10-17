# 0001: Start with a minimal Python foundation

- Status: Accepted
- Date: 2026-10-17

## Context

DemandOS is developed incrementally alongside learning sessions. Season 1 needs
a reproducible package structure without prematurely implementing later stages.

## Decision

Use Python 3.11+, setuptools packaging, and a `src/demandos` package. Keep runtime
dependencies empty until a concrete feature needs them. Preserve empty working
directories with `.gitkeep` files and exclude dataset contents from version
control. Document the distinction between observed sales and true demand.

## Consequences

Developers install the package before running checks. The initial check confirms
that installed package metadata and the import are available. It provides no
evidence about forecasting quality, data quality, or inventory performance.
