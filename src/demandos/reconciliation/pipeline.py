"""Execute the real bounded experiment, hard correctness gates, and publication."""

from importlib.metadata import version
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
import tomllib

import numpy as np
import pandas as pd
from scipy import sparse

from demandos.warehouse.cli import settings, connection, build_artifacts
from demandos.features.pipeline import totals
from demandos.backtesting.pipeline import publish
from demandos.backtesting.metrics import rmsse_scale, summarize, common_cases, horizon_band
from demandos.data.pipeline import fingerprint
from demandos.eda.pipeline import json_ready
from demandos.nbeats.training import load
from demandos.nbeats.pipeline import validate_reference_cutoff
from demandos.neural.sequences import Panel
from .hierarchy import KEYS, observed_matrix
from .experiment import (prepare_data, verify_aggregates, direct_history, base_matrix,
                         long_matrix, verify_reference, historical_residuals)
from .covariance import estimate, eligible
from .methods import reconcile


def load_config(path):
    with Path(path).open("rb") as handle:
        document = tomllib.load(handle)
    if set(document) != {"reconciliation"}:
        raise ValueError("Expected [reconciliation]")
    config = document["reconciliation"]
    expected = {"share_days", "max_bottom", "calibration_start", "calibration_origins", "calibration_stride",
                "calibration_epochs", "covariance_cutoff", "shrinkage", "ridge", "min_covariance_origins", "min_covariance_rows"}
    if set(config) != expected:
        raise ValueError("Missing/unknown reconciliation settings")
    for name in ["share_days", "max_bottom", "calibration_origins", "calibration_stride", "calibration_epochs",
                 "min_covariance_origins", "min_covariance_rows"]:
        if type(config[name]) is not int or config[name] <= 0:
            raise ValueError(f"Positive integer required: {name}")
    if config["max_bottom"] > 256 or config["calibration_origins"] > 32 or config["calibration_epochs"] > 3:
        raise ValueError("Experiment exceeds bounded execution caps")
    if not 0 < config["shrinkage"] <= 1 or not 0 < config["ridge"] <= .1:
        raise ValueError("Invalid fixed covariance regularization")
    return config


def fit_covariance(residuals, hierarchy, config):
    return estimate(residuals, hierarchy.nodes.node_id, config["covariance_cutoff"], hierarchy.nodes.members,
                    shrinkage=config["shrinkage"], ridge=config["ridge"],
                    min_origins=config["min_covariance_origins"], min_rows=config["min_covariance_rows"])


def preserve_continuity(base, hierarchy, reference, origin):
    """Compare native inference first, then preserve original serialized predictions exactly."""
    base = base.copy()
    selected = reference[reference.origin.eq(origin)]
    lookup = {(r.store_id, r.item_id): int(r.bottom_col) for r in hierarchy.bottom.itertuples()}
    for key, rows in selected.groupby(KEYS):
        rows = rows.sort_values("horizon")
        if not rows.horizon.tolist() == list(range(1, 29)) or not rows.target_date.eq(origin+pd.to_timedelta(rows.horizon, unit="D")).all():
            raise ValueError("Reference target keys differ")
        node = hierarchy.bottom_rows[lookup[key]]
        np.testing.assert_allclose(base[node], rows.prediction, rtol=2e-6, atol=1e-4)
        base[node] = rows.prediction.to_numpy(float)
    return base


def quality(hierarchy, matrix, method, origin):
    gap = np.abs(matrix-hierarchy.S@matrix[hierarchy.bottom_rows])
    return {"origin_date": origin, "method": method, "max_coherence_gap": float(gap.max()),
            "mean_coherence_gap": float(gap.mean()), "negative_nodes": int((matrix < -1e-10).sum()),
            "negative_bottom": int((matrix[hierarchy.bottom_rows] < -1e-10).sum()),
            "minimum_forecast": float(matrix.min()), "observations": matrix.size}


def future_check(data, origin, model, base, expected, residuals, covariance, reference, config):
    """Mutate all future labels/prices, rebuild every permitted input, and solve again."""
    hierarchy = data["hierarchy"]
    dates, original = data["dates"], data["observed"]
    checked = []
    for change in ("future_sales", "future_prices"):
        daily = data["daily"].copy()
        future = daily.date.gt(origin)
        if change == "future_sales":
            daily.loc[future, "units_sold"] += 1000000
        else:
            daily.loc[future, "sell_price"] = 99999.
        changed_dates, changed = observed_matrix(daily, hierarchy)
        pd.testing.assert_index_equal(dates, changed_dates)
        np.testing.assert_array_equal(original[:, dates <= origin], changed[:, dates <= origin])
        aggregates = hierarchy.S@changed
        new_base, _, _ = base_matrix(Panel(daily, data["calendar"], data["metadata"]), data, origin,
                                    model, data["neural_config"], aggregates, dates)
        new_base = preserve_continuity(new_base, hierarchy, reference, origin)
        np.testing.assert_array_equal(base, new_base)
        rebuilt = residuals.copy()
        row = rebuilt.node_row.to_numpy(int)
        col = dates.get_indexer(pd.to_datetime(rebuilt.target_date))
        if (col < 0).any():
            raise ValueError("Residual target missing from altered source")
        rebuilt["actual"] = aggregates[row, col]
        rebuilt["error"] = rebuilt.actual-rebuilt.prediction
        _, old_errors = eligible(residuals, hierarchy.nodes.node_id, config["covariance_cutoff"])
        _, new_errors = eligible(rebuilt, hierarchy.nodes.node_id, config["covariance_cutoff"])
        pd.testing.assert_frame_equal(old_errors, new_errors)
        new_covariance = fit_covariance(rebuilt, hierarchy, config)
        np.testing.assert_array_equal(covariance["W"], new_covariance["W"])
        take = (dates <= origin) & (dates > origin-pd.Timedelta(days=config["share_days"]))
        prediction, _, allocation = reconcile(hierarchy, new_base, changed[:, take], new_covariance, maximum=config["max_bottom"])
        _, _, original_allocation = reconcile(hierarchy, base, original[:, take], covariance, maximum=config["max_bottom"])
        for name in ["top", "middle"]:
            np.testing.assert_array_equal(allocation[name], original_allocation[name])
        for name in expected:
            np.testing.assert_allclose(expected[name], prediction[name], rtol=0, atol=1e-10)
        checked.append({"origin_date": origin, "perturbation": change, "future_rows_mutated": int(future.sum()),
                        "unchanged_prefix": True, "unchanged_base": True, "unchanged_shares": True,
                        "unchanged_eligible_residuals": True, "unchanged_covariance": True,
                        "unchanged_all_method_forecasts": True})
    return checked


def peak_memory():
    """OS process high-water mark, including Python, Torch and DuckDB allocations."""
    import os
    if os.name != "nt":
        import resource
        import sys
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak/(1024*1024 if sys.platform == "darwin" else 1024)
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)]+[(name, ctypes.c_size_t) for name in
            ["PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
             "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage"]]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise OSError(ctypes.get_last_error(), "Cannot measure process peak memory")
    return counters.PeakWorkingSetSize/1024**2


def run(root, config):
    start = perf_counter()
    root, source, warehouse, env = settings(root)
    build_artifacts(root, source, warehouse)
    paths = [source, root/"data/interim/sales_long.parquet", *sorted((root/"data/raw").glob("*.csv"))]
    sources = {p.relative_to(root).as_posix(): fingerprint(p) for p in paths}
    target = root/"data/reconciliation/m5_cohort_h28"
    with connection(warehouse, env) as con, TemporaryDirectory(prefix=".reconciliation-", dir=root/"data") as temporary:
        temporary = Path(temporary)
        (temporary/"tables").mkdir()
        initial = totals(con)
        expected = json.loads((warehouse/"analytics_quality_report.json").read_text())["fact"]
        if any(str(value) != str(expected[name]) for name, value in initial.items()):
            raise ValueError("Warehouse baseline changed")
        print("Preparing real hierarchy/cohort and checking all 12 groupings", flush=True)
        data = prepare_data(con, root, config)
        hierarchy = data["hierarchy"]
        tables = {"aggregate_checks": pd.concat([verify_aggregates(con, data["full"]),
                     verify_aggregates(con, hierarchy, cohort=True)], ignore_index=True)}
        histories, aggregate = direct_history(con, hierarchy)
        np.testing.assert_array_equal(aggregate, hierarchy.S@data["observed"])
        tables["aggregate_histories"] = histories.merge(hierarchy.nodes[["node_row", "node_id", "level"]], on="node_row", validate="many_to_one")
        reference, provenance = verify_reference(root, data, sources)
        print(f"Full {data['full'].S.shape}; cohort {hierarchy.S.shape}, {hierarchy.S.nnz} memberships", flush=True)
        residuals, fits = historical_residuals(data, aggregate, config, temporary/"models/calibration")
        covariance = fit_covariance(residuals, hierarchy, config)
        tables["historical_residuals"] = residuals
        np.savez_compressed(temporary/"covariance.npz", W=covariance["W"], diagonal=covariance["diagonal"], node_id=hierarchy.nodes.node_id.to_numpy(str))
        frames, bases, diagnostics, allocations, perturbations, timings, checkpoints = [], [], [], [], [], [], []
        for origin in data["origins"]:
            tick = perf_counter()
            directory = root/"data/nbeats/global_h28/models/interpretable"/f"origin_{origin:%Y%m%d}"
            model, metadata = load(directory)
            validate_reference_cutoff(metadata, origin)
            if metadata["variant"] != "interpretable":
                raise ValueError("Wrong N-BEATS checkpoint architecture")
            checkpoints.append({"origin_date": origin, "path": directory.relative_to(root).as_posix(),
                                "metadata": metadata, "metadata_fingerprint": fingerprint(directory/"metadata.json"),
                                "weights": fingerprint(directory/"weights.pt")})
            base, dataset, raw = base_matrix(data["panel"], data, origin, model, data["neural_config"], aggregate, data["dates"])
            base = preserve_continuity(base, hierarchy, reference, origin)
            base_table = long_matrix(base, hierarchy, origin)
            base_table["source_model"] = np.where(base_table.level.eq("ITEM_STORE"), "nbeats_interpretable_checkpoint", "seasonal_naive_7_direct_aggregate")
            cached_bottom = hierarchy.bottom.merge(data["evaluation_cohort"][KEYS], on=KEYS, validate="one_to_one").bottom_col
            cached_nodes = hierarchy.nodes.node_id.iloc[hierarchy.bottom_rows[cached_bottom]]
            base_table["prediction_source"] = np.where(base_table.node_id.isin(cached_nodes),
                "verified_prior_prediction", np.where(base_table.level.eq("ITEM_STORE"), "native_checkpoint_inference", "direct_aggregate_history"))
            base_table["fit_cutoff"] = np.where(base_table.level.eq("ITEM_STORE"), metadata["fit_cutoff"], str(origin.date()))
            base_table["fit_target_max"] = np.where(base_table.level.eq("ITEM_STORE"), metadata["fit_target_max"], str(origin.date()))
            base_table["history_cutoff"] = origin
            base_table["raw_prediction"] = base_table.prediction
            base_table.loc[base_table.level.eq("ITEM_STORE"), "raw_prediction"] = raw.ravel()
            bases.append(base_table)
            take = (data["dates"] <= origin) & (data["dates"] > origin-pd.Timedelta(days=config["share_days"]))
            output, solver, allocation = reconcile(hierarchy, base, data["observed"][:, take], covariance, maximum=config["max_bottom"])
            timings.extend({"origin_date": origin, **row} for row in solver)
            diagnostics.append(quality(hierarchy, base, "independent_base", origin))
            for method, matrix in output.items():
                diagnostics.append(quality(hierarchy, matrix, method, origin))
                values = long_matrix(matrix, hierarchy, origin)
                values["model"] = method
                frames.append(values)
            for kind in ["top", "middle"]:
                group_rows = hierarchy.rows("TOTAL" if kind == "top" else "STORE_DEPARTMENT")
                weights = allocation[kind]
                for col, group in enumerate(group_rows):
                    members = hierarchy.S[group].indices
                    for bottom in members:
                        allocations.append({"origin_date": origin, "kind": kind, "group_node_id": hierarchy.nodes.node_id[group],
                                            "bottom_col": bottom, "share": weights[bottom, col],
                                            "fit_end": origin, "fit_start": data["dates"][take].min(),
                                            **allocation[kind+"_status"].iloc[col].to_dict()})
            perturbations.extend(future_check(data, origin, model, base, output, residuals, covariance, reference, config))
            print(f"Reconciled {origin.date()}: {len(output)} methods; future-sales/price gates passed ({perf_counter()-tick:.1f}s)", flush=True)
        forecasts = pd.concat(frames, ignore_index=True)
        bases = pd.concat(bases, ignore_index=True)
        forecasts["actual"] = aggregate[forecasts.node_row.to_numpy(), data["dates"].get_indexer(forecasts.target_date)]
        scales = []
        for origin in data["origins"]:
            for node, history in enumerate(aggregate[:, data["dates"] <= origin]):
                scales.append({"node_id": hierarchy.nodes.node_id[node], "origin_date": origin, **rmsse_scale(history)})
        forecasts = forecasts.merge(pd.DataFrame(scales), on=["node_id", "origin_date"], validate="many_to_one")
        forecasts = forecasts.assign(series_id=forecasts.node_id, origin=forecasts.origin_date, window="expanding", status="ok")
        forecasts["horizon_band"] = horizon_band(forecasts.horizon)
        methods = sorted(forecasts.model.unique())
        common = common_cases(forecasts, methods)
        expected_rows = len(hierarchy.nodes)*3*28*len(methods)
        if len(common) != expected_rows or forecasts.duplicated(["node_id", "origin_date", "target_date", "horizon", "model"]).any():
            raise ValueError("Reconciliation forecast coverage/keys differ")
        original_nodes = hierarchy.bottom.merge(data["evaluation_cohort"][KEYS], on=KEYS, validate="one_to_one").bottom_col
        node_ids = hierarchy.nodes.node_id.iloc[hierarchy.bottom_rows[original_nodes]].tolist()
        continuity = forecasts[forecasts.node_id.isin(node_ids)].copy()
        # Check original-model errors and train-only scales agree on the exact continuity grid.
        original = continuity[continuity.model.eq("bottom_up")].merge(hierarchy.nodes[["node_id", *KEYS]], on="node_id", validate="many_to_one")
        joined = original.merge(reference, on=[*KEYS, "origin", "target_date", "horizon"], validate="one_to_one", suffixes=("", "_reference"))
        if len(joined) != 1092:
            raise ValueError("Continuity reference alignment incomplete")
        for column in ["prediction", "actual", "scale"]:
            np.testing.assert_allclose(joined[column], joined[column+"_reference"], rtol=1e-12, atol=1e-12)
        tables.update(forecasts=forecasts, independent_bases=bases, allocation_shares=pd.DataFrame(allocations),
                      coherence=pd.DataFrame(diagnostics), invariance=pd.DataFrame(perturbations),
                      solver_timings=pd.DataFrame(timings), scales=pd.DataFrame(scales), continuity=continuity)
        for name, groups in {"metrics_overall": ["model"], "metrics_level": ["level", "model"],
                "metrics_origin": ["origin", "model"], "metrics_level_origin": ["level", "origin", "model"],
                "metrics_horizon": ["horizon", "model"], "metrics_band": ["horizon_band", "model"]}.items():
            tables[name] = summarize(forecasts, groups)
        tables["metrics_continuity"] = summarize(continuity, ["model"])
        for label, h in [("full", data["full"]), ("cohort", hierarchy)]:
            sparse.save_npz(temporary/f"{label}_S.npz", h.S)
            tables[label+"_bottom_map"], tables[label+"_node_map"], tables[label+"_membership"] = h.bottom, h.nodes, h.membership
        tables["training_cohort"] = data["training_cohort"]
        tables["continuity_cohort"] = data["evaluation_cohort"]
        after = {p.relative_to(root).as_posix(): fingerprint(p) for p in paths}
        if after != sources or totals(con) != initial:
            raise ValueError("Source files/warehouse totals mutated")
        for checkpoint in checkpoints:
            if fingerprint(root/checkpoint["path"]/"weights.pt") != checkpoint["weights"]:
                raise ValueError("Reference model mutated")
            if fingerprint(root/checkpoint["path"]/"metadata.json") != checkpoint["metadata_fingerprint"]:
                raise ValueError("Reference checkpoint metadata mutated")
        for name, filename in [("summary", "summary.json"), ("forecasts", "tables/forecasts.parquet")]:
            if fingerprint(root/"data/nbeats/global_h28"/filename) != provenance[name]:
                raise ValueError("Prior N-BEATS reference artifact mutated")
        for name, table in tables.items():
            table.to_parquet(temporary/"tables"/f"{name}.parquet", index=False)
        summary = {"status": "passed", "config": config, "full_hierarchy": data["full"].check(full=True),
            "cohort_hierarchy": hierarchy.check(), "cohort_rule": "Union of 96 deterministic training series selected through 2014-01-04 and the exact 13 continuity series selected through 2016-01-31; 7 overlap. Both cutoffs precede all evaluation targets.",
            "cohort_bottom": len(hierarchy.bottom), "continuity_bottom": 13,
            "expanded_training_selection_cutoff": data["boost"].selection_cutoff,
            "continuity_selection_cutoff": data["evaluation_cohort"].segment_fit_end.max(),
            "origins": data["origins"], "calibration_fits": fits, "checkpoint_provenance": checkpoints,
            "reference_provenance": provenance, "covariance": covariance["diagnostics"], "covariance_method": covariance["method"],
            "base_forecasts": "Valid saved interpretable N-BEATS inference for all 102; original 13 cached predictions preserved after agreement check. Other nodes: direct observed-aggregate seasonal naive, period 7.",
            "historical_recipe": "Independent historical N-BEATS refits with fixed two-epoch budget; no later checkpoint or later validation selection. Architecture/settings fixed, not tuned on calibration or evaluation losses.",
            "methods": methods, "common_cases_per_method": expected_rows//len(methods), "coverage": 1.,
            "skipped": 0, "fallback": covariance["diagnostics"]["fallback"], "invariance_checks": perturbations,
            "inputs_unchanged": True, "warehouse_totals": initial, "source_fingerprints": sources,
            "metrics_overall": tables["metrics_overall"].to_dict("records"),
            "metrics_level": tables["metrics_level"].to_dict("records"),
            "metrics_continuity": tables["metrics_continuity"].to_dict("records"),
            "coherence": diagnostics, "table_rows": {name: len(t) for name, t in tables.items()},
            "runtime_seconds": perf_counter()-start, "peak_process_mib": peak_memory(),
            "versions": {name: version(name) for name in ["numpy", "scipy", "pandas", "torch", "duckdb"]},
            "limitations": ["Cohort-induced totals are not full Walmart/state/store totals; full sparse membership is verified but full-panel forecasts are not produced.",
                "Retrospective bounded M5 validation-history benchmark, not official M5 WRMSSE or a production model ranking.",
                "MinT uses fixed 90% diagonal shrinkage only with sufficient paired rows/origins and full empirical rank. Singular/underdetermined covariance is explicitly downgraded to diagonal WLS; no MinT result is claimed for that fallback.",
                "All-level pooled WAPE counts each bottom observation 12 times through its memberships; macro RMSSE weights each node-origin equally. Only the separate 13-bottom continuity view is comparable with prior seasons.",
                "Historical two-epoch fixed refits and evaluation saved refits share architecture/optimizer/budget, not fitted weights; no historical epoch tuning using later validation outcomes.",
                "Shares are trailing-year observed-sales proportions. New unsupported descendants receive zero share; all-zero supported groups allocate equally; absent/missing supported records fail explicitly.",
                "Linear reconciliations may be negative; these are retained for scoring. Only the constrained method guarantees nonnegative coherence."]}
        from .report import artifacts
        artifacts(summary, tables, temporary)
        (temporary/"summary.json").write_text(json.dumps(json_ready(summary), default=str, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        publish(temporary, target)
    return summary, target
