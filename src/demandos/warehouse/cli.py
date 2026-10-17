"""Portable dbt execution plus bounded-result SQL verification and timings."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

from demandos.data.pipeline import fingerprint


MARTS = ("mart_store_daily", "mart_category_daily", "mart_product_daily", "mart_store_performance")
MODELS = ("stg_daily_sales", "dim_product", "dim_store", "dim_calendar", "dim_price_week",
          "fact_daily_sales", *MARTS)


def settings(root):
    """Resolve paths only at runtime; no machine-specific profile is written."""
    root = Path(root).resolve()
    source = root / "data/processed/daily_sales.parquet"
    if not source.is_file():
        raise FileNotFoundError(f"Required Season 2 output missing: {source}")
    if not (root / "dbt/dbt_project.yml").is_file():
        raise FileNotFoundError(f"dbt project missing under {root}")
    output = root / "data/warehouse"
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({
        "DEMANDOS_PARQUET_PATH": source.as_posix(),
        "DEMANDOS_WAREHOUSE_PATH": (output / "demandos.duckdb").as_posix(),
        "DEMANDOS_SPILL_PATH": (output / "spill").as_posix(),
        "DBT_SEND_ANONYMOUS_USAGE_STATS": "false",
        "DBT_USE_COLORS": "false",
    })
    return root, source, output, env


def invoke_dbt(root, env, command):
    """Isolate each invocation and close its database handles before verification."""
    code = ("from dbt.cli.main import dbtRunner; import sys; "
            "result = dbtRunner().invoke(sys.argv[1:]); "
            "raise SystemExit(0 if result.success else 1)")
    args = [sys.executable, "-c", code, *command.split(),
            "--project-dir", str(root / "dbt"), "--profiles-dir", str(root / "dbt")]
    subprocess.run(args, cwd=root, env=env, check=True)


def connection(output, env):
    import duckdb

    con = duckdb.connect(str(output / "demandos.duckdb"), read_only=True)
    con.execute("set threads = 2")
    con.execute("set memory_limit = ?", [env.get("DEMANDOS_MEMORY_LIMIT", "2GB")])
    con.execute("set temp_directory = ?", [env["DEMANDOS_SPILL_PATH"]])
    con.execute("set preserve_insertion_order = false")
    return con


def timed_query(con, sql, parameters=None):
    started = perf_counter()
    cursor = con.execute(sql, parameters or [])
    names = [column[0] for column in cursor.description]
    # Called only for aggregate summaries or bounded teaching results.
    rows = [dict(zip(names, row)) for row in cursor.fetchall()]
    return rows, perf_counter() - started


def project_fingerprint(root):
    """Bind successful tests to the exact portable dbt project, not docs artifacts."""
    digest = hashlib.sha256()
    paths = [root / "dbt/dbt_project.yml", root / "dbt/profiles.yml"]
    for directory in ("models", "tests", "macros"):
        paths.extend(path for path in (root / "dbt" / directory).rglob("*")
                     if path.is_file() and path.suffix in (".sql", ".yml", ".yaml"))
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_artifacts(root, source, output):
    record = json.loads((output / "dbt_build_record.json").read_text(encoding="utf-8"))
    if (record.get("status") != "passed"
            or record.get("source") != fingerprint(source)
            or record.get("project_sha256") != project_fingerprint(root)):
        raise ValueError("Source/project changed or latest build did not pass; run warehouse build")
    return record["run_results"]


def examples(root, con):
    results = {}
    for path in sorted((root / "sql/examples").glob("*.sql")):
        rows, seconds = timed_query(con, path.read_text(encoding="utf-8"))
        results[path.stem] = {"seconds": seconds, "result_rows": len(rows), "results": rows}
    if not results:
        raise ValueError("No SQL examples found")
    return results


def verify(root, source, output, env):
    artifacts = build_artifacts(root, source, output)
    tests = [row for row in artifacts["results"] if row["unique_id"].startswith("test.")]
    if not tests or any(row["status"] != "pass" for row in tests):
        raise ValueError("Latest dbt invocation must contain successful data tests")
    metrics_sql = """
        select count(*) as daily_rows, min(date) as date_min, max(date) as date_max,
            count(distinct item_id) as products, count(distinct store_id) as stores,
            count(distinct (store_id, item_id)) as series,
            sum(units_sold) as units_sold, sum(observed_revenue) as known_revenue,
            count(*) filter (where units_sold = 0) as zero_sales_count,
            count(*) filter (where sell_price is null) as missing_price_count,
            count(*) filter (where observed_revenue is null) as unknown_revenue_count
        from main.fact_daily_sales
    """
    with connection(output, env) as con:
        rows, fact_seconds = timed_query(con, metrics_sql)
        fact = rows[0]
        baseline, source_seconds = timed_query(con, """
            select count(*) as daily_rows, sum(units_sold) as units_sold,
                sum(case when units_sold = 0 then 0
                         else units_sold * cast(sell_price as decimal(12, 4)) end) as known_revenue
            from read_parquet(?)
        """, [str(source)])
        for key in ("daily_rows", "units_sold", "known_revenue"):
            if fact[key] != baseline[0][key]:
                raise ValueError(f"Source/fact mismatch: {key}")
        # This runner is for the full M5 validation history, not a synthetic fixture.
        if fact["daily_rows"] != 58327370:
            raise ValueError("Full-data warehouse requires 58,327,370 observations")
        ingestion_path = root / "data/processed/quality_report.json"
        if ingestion_path.exists():
            ingestion = json.loads(ingestion_path.read_text(encoding="utf-8"))
            for key, old in (("daily_rows", "daily_rows"), ("zero_sales_count", "zero_sales_rows"),
                             ("missing_price_count", "missing_prices"), ("unknown_revenue_count", "positive_sales_missing_price")):
                if fact[key] != ingestion[old]:
                    raise ValueError(f"Season 2/3 mismatch: {key}")
        model_counts = {}
        for name in MODELS:
            model_counts[name] = con.execute(f"select count(*) from main.{name}").fetchone()[0]
        reconciliation = {}
        for name in MARTS:
            totals, seconds = timed_query(con, f"""
                select sum(observation_count) as daily_rows, sum(units_sold) as units_sold,
                    sum(known_revenue) as known_revenue, sum(zero_sales_count) as zero_sales_count,
                    sum(missing_price_count) as missing_price_count,
                    sum(unknown_revenue_count) as unknown_revenue_count
                from main.{name}
            """)
            for key, value in totals[0].items():
                if value != fact[key]:
                    raise ValueError(f"Mart reconciliation failed: {name}.{key}")
            reconciliation[name] = {"status": "passed", "seconds": seconds, "totals": totals[0]}
        example_results = examples(root, con)
        query = """select store_id, sum(units_sold) as units
            from main.fact_daily_sales where date >= date '2016-01-01'
            group by store_id order by units desc"""
        _, representative_seconds = timed_query(con, query)
        plan = con.execute("explain " + query).fetchall()
        (output / "representative_explain.txt").write_text("\n".join(str(row[-1]) for row in plan), encoding="utf-8")
    fact["price_completeness"] = 1 - fact["missing_price_count"] / fact["daily_rows"]
    report = {
        "status": "passed", "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": fingerprint(source), "source_path": "data/processed/daily_sales.parquet",
        "fact": fact, "model_rows": model_counts, "mart_reconciliation": reconciliation,
        "key_integrity": "passed dbt dimension uniqueness, relationships, daily grain and metadata tests",
        "dbt_tests": {"passed": len(tests), "failed": 0,
                      "results": [{"id": r["unique_id"], "status": r["status"]} for r in tests]},
        "dbt_node_seconds": {row["unique_id"]: row["execution_time"] for row in artifacts["results"]},
        "dbt_elapsed_seconds": artifacts["elapsed_time"],
        "query_seconds": {"fact_quality_summary": fact_seconds, "source_reconciliation": source_seconds,
                          "representative_store_query": representative_seconds},
        "examples": example_results,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "logical_cpus": os.cpu_count(), "duckdb_threads": 2, "dbt_workers": 1,
                        "memory_limit": env.get("DEMANDOS_MEMORY_LIMIT", "2GB"),
                        "versions": {name: version(name) for name in ("duckdb", "dbt-core", "dbt-duckdb")}},
        "limitations": ["Sales are observed, not uncensored demand.",
                        "Historical selling-price revenue is not profit.",
                        "Zero units imply zero recorded revenue even if price is unknown; positive units with unknown price imply unknown revenue.",
                        "Dimensions cover only metadata and known prices present in the observed history.",
                        "Local timings depend on hardware, cache state and competing processes."],
    }
    temporary = output / "analytics_quality_report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(output / "analytics_quality_report.json")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build and verify the local DemandOS dbt warehouse")
    parser.add_argument("command", choices=["build", "check", "examples", "docs"])
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root")
    args = parser.parse_args(argv)
    try:
        root, source, output, env = settings(args.root)
        if args.command == "docs":
            invoke_dbt(root, env, "docs generate")
            return 0
        if args.command == "examples":
            with connection(output, env) as con:
                print(json.dumps(examples(root, con), indent=2, default=str))
            return 0
        if args.command == "build":
            record_path = output / "dbt_build_record.json"
            record_path.write_text(json.dumps({"status": "building"}), encoding="utf-8")
            invoke_dbt(root, env, "build")
            record = {
                "status": "passed", "source": fingerprint(source),
                "project_sha256": project_fingerprint(root),
                "run_results": json.loads((root / "dbt/target/run_results.json").read_text(encoding="utf-8")),
            }
            record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        report = verify(root, source, output, env)
        print(f"PASS: {report['fact']['daily_rows']:,} daily rows; {report['dbt_tests']['passed']} dbt tests")
        print(f"Report: {output / 'analytics_quality_report.json'}")
        return 0
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError, ImportError) as error:
        print(f"Warehouse stopped: {error}", file=sys.stderr)
        return 1
