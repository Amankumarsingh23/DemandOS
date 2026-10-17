"""Synthetic warehouse integration checks, separate from real M5 verification."""

from decimal import Decimal
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import pandas as pd


HAS_ANALYTICS = importlib.util.find_spec("duckdb") is not None and importlib.util.find_spec("dbt") is not None


@unittest.skipUnless(HAS_ANALYTICS, "Install .[analytics] for warehouse integration tests")
class WarehouseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = TemporaryDirectory(prefix="demandos-synthetic-")
        cls.addClassCleanup(cls.temporary.cleanup)
        directory = Path(cls.temporary.name)
        cls.database = directory / "synthetic.duckdb"
        source = directory / "synthetic.parquet"
        rows = []
        for day in (1, 2):
            for item in ("synthetic_a", "synthetic_b"):
                rows.append({
                    "id": item + "_store", "item_id": item, "dept_id": "dept",
                    "cat_id": "cat", "store_id": "store", "state_id": "state",
                    "d": f"d_{day}", "date": pd.Timestamp("2011-01-29") + pd.Timedelta(days=day - 1),
                    "wm_yr_wk": 11101, "weekday": "Saturday" if day == 1 else "Sunday",
                    "wday": day, "month": 1, "year": 2011,
                    "event_name_1": None, "event_type_1": None,
                    "event_name_2": None, "event_type_2": None,
                    "snap_CA": 0, "snap_TX": 0, "snap_WI": 0,
                    "units_sold": 0 if day == 1 else (3 if item == "synthetic_a" else 2),
                    "sell_price": 2.5 if item == "synthetic_a" else None,
                })
        pd.DataFrame(rows).to_parquet(source, index=False)
        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ, DEMANDOS_PARQUET_PATH=source.as_posix(),
                   DEMANDOS_WAREHOUSE_PATH=cls.database.as_posix(),
                   DEMANDOS_SPILL_PATH=(directory / "spill").as_posix(),
                   DBT_SEND_ANONYMOUS_USAGE_STATS="false", DBT_USE_COLORS="false")
        code = ("from dbt.cli.main import dbtRunner; import sys; "
                "r = dbtRunner().invoke(sys.argv[1:]); raise SystemExit(0 if r.success else 1)")
        result = subprocess.run([
            sys.executable, "-c", code, "build", "--project-dir", str(root / "dbt"),
            "--profiles-dir", str(root / "dbt"), "--target-path", str(directory / "target"),
            "--log-path", str(directory / "logs"),
            "--vars", json.dumps({"expected_fact_rows": 4, "expected_sales_days": 2}),
        ], capture_output=True, text=True, env=env, cwd=root, timeout=180)
        if result.returncode:
            raise AssertionError("Synthetic dbt build failed:\n" + result.stdout + result.stderr)
        cls.dbt_results = json.loads((directory / "target/run_results.json").read_text())
        cls.manifest = json.loads((directory / "target/manifest.json").read_text())

    def query(self, sql):
        import duckdb

        with duckdb.connect(str(self.database), read_only=True) as con:
            return con.execute(sql).fetchall()

    def test_revenue_null_semantics(self):
        result = self.query("""select units_sold, sell_price, observed_revenue
            from fact_daily_sales where item_id = 'synthetic_b' order by date""")
        self.assertEqual(result, [(0, None, Decimal("0.0000")), (2, None, None)])

    def test_mart_values_and_completeness(self):
        result = self.query("""select units_sold, known_revenue, revenue,
            active_selling_items, zero_sales_count, missing_price_count,
            unknown_revenue_count, price_completeness
            from mart_store_daily order by date""")
        self.assertEqual(result, [(0, Decimal("0.0000"), Decimal("0.0000"), 0, 2, 1, 0, 0.5),
                                  (5, Decimal("7.5000"), None, 2, 0, 1, 1, 0.5)])
        result = self.query("select units_sold, known_revenue, revenue, active_selling_items from mart_store_performance")
        self.assertEqual(result, [(5, Decimal("7.5000"), None, 2)])

    def test_model_grains_and_materialization(self):
        for name, count in (("fact_daily_sales", 4), ("dim_product", 2), ("dim_store", 1),
                            ("dim_calendar", 2), ("dim_price_week", 1), ("mart_product_daily", 4)):
            self.assertEqual(self.query(f"select count(*) from {name}")[0][0], count)
        views = self.query("select table_name from information_schema.views where table_schema = 'main'")
        self.assertIn(("fact_daily_sales",), views)
        self.assertIn(("stg_daily_sales",), views)

    def test_dbt_contracts_pass_on_synthetic_build(self):
        tests = [row for row in self.dbt_results["results"] if row["unique_id"].startswith("test.")]
        self.assertGreater(len(tests), 20)
        self.assertTrue(all(row["status"] == "pass" for row in tests))

    def test_contracts_detect_bad_dimensions_totals_and_nulls(self):
        import duckdb

        mutations = [
            ("unique_dim_product_item_id", "insert into dim_product select * from dim_product limit 1"),
            ("mart_reconciliation", "update mart_store_daily set units_sold = units_sold + 1"),
            ("mart_null_semantics", "update mart_store_daily set revenue = 0 where unknown_revenue_count > 0"),
        ]
        with duckdb.connect(str(self.database)) as con:
            for name, mutation in mutations:
                with self.subTest(contract=name):
                    node = next(node for node in self.manifest["nodes"].values()
                                if node["resource_type"] == "test" and node["name"] == name)
                    con.execute("begin")
                    try:
                        con.execute(mutation)
                        self.assertTrue(con.execute(node["compiled_code"]).fetchall())
                    finally:
                        con.execute("rollback")
