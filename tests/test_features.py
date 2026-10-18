"""Deterministic synthetic feature fixtures, separate from actual M5 verification."""

from dataclasses import replace
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd


AVAILABLE = importlib.util.find_spec("duckdb") is not None


def fixture(periods=150):
    import duckdb
    from demandos.features.config import FeatureConfig
    from demandos.features.encoding import fit_categories
    config = FeatureConfig(training_origins=("2016-02-28",), mapping_fit_end="2016-02-27")
    dates = pd.date_range("2016-01-01", periods=periods)
    sales = pd.DataFrame([{"store_id": store, "item_id": "I", "state_id": state, "dept_id": "D", "cat_id": "C",
                           "date": when, "units_sold": index + 1 + offset}
                          for store, state, offset in (("CA_1", "CA", 0), ("TX_1", "TX", 1000), ("WI_1", "WI", 2000))
                          for index, when in enumerate(dates)])
    calendar = pd.DataFrame({"date": dates, "wm_yr_wk": np.arange(periods) // 7 + 100,
                             "event_name_1": ["synthetic leap event" if d == pd.Timestamp("2016-02-29") else None for d in dates],
                             "event_type_1": ["Cultural" if d == pd.Timestamp("2016-02-29") else None for d in dates],
                             "event_name_2": None, "event_type_2": None,
                             "snap_CA": [d.day % 2 for d in dates], "snap_TX": [1 - d.day % 2 for d in dates], "snap_WI": 0})
    keys = sales[["store_id", "item_id", "state_id", "dept_id", "cat_id"]].drop_duplicates().assign(first_observed_date=dates[0])
    prices = pd.DataFrame([{"store_id": store, "item_id": "I", "wm_yr_wk": week, "sell_price": float(week - 95)}
                           for store in keys.store_id for week in calendar.wm_yr_wk.unique()])
    con = duckdb.connect()
    con.register("feature_sales", sales[["store_id", "item_id", "date", "units_sold"]])
    con.register("feature_keys", keys)
    con.register("calendar_fixture", calendar)
    con.execute("""create view feature_calendar as select * replace(
        cast(event_name_1 as varchar) as event_name_1, cast(event_type_1 as varchar) as event_type_1,
        cast(event_name_2 as varchar) as event_name_2, cast(event_type_2 as varchar) as event_type_2) from calendar_fixture""")
    con.register("feature_prices", prices)
    mapping = fit_categories(sales, config.mapping_fit_end)
    return con, config, sales, calendar, keys, prices, mapping


@unittest.skipUnless(AVAILABLE, "Install analytics extras")
class FeatureTests(unittest.TestCase):
    def test_exact_lags_rolling_windows_and_series_isolation(self):
        from demandos.features.sql import origin_features
        con, config, sales, _, _, _, _ = fixture()
        with con:
            origin = pd.Timestamp("2016-02-28")
            result = origin_features(con, origin, config)
            for row in result.itertuples():
                expected = sales[sales.store_id.eq(row.store_id) & sales.date.le(origin)].units_sold
                self.assertEqual(row.sales_at_origin, expected.iloc[-1])
                for lag in (1, 7, 14, 28, 56):
                    self.assertEqual(getattr(row, f"origin_lag_{lag}"), expected.iloc[-lag - 1])
                    self.assertEqual(getattr(row, f"origin_lag_{lag}_source_date"), origin - pd.Timedelta(days=lag))
                for w in (7, 14, 28):
                    tail = expected.tail(w)
                    self.assertEqual(getattr(row, f"rolling_observations_{w}"), w)
                    self.assertAlmostEqual(getattr(row, f"rolling_mean_{w}_at_origin"), tail.mean())
                    self.assertEqual(getattr(row, f"rolling_sum_{w}_at_origin"), tail.sum())
                    self.assertAlmostEqual(getattr(row, f"rolling_std_{w}_at_origin"), tail.std(ddof=0))
                    self.assertEqual(getattr(row, f"rolling_max_{w}_at_origin"), tail.max())
                self.assertAlmostEqual(row.level_change_7, expected.tail(7).mean() - expected.iloc[-14:-7].mean())

    def test_zero_sales_short_history_missing_dates_and_duplicates(self):
        from demandos.features.sql import origin_features
        con, config, sales, _, _, _, _ = fixture()
        with con:
            empty_sales = sales.assign(units_sold=0)
            con.register("feature_sales", empty_sales)
            full = origin_features(con, "2016-02-28", config)
            self.assertTrue(full.rolling_zero_fraction_7_at_origin.eq(1).all())
            self.assertTrue(full.rolling_positive_count_7_at_origin.eq(0).all())
            self.assertTrue(full.level_change_7.eq(0).all())
            self.assertTrue(full.relative_level_change_7.isna().all())
            short = origin_features(con, "2016-01-03", config)
            self.assertTrue(short.rolling_mean_7_at_origin.isna().all())
            self.assertTrue(short.rolling_observations_7.eq(3).all())
            self.assertTrue(short.origin_lag_7.isna().all())
            for bad in (sales.drop(index=55), pd.concat([sales, sales.iloc[[55]]]), sales.assign(units_sold=np.nan)):
                con.register("feature_sales", bad)
                with self.assertRaises(ValueError):
                    origin_features(con, "2016-02-28", config)

    def test_target_calendar_snap_event_and_leap_boundaries(self):
        from demandos.features.validation import generate
        con, config, _, calendar, _, _, mapping = fixture()
        with con:
            batch = generate(con, ["2016-02-28", "2016-03-27"], config, mapping)
            frame = batch.features
            self.assertEqual(len(frame), 3 * 2 * 28)
            self.assertTrue(frame.target_date.eq(frame.origin_date + pd.to_timedelta(frame.horizon.astype(int), unit="D")).all())
            leap = frame[frame.target_date.eq(pd.Timestamp("2016-02-29"))]
            self.assertTrue(leap.target_weekday.eq(0).all())
            self.assertTrue(leap.target_event_cultural.all())
            self.assertTrue(leap.target_has_event.all())
            self.assertEqual(set(leap.target_snap), {0, 1})
            for state in ("CA", "TX", "WI"):
                selected = leap[leap.state_id.eq(state)]
                expected = calendar.loc[calendar.date.eq(pd.Timestamp("2016-02-29")), f"snap_{state}"].iloc[0]
                self.assertTrue(selected.target_snap.eq(expected).all())
            self.assertEqual(frame.target_date.max(), pd.Timestamp("2016-04-24"))
            self.assertFalse(frame[frame.target_date.eq(pd.Timestamp("2016-03-01"))].target_has_event.any())

    def test_entire_vector_future_sales_invariance_and_separate_labels(self):
        from demandos.features.validation import generate
        con, config, sales, _, _, _, mapping = fixture()
        with con:
            origin = pd.Timestamp("2016-02-28")
            first = generate(con, [origin], config, mapping, inference_only=False, label_cutoff="2016-03-27")
            changed = sales.copy(deep=True)
            changed.loc[changed.date.gt(origin), "units_sold"] = 1000000
            con.register("feature_sales", changed)
            second = generate(con, [origin], config, mapping, inference_only=False, label_cutoff="2016-03-27")
            pd.testing.assert_frame_equal(first.features, second.features)
            self.assertFalse(first.labels.target_units_sold.equals(second.labels.target_units_sold))
            for h in (1, 7, 14, 28):
                self.assertTrue(first.features.horizon.eq(h).any())
            self.assertNotIn("target_units_sold", first.predictive_columns)
            self.assertNotIn("target_units_sold", first.features)

    def test_inference_requires_no_future_actual_sales(self):
        from demandos.features.validation import generate
        con, config, sales, _, _, _, mapping = fixture()
        with con:
            origin = pd.Timestamp("2016-02-28")
            full = generate(con, [origin], config, mapping)
            con.register("feature_sales", sales[sales.date.le(origin)])
            prefix = generate(con, [origin], config, mapping)
            pd.testing.assert_frame_equal(full.features, prefix.features)
            self.assertIsNone(prefix.labels)
            with self.assertRaises(ValueError):
                generate(con, [origin], config, mapping, inference_only=False, label_cutoff="2016-02-28")

    def test_future_fitted_or_target_relative_features_rejected(self):
        from demandos.features.validation import generate, validate_features
        con, config, _, _, _, _, mapping = fixture()
        with con:
            batch = generate(con, ["2016-02-28"], config, mapping)
            for invalid in (batch.features.assign(target_actual=9), batch.features.assign(segment_fit_end=pd.Timestamp("2016-03-27")),
                            batch.features.assign(mapping_fit_end=pd.Timestamp("2016-03-27"))):
                with self.assertRaises(ValueError):
                    validate_features(invalid, config, tuple(range(1, 29)))
            leaking = batch.features.copy()
            leaking["origin_lag_7_source_date"] = leaking.target_date - pd.Timedelta(days=7)
            with self.assertRaises(ValueError):
                validate_features(leaking, config, tuple(range(1, 29)))

    def test_category_fit_is_training_only_stable_and_unknown(self):
        from demandos.features.encoding import fit_categories
        con, config, sales, _, _, _, mapping = fixture()
        con.close()
        changed = sales.copy()
        changed.loc[changed.date.gt(pd.Timestamp(config.mapping_fit_end)), "item_id"] = "future_item"
        repeated = fit_categories(changed.sample(frac=1, random_state=1), config.mapping_fit_end)
        self.assertEqual(mapping.as_dict(), repeated.as_dict())
        rows = sales.iloc[:1].rename(columns={"date": "origin_date"}).assign(origin_date=pd.Timestamp("2016-02-28"), item_id="unseen")
        self.assertEqual(mapping.transform(rows).item_id_code.iloc[0], 0)
        with self.assertRaises(ValueError):
            mapping.transform(rows.assign(origin_date=pd.Timestamp("2016-01-01")))

    def test_historical_price_provenance_and_future_price_invariance(self):
        from demandos.features.sql import validate_price_evidence
        from demandos.features.validation import generate, validate_features
        con, config, _, calendar, _, prices, mapping = fixture()
        origin = pd.Timestamp("2016-02-28")
        evidence = calendar.iloc[::7][["date", "wm_yr_wk"]].rename(columns={"date": "observed_at"}).assign(store_id="CA_1", item_id="I")
        evidence["available_at"] = evidence.observed_at
        with con:
            verified = validate_price_evidence(con, evidence)
            first = generate(con, [origin], config, mapping, verified_prices=verified)
            known = first.features[first.features.store_id.eq("CA_1")]
            past = verified[pd.to_datetime(verified.observed_at).le(origin)].sort_values("observed_at")
            self.assertTrue(known.last_known_price.eq(float(past.iloc[-1].sell_price)).all())
            self.assertAlmostEqual(known.historical_price_change.iloc[0], float(past.iloc[-1].sell_price / past.iloc[-2].sell_price - 1))
            future_weeks = set(evidence.loc[evidence.observed_at.gt(origin), "wm_yr_wk"])
            changed = prices.copy()
            changed.loc[changed.wm_yr_wk.isin(future_weeks), "sell_price"] = 9999.
            con.register("feature_prices", changed)
            second = generate(con, [origin], config, mapping, verified_prices=validate_price_evidence(con, evidence))
            pd.testing.assert_frame_equal(first.features, second.features)
            bad = first.features.copy()
            mask = bad.store_id.eq("CA_1")
            bad.loc[mask, "last_price_available_at"] = origin + pd.Timedelta(days=1)
            with self.assertRaises(ValueError):
                validate_features(bad, config, tuple(range(1, 29)))
            future = evidence.copy()
            future["available_at"] = origin + pd.Timedelta(days=1)
            unavailable = generate(con, [origin], config, mapping, verified_prices=validate_price_evidence(con, future))
            self.assertTrue(unavailable.features.last_known_price.isna().all())
            with self.assertRaises(ValueError):
                validate_price_evidence(con, pd.concat([evidence, evidence.iloc[:1]]))
            con.register("feature_prices", pd.concat([prices, prices.iloc[:1]]))
            with self.assertRaises(ValueError):
                validate_price_evidence(con, evidence)

    def test_price_zero_denominators_and_short_known_history(self):
        from demandos.features.sql import price_features
        keys = pd.DataFrame({"store_id": ["CA_1"], "item_id": ["I"]})
        evidence = pd.DataFrame({"store_id": "CA_1", "item_id": "I", "wm_yr_wk": [1, 2],
                                 "observed_at": pd.to_datetime(["2016-01-01", "2016-01-08"]),
                                 "available_at": pd.to_datetime(["2016-01-01", "2016-01-08"]),
                                 "sell_price": [0., 2.]})
        frame = price_features(keys, "2016-01-09", evidence)
        self.assertEqual(frame.last_known_price.iloc[0], 2.)
        self.assertEqual(frame.prior_known_price.iloc[0], 0.)
        self.assertTrue(frame.historical_price_change.isna().all())
        self.assertTrue(frame.historical_relative_price_4.isna().all())

    def test_invalid_configuration_origins_and_horizons(self):
        from demandos.features.config import FeatureConfig
        from demandos.features.validation import generate
        for kwargs in ({"lags": [1, 7]}, {"windows": [7, 14]}, {"max_expanded_series": 2000},
                       {"training_origins": ["2015-10-01"]}, {"mapping_fit_end": "2016-01-01"}):
            with self.assertRaises(ValueError):
                FeatureConfig(**kwargs)
        con, config, _, _, _, _, mapping = fixture()
        with con:
            for horizons in ((0,), (29,), (1, 1)):
                with self.assertRaises(ValueError):
                    generate(con, ["2016-02-28"], config, mapping, horizons=horizons)
            with self.assertRaises(ValueError):
                generate(con, ["2016-02-28", "2016-02-28"], config, mapping)

    def test_calendar_fanout_missing_match_and_default_price_nulls(self):
        from demandos.features.validation import generate
        con, config, _, calendar, _, _, mapping = fixture()
        with con:
            batch = generate(con, ["2016-02-28"], config, mapping)
            self.assertTrue(batch.features.last_known_price.isna().all())
            self.assertTrue(batch.features.last_price_missing.all())
            for invalid in (pd.concat([calendar, calendar[calendar.date.eq(pd.Timestamp("2016-02-29"))]]),
                            calendar[calendar.date.ne(pd.Timestamp("2016-02-29"))]):
                con.register("calendar_fixture", invalid)
                with self.assertRaises(ValueError):
                    generate(con, ["2016-02-28"], config, mapping)

    def test_contract_parquet_and_input_immutability(self):
        import pyarrow.parquet as pq
        from demandos.features.validation import generate
        from demandos.features.contracts import feature_dictionary, write_parquet
        con, config, sales, _, _, _, mapping = fixture()
        original = sales.copy(deep=True)
        with con, TemporaryDirectory() as directory:
            batch = generate(con, ["2016-02-28"], config, mapping)
            path = Path(directory) / "features.parquet"
            write_parquet(batch.features, path, feature_dictionary(config))
            self.assertEqual(pq.read_table(path).num_rows, 84)
            self.assertEqual(str(pq.read_schema(path).field("origin_date").type), "date32[day]")
            self.assertEqual(set(batch.features), {entry["name"] for entry in feature_dictionary(config)})
            pd.testing.assert_frame_equal(sales, original)


if __name__ == "__main__":
    unittest.main()
