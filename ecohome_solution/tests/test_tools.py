from datetime import date, timedelta
import tempfile
import unittest
from pathlib import Path

from models.energy import get_day_energy_data, get_usage_history, initialize_database
from tools import (
    calculate_roi,
    optimize_device_schedule,
    optimize_device_schedule_this_week,
    resolve_target_date,
    retrieve_energy_tips,
    summarize_usage_history,
)


class EnergyToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database = Path(self.temporary_directory.name) / "energy.db"
        initialize_database(self.database)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_database_seeds_history_and_forecasts(self) -> None:
        self.assertTrue(get_usage_history(days=7, db_path=self.database))
        self.assertEqual(len(get_day_energy_data(date.today() + timedelta(days=1), self.database)), 24)

    def test_schedule_calculates_cost_and_savings(self) -> None:
        result = optimize_device_schedule("Dishwasher", date.today() + timedelta(days=1), self.database)
        self.assertEqual(result["device"], "Dishwasher")
        self.assertLess(result["start"], result["end"])
        self.assertGreaterEqual(result["estimated_savings_usd"], 0)
        self.assertIn("18:00", result["benchmark"])

    def test_week_schedule_compares_forecast_days(self) -> None:
        result = optimize_device_schedule_this_week("Pool Pump", self.database)
        self.assertIn(result["date"], {
            (date.today() + timedelta(days=offset)).isoformat() for offset in range(7)
        })
        self.assertIn("seven days", result["selection_criterion"])

    def test_roi_reports_simple_payback_and_incentives(self) -> None:
        result = calculate_roi(1200, 400, incentives_usd=200)
        self.assertEqual(result["net_investment_usd"], 1000)
        self.assertEqual(result["simple_payback_years"], 2.5)
        self.assertEqual(result["first_year_roi_percent"], -60.0)

    def test_history_is_aggregated_by_highest_consumption(self) -> None:
        summary = summarize_usage_history(db_path=self.database)
        self.assertTrue(summary)
        self.assertGreaterEqual(summary[0]["consumption_kwh"], summary[-1]["consumption_kwh"])

    def test_weekday_and_relative_date_resolution(self) -> None:
        anchor = date(2026, 9, 30)
        self.assertEqual(resolve_target_date("charge it tomorrow", anchor), date(2026, 10, 1))
        self.assertEqual(resolve_target_date("Wednesday afternoon", anchor), date(2026, 10, 7))

    def test_retrieval_initializes_missing_vectorstore(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            docs = Path(temporary_directory)
            (docs / "solar.txt").write_text("Schedule EV charging when rooftop solar is producing.", encoding="utf-8")
            (docs / "unrelated.txt").write_text("Set a comfortable thermostat temperature in winter.", encoding="utf-8")
            results = retrieve_energy_tips(
                "rooftop solar EV charging", top_k=1, documents_dir=docs,
                vectorstore_dir=docs / "missing-vectorstore",
            )
            self.assertEqual(results["count"], 1)
            self.assertEqual(results["results"][0]["source"], "solar.txt")
            self.assertTrue((docs / "missing-vectorstore" / "chroma.sqlite3").exists())


if __name__ == "__main__":
    unittest.main()
