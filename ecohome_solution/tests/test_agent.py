import unittest

from agent import ask_energy_advisor
from models.energy import initialize_database
from tools import create_rag_index


class EnergyAdvisorIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        initialize_database()
        cls.indexed_passages = create_rag_index()

    def test_ev_schedule_uses_cost_and_solar_evidence(self) -> None:
        answer = ask_energy_advisor(
            "When should I charge my electric car tomorrow to minimize cost and maximize solar power?"
        )
        self.assertIn("EV Charger", answer)
        self.assertIn("estimated savings", answer)
        self.assertIn("solar", answer.lower())
        self.assertIn("Knowledge base: [", answer)

    def test_thermostat_advice_uses_weather_context(self) -> None:
        answer = ask_energy_advisor(
            "What temperature should I set my thermostat on Wednesday afternoon if electricity prices spike?"
        )
        self.assertIn("setpoint", answer)
        self.assertIn("forecast averages", answer.lower())
        self.assertIn("tip_hvac_optimization.txt", answer)

    def test_weekly_pool_schedule_returns_selected_forecast_date(self) -> None:
        answer = ask_energy_advisor(
            "What is the best time to run my pool pump this week based on the weather forecast?"
        )
        self.assertIn("Pool Pump", answer)
        self.assertIn("Forecast:", answer)

    def test_usage_advice_uses_historical_device_data(self) -> None:
        answer = ask_energy_advisor(
            "Suggest three ways I can reduce energy use based on my usage history."
        )
        self.assertIn("highest measured load", answer)
        self.assertIn("Three practical actions", answer)

    def test_rag_index_contains_project_guidance(self) -> None:
        self.assertGreaterEqual(self.indexed_passages, 7)

    def test_roi_inputs_are_reflected_in_response(self) -> None:
        answer = ask_energy_advisor(
            "Estimate the ROI and payback for a home battery.",
            investment_cost_usd=1200,
            annual_savings_usd=400,
            incentives_usd=200,
        )
        self.assertIn("net cost $1000.00", answer)
        self.assertIn("simple payback 2.5 years", answer)


if __name__ == "__main__":
    unittest.main()
