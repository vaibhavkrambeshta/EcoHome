"""LangGraph Energy Advisor with transparent, data-backed recommendations."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from models.energy import DEFAULT_DB_PATH
from tools import (
    calculate_roi,
    get_electricity_prices,
    get_solar_generation,
    get_weather_forecast,
    optimize_device_schedule,
    optimize_device_schedule_this_week,
    resolve_target_date,
    retrieve_energy_tips,
    summarize_usage_history,
)

ENERGY_ADVISOR_SYSTEM_PROMPT = """You are EcoHome's Energy Advisor. Give concise, practical, personalized guidance using the household's history and the available weather, tariff, and solar tools.

Ground every numeric claim in retrieved data or clearly labeled user assumptions. Never present the bundled sample forecast or time-of-use prices as live data. Compare savings against an explicitly named baseline, state the time horizon, and describe savings, solar use, and carbon impact as estimates. Use 0.4 kg CO2e per solar-offset kWh only as the documented illustrative assumption.

Use retrieved energy-saving guidance when relevant and cite its source filename. Do not invent document citations, tariff rates, solar production, or household history. If data is missing, stale, or inconsistent, say so and ask for the needed detail instead of guessing.

Prioritize occupant comfort, health, food safety, EV departure charge requirements, pool circulation needs, equipment limits, and battery backup reserves over bill savings. Treat thermostat setpoints as optional starting points, not medical or safety advice. Do not recommend delaying a required load past its safe or user-required deadline.

Calculate equipment ROI or payback only when investment cost, expected annual savings, and any incentives are supplied. Label simple payback as excluding financing, maintenance, degradation, and tax effects. Avoid promising savings or emissions reductions."""


class AdvisorState(TypedDict, total=False):
    question: str
    target_date: date
    intent: str
    device: str
    context: dict[str, Any]
    answer: str
    investment_cost_usd: float
    annual_savings_usd: float
    incentives_usd: float


_DEVICE_TERMS = (
    ("EV Charger", ("ev", "electric car", "electric vehicle", "car charge", "charge my car")),
    ("Dishwasher", ("dishwasher", "dish washer")),
    ("Pool Pump", ("pool pump", "pool")),
    ("HVAC", ("thermostat", "hvac", "heating", "cooling", "air conditioning", " a/c")),
    ("Washer", ("washer", "washing machine", "laundry")),
)


def _classify(question: str) -> tuple[str, str | None]:
    lowered = question.lower()
    # Match EV as a word, not a substring of unrelated words such as "every".
    device = next(
        (
            name
            for name, terms in _DEVICE_TERMS
            if any(
                bool(re.search(r"\bev\b", lowered)) if term == "ev" else term in lowered
                for term in terms
            )
        ),
        None,
    )
    if device == "Pool Pump" and "pump" not in lowered:
        device = None
    if device == "Washer" and "dryer" in lowered:
        device = "Washer"
    if any(word in lowered for word in ("history", "usage history", "reduce", "three ways", "ways to")):
        return "historical_advice", device
    if device == "HVAC" and any(word in lowered for word in ("temperature", "thermostat", "set")):
        return "thermostat", device
    if device:
        return "schedule", device
    return "general_advice", None


def _format_citations(tips: list[dict[str, str]]) -> str:
    if not tips:
        return ""
    unique_sources = list(dict.fromkeys(tip["source"] for tip in tips))
    return "\n\nKnowledge base: " + ", ".join(f"[{source}]" for source in unique_sources)


def build_energy_advisor(db_path: str | Path | None = None):
    """Build the compiled LangGraph advisor. No external LLM credentials are needed."""
    database = Path(db_path) if db_path else DEFAULT_DB_PATH

    def classify_request(state: AdvisorState) -> dict[str, Any]:
        intent, device = _classify(state["question"])
        return {
            "intent": intent,
            "device": device or "",
            "target_date": resolve_target_date(state["question"]),
        }

    def gather_context(state: AdvisorState) -> dict[str, Any]:
        target_day = state["target_date"]
        context: dict[str, Any] = {"warnings": []}
        if state["intent"] == "schedule" and state.get("device"):
            try:
                # A weekly optimizer chooses both a date and a time; use that date below.
                if "week" in state["question"].lower():
                    context["schedule"] = optimize_device_schedule_this_week(
                        state["device"], database
                    )
                else:
                    context["schedule"] = optimize_device_schedule(
                        state["device"], target_day, database
                    )
                target_day = date.fromisoformat(context["schedule"]["date"])
            except (OSError, ValueError, RuntimeError) as error:
                context["warnings"].append(str(error))
        for key, tool in (
            ("weather", get_weather_forecast),
            ("prices", get_electricity_prices),
            ("solar", get_solar_generation),
        ):
            try:
                context[key] = tool(target_day, database)
            except (OSError, ValueError, RuntimeError) as error:
                context["warnings"].append(str(error))
        try:
            context["tips"] = retrieve_energy_tips(state["question"])
        except (OSError, ValueError, RuntimeError) as error:
            context["tips"] = []
            context["warnings"].append(f"Knowledge retrieval unavailable: {error}")
        if state["intent"] in ("historical_advice", "general_advice"):
            try:
                context["history"] = summarize_usage_history(db_path=database)
            except (OSError, ValueError, RuntimeError) as error:
                context["warnings"].append(f"Usage history unavailable: {error}")
        context["recommendation_date"] = target_day
        # Avoid presenting payback unless both user-provided financial inputs exist.
        if "investment_cost_usd" in state and "annual_savings_usd" in state:
            try:
                context["roi"] = calculate_roi(
                    state["investment_cost_usd"],
                    state["annual_savings_usd"],
                    state.get("incentives_usd", 0.0),
                )
            except (TypeError, ValueError) as error:
                context["warnings"].append(f"ROI inputs are invalid: {error}")
        return {"context": context}

    def compose_recommendation(state: AdvisorState) -> dict[str, str]:
        context = state["context"]
        target_day = context.get("recommendation_date", state["target_date"])
        lines = [
            f"EcoHome Energy Advisor | {target_day.strftime('%A, %B')} {target_day.day}, {target_day.year}"
        ]
        schedule = context.get("schedule")
        if schedule:
            lines.append(
                f"Run the {schedule['device']} from {schedule['start']} to {schedule['end']}. "
                f"Estimated energy cost: ${schedule['estimated_cost_usd']:.2f}; "
                f"estimated savings: ${schedule['estimated_savings_usd']:.2f} versus an "
                "18:00 start at the same duration."
            )
            lines.append(
                f"The estimate uses {schedule['solar_used_kwh']:.1f} kWh of forecast solar "
                f"and {schedule['estimated_grid_energy_kwh']:.1f} kWh of grid energy. "
                f"Estimated solar-related avoided emissions: "
                f"{schedule['estimated_avoided_carbon_kg']:.1f} kg CO2e "
                "(0.4 kg CO2e/kWh assumption)."
            )
            weather = context.get("weather")
            if weather:
                lines.append(
                    f"Forecast: {weather['conditions']}, {weather['cloud_cover_pct']}% average "
                    f"cloud cover, about {weather['daylight_solar_kwh']:.1f} kWh of daylight solar."
                )
        roi = context.get("roi")
        if roi:
            lines.append(
                f"Investment estimate: net cost ${roi['net_investment_usd']:.2f}; "
                f"annual savings ${roi['annual_savings_usd']:.2f}; "
                f"simple payback {roi['simple_payback_years']} years; "
                f"first-year ROI {roi['first_year_roi_percent']}%. "
                "This simple estimate excludes financing, maintenance, degradation, and tax effects."
            )
        elif any(word in state["question"].lower() for word in ("roi", "payback", "return on investment")):
            lines.append(
                "For an ROI estimate, provide the upfront investment, expected annual savings, "
                "and any incentives using the investment_cost_usd, annual_savings_usd, and "
                "incentives_usd inputs."
            )
        elif state["intent"] == "thermostat":
            weather = context.get("weather")
            hot_day = weather and weather["average_temperature_c"] >= 22
            setting = "78 F (26 C)" if hot_day else "68 F (20 C)"
            action = "cooling" if setting.startswith("78") else "heating"
            lines.append(
                f"For the afternoon peak, try a {setting} {action} setpoint if it suits "
                "your comfort and health needs; use a modest pre-cool/pre-heat before the "
                "price spike, then relax the setpoint during the peak. Avoid abrupt recovery swings."
            )
            if context.get("weather"):
                lines.append(
                    f"The forecast averages {context['weather']['average_temperature_c']:.1f} C "
                    f"with {context['weather']['conditions']} conditions."
                )
        else:
            history = context.get("history", [])
            if history:
                biggest = history[0]
                lines.append(
                    f"Your highest measured load over the available history is {biggest['device']} "
                    f"at {biggest['consumption_kwh']:.1f} kWh and ${biggest['cost_usd']:.2f}. "
                    "Start by scheduling flexible loads outside the 17:00-20:00 peak and checking "
                    "whether the largest recurring load has an efficient schedule."
                )
                if state["intent"] == "historical_advice":
                    lines.append("Three practical actions based on that pattern:")
                    lines.append(
                        "1. Review the largest-load device's schedule and reduce unnecessary run time.\n"
                        "2. Move EV, dishwasher, or laundry operation away from 17:00-20:00.\n"
                        "3. Run flexible loads during forecast solar surplus when they are home-safe."
                    )
            else:
                lines.append(
                    "I couldn't find a scheduleable device in the question. Ask about an EV, "
                    "dishwasher, pool pump, HVAC, or washer, or request a usage-history review."
                )
        if context.get("warnings"):
            lines.append("Data note: " + " ".join(dict.fromkeys(context["warnings"])))
        citations = _format_citations(context.get("tips", []))
        if citations:
            lines.append(citations.lstrip())
        else:
            lines.append("Knowledge base: no matching tip was found; run 02_rag_setup.ipynb to build the index.")
        lines.append("Estimates use the bundled sample tariff and forecast data, not live utility or weather feeds.")
        return {"answer": "\n\n".join(lines)}

    # Keep the graph stages explicit: classify, gather evidence, then compose the answer.
    graph = StateGraph(AdvisorState)
    graph.add_node("classify_request", classify_request)
    graph.add_node("gather_context", gather_context)
    graph.add_node("compose_recommendation", compose_recommendation)
    graph.add_edge(START, "classify_request")
    graph.add_edge("classify_request", "gather_context")
    graph.add_edge("gather_context", "compose_recommendation")
    graph.add_edge("compose_recommendation", END)
    return graph.compile()


def ask_energy_advisor(
    question: str,
    db_path: str | Path | None = None,
    *,
    investment_cost_usd: float | None = None,
    annual_savings_usd: float | None = None,
    incentives_usd: float = 0.0,
) -> str:
    """Run the LangGraph advisor and return a cited, data-grounded response."""
    cleaned = question.strip()
    if not cleaned:
        raise ValueError("Please provide an energy optimization question.")
    inputs: AdvisorState = {"question": cleaned}
    if investment_cost_usd is not None:
        inputs["investment_cost_usd"] = investment_cost_usd
    if annual_savings_usd is not None:
        inputs["annual_savings_usd"] = annual_savings_usd
    if incentives_usd:
        inputs["incentives_usd"] = incentives_usd
    result = build_energy_advisor(db_path).invoke(inputs)
    return result["answer"]
