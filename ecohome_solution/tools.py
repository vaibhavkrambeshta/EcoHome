"""Data, scheduling, and retrieval tools used by the EcoHome advisor."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from models.energy import DEFAULT_DB_PATH, get_day_energy_data, get_usage_history

PROJECT_ROOT = Path(__file__).resolve().parent
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
VECTORSTORE_DIR = PROJECT_ROOT / "data" / "vectorstore"
EMBEDDING_DIMENSION = 192
DEVICE_SPECS: dict[str, dict[str, float]] = {
    "EV Charger": {"power_kw": 6.0, "duration_hours": 4.0},
    "Dishwasher": {"power_kw": 1.2, "duration_hours": 1.0},
    "Pool Pump": {"power_kw": 0.9, "duration_hours": 4.0},
    "HVAC": {"power_kw": 1.8, "duration_hours": 2.0},
    "Washer": {"power_kw": 0.6, "duration_hours": 1.0},
}


def get_weather_forecast(target_day: date, db_path: str | Path | None = None) -> dict[str, Any]:
    """Summarize the locally seeded hourly weather and solar conditions."""
    rows = get_day_energy_data(target_day, db_path)
    if not rows:
        raise ValueError(f"No forecast data for {target_day}; run 01_db_setup.ipynb first.")
    daylight = [row for row in rows if 9 <= int(row["timestamp"][11:13]) <= 16]
    return {
        "date": target_day.isoformat(),
        "conditions": Counter(row["condition"] for row in rows).most_common(1)[0][0],
        "average_temperature_c": round(sum(row["temperature_c"] for row in rows) / len(rows), 1),
        "cloud_cover_pct": round(sum(row["cloud_cover_pct"] for row in rows) / len(rows)),
        "daylight_solar_kwh": round(sum(row["production_kwh"] for row in daylight), 2),
    }


def get_electricity_prices(target_day: date, db_path: str | Path | None = None) -> list[dict[str, Any]]:
    """Return the day's time-of-use electricity prices."""
    rows = get_day_energy_data(target_day, db_path)
    if not rows:
        raise ValueError(f"No price data for {target_day}; run 01_db_setup.ipynb first.")
    return [{"timestamp": row["timestamp"], "price_per_kwh": row["price_per_kwh"]} for row in rows]


def get_solar_generation(target_day: date, db_path: str | Path | None = None) -> list[dict[str, Any]]:
    """Return hourly solar production forecasts for a day."""
    rows = get_day_energy_data(target_day, db_path)
    if not rows:
        raise ValueError(f"No solar data for {target_day}; run 01_db_setup.ipynb first.")
    return [{"timestamp": row["timestamp"], "production_kwh": row["production_kwh"]} for row in rows]


def optimize_device_schedule(
    device: str,
    target_day: date,
    db_path: str | Path | None = None,
    duration_hours: int | None = None,
) -> dict[str, Any]:
    """Choose the lowest-cost contiguous window using prices and solar surplus."""
    canonical = next((name for name in DEVICE_SPECS if name.lower() == device.lower()), None)
    if canonical is None:
        raise ValueError(f"Unsupported device {device!r}; choose one of {', '.join(DEVICE_SPECS)}.")
    spec = DEVICE_SPECS[canonical]
    duration = duration_hours or int(spec["duration_hours"])
    if duration < 1 or duration > 24:
        raise ValueError("duration_hours must be between 1 and 24")
    rows = get_day_energy_data(target_day, db_path)
    if len(rows) < 24:
        raise ValueError(f"Incomplete forecast data for {target_day}; run 01_db_setup.ipynb first.")

    device_kw = spec["power_kw"]
    base_load_kw = 0.55
    candidates: list[dict[str, Any]] = []
    for start_hour in range(24 - duration + 1):
        slot = rows[start_hour : start_hour + duration]
        incremental_cost = 0.0
        solar_used = 0.0
        grid_energy = 0.0
        for row in slot:
            # Allocate only solar left after the assumed essential household load.
            solar_surplus = max(0.0, row["production_kwh"] - base_load_kw)
            solar_offset = min(device_kw, solar_surplus)
            grid_kwh = device_kw - solar_offset
            incremental_cost += grid_kwh * row["price_per_kwh"]
            solar_used += solar_offset
            grid_energy += grid_kwh
        candidates.append({
            "start_hour": start_hour,
            "cost_usd": incremental_cost,
            "solar_used_kwh": solar_used,
            "grid_energy_kwh": grid_energy,
        })
    best = min(candidates, key=lambda item: item["cost_usd"])
    # Report savings against the same device run beginning at the evening peak.
    benchmark_hour = 18
    benchmark = next(item for item in candidates if item["start_hour"] == benchmark_hour)
    savings = max(0.0, benchmark["cost_usd"] - best["cost_usd"])
    solar_offset_carbon_kg = best["solar_used_kwh"] * 0.4
    return {
        "device": canonical,
        "date": target_day.isoformat(),
        "start": f"{best['start_hour']:02d}:00",
        "end": f"{best['start_hour'] + duration:02d}:00",
        "duration_hours": duration,
        "estimated_cost_usd": round(best["cost_usd"], 2),
        "estimated_savings_usd": round(savings, 2),
        "solar_used_kwh": round(best["solar_used_kwh"], 2),
        "estimated_grid_energy_kwh": round(best["grid_energy_kwh"], 2),
        "estimated_avoided_carbon_kg": round(solar_offset_carbon_kg, 2),
        "benchmark": "same-duration run beginning at 18:00, using forecast tariffs and solar",
    }


def optimize_device_schedule_this_week(
    device: str,
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    """Choose the lowest-cost daily schedule over the next seven forecast days."""
    # Compare every available date instead of treating "this week" as tomorrow.
    daily_options = [
        optimize_device_schedule(device, date.today() + timedelta(days=offset), db_path)
        for offset in range(7)
    ]
    best = min(daily_options, key=lambda option: option["estimated_cost_usd"])
    best["selection_criterion"] = "lowest estimated device energy cost across the next seven days"
    return best


def calculate_roi(
    investment_cost_usd: float,
    annual_savings_usd: float,
    incentives_usd: float = 0.0,
) -> dict[str, float | None]:
    """Calculate simple payback and first-year ROI from user-provided assumptions."""
    if investment_cost_usd < 0 or annual_savings_usd < 0 or incentives_usd < 0:
        raise ValueError("Investment, annual savings, and incentives must not be negative.")
    net_investment = max(0.0, investment_cost_usd - incentives_usd)
    payback_years = net_investment / annual_savings_usd if annual_savings_usd else None
    roi_percent = (
        (annual_savings_usd - net_investment) / net_investment * 100
        if net_investment
        else None
    )
    return {
        "net_investment_usd": round(net_investment, 2),
        "annual_savings_usd": round(annual_savings_usd, 2),
        "simple_payback_years": round(payback_years, 2) if payback_years is not None else None,
        "first_year_roi_percent": round(roi_percent, 1) if roi_percent is not None else None,
    }


def summarize_usage_history(
    days: int = 7,
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Aggregate past device consumption and cost, highest use first."""
    rows = get_usage_history(days=days, db_path=db_path)
    totals: dict[str, dict[str, float]] = {}
    for row in rows:
        total = totals.setdefault(row["device"], {"consumption_kwh": 0.0, "cost_usd": 0.0})
        total["consumption_kwh"] += row["consumption_kwh"]
        total["cost_usd"] += row["cost_usd"]
    return [
        {"device": name, "consumption_kwh": round(values["consumption_kwh"], 2), "cost_usd": round(values["cost_usd"], 2)}
        for name, values in sorted(totals.items(), key=lambda item: item[1]["consumption_kwh"], reverse=True)
    ]


def _hashed_embedding(text: str) -> list[float]:
    """Create a stable, normalized feature-hash embedding without model downloads."""
    vector = [0.0] * EMBEDDING_DIMENSION
    words = re.findall(r"[a-z0-9]+", text.lower())
    for word, count in Counter(words).items():
        # Hash tokens into signed buckets so indexing stays deterministic and local.
        digest = hashlib.sha256(word.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % EMBEDDING_DIMENSION
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[index] += sign * (1 + math.log(count))
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


def create_rag_index(
    documents_dir: str | Path = DOCUMENTS_DIR,
    vectorstore_dir: str | Path = VECTORSTORE_DIR,
) -> int:
    """Index local energy guidance in Chroma using deterministic local vectors."""
    try:
        import chromadb
    except ImportError as error:
        raise RuntimeError("Install project requirements to build the Chroma knowledge index.") from error
    documents: list[str] = []
    metadatas: list[dict[str, str]] = []
    ids: list[str] = []
    for path in sorted(Path(documents_dir).glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        for index, paragraph in enumerate(paragraphs):
            documents.append(paragraph)
            metadatas.append({"source": path.name})
            ids.append(f"{path.stem}-{index}")
    if not documents:
        raise ValueError(f"No .txt knowledge documents found in {documents_dir}.")
    with chromadb.PersistentClient(path=str(vectorstore_dir)) as client:
        # Rebuild the collection so removed or edited source paragraphs cannot linger.
        if any(getattr(item, "name", item) == "energy_tips" for item in client.list_collections()):
            client.delete_collection("energy_tips")
        collection = client.create_collection("energy_tips", metadata={"hnsw:space": "cosine"})
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=[_hashed_embedding(text) for text in documents],
        )
    return len(documents)


def retrieve_energy_tips(
    query: str,
    top_k: int = 3,
    vectorstore_dir: str | Path = VECTORSTORE_DIR,
    documents_dir: str | Path = DOCUMENTS_DIR,
) -> dict[str, Any]:
    """Return relevant cited tips, building the local Chroma index when needed."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    store_path = Path(vectorstore_dir)
    import chromadb

    with chromadb.PersistentClient(path=str(store_path)) as client:
        collections = client.list_collections()
        if not any(getattr(item, "name", item) == "energy_tips" for item in collections):
            create_rag_index(documents_dir=documents_dir, vectorstore_dir=store_path)
        collection = client.get_collection("energy_tips")
        result = collection.query(
            query_embeddings=[_hashed_embedding(query)],
            n_results=top_k,
            include=["documents", "metadatas"],
        )
    vector_results = [
        {"text": text, "source": metadata["source"]}
        for text, metadata in zip(result["documents"][0], result["metadatas"][0])
    ]

    stop_words = {
        "a", "an", "and", "are", "as", "based", "can", "for", "how", "i",
        "in", "is", "it", "my", "of", "on", "or", "the", "to", "what",
        "when", "with",
    }
    query_words = set(re.findall(r"[a-z0-9]+", query.lower())) - stop_words
    lexical_scores: dict[tuple[str, str], int] = {}
    for path in sorted(Path(documents_dir).glob("*.txt")):
        for paragraph in re.split(r"\n\s*\n", path.read_text(encoding="utf-8")):
            text = paragraph.strip()
            if text:
                document_words = set(re.findall(r"[a-z0-9]+", text.lower()))
                score = len(query_words & document_words)
                if score:
                    lexical_scores[(path.name, text)] = score

    vector_ranks = {
        (tip["source"], tip["text"]): rank for rank, tip in enumerate(vector_results)
    }
    candidates = set(vector_ranks) | set(lexical_scores)
    ordered = sorted(
        candidates,
        key=lambda key: (
            -lexical_scores.get(key, 0),
            vector_ranks.get(key, len(vector_ranks)),
            key[0],
            key[1],
        ),
    )
    results = [
        {"source": source, "text": text}
        for source, text in ordered[:top_k]
    ]
    return {"query": query, "count": len(results), "results": results}


def resolve_target_date(question: str, today: date | None = None) -> date:
    """Resolve common relative dates and weekday names; default to tomorrow."""
    anchor = today or date.today()
    lowered = question.lower()
    if "today" in lowered:
        return anchor
    if "tomorrow" in lowered:
        return anchor + timedelta(days=1)
    weekdays = {name.lower(): index for index, name in enumerate(("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"))}
    for name, weekday in weekdays.items():
        if re.search(rf"\b{name}\b", lowered):
            delta = (weekday - anchor.weekday()) % 7
            return anchor + timedelta(days=delta or 7)
    return anchor + timedelta(days=1)
