"""SQLite storage and deterministic sample data for EcoHome."""

from __future__ import annotations

import math
import random
import sqlite3
from contextlib import closing
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "energy_data.db"


def connect_database(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Open the project database, creating its parent directory if needed."""
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database(
    db_path: str | Path | None = None,
    today: date | None = None,
) -> Path:
    """Create schema and seed seven days of history plus seven forecast days."""
    anchor = today or date.today()
    rng = random.Random(41)
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    with closing(connect_database(path)) as connection, connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS energy_readings (
                id INTEGER PRIMARY KEY,
                timestamp TEXT NOT NULL,
                device TEXT NOT NULL,
                consumption_kwh REAL NOT NULL,
                cost_usd REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_energy_readings_time_device
                ON energy_readings(timestamp, device);
            CREATE TABLE IF NOT EXISTS solar_generation (
                timestamp TEXT PRIMARY KEY,
                production_kwh REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS electricity_prices (
                timestamp TEXT PRIMARY KEY,
                price_per_kwh REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS weather_forecasts (
                timestamp TEXT PRIMARY KEY,
                condition TEXT NOT NULL,
                temperature_c REAL NOT NULL,
                cloud_cover_pct INTEGER NOT NULL,
                solar_factor REAL NOT NULL
            );
            """
        )
        connection.execute("DELETE FROM energy_readings")
        connection.execute("DELETE FROM solar_generation")
        connection.execute("DELETE FROM electricity_prices")
        connection.execute("DELETE FROM weather_forecasts")

        # Keep a full week of usage history and a full week of future forecasts.
        start_day = anchor - timedelta(days=6)
        end_day = anchor + timedelta(days=7)
        now = datetime.now()
        for offset in range((end_day - start_day).days + 1):
            current_day = start_day + timedelta(days=offset)
            weather_seed = (current_day.toordinal() * 17) % 100
            cloud_cover = 15 + (weather_seed % 70)
            solar_factor = max(0.18, 1.0 - cloud_cover / 115)
            condition = "clear" if cloud_cover < 30 else "partly cloudy" if cloud_cover < 65 else "cloudy"
            for hour in range(24):
                timestamp = datetime.combine(current_day, time(hour=hour))
                stamp = timestamp.isoformat(timespec="minutes")
                price = _price_for_hour(hour)
                solar_shape = max(0.0, math.sin(math.pi * (hour - 6) / 12))
                solar_kwh = round(5.2 * solar_shape * solar_factor, 3)
                temperature = round(13 + 9 * math.sin(math.pi * (hour - 7) / 12) + (offset % 5), 1)
                connection.execute(
                    "INSERT INTO electricity_prices(timestamp, price_per_kwh) VALUES (?, ?)",
                    (stamp, price),
                )
                connection.execute(
                    "INSERT INTO weather_forecasts(timestamp, condition, temperature_c, cloud_cover_pct, solar_factor) VALUES (?, ?, ?, ?, ?)",
                    (stamp, condition, temperature, cloud_cover, round(solar_factor, 3)),
                )
                connection.execute(
                    "INSERT INTO solar_generation(timestamp, production_kwh) VALUES (?, ?)",
                    (stamp, solar_kwh),
                )
                # Forecast rows remain available for planning but are not historical usage.
                if timestamp <= now:
                    for device, kwh in _sample_device_usage(hour, rng).items():
                        if kwh > 0:
                            connection.execute(
                                "INSERT INTO energy_readings(timestamp, device, consumption_kwh, cost_usd) VALUES (?, ?, ?, ?)",
                                (stamp, device, kwh, round(kwh * price, 4)),
                            )
    return path


def _price_for_hour(hour: int) -> float:
    # These deterministic tiers make peak and off-peak scheduling reproducible.
    if 17 <= hour <= 20:
        return 0.39
    if 0 <= hour <= 5:
        return 0.12
    if 10 <= hour <= 15:
        return 0.16
    return 0.24


def _sample_device_usage(hour: int, rng: random.Random) -> dict[str, float]:
    return {
        "HVAC": round((0.55 if 7 <= hour <= 22 else 0.25) + rng.uniform(0, 0.18), 3),
        "EV Charger": 5.8 if hour in (21, 22) and rng.random() < 0.65 else 0.0,
        "Dishwasher": 1.15 if hour == 20 and rng.random() < 0.35 else 0.0,
        "Pool Pump": 0.9 if 9 <= hour <= 12 else 0.0,
        "Lighting": 0.22 if hour >= 19 or hour <= 5 else 0.04,
    }


def get_usage_history(
    device: str | None = None,
    days: int = 7,
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Return hourly device consumption and cost for the requested history."""
    if days < 1:
        raise ValueError("days must be at least 1")
    since = (datetime.now() - timedelta(days=days)).isoformat(timespec="minutes")
    query = "SELECT timestamp, device, consumption_kwh, cost_usd FROM energy_readings WHERE timestamp >= ?"
    parameters: list[Any] = [since]
    if device:
        query += " AND lower(device) = lower(?)"
        parameters.append(device)
    query += " ORDER BY timestamp"
    with closing(connect_database(db_path)) as connection, connection:
        return [dict(row) for row in connection.execute(query, parameters).fetchall()]


def get_day_energy_data(
    target_day: date,
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Join hourly weather, solar generation, and prices for a calendar day."""
    start = datetime.combine(target_day, time.min).isoformat(timespec="minutes")
    end = (datetime.combine(target_day, time.min) + timedelta(days=1)).isoformat(timespec="minutes")
    query = """
        SELECT p.timestamp, p.price_per_kwh, s.production_kwh,
               w.condition, w.temperature_c, w.cloud_cover_pct
        FROM electricity_prices p
        JOIN solar_generation s USING (timestamp)
        JOIN weather_forecasts w USING (timestamp)
        WHERE p.timestamp >= ? AND p.timestamp < ?
        ORDER BY p.timestamp
    """
    with closing(connect_database(db_path)) as connection, connection:
        return [dict(row) for row in connection.execute(query, (start, end)).fetchall()]
