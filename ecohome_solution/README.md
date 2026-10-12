# EcoHome Energy Advisor

EcoHome is a local, LangGraph-orchestrated energy advisor for homes with rooftop solar, EVs, and flexible household loads. It analyzes sample historical usage, hourly weather/solar forecasts, and time-of-use prices; retrieves cited energy guidance; and recommends device schedules with estimated cost, solar use, savings, and avoided emissions.

This repository is a Python application and notebook workflow. It does not start a web server. The default graph remains deterministic and requires no hosted LLM or API key. To use a LangChain-compatible chat model, pass it to `build_energy_advisor(model=...)`; the model-backed `Agent` receives the question, gathered context, and `ENERGY_ADVISOR_SYSTEM_PROMPT` instructions.

Fellow mates
Brindha
Divya
Parvathy
Vimala

## Capabilities

- Schedule EV charging, dishwashers, pool pumps, HVAC, and washers against hourly prices and forecast solar.
- Compare flexible-device schedules over one day or, for “this week” requests, across the next seven forecast days.
- Query and summarize recent device consumption and costs for historical recommendations.
- Retrieve relevant advice from a local Chroma knowledge base, initialize a missing index automatically, and cite source filenames.
- Estimate schedule cost and savings against a same-duration run starting at 18:00.
- Estimate solar-offset emissions using an illustrative factor of 0.4 kg CO2e/kWh.
- Calculate simple payback and first-year ROI when the user supplies investment, annual savings, and incentives.
- Resolve relative dates such as “tomorrow” and weekday names.

## Project layout

```text
ecohome_solution/
|-- agent.py                     # LangGraph request workflow and response policy
|-- tools.py                     # Weather, pricing, solar, history, scheduling, RAG, ROI
|-- models/
|   |-- __init__.py
|   `-- energy.py                 # SQLite schema, sample data, database queries
|-- data/
|   |-- documents/                # Local energy-saving knowledge base
|   |-- energy_data.db            # Created by database setup
|   `-- vectorstore/               # Created by RAG setup
|-- tests/                        # Unit and end-to-end integration tests
|-- 01_db_setup.ipynb             # Initialize database and inspect sample rows
|-- 02_rag_setup.ipynb            # Index and query guidance documents
|-- 03_run_and_evaluate.ipynb     # Run example user questions
|-- requirements.txt
`-- README.md
```

The included knowledge base contains guidance for HVAC optimization, smart-home automation, renewable-energy integration, seasonal energy management, energy storage, general energy savings, and device-scheduling best practices. Add `.txt` documents under `data/documents/` and rerun the RAG setup to re-index them.

## Requirements and installation

Use Python 3.10 or newer. Run these commands in PowerShell from the `ecohome_solution` directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name ecohome --display-name "Python (EcoHome)"
```

If PowerShell blocks environment activation, allow it for the current terminal with `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again. Alternatively, invoke the environment's Python directly as `\.venv\Scripts\python.exe` without activating it.

Dependencies include LangGraph for graph orchestration, ChromaDB for persistent local retrieval, and Jupyter/ipykernel for the notebooks. There is no API key or external service configuration in the current sample implementation.

## Initialize and run with notebooks

Open the three notebooks in VS Code. Select the `Python (EcoHome)` kernel for each notebook, then run code cells from top to bottom in this order:

1. `01_db_setup.ipynb` creates `data/energy_data.db`. It seeds seven days of hourly device history and hourly price, weather, and solar records extending through the next seven days. The notebook also prints table counts and device totals.
2. `02_rag_setup.ipynb` indexes paragraphs from `data/documents/*.txt` into the persistent Chroma collection and demonstrates a retrieval query.
3. `03_run_and_evaluate.ipynb` runs ten structured scenarios and reports response-quality and tool-usage scores, their explanations, strengths, weaknesses, and improvement recommendations.

The database setup is repeatable but resets and reseeds the sample tables. Do not rerun it if you have replaced the sample database with household data you want to keep.

## Run the advisor from Python

After installing dependencies and running the two setup notebooks, open a terminal in `ecohome_solution` and start Python:

```powershell
python
```

Then call the advisor in the Python prompt:

```python
from agent import ask_energy_advisor

print(ask_energy_advisor(
    "When should I charge my electric car tomorrow to minimize cost and maximize solar power?"
))
print(ask_energy_advisor(
    "What is the best time to run my pool pump this week based on the weather forecast?"
))
```

The function returns a text recommendation with the target date, schedule (when applicable), estimated cost/savings, forecast context, and retrieved source filenames. For thermostat questions it gives a weather-informed starting point with comfort and health caveats. For history questions it summarizes the largest measured load and proposes three actions.

ROI can be calculated when the inputs are known:

```python
print(ask_energy_advisor(
    "Estimate ROI and payback for a home battery.",
    investment_cost_usd=1200,
    annual_savings_usd=400,
    incentives_usd=200,
))
```

The ROI is simple payback and first-year ROI; it excludes financing, maintenance, degradation, and taxes. Without these inputs the advisor asks for assumptions rather than inventing a payback estimate.

## Workflow and tools

The graph in `agent.py` classifies the question, gathers relevant context, and composes a response. Without a model it uses its deterministic response composer. Supply a chat model to enable the configured LLM-backed response:

```python
from agent import build_energy_advisor

advisor = build_energy_advisor(model=your_langchain_compatible_chat_model)
result = advisor.invoke({"question": "When should I charge my EV tomorrow?"})
print(result["answer"])
```

The model receives the user question and gathered tool evidence under `ENERGY_ADVISOR_SYSTEM_PROMPT`. The independent `Agent(instructions, model).invoke(question, context)` interface is also available for callers that already have context. `tools.py` provides these operations:

- `get_weather_forecast`, `get_electricity_prices`, and `get_solar_generation` read the corresponding hourly forecast data.
- `get_usage_history` and `summarize_usage_history` query past device consumption/cost.
- `optimize_device_schedule` evaluates contiguous hourly windows; `optimize_device_schedule_this_week` compares candidate days.
- `create_rag_index` and `retrieve_energy_tips` build/query the local knowledge index. Retrieval returns a dictionary with the original query, result count, and cited result records; a missing collection is built from local documents on demand.
- `calculate_roi` computes user-input-based payback and first-year ROI.

`ENERGY_ADVISOR_SYSTEM_PROMPT` sets policy for grounding, citations, safety, and estimation for both the deterministic guidance path and any configured model-backed response. Provider packages and credentials are supplied by the application and should remain outside source control.

### SQLite data model

| Table | Fields | Purpose |
|---|---|---|
| `energy_readings` | `timestamp`, `device`, `consumption_kwh`, `cost_usd` | Historical hourly household device readings. |
| `solar_generation` | `timestamp`, `production_kwh` | Hourly measured or sample solar production. |
| `electricity_prices` | `timestamp`, `price_per_kwh` | Hourly import tariff. |
| `weather_forecasts` | `timestamp`, `condition`, `temperature_c`, `cloud_cover_pct`, `solar_factor` | Hourly forecast context and solar-generation factor. |

Default scheduling device assumptions are: EV charger 6.0 kW for 4 hours, dishwasher 1.2 kW for 1 hour, pool pump 0.9 kW for 4 hours, HVAC 1.8 kW for 2 hours, and washer 0.6 kW for 1 hour. The device duration can be overridden for a one-day schedule through the Python tool API.

## Data assumptions and limitations

- Weather and tariffs are deterministic synthetic examples, not live forecasts or utility rates. The sample tariff is $0.12/kWh from 00:00-05:00, $0.39/kWh from 17:00-20:00, $0.16/kWh from 10:00-15:00, and $0.24/kWh otherwise.
- Solar production is generated from a daylight curve and a synthetic cloud factor. Scheduling assumes an essential household base load of 0.55 kW when estimating solar surplus.
- Estimated savings compare the selected schedule with a same-duration schedule starting at 18:00. They are not bill guarantees and do not account for every utility fee, export credit, or household constraint.
- Estimated avoided emissions use 0.4 kg CO2e per solar-offset kWh. Real marginal grid emissions vary by place and time.
- Chroma uses deterministic local feature-hash vectors to avoid downloading an embedding model. This is a lightweight retrieval baseline, not production-grade semantic embedding. If no Chroma index exists, retrieval falls back to lexical ranking over the text files.
- A real deployment should connect authenticated weather and utility-rate sources, use household-specific timezone and device constraints, validate recommendations with the user, and use measured solar/inverter data.

## Tests

From `ecohome_solution`, run:

```powershell
python -m unittest discover -s tests -v
```

The tests cover database seeding, history aggregation, date resolution, daily/weekly schedules, savings/ROI calculations, local retrieval, and end-to-end LangGraph responses. Integration tests rebuild the Chroma index and reseed the default sample database; back up any non-sample data before running them.
