# AI-Powered Cybersecurity Threat Detection & Alert Explanation System

## Overview
A **defensive** cybersecurity monitoring project that analyzes network/log
data, detects anomalies with machine learning, scores risk deterministically,
classifies threats when labeled data is available, and uses the **Google
Gemini API** to generate a plain-language explanation of each alert for a
SOC analyst or student.

This is an educational/demo project. It does **not** implement or teach any
offensive capability (no exploit generation, malware, credential theft,
unauthorized scanning, attack execution, persistence, or evasion). It only
analyzes datasets/logs you supply.

## Problem Statement
Raw security logs are voluminous and hard to interpret quickly. Analysts
need (1) reliable anomaly detection, (2) a consistent way to prioritize
alerts, and (3) clear, non-hallucinated explanations of *why* something
was flagged — without an AI model pretending to be the detector itself.

## Objectives
- Detect anomalous network/log events using Isolation Forest.
- Classify threat category with Random Forest, when labels exist.
- Compute a transparent, deterministic 0–100 "Application Risk Score."
- Use Gemini strictly as an **explainer** of ML output — never the detector.
- Present everything in a SOC-style Streamlit dashboard.

## Architecture
```
DATASET / LOG FILE
      ↓
DATA PREPROCESSING (preprocessing/clean.py)
      ↓
FEATURE ENGINEERING (preprocessing/features.py)
      ↓
ML ANOMALY DETECTION (detection/anomaly_detector.py — Isolation Forest)
      ↓
THREAT CLASSIFICATION (detection/classifier.py — Random Forest, optional)
      ↓
RISK SCORING (detection/risk_score.py)
      ↓
GEMINI AI EXPLANATION (ai/gemini_client.py — on request only)
      ↓
SQLITE DATABASE (database/db.py)
      ↓
FASTAPI (api/main.py)  ⇄  STREAMLIT DASHBOARD (dashboard/app.py)
```

**Design rule:** ML is deterministic and independently testable. Gemini
never makes the detection decision — it only explains evidence the ML
pipeline already produced, and it is instructed never to invent facts
(IPs, ports, timestamps, statistics) it wasn't given.

## Technology Stack
Python 3.11+, Pandas, NumPy, Scikit-learn (Isolation Forest, Random Forest),
Joblib, FastAPI, Uvicorn, Streamlit, Plotly, SQLite, python-dotenv,
google-genai (official Gemini SDK). XGBoost and PyTorch are optional and
not required for the base implementation.

## Dataset Setup
A sample dataset is included at `data/sample/sample_events.csv` for demo
mode. Expected columns (not all are required — the system adapts to what's
present):

| Column | Type | Notes |
|---|---|---|
| timestamp | string | ISO-like timestamp |
| source_ip / destination_ip | string | |
| destination_port | int | |
| protocol | string | e.g. TCP/UDP |
| bytes_sent / bytes_received | float | |
| packet_count | float | |
| flow_duration | float | seconds |
| failed_logins | float | |
| label | string | optional — enables threat classification |

To use your own dataset, place a CSV in `data/raw/` and pass its path to
`train.py --data <path>`, or upload it via `POST /upload`.

## Installation
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Environment Configuration
```bash
cp .env.example .env
# then edit .env and set GEMINI_API_KEY
```
Required variables (see `.env.example`):
- `GEMINI_API_KEY` — your Gemini API key (never hard-code, never commit)
- `GEMINI_MODEL` — e.g. `gemini-2.0-flash`
- `DATABASE_URL` — SQLite file path
- `MODEL_PATH` — models directory

## Model Training
```bash
python train.py --data data/sample/sample_events.csv
```
This cleans data, engineers features, fits and saves the preprocessing
pipeline, trains Isolation Forest, and trains Random Forest **only if**
a `label` column with values is present. Outputs go to `models/`:
`preprocessor.joblib`, `isolation_forest.joblib`, `random_forest.joblib`.

Models are trained **offline via this script** — the dashboard/API never
retrains on startup, only loads existing model files.

## Running FastAPI
```bash
uvicorn api.main:app --reload
```
Visit `http://localhost:8000/docs` for interactive API documentation.

## Running Streamlit
In a second terminal (with the API already running):
```bash
streamlit run dashboard/app.py
```

## API Endpoints
| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Service + model load status |
| GET | `/models/status` | Which models are loaded |
| POST | `/upload` | Upload a CSV/JSON dataset, get validation report |
| POST | `/analyze` | Run full pipeline on one event, create an alert |
| GET | `/alerts` | List alerts (filter by severity/category/status) |
| GET | `/alerts/{id}` | Get one alert |
| POST | `/alerts/{id}/explain` | Generate (and store) a Gemini explanation |
| GET | `/statistics` | Aggregate counts for the dashboard |

Gemini is called **only** from `/alerts/{id}/explain`, on demand — never
automatically for every normal event.

## Database Structure
SQLite tables: `users`, `network_events`, `alerts`, `predictions`,
`ai_explanations`. All queries in `database/db.py` are parameterized —
no string-concatenated SQL.

## ML Algorithms
- **Isolation Forest** (unsupervised): flags anomalies without needing
  labeled attack data. Primary detector.
- **Random Forest** (supervised, optional): classifies threat category
  when the dataset has a `label` column. Uses `class_weight="balanced"`
  and macro-averaged metrics to handle imbalance.

## Gemini Integration
`ai/gemini_client.py` calls the official `google-genai` SDK. Only a
whitelisted set of already-computed fields (see `ALLOWED_ALERT_FIELDS`)
is sent — never raw sensitive data, never the API key to the frontend.
The system prompt (`ai/prompts.py`) explicitly forbids inventing evidence,
requires distinguishing observed data / ML prediction / interpretation,
and forbids offensive content of any kind. If Gemini fails or the API key
is missing, the underlying ML alert is still returned and displayed.

## Evaluation Metrics
`train.py` prints accuracy, macro precision/recall/F1, a confusion matrix,
and a full classification report for the classifier.

**On class imbalance:** cybersecurity datasets typically have far more
normal events than attacks. Raw accuracy can look excellent (e.g. 98%)
while missing nearly every real attack, because predicting "normal" for
everything already scores high. That's why this project reports
macro-averaged precision/recall/F1 and a confusion matrix rather than
accuracy alone, and uses `class_weight="balanced"` during training.

For anomaly detection specifically: false positives (normal flagged as
anomalous) create alert fatigue; false negatives (real anomalies missed)
are the more dangerous failure mode. Isolation Forest's `contamination`
parameter should be tuned against a validation set representative of your
real traffic, not left at a default guess.

## Limitations
- Isolation Forest and Random Forest are trained on whatever dataset you
  provide — quality and representativeness of that data drives quality
  of results. This is not a substitute for a production SIEM/EDR.
- The risk score is an **Application Risk Score**, not an
  industry-standard score like CVSS — it's a transparent, configurable
  heuristic for prioritization within this app.
- No real-time packet capture; the system analyzes supplied
  datasets/logs only.
- Gemini explanations depend on prompt-following behavior of the model;
  always verify explanations against the raw alert evidence.

## Future Improvements
- Optional Autoencoder-based anomaly detection (PyTorch) as an ensemble
  member alongside Isolation Forest.
- XGBoost as an alternative/ensemble classifier.
- Role-based access control and authenticated API endpoints.
- Streaming ingestion for near-real-time log tailing.

## Ethical / Security Considerations
- This project is defensive-only by design and explicitly excludes
  offensive capabilities.
- API keys are read from `.env` (never hard-coded, never logged, never
  sent to the frontend) — see `.gitignore`.
- All SQL is parameterized to prevent injection.
- Gemini is constrained to explain, not decide, and is instructed to
  say "Not available from the supplied event data" rather than guess.
- Demo mode data is clearly labeled as a demo/public dataset, not live
  production traffic.
