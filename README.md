# Feature Selection App

> An interactive Streamlit app for intelligent feature selection on high-dimensional datasets — with variance filtering, correlation pruning, and model-based importance ranking.

**Live Demo → [wyvccqyvf8fxg2mmgs82g5.streamlit.app](https://wyvccqyvf8fxg2mmgs82g5.streamlit.app/)**

---

## What it does

Feature selection is one of the most critical (and often overlooked) steps in the ML pipeline. This app automates a 3-stage pipeline that takes a high-dimensional dataset and outputs only the features that matter:

| Stage | Method | What it removes |
|---|---|---|
| 1 | Variance filter | Near-constant columns with almost no signal |
| 2 | Correlation filter | Redundant features strongly correlated with another |
| 3 | Model-based selection | Low-importance features ranked by a Random Forest |

The app auto-detects whether your target is a **classification** or **regression** problem and switches the underlying model accordingly — no configuration needed.

---

## Features

- **Synthetic dataset mode** — generate a configurable high-dimensional classification dataset instantly, no upload required
- **CSV upload mode** — bring your own dataset; non-numeric columns are handled automatically
- **Auto task detection** — uses `RandomForestClassifier` for categorical targets and `RandomForestRegressor` for continuous ones
- **Interactive controls** — tune variance threshold, correlation threshold, importance threshold, and number of trees from the sidebar
- **Feature importance chart** — visualise which features survived and why, with a styled top-20 bar chart
- **One-click download** — export the cleaned, selected-feature dataset as CSV

---

## Tech stack

- **Python 3.10+**
- **Streamlit** — UI and deployment
- **scikit-learn** — `VarianceThreshold`, `SelectFromModel`, `RandomForestClassifier/Regressor`
- **pandas / numpy** — data handling and preprocessing

---

## Run locally

```bash
# 1. Clone the repo
git clone https://github.com/thukralhanshika64-design/feature-engineering-.git
cd feature-engineering-

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

Open the URL shown in your terminal (usually `http://localhost:8501`).

---

## Project structure

```
feature-engineering-/
├── app.py                # Streamlit UI — layout, controls, pipeline orchestration
├── feature_selection.py  # Core logic — variance, correlation, model-based filters
├── requirements.txt      # Pinned dependencies for local and cloud deployment
├── data.ipynb            # Exploratory data analysis notebook
└── README.md
```

---

## How the pipeline works

```
Raw dataset (N features)
        │
        ▼
┌─────────────────────┐
│   Variance filter   │  Drops features below variance threshold
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│ Correlation filter  │  Drops one of each highly correlated pair
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Random Forest      │  Fits model, ranks feature importances
│  (auto: clf/reg)    │  Keeps features above median/mean threshold
└─────────────────────┘
        │
        ▼
Selected features (download as CSV)
```

---

## Deploy your own copy

### Streamlit Community Cloud (free)

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → connect your fork → set main file to `app.py`
4. Deploy — done in under 2 minutes

---

## Author

**Hanshika Thukral**
[GitHub](https://github.com/thukralhanshika64-design) · [LinkedIn](https://linkedin.com/in/hanshika-thukral)
