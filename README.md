# ECE-1513 MLB Team Season Win Prediction

Predicting MLB team season win totals from previous-season statistics, benchmarked against the Pythagorean Win Expectation.

**Course:** ECE1513 — Introduction to Machine Learning (Winter 2026)

## Dataset

[Lahman Baseball Database](https://sabr.app.box.com/s/y1prhc795jk8zvmelfd3jq7tl389y6cd) — `Teams` table.

Download `Teams.csv` and place it in the `data/` directory.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Run the shared data pipeline

TODO

### 2. Compute the Pythagorean baseline

TODO

### 3. Run individual model notebooks

Open the notebooks in `notebooks/` for each model's training, tuning, and evaluation.

## Project Structure

```
ece-1513-baseball-analytics/
├── data/                   # Teams.csv (gitignored)
├── src/                    # Shared pipeline + individual model modules
├── notebooks/              # One notebook per model
├── results/                # Per-model plots and metrics
├── report/                 # LaTeX report (mandatory template)
├── presentation/           # Slide deck
├── requirements.txt
└── README.md
```

## References

- Lahman, S. (2024). Lahman's Baseball Database. https://sabr.org/lahman-database/
- James, B. (1980). Pythagorean Win Expectation.
