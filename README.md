# GrooveMatch – Spotify Song Recommendation System

A content-based music recommendation system that finds songs similar to a given seed song using cosine similarity across 9 Spotify audio features. Also benchmarks a Splay Tree vs. Red-Black Tree for ranking similarity scores.

---

## Team Members

- Jason Peerboccus
- Prem Patel
- Jenish Patel

---

## Files

- `groovematch.py` – data pipeline, cosine similarity engine, benchmarking framework, and visualizations
- `Splay_Tree.py` – Splay Tree implementation
- `RB_Tree.py` – Red-Black Tree implementation
- `tests.py` – unit tests for RedBlackTree

---

## Requirements

Install dependencies:

```
pip install kagglehub numpy pandas matplotlib scikit-learn
```

---

## How to Run

```
python groovematch.py
```

Or with a custom seed song:

```
python groovematch.py "Blinding Lights"
```

On first run, the Spotify dataset will be downloaded automatically via kagglehub (~230k tracks). Subsequent runs use the cached version.

---

## Output

- Top 10 recommended songs printed to terminal
- Splay Tree vs. Red-Black Tree benchmark results printed to terminal
- 5 charts saved to `groovematch_plots/`:
  - `benchmark.png` – insert time, retrieval time, memory usage
  - `radar.png` – audio feature profile of seed vs. top 5
  - `score_distribution.png` – histogram of all similarity scores
  - `feature_correlation.png` – correlation heatmap of audio features
  - `top_k_scores.png` – horizontal bar chart of top 10 results

---

## Dataset

SpotifyFeatures.csv – [Ultimate Spotify Tracks DB](https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db) via Kaggle
