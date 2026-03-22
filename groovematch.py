"""
groovematch.py
GrooveMatch – Full Pipeline (Jason's Components)
Team 58 | DSA Final Project

Sections
--------
1. Configuration & Constants
2. Data Preprocessing
3. Cosine Similarity Pipeline
4. Benchmarking
5. Visualization
6. Main entry point (CLI demo)

Usage
-----
    python groovematch.py  (copy and paste in the terminal to get the outputs)                                # uses default CSV path + "Bohemian Rhapsody"
    python groovematch.py <csv_path> "<seed_song>"         # custom path and seed song
"""

# ══════════════════════════════════════════════════════════════════════════════
# Imports
# ══════════════════════════════════════════════════════════════════════════════

import os
import sys
import time
import heapq
import random
import statistics
import tracemalloc
from dataclasses import dataclass, field
from math import pi

import kagglehub
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, normalize
from RB_Tree import RedBlackTree


matplotlib.use("Agg")  # non-interactive backend (safe for Kaggle / terminal)

# "top_k_scores" could not render non english characters by default, so we set a font family that includes support
plt.rcParams["font.family"] = ["Segoe UI", "Malgun Gothic", "Meiryo", "DejaVu Sans", "sans-serif"]


# ══════════════════════════════════════════════════════════════════════════════
# 1 · CONFIGURATION & CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# Audio features used for cosine similarity
AUDIO_FEATURES = [
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "liveness",
    "loudness",
    "speechiness",
    "tempo",
    "valence",
]

# Spotify-inspired dark theme colours
SPLAY_COLOR  = "#1DB954"   # Spotify green  → Splay Tree
RB_COLOR     = "#E91429"   # Spotify red    → Red-Black Tree
BG_COLOR     = "#191414"   # Dark background
TEXT_COLOR   = "#FFFFFF"
ACCENT       = "#535353"


DEFAULT_SEED = "Jewelry"
TOP_K        = 10


# ══════════════════════════════════════════════════════════════════════════════
# 2 · DATA PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    """
    Load SpotifyFeatures.csv via kagglehub (downloads automatically on
    first run, uses the cached version on subsequent runs).

    Requires:  pip install kagglehub
    """
    print("[load]  Fetching dataset via kagglehub…")
    path = kagglehub.dataset_download("zaheenhamidani/ultimate-spotify-tracks-db")
    print(f"[load]  Path to dataset files: {path}")
    csv = os.path.join(path, "SpotifyFeatures.csv")
    df  = pd.read_csv(csv)
    print(f"[load]  Raw shape: {df.shape}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataset:
      1. Drop rows missing track_name, artist_name, or any audio feature.
      2. Drop exact duplicate (track_name, artist_name) pairs.
      3. Ensure all audio features are numeric.
      4. Clamp [0,1]-bounded features to their valid range.

    Parameters
    ----------
    df : pd.DataFrame  –  raw dataset

    Returns
    -------
    pd.DataFrame  –  cleaned dataset with reset index
    """
    df = df.copy()

    # 1. Drop rows with missing critical fields
    critical_cols = ["track_name", "artist_name"] + AUDIO_FEATURES
    before = len(df)
    df.dropna(subset=critical_cols, inplace=True)
    print(f"[clean] Dropped {before - len(df)} rows with missing values.")

    # 2. Drop duplicate (track_name, artist_name) pairs
    before = len(df)
    df.drop_duplicates(subset=["track_name", "artist_name"], keep="first", inplace=True)
    print(f"[clean] Dropped {before - len(df)} duplicate track–artist pairs.")

    # 3. Coerce audio features to float
    for col in AUDIO_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df.dropna(subset=AUDIO_FEATURES, inplace=True)
    print(f"[clean] Dropped {before - len(df)} rows after numeric coercion.")

    # 4. Clamp [0,1]-bounded features
    unit_features = [
        "acousticness", "danceability", "energy",
        "instrumentalness", "liveness", "speechiness", "valence",
    ]
    for col in unit_features:
        df[col] = df[col].clip(0.0, 1.0)

    df.reset_index(drop=True, inplace=True)
    print(f"[clean] Final shape after cleaning: {df.shape}")
    return df


def normalize_features(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Min-Max normalize the audio features so every dimension is in [0, 1].
    Loudness and tempo have wider natural ranges, so normalization is
    essential before computing cosine similarity.

    Returns
    -------
    df_out        : pd.DataFrame  –  normalized dataset
    feature_matrix: np.ndarray   –  shape (n_tracks, 9)
    """
    df_out = df.copy()
    scaler = MinMaxScaler()
    feature_matrix = scaler.fit_transform(df_out[AUDIO_FEATURES].values)
    df_out[AUDIO_FEATURES] = feature_matrix
    print(f"[norm]  Feature matrix shape: {feature_matrix.shape}")
    return df_out, feature_matrix


def preprocess() -> tuple[pd.DataFrame, np.ndarray]:
    """
    Full preprocessing pipeline: load → clean → normalize.

    Returns
    -------
    df            : pd.DataFrame  –  cleaned + normalized dataset
    feature_matrix: np.ndarray   –  normalized audio feature matrix (n × 9)
    """
    raw     = load_data()
    cleaned = clean_data(raw)
    df, feature_matrix = normalize_features(cleaned)
    return df, feature_matrix


# ══════════════════════════════════════════════════════════════════════════════
# 3 · COSINE SIMILARITY PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def get_song_vector(df: pd.DataFrame, track_name: str) -> tuple[np.ndarray, int]:
    """
    Look up a song in the dataset and return its normalized feature vector
    and its DataFrame row index.

    Raises ValueError if the song is not found.
    """
    mask    = df["track_name"].str.lower() == track_name.lower()
    matches = df[mask]

    if matches.empty:
        raise ValueError(
            f"Song '{track_name}' not found in dataset. "
            "Use build_manual_vector() to enter features manually."
        )

    idx        = matches.index[0]
    raw_vector = df.loc[idx, AUDIO_FEATURES].values.astype(float)
    vector     = raw_vector / (np.linalg.norm(raw_vector) + 1e-9)
    return vector, int(idx)


def build_manual_vector(features: dict) -> np.ndarray:
    """
    Build a normalized feature vector from manually entered audio features.
    All keys in AUDIO_FEATURES must be present.

    Example
    -------
    v = build_manual_vector({"acousticness": 0.1, "danceability": 0.8, ...})
    """
    for key in AUDIO_FEATURES:
        if key not in features:
            raise ValueError(f"Missing feature: '{key}'. Required: {AUDIO_FEATURES}")
    raw = np.array([features[key] for key in AUDIO_FEATURES], dtype=float)
    return raw / (np.linalg.norm(raw) + 1e-9)


def compute_similarity_scores(
    seed_vector: np.ndarray,
    feature_matrix: np.ndarray,
    seed_idx: int | None = None,
) -> list[tuple[float, int]]:
    """
    Compute cosine similarity between the seed vector and every row in
    feature_matrix using a single NumPy dot product (O(n), vectorized).

    Parameters
    ----------
    seed_vector    : shape (9,)   – L2-normalized seed feature vector
    feature_matrix : shape (n,9)  – dataset feature matrix
    seed_idx       : row index of the seed song (excluded from results)

    Returns
    -------
    list of (score, row_index) sorted descending by score
    """
    normed_matrix = normalize(feature_matrix, norm="l2")
    scores        = normed_matrix @ seed_vector  # shape (n,)

    if seed_idx is not None:
        scores[seed_idx] = -1.0

    scored = [(float(scores[i]), i) for i in range(len(scores))]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def get_top_k_results(
    scored: list[tuple[float, int]],
    df: pd.DataFrame,
    k: int = 10,
) -> pd.DataFrame:
    """
    Extract top-k results from a sorted score list.

    Returns
    -------
    pd.DataFrame  –  columns: rank, track_name, artist_name, similarity_score
    """
    rows = []
    for rank, (score, idx) in enumerate(scored[:k], start=1):
        rows.append({
            "rank"            : rank,
            "track_name"      : df.loc[idx, "track_name"],
            "artist_name"     : df.loc[idx, "artist_name"],
            "similarity_score": round(score, 4),
        })
    return pd.DataFrame(rows)


def recommend(
    track_name: str,
    df: pd.DataFrame,
    feature_matrix: np.ndarray,
    k: int = 10,
) -> pd.DataFrame:
    """
    End-to-end recommendation: look up seed song → cosine scores → top-k.
    """
    seed_vector, seed_idx = get_song_vector(df, track_name)
    scored                = compute_similarity_scores(seed_vector, feature_matrix, seed_idx)
    return get_top_k_results(scored, df, k)


# ══════════════════════════════════════════════════════════════════════════════
# 4 · BENCHMARKING
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TreeBenchmark:
    name: str
    insert_time_ms  : float = 0.0
    retrieval_time_ms: float = 0.0
    peak_memory_kb  : float = 0.0
    top_k_results   : list  = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"{self.name:<20} "
            f"Insert: {self.insert_time_ms:>8.3f} ms  |  "
            f"Retrieval: {self.retrieval_time_ms:>8.3f} ms  |  "
            f"Memory: {self.peak_memory_kb:>8.1f} KB"
        )


def _measure(fn, *args, **kwargs) -> tuple:
    """Run fn(*args) and return (elapsed_ms, peak_kb, return_value)."""
    tracemalloc.start()
    t0     = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed_ms  = (time.perf_counter() - t0) * 1_000
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return elapsed_ms, peak_bytes / 1_024, result


def _bulk_insert(tree, scored_pairs: list[tuple[float, int]]) -> None:
    for score, idx in scored_pairs:
        tree.insert(score, idx)


def benchmark_tree(
    name: str,
    tree,
    scored_pairs: list[tuple[float, int]],
    k: int = 10,
    n_retrieval_trials: int = 5,
) -> TreeBenchmark:
    """
    Benchmark a single tree implementation.

    The tree must implement:
        tree.insert(score: float, idx: int) → None
        tree.top_k(k: int) → list[(score, idx)]
    """
    bench = TreeBenchmark(name=name)

    # Insertion
    insert_ms, insert_kb, _ = _measure(_bulk_insert, tree, scored_pairs)
    bench.insert_time_ms  = insert_ms
    bench.peak_memory_kb  = insert_kb

    # Retrieval (median of n trials)
    trial_times   = []
    top_k_result  = []
    for _ in range(n_retrieval_trials):
        ret_ms, _, top_k_result = _measure(tree.top_k, k)
        trial_times.append(ret_ms)

    bench.retrieval_time_ms = statistics.median(trial_times)
    bench.top_k_results     = top_k_result
    return bench


def run_benchmark(
    scored_pairs: list[tuple[float, int]],
    splay_tree,
    rb_tree,
    k: int = 10,
) -> dict[str, TreeBenchmark]:
    """
    Run benchmarks for both trees and print a summary table.

    Returns
    -------
    dict with keys 'splay' and 'rb', each a TreeBenchmark instance.
    """
    print(f"\nBenchmarking with {len(scored_pairs):,} similarity scores…")

    results = {
        "splay": benchmark_tree("Splay Tree",      splay_tree, scored_pairs, k),
        "rb"   : benchmark_tree("Red-Black Tree",  rb_tree,    scored_pairs, k),
    }
    _print_benchmark_summary(results)
    return results


def _print_benchmark_summary(results: dict[str, TreeBenchmark]) -> None:
    splay, rb = results["splay"], results["rb"]
    print("\n" + "=" * 70)
    print("  BENCHMARK RESULTS")
    print("=" * 70)
    print(f"{'Structure':<20} {'Insert (ms)':>12} {'Retrieval (ms)':>16} {'Memory (KB)':>13}")
    print("-" * 70)
    for b in [splay, rb]:
        print(f"{b.name:<20} {b.insert_time_ms:>12.3f} {b.retrieval_time_ms:>16.3f} {b.peak_memory_kb:>13.1f}")
    print("=" * 70)
    print(f"  Faster insertion : {'Splay Tree' if splay.insert_time_ms < rb.insert_time_ms else 'Red-Black Tree'}")
    print(f"  Faster retrieval : {'Splay Tree' if splay.retrieval_time_ms < rb.retrieval_time_ms else 'Red-Black Tree'}")
    print(f"  Lower memory     : {'Splay Tree' if splay.peak_memory_kb < rb.peak_memory_kb else 'Red-Black Tree'}")
    print("=" * 70)


def benchmark_to_dataframe(results: dict[str, TreeBenchmark]) -> pd.DataFrame:
    """Convert benchmark results to a DataFrame for visualization / export."""
    return pd.DataFrame([
        {
            "structure"   : b.name,
            "insert_ms"   : round(b.insert_time_ms, 4),
            "retrieval_ms": round(b.retrieval_time_ms, 4),
            "memory_kb"   : round(b.peak_memory_kb, 2),
        }
        for b in results.values()
    ])


# ══════════════════════════════════════════════════════════════════════════════
# 5 · VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def _apply_spotify_theme(ax, fig=None) -> None:
    """Apply Spotify-inspired dark theme to a matplotlib axes."""
    if fig:
        fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(ACCENT)


def plot_benchmark(benchmark_df: pd.DataFrame, save_path: str = "benchmark.png") -> None:
    """Three-panel bar chart: Insert Time | Retrieval Time | Memory Usage."""
    metrics = [
        ("insert_ms",    "Insertion Time (ms)", "ms"),
        ("retrieval_ms", "Retrieval Time (ms)", "ms"),
        ("memory_kb",    "Peak Memory (KB)",    "KB"),
    ]
    colors = [SPLAY_COLOR, RB_COLOR]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("GrooveMatch – Data Structure Benchmark", color=TEXT_COLOR, fontsize=15, fontweight="bold")
    fig.patch.set_facecolor(BG_COLOR)

    for ax, (col, title, unit) in zip(axes, metrics):
        values = benchmark_df[col].values
        bars   = ax.bar(benchmark_df["structure"], values, color=colors, width=0.5, edgecolor=ACCENT)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(unit)
        _apply_spotify_theme(ax)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                    f"{val:.3f}", ha="center", va="bottom", color=TEXT_COLOR, fontsize=9)
        ax.set_ylim(0, max(values) * 1.25)
        ax.set_xticklabels(benchmark_df["structure"], rotation=10, ha="right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[viz] Benchmark chart saved → {save_path}")


def plot_radar(
    seed_name: str,
    seed_features: dict,
    top5_df: pd.DataFrame,
    save_path: str = "radar.png",
) -> None:
    """Radar chart: seed song audio profile vs. average of top-5 recommendations."""
    labels = AUDIO_FEATURES
    N      = len(labels)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    seed_vals = [seed_features[f] for f in labels] + [seed_features[labels[0]]]
    avg_vals  = top5_df[labels].mean().tolist() + [top5_df[labels[0]].mean()]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    ax.plot(angles, seed_vals, color=SPLAY_COLOR, linewidth=2, label=seed_name)
    ax.fill(angles, seed_vals, color=SPLAY_COLOR, alpha=0.25)
    ax.plot(angles, avg_vals,  color=RB_COLOR,    linewidth=2, label="Top-5 Average")
    ax.fill(angles, avg_vals,  color=RB_COLOR,    alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color=TEXT_COLOR, fontsize=9)
    ax.yaxis.set_tick_params(labelcolor=ACCENT)
    ax.set_title("Audio Feature Profile: Seed vs Top-5", color=TEXT_COLOR, fontsize=12, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1),
              labelcolor=TEXT_COLOR, facecolor=BG_COLOR, edgecolor=ACCENT)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[viz] Radar chart saved → {save_path}")


def plot_score_distribution(
    scored_pairs: list[tuple[float, int]],
    save_path: str = "score_distribution.png",
) -> None:
    """Histogram of all cosine similarity scores."""
    scores = [s for s, _ in scored_pairs]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(scores, bins=80, color=SPLAY_COLOR, edgecolor=BG_COLOR, alpha=0.9)
    ax.axvline(scores[0], color=RB_COLOR, linewidth=1.5, linestyle="--",
               label=f"Top score: {scores[0]:.4f}")
    ax.set_xlabel("Cosine Similarity Score")
    ax.set_ylabel("Number of Tracks")
    ax.set_title("Distribution of Similarity Scores")
    ax.legend(labelcolor=TEXT_COLOR, facecolor=BG_COLOR, edgecolor=ACCENT)
    _apply_spotify_theme(ax, fig)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[viz] Score distribution saved → {save_path}")


def plot_feature_correlation(
    df: pd.DataFrame,
    save_path: str = "feature_correlation.png",
) -> None:
    """Heatmap of Pearson correlations between the 9 audio features."""
    corr = df[AUDIO_FEATURES].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    _apply_spotify_theme(ax, fig)

    cax  = ax.matshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1)
    cbar = plt.colorbar(cax)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COLOR)

    ax.set_xticks(range(len(AUDIO_FEATURES)))
    ax.set_yticks(range(len(AUDIO_FEATURES)))
    ax.set_xticklabels(AUDIO_FEATURES, rotation=45, ha="left", fontsize=9, color=TEXT_COLOR)
    ax.set_yticklabels(AUDIO_FEATURES, fontsize=9, color=TEXT_COLOR)
    ax.set_title("Audio Feature Correlation Matrix", pad=20, color=TEXT_COLOR)

    for i in range(len(AUDIO_FEATURES)):
        for j in range(len(AUDIO_FEATURES)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if abs(corr.values[i, j]) < 0.5 else "white")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[viz] Correlation heatmap saved → {save_path}")


def plot_top_k_scores(
    results_df: pd.DataFrame,
    seed_name: str,
    save_path: str = "top_k_scores.png",
) -> None:
    """Horizontal bar chart of top-k similarity scores."""
    labels     = [f"{r['track_name']} – {r['artist_name']}" for _, r in results_df.iterrows()]
    scores     = results_df["similarity_score"].values
    bar_colors = [SPLAY_COLOR if i < 3 else "#1a7a3c" for i in range(len(scores))]

    fig, ax = plt.subplots(figsize=(10, 6))
    _apply_spotify_theme(ax, fig)
    bars = ax.barh(labels[::-1], scores[::-1], color=bar_colors[::-1], edgecolor=ACCENT)
    ax.set_xlabel("Cosine Similarity Score")
    ax.set_title(f"Top-{len(results_df)} Recommendations for '{seed_name}'", fontsize=12)
    ax.set_xlim(min(scores) * 0.97, 1.01)

    for bar, score in zip(bars, scores[::-1]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{score:.4f}", va="center", ha="left", color=TEXT_COLOR, fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[viz] Top-k bar chart saved → {save_path}")


def generate_all_plots(
    df: pd.DataFrame,
    scored_pairs: list[tuple[float, int]],
    results_df: pd.DataFrame,
    seed_name: str,
    seed_idx: int,
    benchmark_df: pd.DataFrame | None = None,
    output_dir: str = "groovematch_plots",
) -> None:
    """
    Generate all five visualization plots and save to output_dir.

    Plots produced
    --------------
    benchmark.png         – Splay vs RB bar chart (skipped if benchmark_df is None)
    radar.png             – Audio feature radar
    score_distribution.png– Histogram of cosine scores
    feature_correlation.png– Feature correlation heatmap
    top_k_scores.png      – Top-k horizontal bar
    """
    os.makedirs(output_dir, exist_ok=True)
    p = lambda name: os.path.join(output_dir, name)

    if benchmark_df is not None:
        plot_benchmark(benchmark_df, save_path=p("benchmark.png"))

    seed_features = {f: df.loc[seed_idx, f] for f in AUDIO_FEATURES}
    top5_indices  = [idx for _, idx in scored_pairs[:5]]
    top5_df       = df.loc[top5_indices, AUDIO_FEATURES]

    plot_radar(seed_name, seed_features, top5_df, save_path=p("radar.png"))
    plot_score_distribution(scored_pairs,          save_path=p("score_distribution.png"))
    plot_feature_correlation(df,                   save_path=p("feature_correlation.png"))
    plot_top_k_scores(results_df, seed_name,       save_path=p("top_k_scores.png"))

    print(f"\n[viz] All plots saved to: {output_dir}/")


# ══════════════════════════════════════════════════════════════════════════════
# 6 · MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

class _StubTree:
    """
    Heap-backed placeholder tree used when SplayTree / RedBlackTree are not
    yet available.  Replace with real implementations from Jenish and Prem:

        from splay_tree import SplayTree
        from red_black_tree import RedBlackTree
        splay_tree = SplayTree()
        rb_tree    = RedBlackTree()
    """
    def __init__(self):
        self._heap: list[tuple[float, int]] = []

    def insert(self, score: float, idx: int) -> None:
        heapq.heappush(self._heap, (-score, idx))

    def top_k(self, k: int) -> list[tuple[float, int]]:
        heap_copy = self._heap[:]
        out = []
        for _ in range(min(k, len(heap_copy))):
            neg_s, i = heapq.heappop(heap_copy)
            out.append((-neg_s, i))
        return out


def main():
    seed_song = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SEED

    # ── Preprocessing ─────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  GrooveMatch – Starting pipeline")
    print("=" * 55)
    df, feature_matrix = preprocess()

    # ── Similarity ────────────────────────────────────────────────────────────
    print(f"\nFinding songs similar to: '{seed_song}'")
    try:
        seed_vector, seed_idx = get_song_vector(df, seed_song)
    except ValueError as e:
        print(f"[error] {e}")
        sys.exit(1)

    t0           = time.perf_counter()
    scored_pairs = compute_similarity_scores(seed_vector, feature_matrix, seed_idx)
    elapsed_ms   = (time.perf_counter() - t0) * 1_000
    print(f"Computed {len(scored_pairs):,} similarity scores in {elapsed_ms:.2f} ms")

    results_df = get_top_k_results(scored_pairs, df, k=TOP_K)

    # ── Results display ───────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Top {TOP_K} Recommendations for: '{seed_song}'")
    print(f"{'='*55}")
    for _, row in results_df.iterrows():
        print(f"  {row['rank']:>2}. {row['track_name']:<35} – {row['artist_name']:<25} [{row['similarity_score']:.4f}]")

    # ── Benchmarking ──────────────────────────────────────────────────────────
    # Swap _StubTree() with SplayTree() / RedBlackTree() when available
    splay_tree = _StubTree()
    rb_tree    = RedBlackTree()

    benchmark_results = run_benchmark(scored_pairs, splay_tree, rb_tree, k=TOP_K)
    benchmark_df      = benchmark_to_dataframe(benchmark_results)

    # ── Visualizations ────────────────────────────────────────────────────────
    generate_all_plots(
        df           = df,
        scored_pairs = scored_pairs,
        results_df   = results_df,
        seed_name    = seed_song,
        seed_idx     = seed_idx,
        benchmark_df = benchmark_df,
        output_dir   = "groovematch_plots",
    )

    print("\n[done] GrooveMatch pipeline complete.")


if __name__ == "__main__":
    main()