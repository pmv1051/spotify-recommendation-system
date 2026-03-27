"""
    python groovematch.py
    python groovematch.py "<seed_song>"

notebook converted to a .py file
"""
import os, sys, time, heapq, statistics, tracemalloc
from math import pi

import kagglehub
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, normalize

matplotlib.use("Agg")

AUDIO_FEATURES = ["acousticness","danceability","energy","instrumentalness",
                   "liveness","loudness","speechiness","tempo","valence"]

SPLAY_COLOR  = "#1DB954"
RB_COLOR     = "#E91429"
BG_COLOR     = "#191414"
TEXT_COLOR   = "#FFFFFF"
ACCENT       = "#535353"
DEFAULT_SEED = "Bohemian Rhapsody"
TOP_K        = 10


def load_data():
    print("[load]  Fetching dataset via kagglehub…")
    path = kagglehub.dataset_download("zaheenhamidani/ultimate-spotify-tracks-db")
    print(f"[load]  Path to dataset files: {path}")
    csv = os.path.join(path, "SpotifyFeatures.csv")
    df  = pd.read_csv(csv)
    print(f"[load]  Raw shape: {df.shape}")
    return df


def clean_data(df):
    df = df.copy()
    needed = ["track_name", "artist_name"] + AUDIO_FEATURES

    before = len(df)
    df.dropna(subset=needed, inplace=True)
    print(f"[clean] Dropped {before - len(df)} rows with missing values.")

    before = len(df)
    df.drop_duplicates(subset=["track_name", "artist_name"], keep="first", inplace=True)
    print(f"[clean] Dropped {before - len(df)} duplicate track–artist pairs.")

    for col in AUDIO_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df.dropna(subset=AUDIO_FEATURES, inplace=True)
    print(f"[clean] Dropped {before - len(df)} rows after numeric coercion.")

    for col in ["acousticness","danceability","energy","instrumentalness","liveness","speechiness","valence"]:
        df[col] = df[col].clip(0.0, 1.0)

    df.reset_index(drop=True, inplace=True)
    print(f"[clean] Final shape after cleaning: {df.shape}")
    return df


def normalize_features(df):
    df_out = df.copy()
    feature_matrix = MinMaxScaler().fit_transform(df_out[AUDIO_FEATURES].values)
    df_out[AUDIO_FEATURES] = feature_matrix
    print(f"[norm]  Feature matrix shape: {feature_matrix.shape}")
    return df_out, feature_matrix


def preprocess():
    raw     = load_data()
    cleaned = clean_data(raw)
    return normalize_features(cleaned)


def get_song_vector(df, track_name):
    matches = df[df["track_name"].str.lower() == track_name.lower()]
    if matches.empty:
        raise ValueError(
            f"Song '{track_name}' not found in dataset. "
            "Use build_manual_vector() to enter features manually."
        )
    idx    = matches.index[0]
    vector = df.loc[idx, AUDIO_FEATURES].values.astype(float)
    return vector / (np.linalg.norm(vector) + 1e-9), int(idx)


def build_manual_vector(features):
    for key in AUDIO_FEATURES:
        if key not in features:
            raise ValueError(f"Missing feature: '{key}'. Required: {AUDIO_FEATURES}")
    raw = np.array([features[key] for key in AUDIO_FEATURES], dtype=float)
    return raw / (np.linalg.norm(raw) + 1e-9)


def compute_similarity_scores(seed_vector, feature_matrix, seed_idx=None):
    scores = normalize(feature_matrix, norm="l2") @ seed_vector
    if seed_idx is not None:
        scores[seed_idx] = -1.0
    scored = [(float(scores[i]), i) for i in range(len(scores))]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def get_top_k_results(scored, df, k=10):
    rows = []
    for rank, (score, idx) in enumerate(scored[:k], start=1):
        rows.append({"rank": rank, "track_name": df.loc[idx, "track_name"],
                     "artist_name": df.loc[idx, "artist_name"],
                     "similarity_score": round(score, 4)})
    return pd.DataFrame(rows)


def recommend(track_name, df, feature_matrix, k=10):
    seed_vector, seed_idx = get_song_vector(df, track_name)
    return get_top_k_results(compute_similarity_scores(seed_vector, feature_matrix, seed_idx), df, k)


class TreeBenchmark:
    def __init__(self, name):
        self.name              = name
        self.insert_time_ms   = 0.0
        self.retrieval_time_ms = 0.0
        self.peak_memory_kb   = 0.0
        self.top_k_results    = []

    def __str__(self):
        return (f"{self.name:<20} Insert: {self.insert_time_ms:>8.3f} ms  |  "
                f"Retrieval: {self.retrieval_time_ms:>8.3f} ms  |  Memory: {self.peak_memory_kb:>8.1f} KB")


def _measure(fn, *args, **kwargs):
    tracemalloc.start()
    t0            = time.perf_counter()
    result        = fn(*args, **kwargs)
    elapsed_ms    = (time.perf_counter() - t0) * 1_000
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return elapsed_ms, peak_bytes / 1_024, result


def _bulk_insert(tree, scored_pairs):
    for score, idx in scored_pairs:
        tree.insert(score, idx)


def benchmark_tree(name, tree, scored_pairs, k=10, n_retrieval_trials=5):
    bench = TreeBenchmark(name)
    bench.insert_time_ms, bench.peak_memory_kb, _ = _measure(_bulk_insert, tree, scored_pairs)

    trial_times, top_k_result = [], []
    for _ in range(n_retrieval_trials):
        ret_ms, _, top_k_result = _measure(tree.top_k, k)
        trial_times.append(ret_ms)

    bench.retrieval_time_ms = statistics.median(trial_times)
    bench.top_k_results     = top_k_result
    return bench


def run_benchmark(scored_pairs, splay_tree, rb_tree, k=10):
    print(f"\nBenchmarking with {len(scored_pairs):,} similarity scores…")
    results = {
        "splay": benchmark_tree("Splay Tree",     splay_tree, scored_pairs, k),
        "rb"   : benchmark_tree("Red-Black Tree", rb_tree,    scored_pairs, k),
    }
    _print_benchmark_summary(results)
    return results


def _print_benchmark_summary(results):
    splay, rb = results["splay"], results["rb"]
    print("\n" + "=" * 70)
    print("  BENCHMARK RESULTS")
    print("=" * 70)
    print(f"{'Structure':<20} {'Insert (ms)':>12} {'Retrieval (ms)':>16} {'Memory (KB)':>13}")
    print("-" * 70)
    for b in [splay, rb]:
        print(f"{b.name:<20} {b.insert_time_ms:>12.3f} {b.retrieval_time_ms:>16.3f} {b.peak_memory_kb:>13.1f}")
    print("=" * 70)
    print(f"  Faster insertion : {'Splay Tree' if splay.insert_time_ms   < rb.insert_time_ms   else 'Red-Black Tree'}")
    print(f"  Faster retrieval : {'Splay Tree' if splay.retrieval_time_ms < rb.retrieval_time_ms else 'Red-Black Tree'}")
    print(f"  Lower memory     : {'Splay Tree' if splay.peak_memory_kb   < rb.peak_memory_kb   else 'Red-Black Tree'}")
    print("=" * 70)


def benchmark_to_dataframe(results):
    return pd.DataFrame([{"structure": b.name, "insert_ms": round(b.insert_time_ms, 4),
                           "retrieval_ms": round(b.retrieval_time_ms, 4),
                           "memory_kb": round(b.peak_memory_kb, 2)}
                          for b in results.values()])


def _apply_spotify_theme(ax, fig=None):
    if fig:
        fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(ACCENT)


def plot_benchmark(benchmark_df, save_path="benchmark.png"):
    metrics = [("insert_ms","Insertion Time (ms)","ms"),
               ("retrieval_ms","Retrieval Time (ms)","ms"),
               ("memory_kb","Peak Memory (KB)","KB")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("GrooveMatch – Data Structure Benchmark", color=TEXT_COLOR, fontsize=15, fontweight="bold")
    fig.patch.set_facecolor(BG_COLOR)
    for ax, (col, title, unit) in zip(axes, metrics):
        values = benchmark_df[col].values
        bars   = ax.bar(benchmark_df["structure"], values, color=[SPLAY_COLOR, RB_COLOR], width=0.5, edgecolor=ACCENT)
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


def plot_radar(seed_name, seed_features, top5_df, save_path="radar.png"):
    N      = len(AUDIO_FEATURES)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    seed_vals = [seed_features[f] for f in AUDIO_FEATURES] + [seed_features[AUDIO_FEATURES[0]]]
    avg_vals  = top5_df[AUDIO_FEATURES].mean().tolist() + [top5_df[AUDIO_FEATURES[0]].mean()]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.plot(angles, seed_vals, color=SPLAY_COLOR, linewidth=2, label=seed_name)
    ax.fill(angles, seed_vals, color=SPLAY_COLOR, alpha=0.25)
    ax.plot(angles, avg_vals,  color=RB_COLOR,    linewidth=2, label="Top-5 Average")
    ax.fill(angles, avg_vals,  color=RB_COLOR,    alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(AUDIO_FEATURES, color=TEXT_COLOR, fontsize=9)
    ax.yaxis.set_tick_params(labelcolor=ACCENT)
    ax.set_title("Audio Feature Profile: Seed vs Top-5", color=TEXT_COLOR, fontsize=12, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), labelcolor=TEXT_COLOR, facecolor=BG_COLOR, edgecolor=ACCENT)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[viz] Radar chart saved → {save_path}")


def plot_score_distribution(scored_pairs, save_path="score_distribution.png"):
    scores = [s for s, _ in scored_pairs]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(scores, bins=80, color=SPLAY_COLOR, edgecolor=BG_COLOR, alpha=0.9)
    ax.axvline(scores[0], color=RB_COLOR, linewidth=1.5, linestyle="--", label=f"Top score: {scores[0]:.4f}")
    ax.set_xlabel("Cosine Similarity Score")
    ax.set_ylabel("Number of Tracks")
    ax.set_title("Distribution of Similarity Scores")
    ax.legend(labelcolor=TEXT_COLOR, facecolor=BG_COLOR, edgecolor=ACCENT)
    _apply_spotify_theme(ax, fig)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[viz] Score distribution saved → {save_path}")


def plot_feature_correlation(df, save_path="feature_correlation.png"):
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
            val = corr.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(val) >= 0.5 else "black")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[viz] Correlation heatmap saved → {save_path}")


def plot_top_k_scores(results_df, seed_name, save_path="top_k_scores.png"):
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


def generate_all_plots(df, scored_pairs, results_df, seed_name, seed_idx, benchmark_df=None, output_dir="groovematch_plots"):
    os.makedirs(output_dir, exist_ok=True)
    p = lambda name: os.path.join(output_dir, name)
    if benchmark_df is not None:
        plot_benchmark(benchmark_df, save_path=p("benchmark.png"))
    seed_features = {f: df.loc[seed_idx, f] for f in AUDIO_FEATURES}
    top5_df       = df.loc[[idx for _, idx in scored_pairs[:5]], AUDIO_FEATURES]
    plot_radar(seed_name, seed_features, top5_df,  save_path=p("radar.png"))
    plot_score_distribution(scored_pairs,           save_path=p("score_distribution.png"))
    plot_feature_correlation(df,                    save_path=p("feature_correlation.png"))
    plot_top_k_scores(results_df, seed_name,        save_path=p("top_k_scores.png"))
    print(f"\n[viz] All plots saved to: {output_dir}/")


class _StubTree:
    def __init__(self):
        self._heap = []

    def insert(self, score, idx):
        heapq.heappush(self._heap, (-score, idx))

    def top_k(self, k):
        heap_copy = self._heap[:]
        out = []
        for _ in range(min(k, len(heap_copy))):
            neg_s, i = heapq.heappop(heap_copy)
            out.append((-neg_s, i))
        return out


def main():
    seed_song = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SEED

    print("\n" + "=" * 55)
    print("  GrooveMatch – Starting pipeline")
    print("=" * 55)

    df, feature_matrix = preprocess()

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

    print(f"\n{'='*55}")
    print(f"  Top {TOP_K} Recommendations for: '{seed_song}'")
    print(f"{'='*55}")
    for _, row in results_df.iterrows():
        print(f"  {row['rank']:>2}. {row['track_name']:<35} – {row['artist_name']:<25} [{row['similarity_score']:.4f}]")

    splay_tree = _StubTree()
    rb_tree    = _StubTree()

    benchmark_results = run_benchmark(scored_pairs, splay_tree, rb_tree, k=TOP_K)
    benchmark_df      = benchmark_to_dataframe(benchmark_results)

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