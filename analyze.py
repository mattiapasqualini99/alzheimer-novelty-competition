"""
Analyse the relationship between novelty (from novelty_pipeline.py) and
journal quality (from merge_journals.py).

Inputs:
  --enriched : data/alzheimer_enriched.jsonl (from merge_journals.py)
  --scores   : results/alzheimer_scores.csv  (from novelty_pipeline.py)

Outputs:
  results/analysis_summary.txt     -- human-readable stats + regression
  results/novelty_by_quartile.png  -- box plot
  results/scatter_sjr_novelty.png  -- scatter of SJR vs novelty
  results/merged_data.csv          -- the joined dataset (for inspection)

What we compute:
  1. Match rate (% of target papers with a journal quartile)
  2. Per-quartile means and medians of novelty measures
  3. Simple OLS regression:
       share_new_unigrams ~ C(quartile) + n_tokens + C(year)
     The n_tokens control matters: longer abstracts mechanically have
     more chances to contain a "new" unigram, so without it the
     coefficient on quartile could just reflect abstract length.

Caveats this first attempt CANNOT address:
  * Causality. This is pure cross-section correlation.
  * Selection bias. PubMed's `retmax` is not a random sample.
  * Year matching. SCImago is from one year; journal quartiles
    fluctuate over time.
  * Sample size. With ~300 target papers across 4 quartiles, every
    cell has ~75 papers if even, less if unbalanced. Statistical
    power is modest.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, no display needed
import matplotlib.pyplot as plt


# Novelty columns we want to summarise
NOVELTY_COLS = [
    "share_new_unigrams",
    "share_new_bigrams",
    "share_new_mesh",
    "share_new_mesh_pairs",
]


def load_enriched(path):
    """Load the JSONL of enriched records into a DataFrame (target rows only)."""
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("is_target"):
                continue
            rows.append({
                "pmid": str(r["pmid"]),
                "year": r["year"],
                "journal_title": r.get("journal_title", ""),
                "journal_iso": r.get("journal_iso", ""),
                "sjr": r.get("sjr"),
                "sjr_quartile": r.get("sjr_quartile", "") or "",
            })
    return pd.DataFrame(rows)


def load_scores(path):
    df = pd.read_csv(path, dtype={"pmid": str})
    return df


def summarise_by_quartile(df, out_path):
    """Print and save a tidy summary table."""
    lines = []
    lines.append("=" * 70)
    lines.append("SAMPLE COMPOSITION")
    lines.append("=" * 70)
    n_total = len(df)
    n_with_q = (df["sjr_quartile"].isin(["Q1", "Q2", "Q3", "Q4"])).sum()
    lines.append(f"Target papers:                  {n_total}")
    lines.append(f"Matched to a journal quartile:  {n_with_q} "
                 f"({100*n_with_q/max(n_total,1):.1f}%)")
    lines.append("")
    counts = df["sjr_quartile"].replace("", "(unmatched)").value_counts()
    lines.append("Counts per quartile:")
    for k, v in counts.items():
        lines.append(f"  {k:<14}: {v}")

    lines.append("")
    lines.append("=" * 70)
    lines.append("NOVELTY BY QUARTILE (target papers, mean / median)")
    lines.append("=" * 70)

    df_with_q = df[df["sjr_quartile"].isin(["Q1", "Q2", "Q3", "Q4"])].copy()
    grouped = df_with_q.groupby("sjr_quartile")

    for col in NOVELTY_COLS:
        lines.append(f"\n{col}:")
        lines.append(f"  {'quartile':<10} {'n':>5} {'mean':>10} {'median':>10} "
                     f"{'std':>10}")
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            sub = grouped.get_group(q) if q in grouped.groups else None
            if sub is None or sub.empty:
                continue
            lines.append(f"  {q:<10} {len(sub):>5} "
                         f"{sub[col].mean():>10.4f} "
                         f"{sub[col].median():>10.4f} "
                         f"{sub[col].std():>10.4f}")

    text = "\n".join(lines)
    print(text)

    with open(out_path, "w") as f:
        f.write(text + "\n")
    return text


def run_regression(df, summary_path):
    """Simple OLS via numpy on pandas dummies.
    Avoids statsmodels dependency for the first attempt."""
    df = df[df["sjr_quartile"].isin(["Q1", "Q2", "Q3", "Q4"])].copy()
    if len(df) < 30:
        print("Too few matched rows for regression.")
        return

    y = df["share_new_unigrams"].values

    # Build design matrix:
    #   intercept + Q2/Q3/Q4 dummies (Q1 = baseline)
    #   + log(n_tokens) to control for abstract length
    #   + year dummies (one omitted)
    import numpy as np
    n = len(df)
    cols = []
    names = []

    cols.append(np.ones(n))
    names.append("intercept")

    for q in ["Q2", "Q3", "Q4"]:
        cols.append((df["sjr_quartile"] == q).astype(float).values)
        names.append(f"quartile_{q}")

    cols.append(np.log(df["n_tokens"].clip(lower=1).values))
    names.append("log_n_tokens")

    years_sorted = sorted(df["year"].dropna().unique())
    # Drop earliest year as baseline
    for y_ in years_sorted[1:]:
        cols.append((df["year"] == y_).astype(float).values)
        names.append(f"year_{int(y_)}")

    X = np.column_stack(cols)
    # OLS: beta = (X'X)^-1 X'y
    beta, res, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    resid = y - yhat
    rss = (resid ** 2).sum()
    tss = ((y - y.mean()) ** 2).sum()
    r2 = 1 - rss / tss
    dof = n - X.shape[1]
    sigma2 = rss / dof if dof > 0 else float("nan")
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t = beta / se

    lines = []
    lines.append("")
    lines.append("=" * 70)
    lines.append("OLS REGRESSION")
    lines.append("=" * 70)
    lines.append(f"Dependent variable: share_new_unigrams")
    lines.append(f"N = {n}, R² = {r2:.4f}, dof = {dof}")
    lines.append("")
    lines.append(f"  {'variable':<22} {'coef':>10} {'se':>10} {'t':>8}")
    for nm, b, s, tt in zip(names, beta, se, t):
        lines.append(f"  {nm:<22} {b:>10.4f} {s:>10.4f} {tt:>8.2f}")
    lines.append("")
    lines.append("Interpretation note:")
    lines.append("  Q1 is the baseline. A NEGATIVE coefficient on Q2/Q3/Q4")
    lines.append("  means those papers have LOWER share_new_unigrams than Q1,")
    lines.append("  i.e. less novel. (After controlling for abstract length")
    lines.append("  and year fixed effects.) |t| > 2 ≈ p < 0.05, but power")
    lines.append("  is modest at this sample size.")

    text = "\n".join(lines)
    print(text)
    with open(summary_path, "a") as f:
        f.write("\n" + text + "\n")


def plot_by_quartile(df, out_path):
    df = df[df["sjr_quartile"].isin(["Q1", "Q2", "Q3", "Q4"])].copy()
    if df.empty:
        print("No data to plot.")
        return
    df = df.sort_values("sjr_quartile")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, col in zip(axes.flatten(), NOVELTY_COLS):
        groups = [df[df["sjr_quartile"] == q][col].values
                  for q in ["Q1", "Q2", "Q3", "Q4"]]
        ax.boxplot(groups, tick_labels=["Q1", "Q2", "Q3", "Q4"])
        ax.set_title(col)
        ax.set_ylabel("share")
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Novelty measures by journal SJR quartile (target papers)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Saved boxplot to {out_path}")


def plot_scatter(df, out_path):
    df = df.dropna(subset=["sjr"]).copy()
    if df.empty:
        print("No SJR values to scatter.")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["sjr"], df["share_new_unigrams"], alpha=0.5, s=20)
    ax.set_xscale("log")
    ax.set_xlabel("SJR score (log scale)")
    ax.set_ylabel("share_new_unigrams")
    ax.set_title("Novelty vs journal SJR score")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Saved scatter to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--enriched", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    enriched = load_enriched(args.enriched)
    scores = load_scores(args.scores)

    print(f"Loaded {len(enriched)} target rows from enriched JSONL")
    print(f"Loaded {len(scores)} scored rows from scores CSV")

    df = enriched.merge(scores, on="pmid", how="inner",
                        suffixes=("_enr", ""))
    print(f"After merge: {len(df)} rows")

    # Save merged for manual inspection
    merged_path = outdir / "merged_data.csv"
    df.to_csv(merged_path, index=False)
    print(f"Saved merged dataset to {merged_path}")

    summary_path = outdir / "analysis_summary.txt"
    summarise_by_quartile(df, summary_path)
    run_regression(df, summary_path)
    plot_by_quartile(df, outdir / "novelty_by_quartile.png")
    plot_scatter(df, outdir / "scatter_sjr_novelty.png")

    print(f"\nAll outputs in {outdir}/")


if __name__ == "__main__":
    main()
