"""
Mechanism tests for the competition-novelty hypothesis.

Tests three observable implications:
1. Cross-network collaboration: % of new author pairs (first-time collaborations)
2. Topical distance: semantic distance between co-authors' prior work (Jaccard on MeSH)
3. Cross-field work: mean topical distance between consecutive papers of same author

Inputs:
  --input : data/alzheimer_enriched.jsonl (target papers only)

Outputs:
  results/mechanism_metrics_by_year.csv
  results/mechanism_plots.png
"""
import argparse
import json
from collections import defaultdict, Counter
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import jaccard


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def jaccard_similarity(set1, set2):
    """Compute Jaccard similarity between two sets. Range [0, 1]."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def compute_mechanism_metrics(records_by_year):
    """
    For each year, compute:
    1. % new collaborations (first-time author pairs)
    2. Mean topical distance between co-authors
    3. Mean topical distance between consecutive papers of same author
    
    Returns dict with yearly metrics.
    """
    results = []
    
    # Build author→mesh mapping (all papers, across all years)
    author_mesh = defaultdict(set)
    for recs in records_by_year.values():
        for rec in recs:
            mesh_set = set(rec.get("mesh", []) or [])
            for author in rec.get("authors", []) or []:
                author_mesh[author].update(mesh_set)
    
    # Build author→papers mapping (in order)
    author_papers = defaultdict(list)
    all_papers_by_year = []
    for year in sorted(records_by_year.keys()):
        for rec in records_by_year[year]:
            all_papers_by_year.append((year, rec))
            for author in rec.get("authors", []) or []:
                author_papers[author].append((year, rec))
    
    # Sort each author's papers by year
    for author in author_papers:
        author_papers[author].sort(key=lambda x: x[0])
    
    # Compute metrics per year
    collaborations_ever = set()  # Track all co-author pairs seen so far
    
    for year in sorted(records_by_year.keys()):
        recs = records_by_year[year]
        n_papers = len(recs)
        
        if n_papers == 0:
            continue
        
        # ---- 1. New collaborations ----
        new_pairs_this_year = 0
        total_pairs_this_year = 0
        
        for rec in recs:
            authors = rec.get("authors", []) or []
            # All pairs of authors in this paper
            for i in range(len(authors)):
                for j in range(i + 1, len(authors)):
                    pair = tuple(sorted([authors[i], authors[j]]))
                    total_pairs_this_year += 1
                    if pair not in collaborations_ever:
                        new_pairs_this_year += 1
                        collaborations_ever.add(pair)
        
        pct_new_collabs = (
            100 * new_pairs_this_year / total_pairs_this_year
            if total_pairs_this_year > 0 else 0
        )
        
        # ---- 2. Topical distance between co-authors ----
        topical_distances = []
        
        for rec in recs:
            authors = rec.get("authors", []) or []
            for i in range(len(authors)):
                for j in range(i + 1, len(authors)):
                    mesh_i = author_mesh.get(authors[i], set())
                    mesh_j = author_mesh.get(authors[j], set())
                    if mesh_i and mesh_j:
                        # Jaccard similarity; convert to distance (1 - similarity)
                        sim = jaccard_similarity(mesh_i, mesh_j)
                        dist = 1 - sim
                        topical_distances.append(dist)
        
        mean_topical_dist_coauthors = (
            np.mean(topical_distances) if topical_distances else 0
        )
        
        # ---- 3. Topical distance between consecutive papers of same author ----
        consecutive_distances = []
        
        for author in author_papers:
            papers_list = author_papers[author]
            # Only consider papers from this year and earlier (backward-looking)
            papers_up_to_year = [
                (y, rec) for y, rec in papers_list if y <= year
            ]
            
            if len(papers_up_to_year) >= 2:
                # Compare consecutive papers
                for k in range(len(papers_up_to_year) - 1):
                    y_prev, rec_prev = papers_up_to_year[k]
                    y_curr, rec_curr = papers_up_to_year[k + 1]
                    
                    mesh_prev = set(rec_prev.get("mesh", []) or [])
                    mesh_curr = set(rec_curr.get("mesh", []) or [])
                    
                    if mesh_prev and mesh_curr:
                        sim = jaccard_similarity(mesh_prev, mesh_curr)
                        dist = 1 - sim
                        consecutive_distances.append(dist)
        
        mean_topical_dist_consecutive = (
            np.mean(consecutive_distances) if consecutive_distances else 0
        )
        
        # ---- Collect results ----
        results.append({
            "year": int(year),
            "n_papers": n_papers,
            "pct_new_collaborations": pct_new_collabs,
            "mean_topical_dist_coauthors": mean_topical_dist_coauthors,
            "mean_topical_dist_consecutive": mean_topical_dist_consecutive,
            "n_coauthor_pairs": total_pairs_this_year,
        })
    
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args()
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading {args.input}...")
    records = load_jsonl(args.input)
    
    # Filter to target papers only
    target_recs = [r for r in records if r.get("is_target")]
    print(f"Total records: {len(records)}, target: {len(target_recs)}")
    
    # Group by year
    by_year = defaultdict(list)
    for rec in target_recs:
        y = rec.get("year")
        if y is not None:
            by_year[y].append(rec)
    
    print(f"Years: {sorted(by_year.keys())}")
    
    metrics = compute_mechanism_metrics(by_year)
    df = pd.DataFrame(metrics)
    
    out_csv = outdir / "mechanism_metrics_by_year.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nMechanism metrics by year:")
    print(df.to_string(index=False))
    print(f"\nWrote to {out_csv}")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: % new collaborations
    axes[0].plot(df["year"], df["pct_new_collaborations"],
                marker="o", linewidth=2, markersize=6, color="C0")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("% new collaborations")
    axes[0].set_title("New Author Pairs\n(higher = exploring new networks)")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(bottom=0)
    
    # Plot 2: Topical distance between co-authors
    axes[1].plot(df["year"], df["mean_topical_dist_coauthors"],
                marker="s", linewidth=2, markersize=6, color="C1")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Mean topical distance")
    axes[1].set_title("Co-author Topic Diversity\n(higher = more diverse collaborators)")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)
    
    # Plot 3: Topical distance between consecutive papers
    axes[2].plot(df["year"], df["mean_topical_dist_consecutive"],
                marker="^", linewidth=2, markersize=6, color="C2")
    axes[2].set_xlabel("Year")
    axes[2].set_ylabel("Mean topical distance")
    axes[2].set_title("Author Topic Migration\n(higher = jumping topics more)")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(0, 1)
    
    fig.suptitle("Mechanism Tests: How Competition Shapes Collaboration and Exploration",
                fontsize=13, fontweight="bold")
    fig.tight_layout()
    
    out_png = outdir / "mechanism_plots.png"
    fig.savefig(out_png, dpi=120)
    print(f"Saved plot to {out_png}")
    
    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print("\nIf competition INCREASES (higher Herfindahl, more Q1 papers):")
    print("  → % new collaborations should DECREASE (less exploration)")
    print("  → Topical distance between co-authors should DECREASE (work with similar people)")
    print("  → Topic migration (consecutive papers) should DECREASE (stay in lane)")
    print("\nIf competition DECREASES (e.g., ARRA funding shock):")
    print("  → All three metrics should INCREASE")
    print("\nExpected direction for Alzheimer 2008→2020 (competition rising):")
    print("  → Downward trends on all three metrics")


if __name__ == "__main__":
    main()
