# Measuring Competition and Novelty in Biomedical Research

A Python pipeline for reproducing the analysis in "Hyper-competition and the Narrowing of Scientific Ideas: Evidence from U.S. Biomedical Research."

## Overview

This pipeline measures two things:

1. **Scientific novelty** — using text-based (unigrams, bigrams, noun-phrases) and MeSH-based metrics following Arts, Melluso & Veugelers (2025)
2. **Research competition** — using author concentration (Herfindahl), prestige concentration (% Q1 papers), funding density, and career insecurity measures

**Key innovation**: backward-rolling reference window. Each target paper's novelty is measured against the cumulative knowledge base *up to its publication year*, not a pooled corpus. This enables year-on-year comparisons and isolates whether novelty changes are real or mechanical.

## Quick Start

```bash
# 1. Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Fetch PubMed data (example: Alzheimer Disease, 2008-2020)
python pubmed_fetch_2.py \
    --mesh "Alzheimer Disease" \
    --target-years 2015-2020 \
    --reference-years 2008-2014 \
    --max-per-year 50 \
    --email your@email.com \
    --output data/alzheimer.jsonl

# 3. Merge journal quality (requires SCImago CSV from scimagojr.com, year 2018)
python merge_journals.py \
    --input data/alzheimer.jsonl \
    --scimago data/scimagojr_2018.csv \
    --output data/alzheimer_enriched.jsonl

# 4. Score novelty (backward-rolling)
python novelty_pipeline.py \
    --input data/alzheimer_enriched.jsonl \
    --output results/alzheimer_scores.csv

# 5. Measure competition
python competition_measures.py \
    --input data/alzheimer_enriched.jsonl \
    --output results/competition_by_year.csv

# 6. Plot trends
python plot_trends.py \
    --novelty results/alzheimer_scores.csv \
    --competition results/competition_by_year.csv \
    --outdir results/

# 7. Full analysis (optional)
python analyze.py \
    --enriched data/alzheimer_enriched.jsonl \
    --scores results/alzheimer_scores.csv \
    --outdir results/
```

All data is public and freely available (PubMed, SCImago). No API key required for small samples.

## Pipeline Architecture

```
PubMed query
    ↓
[pubmed_fetch_2.py]         ← Abstracts, MeSH, authors, journal info
    ↓
[merge_journals.py]         ← + SJR score, journal quartile
    ↓
[novelty_pipeline.py]       ← Novelty scores (backward-rolling)
    ↓
[competition_measures.py]   ← Competition measures by year
    ↓
[plot_trends.py]            ← Visualization + correlations
    ↓
[analyze.py] (optional)     ← Extended analysis
```

## Novelty Metrics

For each target paper, we compute:

- **`share_new_unigrams`** — Fraction of word unigrams that don't appear in papers from prior years
- **`share_new_bigrams`** — Fraction of word pairs that are novel
- **`share_new_mesh`** — Fraction of MeSH descriptors that are new
- **`share_new_mesh_pairs`** — Fraction of MeSH co-occurrences that haven't appeared before
- **`ref_n_papers_at_score`** — Size of reference corpus when each paper was scored (diagnostic)

Example interpretation: `share_new_unigrams = 0.35` means 35% of the word unigrams in this paper's abstract have no precedent in the pre-publication literature base.

## Competition Metrics

Computed annually per field:

- **`herfindahl_authors`** — Concentration of author productivity. Range [0, 1]. Higher = fewer authors dominate field
- **`top10pct_authors_share`** — Fraction of papers from the top 10% most productive authors
- **`n_unique_authors`** — Total unique authors (diversity proxy)
- **`pct_papers_in_q1`** — % of papers in top-quartile journals (prestige concentration)
- **`pct_papers_matched`** — % matched to SCImago (diagnostic)

## Outputs

- **`alzheimer_scores.csv`** — One row per target paper with novelty scores
- **`competition_by_year.csv`** — One row per year with competition measures
- **`novelty_vs_competition_trends.png`** — Visualization: competition vs novelty over time
- **`analysis_summary.txt`** — Descriptive stats, regression coefficients, interpretation
- **`merged_data.csv`** — Full joined dataset for manual inspection

## Key Design Choices

### Backward-rolling reference
Standard approach: all reference papers form one pool; every target paper is scored against it.
**Our approach**: each target paper at year *t* is scored against papers from years < *t* only. This:
- Enables year-on-year comparisons
- Avoids artificial deflation of novelty for recent papers
- Accounts for vocabulary introduced in intervening years

### Why multiple novelty measures?
Each has different biases:
- **Text-based** (unigrams, bigrams): captures early-stage vocabulary innovation, sensitive to tokenization
- **MeSH-based**: captures formal codification by PubMed indexers, less granular
- **CD-index** (if added): measures disruption/impact, citation-dependent

If all decline together, signal is robust. Divergences reveal which aspects of novelty change.

### Testing the mechanism
We measure observable implications of the "competition causes caution" hypothesis:
1. **Cross-network collaboration** — Do scientists branch out when competition relaxes?
2. **Cross-field collaboration** — Do topic jumps increase?
3. **Topical diversity** — Do author research agendas broaden?

These are testable directly from PubMed/author data.

## Validation

Run the smoke test to verify the pipeline works:
```bash
python test_pipeline.py
```

This scores 6 synthetic papers spanning 2010--2018. Expected output: papers introducing new vocabulary (e.g., CRISPR in 2015) score high novelty; papers using the same vocabulary later (CRISPR in 2018) score lower (because vocabulary is no longer new). This demonstrates backward-rolling works correctly.

## Limitations (v1)

- **Small sample** (50 papers/year) — reference vocabulary is tiny; novelty is inflated
- **Single-year SCImago** — journal quartiles change; using 2018 quartiles for all years is approximate
- **No noun-phrase extraction yet** — uses raw unigrams/bigrams, more noise than AMV
- **Cross-section only** — no causal claims from the main analysis alone (ARRA DiD addresses this in full paper)

## For the full paper

These analyses are built for the ARRA difference-in-differences test:
1. Expand to full NIH dataset (all fields, 1985-2018)
2. Add official NIH success rates and funding data
3. Exploit ARRA 2009 as exogenous shock to competition
4. Test: fields with larger ARRA treatment → larger novelty increases post-2009?

This pipeline is the methodological foundation.

## References

- Arts, S., Melluso, N., & Veugelers, R. (2025). Beyond Citations: Measuring Novel Scientific Ideas and their Impact in Publication Text. Review of Economics and Statistics, 1-33.
- Callaway, B., Goodman-Bacon, A., & Sant'Anna, P. H. (2024). Event-Studies with a Continuous Treatment. NBER Working Paper No. w32118.
- Funk, R. J., & Owen-Smith, J. (2017). A Dynamic Network Measure of Technological Change. Management Science, 63(3), 791-817.
- Park, M., Leahey, E., & Funk, R. J. (2023). Papers and Patents Are Becoming Less Disruptive Over Time. Nature, 613(7942), 138-144.
- Packalen, M., & Bhattacharya, J. (2020). NIH Funding and the Pursuit of Edge Science. PNAS, 117(22), 12011-12016.

## Contact & Questions

For questions about implementation, data, or methodology, contact: m.pasqualini@tue.nl

---

**Status**: First attempt / proof of concept. Ready for feedback and iteration.
