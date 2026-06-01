import argparse
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

parser = argparse.ArgumentParser()
parser.add_argument("--novelty", required=True)
parser.add_argument("--competition", required=True)
parser.add_argument("--outdir", required=True)
args = parser.parse_args()

# Load data
novelty_df = pd.read_csv(args.novelty)
comp_df = pd.read_csv(args.competition)

# Aggregate novelty by year
novelty_by_year = novelty_df.groupby('year').agg({
    'share_new_unigrams': 'mean',
    'share_new_bigrams': 'mean',
    'share_new_mesh': 'mean'
}).reset_index()

# Merge with competition
merged = pd.merge(novelty_by_year, comp_df, on='year', how='inner')

print("Merged data:")
print(merged)

# Create figure with 2 rows, 3 cols
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Novelty vs Competition Trends', fontsize=14, fontweight='bold')

# Row 1: Competition trends
axes[0, 0].plot(merged['year'], merged['herfindahl_authors'], marker='o', color='blue')
axes[0, 0].set_title('Author Concentration (Herfindahl)')
axes[0, 0].set_ylabel('Herfindahl Index')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(merged['year'], merged['pct_papers_q1'], marker='o', color='green')
axes[0, 1].set_title('Prestige Concentration (Q1 %)')
axes[0, 1].set_ylabel('% Papers in Q1')
axes[0, 1].grid(True, alpha=0.3)

axes[0, 2].plot(merged['year'], merged['n_unique_authors'], marker='o', color='red')
axes[0, 2].set_title('Author Diversity')
axes[0, 2].set_ylabel('# Unique Authors')
axes[0, 2].grid(True, alpha=0.3)

# Row 2: Novelty trends
axes[1, 0].plot(merged['year'], merged['share_new_unigrams'], marker='s', color='blue', label='unigrams')
axes[1, 0].plot(merged['year'], merged['share_new_bigrams'], marker='s', color='green', label='bigrams')
axes[1, 0].plot(merged['year'], merged['share_new_mesh'], marker='s', color='red', label='MeSH')
axes[1, 0].set_title('Novelty: Text Measures')
axes[1, 0].set_ylabel('Share of New Terms')
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].axis('off')
axes[1, 2].axis('off')

# Add correlation text
corr_text = "Correlations (Spearman):\n"
for col in ['share_new_unigrams', 'share_new_bigrams']:
    if col in merged.columns and 'herfindahl_authors' in merged.columns:
        rho, pval = spearmanr(merged['herfindahl_authors'], merged[col])
        corr_text += f"{col} vs Herfindahl: ρ={rho:.3f} (p={pval:.3f})\n"

axes[1, 1].text(0.1, 0.5, corr_text, fontsize=10, family='monospace')

plt.tight_layout()
plt.savefig(f"{args.outdir}/novelty_vs_competition_trends.png", dpi=150)
print(f"\nSaved to {args.outdir}/novelty_vs_competition_trends.png")
plt.close()
