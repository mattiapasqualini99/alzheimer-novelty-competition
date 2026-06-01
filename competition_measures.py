import argparse
import json
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    records = []
    with open(args.input) as f:
        for line in f:
            records.append(json.loads(line))
    
    df = pd.DataFrame(records)
    
    # Group by year
    results = []
    for year in sorted(df['year'].unique()):
        year_df = df[df['year'] == year]
        target_df = year_df[year_df['is_target'] == True]
        
        if len(target_df) == 0:
            continue
        
        # Herfindahl author concentration
        author_counts = {}
        for authors in target_df['authors']:
            for author in authors:
                author_counts[author] = author_counts.get(author, 0) + 1
        
        total = sum(author_counts.values())
        herfindahl = sum((count/total)**2 for count in author_counts.values()) if total > 0 else 0
        
        # Prestige concentration (% in Q1)
        q1_count = len(target_df[target_df['sjr_quartile'] == 'Q1'])
        pct_q1 = 100 * q1_count / len(target_df)
        
        # Author diversity
        unique_authors = len(author_counts)
        
        results.append({
            'year': year,
            'n_papers': len(target_df),
            'herfindahl_authors': round(herfindahl, 4),
            'pct_papers_q1': round(pct_q1, 2),
            'n_unique_authors': unique_authors,
            'avg_papers_per_author': round(total / unique_authors, 2) if unique_authors > 0 else 0
        })
    
    result_df = pd.DataFrame(results)
    result_df.to_csv(args.output, index=False)
    print(result_df.to_string())
    print(f"\nWrote to {args.output}")

if __name__ == "__main__":
    main()
