"""
Merge a PubMed JSONL (output of pubmed_fetch_2.py v2) with SCImago
journal quality data, adding `sjr` (continuous score) and
`sjr_quartile` (Q1/Q2/Q3/Q4) to each record.

SCImago data:
  Download from https://www.scimagojr.com/journalrank.php
    1. Set year (we recommend 2018 for the first attempt, middle of
       a 2015-2020 target range -- journal quartiles are pretty stable
       year-to-year, so a single-year proxy works for a first pass)
    2. Don't filter by category (we want all journals)
    3. Click "Download data" at the top right
    4. You get a SEMICOLON-separated CSV named like "scimagojr 2018.csv"

The columns we use:
    "Issn"             -- one or more ISSNs separated by commas, with
                          NO hyphens (e.g. "00280836, 14764687")
    "SJR"              -- continuous score, decimal with comma separator
                          in European locale (e.g. "1,234")
    "SJR Best Quartile" -- "Q1" / "Q2" / "Q3" / "Q4" / "-"

We match by stripping hyphens from PubMed ISSNs and trying all
PubMed ISSNs (print, electronic, linking) against all SCImago ISSNs
for each journal.

Usage:
    python merge_journals.py \\
        --input data/alzheimer.jsonl \\
        --scimago "data/scimagojr 2018.csv" \\
        --output data/alzheimer_enriched.jsonl
"""
import argparse
import csv
import json
import sys
from pathlib import Path


def normalise_issn(s):
    """Strip hyphens and whitespace, uppercase."""
    return (s or "").replace("-", "").replace(" ", "").upper().strip()


def load_scimago(path):
    """
    Return dict: issn (no hyphen) -> {sjr: float|None, quartile: str|"-"}.

    SCImago format quirks:
      * Field separator is ';' (semicolon), not ','
      * Numeric values use ',' as decimal separator (European locale)
      * The "Issn" column lists multiple ISSNs separated by ', '
      * Some rows have empty SJR / quartile fields ('-' or '')
    """
    issn_to_info = {}
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        # Sanity check: column names we need
        needed = {"Issn", "SJR", "SJR Best Quartile"}
        missing = needed - set(reader.fieldnames or [])
        if missing:
            print(f"ERROR: SCImago file missing columns: {missing}",
                  file=sys.stderr)
            print(f"Available columns: {reader.fieldnames}", file=sys.stderr)
            sys.exit(1)

        n_rows = 0
        n_with_quartile = 0
        for row in reader:
            n_rows += 1
            quartile = (row.get("SJR Best Quartile") or "").strip()
            sjr_raw = (row.get("SJR") or "").strip()

            sjr = None
            if sjr_raw and sjr_raw != "-":
                # European decimal: "1,234" -> 1.234
                try:
                    sjr = float(sjr_raw.replace(",", "."))
                except ValueError:
                    sjr = None

            if quartile and quartile != "-":
                n_with_quartile += 1

            issn_list = (row.get("Issn") or "").split(",")
            info = {"sjr": sjr, "quartile": quartile or ""}
            for issn in issn_list:
                norm = normalise_issn(issn)
                if len(norm) == 8:  # ISSNs are 8 chars without hyphen
                    issn_to_info[norm] = info

        print(f"SCImago: loaded {n_rows} journal entries, "
              f"{n_with_quartile} with quartile assigned, "
              f"{len(issn_to_info)} ISSN keys", flush=True)

    return issn_to_info


def enrich_record(rec, issn_to_info):
    """Add sjr / sjr_quartile to a record. Returns (rec, matched: bool)."""
    candidates = [
        rec.get("issn_print", ""),
        rec.get("issn_electronic", ""),
        rec.get("issn_linking", ""),
    ]
    for issn in candidates:
        key = normalise_issn(issn)
        if len(key) == 8 and key in issn_to_info:
            info = issn_to_info[key]
            rec["sjr"] = info["sjr"]
            rec["sjr_quartile"] = info["quartile"]
            return rec, True
    rec["sjr"] = None
    rec["sjr_quartile"] = ""
    return rec, False


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True)
    parser.add_argument("--scimago", required=True,
        help="SCImago CSV (semicolon-separated, downloaded from "
             "scimagojr.com)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    issn_to_info = load_scimago(args.scimago)

    n_total = 0
    n_target_total = 0
    n_target_matched = 0
    quartile_counts = {"Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0, "": 0}

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.input) as fin, open(args.output, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            rec = json.loads(line)
            rec, matched = enrich_record(rec, issn_to_info)
            n_total += 1
            if rec.get("is_target"):
                n_target_total += 1
                if matched:
                    n_target_matched += 1
                q = rec.get("sjr_quartile", "") or ""
                quartile_counts[q] = quartile_counts.get(q, 0) + 1
            fout.write(json.dumps(rec) + "\n")

    print(f"\nTotal records:           {n_total}")
    print(f"Target records:          {n_target_total}")
    print(f"Target matched to SJR:   {n_target_matched} "
          f"({100*n_target_matched/max(n_target_total,1):.1f}%)")
    print(f"\nTarget papers by quartile:")
    for q in ["Q1", "Q2", "Q3", "Q4", ""]:
        label = q if q else "(unmatched)"
        print(f"  {label:<14}: {quartile_counts.get(q, 0)}")
    print(f"\nWrote enriched file to {args.output}")

    # Quick diagnostic if match rate is low
    if n_target_matched / max(n_target_total, 1) < 0.5:
        print("\nWARNING: less than half of target papers matched.")
        print("Possible causes:")
        print("  * SCImago year mismatch (download a closer year)")
        print("  * ISSN format issues (check a few records by hand)")
        print("  * Many journals are in PubMed but not in SCImago "
              "(small/specialised journals)")


if __name__ == "__main__":
    main()
