"""
Fetch a small PubMed sample for the AMV-style novelty pipeline.

v2 changes:
  * Now also extracts journal title, ISO abbreviation, and ISSNs
    (print, electronic, linking) -- needed for merging with SCImago
    journal quality data downstream.

Pulls papers matching a MeSH term across a date range, extracts
title / abstract / year / MeSH terms / journal info, and writes one
JSON record per line (JSONL).

Each record is flagged as `is_target` (paper we want to score) or
`is_reference` (paper used to define the prior vocabulary).

Usage:
    python pubmed_fetch_2.py \\
        --mesh "Alzheimer Disease" \\
        --target-years 2015-2020 \\
        --reference-years 2008-2014 \\
        --max-per-year 50 \\
        --email you@example.com \\
        --output data/alzheimer.jsonl
"""
import argparse
import json
import time
from pathlib import Path

from Bio import Entrez


def search_pmids(query, year, retmax, email, api_key=None):
    """Return up to `retmax` PMIDs matching `query` in a single year."""
    Entrez.email = email
    if api_key:
        Entrez.api_key = api_key
    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        mindate=str(year),
        maxdate=str(year),
        datetype="pdat",
        retmax=retmax,
    )
    result = Entrez.read(handle)
    handle.close()
    return result["IdList"]


def fetch_records(pmids, email, api_key=None, batch_size=50):
    """Yield parsed dicts for `pmids`, in batches of `batch_size`."""
    Entrez.email = email
    if api_key:
        Entrez.api_key = api_key
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i + batch_size]
        handle = Entrez.efetch(
            db="pubmed", id=batch, rettype="xml", retmode="xml",
        )
        records = Entrez.read(handle)
        handle.close()
        for article in records.get("PubmedArticle", []):
            parsed = parse_record(article)
            if parsed is not None:
                yield parsed
        time.sleep(0.4)


def _safe_str(x):
    return str(x) if x is not None else ""


def parse_record(article):
    """Pick out PMID, year, title, abstract, MeSH terms, journal info."""
    citation = article["MedlineCitation"]
    pmid = str(citation["PMID"])
    art = citation["Article"]

    title = _safe_str(art.get("ArticleTitle", "")).strip()

    abstract_parts = art.get("Abstract", {}).get("AbstractText", []) or []
    abstract = " ".join(_safe_str(part) for part in abstract_parts).strip()
    if not abstract:
        return None

    # ---- Publication year ----
    pub_year = None
    pd = art.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
    if "Year" in pd:
        try:
            pub_year = int(pd["Year"])
        except ValueError:
            pass
    elif "MedlineDate" in pd:
        try:
            pub_year = int(_safe_str(pd["MedlineDate"]).split()[0])
        except (ValueError, IndexError):
            pass

    # ---- MeSH descriptors ----
    mesh_list = citation.get("MeshHeadingList", []) or []
    mesh_terms = []
    for m in mesh_list:
        desc = m.get("DescriptorName")
        if desc is not None:
            mesh_terms.append(str(desc))

    # ---- Journal info (NEW in v2) ----
    journal = art.get("Journal", {})
    journal_title = _safe_str(journal.get("Title", ""))
    journal_iso = _safe_str(journal.get("ISOAbbreviation", ""))

    # ISSN in <Journal> is a single StringElement with IssnType attribute
    issn_print = ""
    issn_electronic = ""
    issn_element = journal.get("ISSN")
    if issn_element is not None:
        issn_type = ""
        try:
            issn_type = issn_element.attributes.get("IssnType", "")
        except AttributeError:
            pass
        if issn_type == "Print":
            issn_print = str(issn_element)
        elif issn_type == "Electronic":
            issn_electronic = str(issn_element)
        else:
            # Unknown type, store as print
            issn_print = str(issn_element)

    # Linking ISSN from MedlineJournalInfo (often the most stable identifier)
    issn_linking = ""
    mji = citation.get("MedlineJournalInfo", {})
    if "ISSNLinking" in mji:
        issn_linking = str(mji["ISSNLinking"])

    return {
        "pmid": pmid,
        "year": pub_year,
        "title": title,
        "abstract": abstract,
        "mesh": mesh_terms,
        "journal_title": journal_title,
        "journal_iso": journal_iso,
        "issn_print": issn_print,
        "issn_electronic": issn_electronic,
        "issn_linking": issn_linking,
    }


def parse_year_range(s):
    a, b = s.split("-")
    return int(a), int(b)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--target-years", required=True, help="e.g. 2015-2020")
    parser.add_argument("--reference-years", required=True, help="e.g. 2008-2014")
    parser.add_argument("--max-per-year", type=int, default=200)
    parser.add_argument("--email", required=True)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    target_start, target_end = parse_year_range(args.target_years)
    ref_start, ref_end = parse_year_range(args.reference_years)

    query = (
        f'"{args.mesh}"[MeSH Major Topic] '
        f'AND English[Language] '
        f'AND hasabstract[Filter]'
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0

    with out_path.open("w") as f:
        for year in range(ref_start, target_end + 1):
            print(f"[{year}] searching...", flush=True)
            pmids = search_pmids(
                query, year, args.max_per_year, args.email, args.api_key,
            )
            print(f"[{year}] {len(pmids)} PMIDs, fetching abstracts...",
                  flush=True)
            n_year = 0
            for rec in fetch_records(pmids, args.email, args.api_key):
                rec["is_target"] = (
                    target_start <= (rec["year"] or 0) <= target_end
                )
                rec["is_reference"] = (
                    ref_start <= (rec["year"] or 0) <= ref_end
                )
                f.write(json.dumps(rec) + "\n")
                n_written += 1
                n_year += 1
            print(f"[{year}] kept {n_year} records "
                  f"(after dropping no-abstract)", flush=True)

    print(f"\nDone. Wrote {n_written} records to {out_path}")


if __name__ == "__main__":
    main()
