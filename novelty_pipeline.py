"""
Compute AMV-style novelty measures on a PubMed sample.

SIMPLIFIED first attempt -- what we DO:
  * Lowercase + tokenize title and abstract with a regex
  * Drop a minimal stoplist and 1-char tokens
  * Build a "reference universe" from all `is_reference=True` records:
      - set of unigrams ever seen
      - set of bigrams ever seen
      - set of MeSH terms ever seen
      - set of MeSH pair co-occurrences (within a single paper) ever seen
  * For each `is_target` paper, count how many of its
    unigrams / bigrams / MeSH terms / MeSH pairs are NEW
    (i.e., absent from the reference universe).
  * Also compute the SHARE of new elements per paper (size-controlled).

What we are NOT doing yet (deliberately, for the first attempt):
  * Noun-phrase extraction (AMV uses spaCy; we use raw n-grams).
    Adding this is Phase 2 -- it cleans up the unigram noise a lot.
  * Year-by-year backward-looking reference. Right now the whole
    reference block is "the past". This means a paper in 2015 and one
    in 2020 see the same reference. AMV updates the reference up to
    year t-1 of each paper. Adding this is Phase 2.
  * Frequency cutoffs. AMV considers an element "seen" only if it
    appeared at least K times in the prior corpus. With a tiny sample,
    a typo appearing once counts as "seen" and kills a candidate
    novelty -- we'll need this once we have more data.

The expected biases of this simplified version, GOOD to surface:
  * Inflated novelty (small reference = many false novelties)
  * Token noise (typos, abbreviations, gene symbols)
  * Author names occasionally leaking into "novel" tokens
"""
import argparse
import csv
import json
import re
from collections import Counter
from itertools import combinations
from pathlib import Path


# Minimal stoplist. The real AMV uses a longer NLTK-derived one.
STOPWORDS = set("""
a about above after again against all am an and any are aren't as at
be because been before being below between both but by
can could couldn't
did didn't do does doesn't doing don't down during
each
few for from further
had hadn't has hasn't have haven't having he he'd he'll he's her here here's
hers herself him himself his how how's
i i'd i'll i'm i've if in into is isn't it it's its itself
just
let's
me more most mustn't my myself
no nor not
of off on once only or other ought our ours ourselves out over own
same shan't she she'd she'll she's should shouldn't so some such
than that that's the their theirs them themselves then there there's these
they they'd they'll they're they've this those through to too
under until up
very
was wasn't we we'd we'll we're we've were weren't what what's when when's
where where's which while who who's whom why why's with won't would
wouldn't
you you'd you'll you're you've your yours yourself yourselves
also however therefore thus may might also using used use one two
""".split())

# Allow alpha or alphanumeric (gene symbols like TREM2) and internal hyphens.
TOKEN_RE = re.compile(r"[a-z][a-z0-9\-]*[a-z0-9]")


def tokenize(text):
    """Lowercase, regex-tokenize, drop stopwords and short tokens."""
    text = (text or "").lower()
    return [t for t in TOKEN_RE.findall(text)
            if t not in STOPWORDS and len(t) > 2]


def bigrams(tokens):
    """Adjacent token pairs."""
    return list(zip(tokens, tokens[1:]))


def build_reference(records):
    """Build the union vocabulary from all reference papers."""
    ref_unigrams = set()
    ref_bigrams = set()
    ref_mesh = set()
    ref_mesh_pairs = set()  # tuples sorted alphabetically
    n_ref = 0
    for r in records:
        if not r.get("is_reference"):
            continue
        n_ref += 1
        toks = tokenize(r["title"] + " " + r["abstract"])
        ref_unigrams.update(toks)
        ref_bigrams.update(bigrams(toks))
        mesh = sorted(set(r.get("mesh", [])))
        ref_mesh.update(mesh)
        for a, b in combinations(mesh, 2):
            ref_mesh_pairs.add((a, b))
    return {
        "n_ref_papers": n_ref,
        "unigrams": ref_unigrams,
        "bigrams": ref_bigrams,
        "mesh": ref_mesh,
        "mesh_pairs": ref_mesh_pairs,
    }


def score_paper(rec, ref):
    """Return a flat dict of novelty scores + a few example novel items."""
    toks = tokenize(rec["title"] + " " + rec["abstract"])
    tok_set = set(toks)
    bg_set = set(bigrams(toks))
    mesh = sorted(set(rec.get("mesh", [])))
    mesh_set = set(mesh)
    mesh_pairs_here = set(combinations(mesh, 2))

    new_unigrams = tok_set - ref["unigrams"]
    new_bigrams = bg_set - ref["bigrams"]
    new_mesh = mesh_set - ref["mesh"]
    new_mesh_pairs = mesh_pairs_here - ref["mesh_pairs"]

    n_u, n_b, n_m, n_mp = len(tok_set), len(bg_set), len(mesh_set), len(mesh_pairs_here)

    return {
        "pmid": rec["pmid"],
        "year": rec["year"],
        "n_tokens": len(toks),
        "n_unique_unigrams": n_u,
        "n_unique_bigrams": n_b,
        "n_mesh": n_m,
        "n_mesh_pairs": n_mp,
        "n_new_unigrams": len(new_unigrams),
        "n_new_bigrams": len(new_bigrams),
        "n_new_mesh": len(new_mesh),
        "n_new_mesh_pairs": len(new_mesh_pairs),
        "share_new_unigrams": len(new_unigrams) / max(n_u, 1),
        "share_new_bigrams": len(new_bigrams) / max(n_b, 1),
        "share_new_mesh": len(new_mesh) / max(n_m, 1),
        "share_new_mesh_pairs": len(new_mesh_pairs) / max(n_mp, 1),
        # Keep top examples for manual inspection -- the most useful
        # output of a first attempt is the eyeball check.
        "examples_new_unigrams": sorted(new_unigrams)[:8],
        "examples_new_bigrams": [" ".join(b) for b in sorted(new_bigrams)[:8]],
        "examples_new_mesh": sorted(new_mesh)[:5],
    }


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_csv(rows, path):
    """Write rows to CSV, joining list fields with semicolons."""
    if not rows:
        Path(path).touch()
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            flat = {k: ("; ".join(v) if isinstance(v, list) else v)
                    for k, v in row.items()}
            writer.writerow(flat)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
        help="JSONL from pubmed_fetch.py")
    parser.add_argument("--output", required=True, help="CSV output path")
    args = parser.parse_args()

    records = load_jsonl(args.input)
    n_ref = sum(1 for r in records if r.get("is_reference"))
    n_tgt = sum(1 for r in records if r.get("is_target"))
    print(f"Loaded {len(records)} records ({n_ref} reference, {n_tgt} target)")

    print("Building reference universe...")
    ref = build_reference(records)
    print(f"  {ref['n_ref_papers']} reference papers")
    print(f"  {len(ref['unigrams']):>7} unigrams")
    print(f"  {len(ref['bigrams']):>7} bigrams")
    print(f"  {len(ref['mesh']):>7} MeSH terms")
    print(f"  {len(ref['mesh_pairs']):>7} MeSH pairs")

    print("Scoring target papers...")
    rows = [score_paper(r, ref) for r in records if r.get("is_target")]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output)
    print(f"Wrote {len(rows)} scored papers to {args.output}")


if __name__ == "__main__":
    main()
