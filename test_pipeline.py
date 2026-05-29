"""
Smoke test for the novelty pipeline.

Builds a tiny hand-crafted corpus of 5 reference + 4 target papers and
runs the scoring. We can eyeball the output to check that:
  * A paper that repeats stuff in the reference scores LOW.
  * A paper with genuinely new vocabulary/MeSH scores HIGH.
  * The example lists make biological sense.

This is NOT a unit test for correctness on real data -- it's a sanity
check that the plumbing works end-to-end before we hit PubMed.
"""
import json
import sys
from pathlib import Path

# Make sibling module importable
sys.path.insert(0, str(Path(__file__).parent))

from novelty_pipeline import build_reference, score_paper


CORPUS = [
    # ===== Reference papers (2010-2014) =====
    {
        "pmid": "R1", "year": 2010,
        "title": "Amyloid plaques and cerebral cortex pathology in Alzheimer disease",
        "abstract": "We examined amyloid plaque accumulation in the cerebral "
                    "cortex of patients with Alzheimer disease. Plaque density "
                    "correlated with cognitive decline.",
        "mesh": ["Alzheimer Disease", "Amyloid", "Cerebral Cortex",
                 "Cognitive Dysfunction"],
        "is_reference": True, "is_target": False,
    },
    {
        "pmid": "R2", "year": 2011,
        "title": "Tau protein and neurodegeneration in mouse models",
        "abstract": "Tau protein aggregation drives neurodegeneration. "
                    "Transgenic mouse models recapitulate the pathology.",
        "mesh": ["Tau Proteins", "Neurodegenerative Diseases",
                 "Mice, Transgenic", "Disease Models, Animal"],
        "is_reference": True, "is_target": False,
    },
    {
        "pmid": "R3", "year": 2012,
        "title": "MRI imaging reveals cortical atrophy in dementia",
        "abstract": "Magnetic resonance imaging scans of dementia patients "
                    "showed extensive cortical atrophy in the temporal lobe.",
        "mesh": ["Magnetic Resonance Imaging", "Dementia", "Atrophy",
                 "Temporal Lobe"],
        "is_reference": True, "is_target": False,
    },
    {
        "pmid": "R4", "year": 2013,
        "title": "Inflammatory markers in Alzheimer disease cerebrospinal fluid",
        "abstract": "Cerebrospinal fluid samples from Alzheimer disease "
                    "patients showed elevated inflammatory cytokine levels.",
        "mesh": ["Alzheimer Disease", "Cerebrospinal Fluid",
                 "Cytokines", "Inflammation"],
        "is_reference": True, "is_target": False,
    },
    {
        "pmid": "R5", "year": 2014,
        "title": "APOE genotype and Alzheimer disease risk",
        "abstract": "The APOE epsilon4 allele is the strongest genetic risk "
                    "factor for late-onset Alzheimer disease.",
        "mesh": ["Alzheimer Disease", "Apolipoproteins E",
                 "Genetic Predisposition to Disease"],
        "is_reference": True, "is_target": False,
    },

    # ===== Target papers (2018) =====

    # T1: deliberately conservative -- recycles reference vocabulary
    {
        "pmid": "T1", "year": 2018,
        "title": "Amyloid plaques and cortical atrophy in Alzheimer disease",
        "abstract": "We confirmed prior findings that amyloid plaque density "
                    "correlates with cortical atrophy in Alzheimer disease "
                    "patients on magnetic resonance imaging.",
        "mesh": ["Alzheimer Disease", "Amyloid", "Atrophy",
                 "Magnetic Resonance Imaging"],
        "is_reference": False, "is_target": True,
    },

    # T2: deliberately novel -- CRISPR + microglia + new genes
    {
        "pmid": "T2", "year": 2018,
        "title": "CRISPR-Cas9 screen of microglial autophagy reveals TREM2 "
                 "as a regulator of phagocytosis in Alzheimer disease",
        "abstract": "Using genome-wide CRISPR-Cas9 screening in iPSC-derived "
                    "microglia, we identified TREM2 and CD33 as central "
                    "regulators of phagocytic clearance. Single-cell RNA "
                    "sequencing revealed transcriptomic heterogeneity.",
        "mesh": ["Alzheimer Disease", "CRISPR-Cas Systems", "Microglia",
                 "Autophagy", "Single-Cell Analysis",
                 "Induced Pluripotent Stem Cells"],
        "is_reference": False, "is_target": True,
    },

    # T3: novel topic combination (gut-brain axis + Alzheimer)
    {
        "pmid": "T3", "year": 2018,
        "title": "Gut microbiome composition modulates neuroinflammation in "
                 "Alzheimer disease mouse models",
        "abstract": "Germ-free transgenic mice exhibited reduced amyloid "
                    "plaque burden. Fecal microbiota transplantation from "
                    "Alzheimer patients restored pathology.",
        "mesh": ["Alzheimer Disease", "Gastrointestinal Microbiome",
                 "Mice, Transgenic", "Amyloid", "Fecal Microbiota Transplantation"],
        "is_reference": False, "is_target": True,
    },

    # T4: mostly known content, one new method
    {
        "pmid": "T4", "year": 2018,
        "title": "Deep learning prediction of Alzheimer disease progression "
                 "from MRI",
        "abstract": "A convolutional neural network trained on magnetic "
                    "resonance imaging scans predicted cognitive decline "
                    "with high accuracy in Alzheimer disease patients.",
        "mesh": ["Alzheimer Disease", "Magnetic Resonance Imaging",
                 "Deep Learning", "Cognitive Dysfunction"],
        "is_reference": False, "is_target": True,
    },
]


def main():
    print("=" * 70)
    print("BUILDING REFERENCE UNIVERSE")
    print("=" * 70)
    ref = build_reference(CORPUS)
    print(f"Reference papers:       {ref['n_ref_papers']}")
    print(f"Unique unigrams:        {len(ref['unigrams'])}")
    print(f"Unique bigrams:         {len(ref['bigrams'])}")
    print(f"Unique MeSH terms:      {len(ref['mesh'])}")
    print(f"Unique MeSH pairs:      {len(ref['mesh_pairs'])}")
    print()

    print("=" * 70)
    print("SCORING TARGET PAPERS")
    print("=" * 70)

    scores = []
    for rec in CORPUS:
        if not rec["is_target"]:
            continue
        s = score_paper(rec, ref)
        scores.append((rec, s))

    # Sort by share_new_unigrams to make the ordering legible
    scores.sort(key=lambda x: x[1]["share_new_unigrams"], reverse=True)

    for rec, s in scores:
        print("-" * 70)
        print(f"[{s['pmid']}] {rec['title']}")
        print(f"  tokens={s['n_tokens']}  unique_unigrams={s['n_unique_unigrams']}  "
              f"bigrams={s['n_unique_bigrams']}  mesh={s['n_mesh']}  "
              f"mesh_pairs={s['n_mesh_pairs']}")
        print(f"  NEW unigrams:    {s['n_new_unigrams']:3d} "
              f"({s['share_new_unigrams']:.0%})  -> {s['examples_new_unigrams']}")
        print(f"  NEW bigrams:     {s['n_new_bigrams']:3d} "
              f"({s['share_new_bigrams']:.0%})  -> {s['examples_new_bigrams']}")
        print(f"  NEW MeSH:        {s['n_new_mesh']:3d} "
              f"({s['share_new_mesh']:.0%})  -> {s['examples_new_mesh']}")
        print(f"  NEW MeSH pairs:  {s['n_new_mesh_pairs']:3d} "
              f"({s['share_new_mesh_pairs']:.0%})")

    print()
    print("=" * 70)
    print("EXPECTED PATTERN")
    print("=" * 70)
    print("T2 (CRISPR/microglia) and T3 (gut microbiome) should rank HIGH;")
    print("T1 (recycles reference vocab) should rank LOW;")
    print("T4 (one new method on known disease) should be in the middle.")


if __name__ == "__main__":
    main()
