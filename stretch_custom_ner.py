"""
Module 6 Week A — Stretch: Custom NER Rules
stretch_custom_ner.py

Extends spaCy's built-in NER with a domain-specific EntityRuler that captures
climate terminology the base model misses or misclassifies.

Custom entity types defined:
  CLIMATE_EVENT  – major climate policy events/summits  (e.g., COP28, COP27)
  POLICY         – named agreements, mechanisms, frameworks  (e.g., Paris Agreement)
  REPORT         – authoritative scientific / institutional reports  (e.g., IPCC AR6)
  THRESHOLD      – quantitative climate targets / danger levels  (e.g., 1.5°C, 2°C)
  AGREEMENT      – international pacts / treaties  (e.g., Kyoto Protocol)

Run:
    python stretch_custom_ner.py
"""

import json
import pandas as pd
import spacy
from spacy.pipeline import EntityRuler

# ── 1. PATTERN LIBRARY ────────────────────────────────────────────────────────
#
# Each dict counts as one pattern entry.  We define 20+ entries covering more
# than 8 distinct concepts and 5 custom labels.
#
# Rules:
#   "phrase" patterns  → simple string matching (lowercased internally by spaCy)
#   "token"  patterns  → list of token-attribute dicts for flexible matching

CLIMATE_PATTERNS = [
    # ── CLIMATE_EVENT (major summits / conferences) ──────────────────────────
    # 1 – COP28  (token: handles casing variants like "cop28")
    {
        "label": "CLIMATE_EVENT",
        "pattern": [{"LOWER": "cop28"}],
        "id": "cop28",
    },
    # 2 – COP27
    {
        "label": "CLIMATE_EVENT",
        "pattern": [{"LOWER": "cop27"}],
        "id": "cop27",
    },
    # 3 – COP (followed by a number, e.g. "COP 29", "COP30")
    {
        "label": "CLIMATE_EVENT",
        "pattern": [{"LOWER": "cop"}, {"IS_DIGIT": True}],
        "id": "cop_numbered",
    },
    # 4 – Bonn Climate Change Conference (phrase)
    {
        "label": "CLIMATE_EVENT",
        "pattern": "Bonn Climate Change Conference",
        "id": "bonn_conference",
    },
    # 5 – Climate Ambition Summit (phrase)
    {
        "label": "CLIMATE_EVENT",
        "pattern": "Climate Ambition Summit",
        "id": "climate_ambition_summit",
    },

    # ── POLICY (named frameworks / mechanisms / schemes) ─────────────────────
    # 6 – Paris Agreement (phrase)  — "Paris" alone must NOT trigger
    {
        "label": "POLICY",
        "pattern": "Paris Agreement",
        "id": "paris_agreement",
    },
    # 7 – Paris Agreement (token pattern — captures "the Paris Agreement")
    {
        "label": "POLICY",
        "pattern": [{"LOWER": "paris"}, {"LOWER": "agreement"}],
        "id": "paris_agreement_tok",
    },
    # 8 – Carbon Border Adjustment Mechanism (phrase)
    {
        "label": "POLICY",
        "pattern": "Carbon Border Adjustment Mechanism",
        "id": "cbam",
    },
    # 9 – Nationally Determined Contributions / NDCs (both forms)
    {
        "label": "POLICY",
        "pattern": [{"LOWER": "nationally"}, {"LOWER": "determined"}, {"LOWER": "contributions"}],
        "id": "ndc_full",
    },
    # 10 – NDC(s) acronym
    {
        "label": "POLICY",
        "pattern": [{"LOWER": {"REGEX": "^ndcs?$"}}],
        "id": "ndc_acronym",
    },
    # 11 – Kyoto Protocol (phrase)
    {
        "label": "POLICY",
        "pattern": "Kyoto Protocol",
        "id": "kyoto_protocol",
    },

    # ── REPORT (scientific / institutional reports) ───────────────────────────
    # 12 – IPCC AR6 / Sixth Assessment Report (token pattern)
    {
        "label": "REPORT",
        "pattern": [{"LOWER": "ipcc"}, {"LOWER": {"REGEX": "^ar[4-7]$"}}],
        "id": "ipcc_ar",
    },
    # 13 – Sixth Assessment Report (phrase)
    {
        "label": "REPORT",
        "pattern": "Sixth Assessment Report",
        "id": "sixth_assessment_report",
    },
    # 14 – State of Food and Agriculture (phrase)
    {
        "label": "REPORT",
        "pattern": "State of Food and Agriculture",
        "id": "sofa_report",
    },
    # 15 – IPCC report (token pattern — generic)
    {
        "label": "REPORT",
        "pattern": [{"LOWER": "ipcc"}, {"LOWER": {"REGEX": "^reports?$"}}],
        "id": "ipcc_report",
    },

    # ── THRESHOLD (quantitative climate danger levels / targets) ─────────────
    # 16 – "1.5 degrees Celsius" / "1.5°C"  (token pattern)
    {
        "label": "THRESHOLD",
        "pattern": [
            {"LOWER": {"REGEX": r"^1[\.,]5$"}},
            {"LOWER": {"REGEX": r"^(degrees?|°c)$"}},
            {"LOWER": "celsius", "OP": "?"},
        ],
        "id": "threshold_1_5c",
    },
    # 17 – "2 degrees Celsius" / "2°C" target
    {
        "label": "THRESHOLD",
        "pattern": [
            {"LOWER": "2"},
            {"LOWER": {"REGEX": r"^(degrees?|°c)$"}},
            {"LOWER": "celsius", "OP": "?"},
        ],
        "id": "threshold_2c",
    },
    # 18 – "net zero" (phrase)
    {
        "label": "THRESHOLD",
        "pattern": "net zero",
        "id": "net_zero",
    },
    # 19 – "net-zero" (token pattern for hyphenated form)
    {
        "label": "THRESHOLD",
        "pattern": [{"LOWER": "net"}, {"LOWER": "-", "OP": "?"}, {"LOWER": "zero"}],
        "id": "net_zero_hyphen",
    },

    # ── AGREEMENT (international pacts / treaties) ────────────────────────────
    # 20 – Glasgow Climate Pact (phrase)
    {
        "label": "AGREEMENT",
        "pattern": "Glasgow Climate Pact",
        "id": "glasgow_climate_pact",
    },
    # 21 – Sunnylands Statement (phrase — from the US-China joint statement)
    {
        "label": "AGREEMENT",
        "pattern": "Sunnylands Statement",
        "id": "sunnylands_statement",
    },
    # 22 – Loss and Damage fund / mechanism
    {
        "label": "AGREEMENT",
        "pattern": [{"LOWER": "loss"}, {"LOWER": "and"}, {"LOWER": "damage"}],
        "id": "loss_and_damage",
    },
]

# ── 2. RULER FACTORY ──────────────────────────────────────────────────────────

def build_ruler_before(nlp):
    """Return an nlp pipeline with EntityRuler inserted BEFORE the NER."""
    if "entity_ruler" in nlp.pipe_names:
        nlp.remove_pipe("entity_ruler")
    ruler = nlp.add_pipe("entity_ruler", before="ner", config={"overwrite_ents": True})
    ruler.add_patterns(CLIMATE_PATTERNS)
    return nlp


def build_ruler_after(nlp):
    """Return an nlp pipeline with EntityRuler inserted AFTER the NER."""
    if "entity_ruler" in nlp.pipe_names:
        nlp.remove_pipe("entity_ruler")
    ruler = nlp.add_pipe("entity_ruler", after="ner", config={"overwrite_ents": False})
    ruler.add_patterns(CLIMATE_PATTERNS)
    return nlp


# ── 3. ENTITY EXTRACTION ──────────────────────────────────────────────────────

def extract_entities(df, nlp, pipeline_label="baseline"):
    """Run nlp over English texts and return a DataFrame of entities."""
    en_df = df[df["language"] == "en"]
    rows = []
    for _, row in en_df.iterrows():
        doc = nlp(row["text"])
        for ent in doc.ents:
            rows.append({
                "text_id": row["id"],
                "entity_text": ent.text,
                "entity_label": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char,
                "pipeline": pipeline_label,
            })
    cols = ["text_id", "entity_text", "entity_label", "start_char", "end_char", "pipeline"]
    return pd.DataFrame(rows, columns=cols)


# ── 4. EVALUATION (standard labels only) ─────────────────────────────────────

STANDARD_LABELS = {"ORG", "GPE", "DATE", "LAW", "MONEY", "PERSON",
                   "QUANTITY", "LOC", "EVENT", "WORK_OF_ART"}


def evaluate_standard(predicted_df, gold_df):
    """
    Compute precision / recall / F1 on overlapping standard spaCy labels only.
    Custom labels (CLIMATE_EVENT, POLICY, REPORT, THRESHOLD, AGREEMENT) are
    excluded so they cannot artificially depress precision.
    """
    pred_std = predicted_df[predicted_df["entity_label"].isin(STANDARD_LABELS)]

    gold_text_ids = set(gold_df["text_id"])
    pred_std = pred_std[pred_std["text_id"].isin(gold_text_ids)]

    pred_set = set(zip(pred_std["text_id"], pred_std["entity_text"], pred_std["entity_label"]))
    gold_set = set(zip(gold_df["text_id"], gold_df["entity_text"], gold_df["entity_label"]))

    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall    = tp / len(gold_set) if gold_set else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) else 0.0)
    return {"precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
            "tp": tp, "pred_count": len(pred_set), "gold_count": len(gold_set)}


# ── 5. BEFORE / AFTER COMPARISON ─────────────────────────────────────────────

def before_after_comparison(baseline_df, ruler_df):
    """Print entity counts by label for baseline vs. ruler pipeline."""
    def label_counts(df):
        return df.groupby("entity_label")["entity_text"].count().sort_values(ascending=False)

    base_counts = label_counts(baseline_df)
    ruler_counts = label_counts(ruler_df)

    all_labels = sorted(set(base_counts.index) | set(ruler_counts.index))
    rows = []
    for lbl in all_labels:
        b = int(base_counts.get(lbl, 0))
        r = int(ruler_counts.get(lbl, 0))
        rows.append({"label": lbl, "baseline": b, "with_ruler": r, "delta": r - b})
    comparison_df = pd.DataFrame(rows)
    return comparison_df


# ── 6. QUALITATIVE EXAMPLES ───────────────────────────────────────────────────

def qualitative_examples(df, ruler_df, n_per_label=3):
    """Surface example texts where each custom-label rule fired."""
    custom_labels = ["CLIMATE_EVENT", "POLICY", "REPORT", "THRESHOLD", "AGREEMENT"]
    text_map = df.set_index("id")["text"].to_dict()
    examples = {}
    for label in custom_labels:
        hits = ruler_df[ruler_df["entity_label"] == label].drop_duplicates("entity_text").head(n_per_label)
        label_examples = []
        for _, hit in hits.iterrows():
            snippet = text_map.get(hit["text_id"], "")
            s, e = hit["start_char"], hit["end_char"]
            context = snippet[max(0, s - 60): e + 60]
            label_examples.append({
                "entity": hit["entity_text"],
                "text_id": hit["text_id"],
                "context": "…" + context + "…",
            })
        examples[label] = label_examples
    return examples


# ── 7. MAIN ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Stretch 6A-S1 — Custom Climate EntityRuler")
    print("=" * 70)

    # Load data
    df   = pd.read_csv("data/climate_articles.csv")
    gold = pd.read_csv("data/gold_entities.csv")
    print(f"\nDataset: {len(df)} articles  |  Gold standard: {len(gold)} entities\n")

    # ── Pipeline 0: Baseline (no ruler) ──────────────────────────────────────
    nlp_base = spacy.load("en_core_web_sm")
    print("Running baseline spaCy NER …")
    base_df = extract_entities(df, nlp_base, pipeline_label="baseline")
    base_metrics = evaluate_standard(base_df, gold)
    print(f"  Baseline    -> P={base_metrics['precision']:.4f}  "
          f"R={base_metrics['recall']:.4f}  F1={base_metrics['f1']:.4f}  "
          f"(TP={base_metrics['tp']}, preds={base_metrics['pred_count']})")

    # ── Pipeline 1: EntityRuler BEFORE NER ───────────────────────────────────
    nlp_before = spacy.load("en_core_web_sm")
    nlp_before = build_ruler_before(nlp_before)
    print("\nRunning EntityRuler BEFORE NER …")
    before_df = extract_entities(df, nlp_before, pipeline_label="ruler_before")
    before_metrics = evaluate_standard(before_df, gold)
    print(f"  Ruler-Before -> P={before_metrics['precision']:.4f}  "
          f"R={before_metrics['recall']:.4f}  F1={before_metrics['f1']:.4f}  "
          f"(TP={before_metrics['tp']}, preds={before_metrics['pred_count']})")

    # ── Pipeline 2: EntityRuler AFTER NER ────────────────────────────────────
    nlp_after = spacy.load("en_core_web_sm")
    nlp_after = build_ruler_after(nlp_after)
    print("\nRunning EntityRuler AFTER NER …")
    after_df = extract_entities(df, nlp_after, pipeline_label="ruler_after")
    after_metrics = evaluate_standard(after_df, gold)
    print(f"  Ruler-After  -> P={after_metrics['precision']:.4f}  "
          f"R={after_metrics['recall']:.4f}  F1={after_metrics['f1']:.4f}  "
          f"(TP={after_metrics['tp']}, preds={after_metrics['pred_count']})")

    # ── Before / After count comparison ──────────────────────────────────────
    print("\n── Entity count comparison (BEFORE ruler vs AFTER ruler) ──")
    comparison_df = before_after_comparison(base_df, before_df)
    print(comparison_df.to_string(index=False))

    # ── Qualitative examples ──────────────────────────────────────────────────
    print("\n── Qualitative examples for custom labels ──")
    examples = qualitative_examples(df, before_df)
    for label, hits in examples.items():
        print(f"\n  [{label}]")
        for h in hits:
            print(f"    text_id={h['text_id']}  entity='{h['entity']}'")
            print(f"      context: {h['context']}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    base_df.to_csv("stretch_baseline_entities.csv", index=False)
    before_df.to_csv("stretch_ruler_before_entities.csv", index=False)
    after_df.to_csv("stretch_ruler_after_entities.csv", index=False)
    comparison_df.to_csv("stretch_comparison.csv", index=False)

    results = {
        "baseline":     base_metrics,
        "ruler_before": before_metrics,
        "ruler_after":  after_metrics,
        "qualitative_examples": examples,
    }
    with open("stretch_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n\nOutputs saved:")
    print("  stretch_baseline_entities.csv")
    print("  stretch_ruler_before_entities.csv")
    print("  stretch_ruler_after_entities.csv")
    print("  stretch_comparison.csv")
    print("  stretch_results.json")
    print("\nDone.")
    return results


if __name__ == "__main__":
    main()
