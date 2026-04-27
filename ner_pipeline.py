"""
Module 6 Week A — Lab: NER Pipeline

Build and compare Named Entity Recognition pipelines using spaCy
and Hugging Face on climate-related text data.

Run: python ner_pipeline.py
"""

import pandas as pd
import numpy as np
import spacy
import unicodedata
from transformers import pipeline as hf_pipeline


def load_data(filepath="data/climate_articles.csv"):
    """Load the climate articles dataset.

    Args:
        filepath: Path to the CSV file.

    Returns:
        DataFrame with columns: id, text, source, language, category.
    """
    return pd.read_csv(filepath)


def explore_data(df):
    """Summarize basic corpus statistics.

    Args:
        df: DataFrame returned by load_data.

    Returns:
        Dictionary with keys:
          'shape': tuple (n_rows, n_cols)
          'lang_counts': dict mapping language code -> row count
          'category_counts': dict mapping category -> row count
          'text_length_stats': dict with 'mean', 'min', 'max' word counts
    """
    word_counts = df['text'].str.split().str.len()
    return {
        'shape': df.shape,
        'lang_counts': df['language'].value_counts().to_dict(),
        'category_counts': df['category'].value_counts().to_dict(),
        'text_length_stats': {
            'mean': float(word_counts.mean()),
            'min': float(word_counts.min()),
            'max': float(word_counts.max())
        }
    }


def preprocess_text(text, nlp):
    """Preprocess a single text string for NLP analysis.

    Normalize Unicode, lowercase, remove punctuation, tokenize,
    and lemmatize using the injected spaCy pipeline.

    Args:
        text: Raw text string.
        nlp: A loaded spaCy Language object (e.g., en_core_web_sm).

    Returns:
        List of cleaned, lemmatized token strings.
    """
    norm_text = unicodedata.normalize('NFC', text)
    doc = nlp(norm_text)
    return [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]


def extract_spacy_entities(df, nlp):
    """Extract named entities from English texts using spaCy NER.

    Args:
        df: DataFrame with columns id, text, language, ...
        nlp: A loaded spaCy Language object.

    Returns:
        DataFrame with columns: text_id, entity_text, entity_label,
        start_char, end_char.
    """
    en_df = df[df['language'] == 'en']
    entities = []
    for _, row in en_df.iterrows():
        doc = nlp(row['text'])
        for ent in doc.ents:
            entities.append({
                'text_id': row['id'],
                'entity_text': ent.text,
                'entity_label': ent.label_,
                'start_char': ent.start_char,
                'end_char': ent.end_char
            })
    return pd.DataFrame(entities, columns=['text_id', 'entity_text', 'entity_label', 'start_char', 'end_char'])


def extract_hf_entities(df, ner_pipeline):
    """Extract named entities from English texts using Hugging Face NER.

    Uses the injected HF pipeline (expected: dslim/bert-base-NER).

    Args:
        df: DataFrame with columns id, text, language, ...
        ner_pipeline: A loaded Hugging Face `pipeline('ner', ...)` object.

    Returns:
        DataFrame with columns: text_id, entity_text, entity_label,
        start_char, end_char.
    """
    en_df = df[df['language'] == 'en']
    entities = []
    for _, row in en_df.iterrows():
        text_id = row['id']
        preds = ner_pipeline(row['text'])

        # Pass 1: merge WordPiece ## subword tokens back into whole words
        merged_words = []
        for p in preds:
            if p['word'].startswith('##') and merged_words:
                merged_words[-1] = dict(merged_words[-1])  # make mutable copy
                merged_words[-1]['word'] += p['word'][2:]
                merged_words[-1]['end'] = p['end']
            else:
                merged_words.append(p.copy())

        # Pass 2: merge consecutive I- tokens into the preceding B- entity span
        merged_spans = []
        for p in merged_words:
            raw_label = p['entity']
            if raw_label.startswith('I-') and merged_spans:
                # extend the previous entity span
                merged_spans[-1]['word'] += ' ' + p['word']
                merged_spans[-1]['end'] = p['end']
            else:
                merged_spans.append(p.copy())

        # Strip B-/I- prefix and collect results
        for p in merged_spans:
            label = p['entity']
            if label.startswith('B-') or label.startswith('I-'):
                label = label[2:]
            entities.append({
                'text_id': text_id,
                'entity_text': p['word'],
                'entity_label': label,
                'start_char': p['start'],
                'end_char': p['end']
            })
    return pd.DataFrame(entities, columns=['text_id', 'entity_text', 'entity_label', 'start_char', 'end_char'])


def compare_ner_outputs(spacy_df, hf_df):
    """Compare entity extraction results from spaCy and Hugging Face.

    Args:
        spacy_df: DataFrame of spaCy entities (from extract_spacy_entities).
        hf_df: DataFrame of HF entities (from extract_hf_entities).

    Returns:
        Dictionary with keys:
          'spacy_counts': dict of entity_label -> count for spaCy
          'hf_counts': dict of entity_label -> count for HF
          'total_spacy': int total entities from spaCy
          'total_hf': int total entities from HF
          'both': set of (text_id, entity_text) tuples found by both systems
          'spacy_only': set of (text_id, entity_text) tuples found only by spaCy
          'hf_only': set of (text_id, entity_text) tuples found only by HF
    """
    spacy_counts = spacy_df['entity_label'].value_counts().to_dict() if not spacy_df.empty else {}
    hf_counts = hf_df['entity_label'].value_counts().to_dict() if not hf_df.empty else {}
    
    spacy_set = set(zip(spacy_df['text_id'], spacy_df['entity_text'])) if not spacy_df.empty else set()
    hf_set = set(zip(hf_df['text_id'], hf_df['entity_text'])) if not hf_df.empty else set()
    
    both = spacy_set.intersection(hf_set)
    spacy_only = spacy_set - hf_set
    hf_only = hf_set - spacy_set
    
    return {
        'spacy_counts': spacy_counts,
        'hf_counts': hf_counts,
        'total_spacy': len(spacy_df),
        'total_hf': len(hf_df),
        'both': both,
        'spacy_only': spacy_only,
        'hf_only': hf_only
    }


def evaluate_ner(predicted_df, gold_df):
    """Evaluate NER predictions against gold-standard annotations.

    Computes entity-level precision, recall, and F1. An entity is a
    true positive if both the entity text and label match a gold entry
    for the same text_id.

    Args:
        predicted_df: DataFrame with columns text_id, entity_text,
                      entity_label.
        gold_df: DataFrame with columns text_id, entity_text,
                 entity_label.

    Returns:
        Dictionary with keys: 'precision', 'recall', 'f1' (floats 0-1).
    """
    if predicted_df.empty or gold_df.empty:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
    gold_text_ids = set(gold_df['text_id'])
    filtered_preds = predicted_df[predicted_df['text_id'].isin(gold_text_ids)]
        
    pred_set = set(zip(filtered_preds['text_id'], filtered_preds['entity_text'], filtered_preds['entity_label']))
    gold_set = set(zip(gold_df['text_id'], gold_df['entity_text'], gold_df['entity_label']))
    
    true_positives = len(pred_set.intersection(gold_set))
    precision = true_positives / len(pred_set) if len(pred_set) > 0 else 0.0
    recall = true_positives / len(gold_set) if len(gold_set) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {'precision': float(precision), 'recall': float(recall), 'f1': float(f1)}


if __name__ == "__main__":

    nlp = spacy.load("en_core_web_sm")
    hf_ner = hf_pipeline("ner", model="dslim/bert-base-NER")


    df = load_data()
    if df is not None:
        summary = explore_data(df)
        if summary is not None:
            print(f"Shape: {summary['shape']}")
            print(f"Languages: {summary['lang_counts']}")
            print(f"Categories: {summary['category_counts']}")
            print(f"Text length (words): {summary['text_length_stats']}")


        sample_row = df[df["language"] == "en"].iloc[0]
        sample_tokens = preprocess_text(sample_row["text"], nlp)
        if sample_tokens is not None:
            print(f"\nSample preprocessed tokens: {sample_tokens[:10]}")


        spacy_entities = extract_spacy_entities(df, nlp)
        if spacy_entities is not None:
            print(f"\nspaCy entities: {len(spacy_entities)} total")

        hf_entities = extract_hf_entities(df, hf_ner)
        if hf_entities is not None:
            print(f"HF entities: {len(hf_entities)} total")

        if spacy_entities is not None and hf_entities is not None:
            comparison = compare_ner_outputs(spacy_entities, hf_entities)
            if comparison is not None:
                print(f"\nBoth systems agreed on {len(comparison['both'])} entities")
                print(f"spaCy-only: {len(comparison['spacy_only'])}")
                print(f"HF-only: {len(comparison['hf_only'])}")

        gold = pd.read_csv("data/gold_entities.csv")
        if spacy_entities is not None:
            metrics_spacy = evaluate_ner(spacy_entities, gold)
            if metrics_spacy is not None:
                print(f"\nspaCy evaluation: {metrics_spacy}")
        
        if hf_entities is not None:
            metrics_hf = evaluate_ner(hf_entities, gold)
            if metrics_hf is not None:
                print(f"HF evaluation: {metrics_hf}")
