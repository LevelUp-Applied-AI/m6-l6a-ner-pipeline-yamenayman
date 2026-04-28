# Custom NER Analysis

## How the Custom Rules Changed the Entity Landscape

Implementing the `EntityRuler` significantly improved the extraction of domain-specific climate terminology compared to the base statistical NER. The base model completely missed specialized acronyms and phrasing (e.g., treating "COP28" as just an ORG or ignoring "IPCC AR6" and "1.5 degrees Celsius" entirely). By introducing custom rules, we successfully captured key entities across five custom labels: `CLIMATE_EVENT`, `POLICY`, `REPORT`, `THRESHOLD`, and `AGREEMENT`. 

**Where they helped:** The rules accurately extracted exact phrases like "Paris Agreement", "Carbon Border Adjustment Mechanism", and "Climate Ambition Summit", which are critical context for climate documents but typically slip past generic models. Using token patterns, we also reliably identified variable forms of entities, such as the "1.5°C" threshold.

**Pipeline Positioning:** When the `EntityRuler` was placed **before** the NER model, it prioritized our custom labels over the base model's predictions, ensuring that "Paris Agreement" was classified as a `POLICY` rather than a generic `LAW` or `ORG`. When placed **after**, the base model's matches took priority, which sometimes overwrote our more specific domain labels with less informative standard labels. 

**Where they introduced noise (False Positives):** Custom rules, especially overly broad token patterns, can introduce noise. For example, generic patterns like `{"LOWER": "loss"}, {"LOWER": "and"}, {"LOWER": "damage"}` could potentially flag casual usage of the words "loss and damage" rather than the formal UN financial mechanism. This highlights the tradeoff in production NER: rule-based systems offer high precision for known vocabulary but can be brittle and require careful constraint to avoid unintended matches in broader text.
