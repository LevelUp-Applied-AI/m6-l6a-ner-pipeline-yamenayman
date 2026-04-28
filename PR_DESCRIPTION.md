
| Entity Type   | spaCy | Hugging Face |
|---------------|------:|-------------:|
| **ORG**       |   184 |          173 |
| **LOC**       |    93 |          283 |
| **GPE**       |   165 |            — |
| **DATE**      |   256 |            — |
| **CARDINAL**  |   138 |            — |
| **PERCENT**   |   103 |            — |
| **QUANTITY**  |    92 |            — |
| **MONEY**     |    63 |            — |
| **PERSON**    |    36 |            — |
| **MISC**      |     — |          103 |
| **PER**       |     — |           15 |
| **NORP**      |    21 |            — |
| **ORDINAL**   |     9 |            — |
| **FAC**       |     9 |            — |
| **EVENT**     |     8 |            — |
| **TIME**      |     8 |            — |
| **WORK_OF_ART**|    6 |            — |
| **PRODUCT**   |     6 |            — |
| **LAW**       |     5 |            — |
| **Total**     | **1202** |      **574** |



## Precision, Recall, and F1 — Gold-Annotated Subset


| System            | Precision | Recall | F1    |
|-------------------|----------:|-------:|------:|
| **spaCy**         |   0.629   | 0.647  | **0.638** |
| **Hugging Face**  |   0.341   | 0.221  | 0.268 |





**spaCy** significantly outperformed Hugging Face on this dataset, achieving an F1 of **0.638** compared to HF's **0.268**. spaCy's strength lies in its diverse label schema — it detected 17 distinct entity types (DATE, CARDINAL, PERCENT, MONEY, etc.) that are highly relevant to scientific and climate texts. **Hugging Face** (`dslim/bert-base-NER`) is restricted to only 4 types: `ORG`, `LOC`, `PER`, and `MISC`, which caused systematic recall failures (0.221) — it simply cannot recognize DATE, GPE, or QUANTITY entities that dominate climate texts. Both systems agreed on **308 entities** out of the full corpus, with HF contributing 241 unique finds versus spaCy's 880, suggesting a hybrid approach could improve overall coverage. For production use on climate or scientific corpora, a domain-adapted transformer model (e.g., fine-tuned on scientific NER datasets) would likely outperform the general-purpose `dslim/bert-base-NER` used here.
