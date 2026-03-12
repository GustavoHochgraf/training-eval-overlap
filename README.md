# Training–Evaluation Overlap: Carolina Corpus × PoetaV2 Benchmarks

Quantifying the overlap between the **Carolina corpus** (pre-training data for a Portuguese LLM based on Qwen) and the evaluation instances used by the **PoetaV2** benchmark framework. The goal is to determine whether — and to what extent — benchmark questions, passages, or answer strings appear verbatim or near-verbatim in the training corpus, which would inflate reported evaluation scores.

## Motivation

Large language models trained on web-scale corpora risk *data contamination*: evaluation examples that leak into the training set produce artificially high benchmark scores and undermine the validity of reported results. This repository implements a systematic, reproducible pipeline to:

1. **Extract** every evaluation instance (question, context, answer) from each PoetaV2 task.
2. **Search** for exact and near-duplicate matches inside the Carolina corpus.
3. **Quantify** contamination rates per task, per split, and overall.
4. **Report** results with statistical confidence intervals following best practices from recent contamination audits (e.g., GPT-4 Technical Report §Appendix C, Llama-3 contamination analysis, Oren et al. 2024).

## Repository Structure

```
├── configs/              # Experiment configuration files (YAML)
├── data/
│   ├── raw/              # Unmodified source data (gitignored)
│   ├── processed/        # Cleaned & tokenized artifacts
│   └── external/         # Third-party reference datasets
├── notebooks/            # Exploratory & reporting notebooks
├── results/
│   ├── tables/           # LaTeX / CSV result tables
│   └── figures/          # Publication-ready plots
├── scripts/              # Entry-point CLI scripts
├── src/contamination/    # Core library (installable package)
├── tests/                # Unit & integration tests
├── pyproject.toml
└── README.md
```

## Quick Start

```bash
# 1. Clone & install
git clone <repo-url> && cd training-eval-overlap
pip install -e ".[dev]"

# 2. Configure paths to Carolina + PoetaV2 data
cp configs/default.yaml configs/local.yaml
# edit configs/local.yaml with your paths

# 3. Run semantic overlap detection (primary method)
python scripts/run_semantic_search.py --config configs/local.yaml

# 4. (Optional) Run n-gram overlap detection
python scripts/run_pipeline.py --config configs/local.yaml
```

## Methodology

Two complementary detection strategies are used:

### A. Semantic Search (primary)

Uses **BAAI/bge-m3** (state-of-the-art free multilingual embedding model) + **FAISS** for efficient nearest-neighbor retrieval.

| Step | Description | Module |
|------|-------------|--------|
| 1. Extraction | Parse every PoetaV2 task into `(id, question, context, answer)` tuples | `src/contamination/extraction.py` |
| 2. Embedding | Encode all Carolina docs and PoetaV2 instances with bge-m3 | `src/contamination/embeddings.py` |
| 3. Retrieval | FAISS cosine similarity search — top-k nearest neighbors per instance | `src/contamination/embeddings.py` |
| 4. Classification | Flag instances above similarity threshold as overlapping | `scripts/run_semantic_search.py` |

### B. N-gram Overlap (complementary)

| Step | Description | Module |
|------|-------------|--------|
| 1. Normalization | Unicode NFKC, lowercasing, whitespace collapse | `src/contamination/normalization.py` |
| 2. N-gram indexing | Build n-gram (n = 8, 13) inverted index over Carolina | `src/contamination/indexing.py` |
| 3. Matching | Exact substring + Jaccard overlap detection | `src/contamination/matching.py` |
| 4. Scoring | Per-instance labels with bootstrap confidence intervals | `src/contamination/scoring.py` |

### Overlap Definitions

- **Exact overlap**: an evaluation instance appears verbatim (after normalization) in the training corpus.
- **Near overlap**: ≥ 70% n-gram overlap (n = 8) between the evaluation instance and any contiguous passage in the training corpus.
- **Semantic overlap**: cosine similarity ≥ 0.85 between the embedding of an evaluation instance and any training passage (captures paraphrases and reformulations).

## References

- Oren, Y. et al. (2024). *Proving Test Set Contamination in Black Box Language Models.* ICLR 2024.
- OpenAI (2023). *GPT-4 Technical Report*, Appendix C — Contamination analysis.
- Groeneveld, D. et al. (2024). *OLMo: Accelerating the Science of Language Models.* ACL 2024.
- Dodge, J. et al. (2021). *Documenting Large Webtext Corpora: A Case Study on the Colossal Clean Crawled Corpus.* EMNLP 2021.
- Carolina Corpus: https://sites.usp.br/corpuscarolina/

## License

MIT
