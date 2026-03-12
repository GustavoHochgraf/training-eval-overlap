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

# 3. Run the contamination pipeline
python scripts/run_pipeline.py --config configs/local.yaml

# 4. Generate report tables & figures
python scripts/generate_report.py --results results/
```

## Methodology

| Step | Description | Module |
|------|-------------|--------|
| 1. Extraction | Parse every PoetaV2 task into `(id, question, context, answer)` tuples | `src/contamination/extraction.py` |
| 2. Normalization | Unicode NFKC, lowercasing, whitespace collapse, punctuation removal | `src/contamination/normalization.py` |
| 3. N-gram indexing | Build n-gram (n = 8, 13) inverted index over the Carolina corpus | `src/contamination/indexing.py` |
| 4. Matching | Exact substring, n-gram overlap (Jaccard), and BM25 retrieval | `src/contamination/matching.py` |
| 5. Scoring | Per-instance contamination label with confidence; aggregate statistics | `src/contamination/scoring.py` |
| 6. Reporting | Tables, plots, and LaTeX snippets for the paper | `src/contamination/reporting.py` |

### Contamination Definitions

Following Oren et al. (2024) *"Proving Test Set Contamination in Black Box Language Models"* and the Llama-3 contamination protocol:

- **Exact contamination**: an evaluation instance appears verbatim (after normalization) in the training corpus.
- **Near contamination**: ≥ 70% n-gram overlap (n = 8) between the evaluation instance and any contiguous passage in the training corpus.
- **Partial contamination**: BM25 top-1 retrieval score exceeds a calibrated threshold, indicating high lexical similarity without verbatim overlap.

## References

- Oren, Y. et al. (2024). *Proving Test Set Contamination in Black Box Language Models.* ICLR 2024.
- OpenAI (2023). *GPT-4 Technical Report*, Appendix C — Contamination analysis.
- Groeneveld, D. et al. (2024). *OLMo: Accelerating the Science of Language Models.* ACL 2024.
- Dodge, J. et al. (2021). *Documenting Large Webtext Corpora: A Case Study on the Colossal Clean Crawled Corpus.* EMNLP 2021.
- Carolina Corpus: https://sites.usp.br/corpuscarolina/

## License

MIT
