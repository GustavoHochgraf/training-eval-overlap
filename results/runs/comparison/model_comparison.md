# Semantic Run Comparison

## Overall

| Run | Model | Overlaps | Total | Rate (%) |
| --- | --- | ---: | ---: | ---: |
| Baseline | `BAAI/bge-m3` | 0 | 11409 | 0.00 |
| Candidate | `intfloat/multilingual-e5-large-instruct` | 3527 | 11409 | 30.91 |

## Agreement

- Overlaps found by both runs: 0
- Overlaps found only by baseline: 0
- Overlaps found only by candidate: 3527

## Tasks With Largest Rate Changes

| Task | Baseline Rate (%) | Candidate Rate (%) | Delta (pp) |
| --- | ---: | ---: | ---: |
| `broverbs_mc_greedy` | 0.00 | 88.89 | 88.89 |
| `tweetsentbr_greedy` | 0.00 | 77.00 | 77.00 |
| `storycloze_pt_greedy` | 0.00 | 74.40 | 74.40 |
| `bigbench_pt_causal_judgment_greedy` | 0.00 | 73.16 | 73.16 |
| `bigbench_pt_analogical_similarity_greedy` | 0.00 | 72.33 | 72.33 |
| `mina_br_greedy` | 0.00 | 72.00 | 72.00 |
| `wsc285_pt_greedy` | 0.00 | 69.47 | 69.47 |
| `bigbench_pt_social_iqa_greedy` | 0.00 | 58.00 | 58.00 |
| `bigbench_pt_bbq_greedy` | 0.00 | 55.00 | 55.00 |
| `pt_hate_speech_greedy` | 0.00 | 54.80 | 54.80 |