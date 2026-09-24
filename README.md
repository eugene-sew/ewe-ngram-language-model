# Ewe N-Gram Language Model

A word-level n-gram language model for **Ewe** (a West African language) built from scratch in Python, with no NLP libraries: only `pandas`, `matplotlib` and the standard library. It learns from about 47,000 Ewe sentences and can score how natural a sentence looks (perplexity) and generate new Ewe-looking text. The models cover orders 1 to 5 (unigram to 5-gram).

The notebook is written to be read: every step is explained, and a separate guide defines every term and walks through the reasoning, so a newcomer can follow it and reuse it as a basis for a technical report.

> **Read this before quoting any number.** The tuning grid for the interpolation weights was too narrow: for the trigram, 4-gram and 5-gram the chosen weight was 0.5, the lowest value searched. On the dev set, widening the grid changes the picture substantially (table below), so the conclusion that 4-/5-grams are much worse than the trigram is largely an artifact. See [`EWE_NGRAM_GUIDE.md`](EWE_NGRAM_GUIDE.md) Section 8.1 and run `wide_grid_check.py` to reproduce it.

## Results

Test-set perplexity from `ewe_ngram_lm.ipynb` as it stands (lower is better; the test set is scored once, after tuning on a separate dev set):

| Model | Context | Test perplexity |
|---|---|---|
| Unigram | none | 595.9 |
| Bigram | 1 word | 167.3 |
| Trigram | 2 words | **139.2** |
| 4-gram | 3 words | 184.6 |
| 5-gram | 4 words | 299.6 |

Effect of the tuning grid, **dev set only** (from `wide_grid_check.py`):

| Order | Notebook grid (0.5 to 0.95) | Wider grid (0.01 to 0.95) |
|---|---|---|
| Trigram | 140.9 | 134.5 |
| 4-gram | 188.4 | 132.0 |
| 5-gram | 307.2 | 131.9 |

With a fitting grid, longer context helps slightly and then plateaus. Re-tune before reporting final numbers.

## What is in this repository

```
ewe_ngram_lm.ipynb        The notebook (saved with outputs)
EWE_NGRAM_GUIDE.md        Guided journal: concepts, glossary, walkthrough, debugging story, limitations,
                          and how to write a technical report from this work
wide_grid_check.py        Reproduces the tuning-grid finding (dev set only, about a minute)
data/EWE_ENGLISH.csv      Ewe-English sentence pairs (only the Ewe side is used)
data/waxal_transcriptions.xlsx   Ewe transcriptions
requirements.txt
```

## Quick start

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Open `ewe_ngram_lm.ipynb` in VS Code or Jupyter (install `jupyterlab` if you use it), choose **Restart kernel and Run All**. It takes about half a minute. No internet connection is needed: all data is in `data/`.

```bash
python wide_grid_check.py    # optional: the tuning-grid check
```

Tested with Python 3.14, pandas 3.0.5, matplotlib 3.11.2, openpyxl 3.1.5.

## The approach in short

1. **Clean** the text: lowercase, keep Ewe letters (`ɔ ɛ ŋ ɖ ƒ`), keep `. , ! ?` as tokens, drop digits and other symbols, add `<s>` and `</s>` sentence markers.
2. **Split** the shuffled sentences 80 / 10 / 10 into train / dev / test (seed 42): 37,704 / 4,713 / 4,714.
3. **Count** all 1- to 5-grams in the training set.
4. **Smooth** with add-k at the unigram level and recursive linear interpolation for higher orders, so a context never seen in training falls back gracefully to a shorter one.
5. **Tune** `k` and one interpolation weight per order on the dev set with a staged grid search.
6. **Evaluate** perplexity on the test set once, and **generate** text by sampling.

## Data

| Source | Rows |
|---|---|
| `data/EWE_ENGLISH.csv` | 28,614 |
| `data/waxal_transcriptions.xlsx` (described in the notebook as University of Ghana, 2023) | 19,151 |

**Provenance and licence are not yet documented.** I have not verified the origin, licence or redistribution terms of the two data files in this repository, which is why the repository is private. Confirm them before making it public or citing the data. The corpus also appears to contain a large share of religious text (about a fifth of lines start with what looks like a Bible verse number), so results may not generalise to everyday Ewe.

## Known limitations

Details and evidence are in Section 8 of the guide.

- **Tuning grid too narrow** (above).
- **Tokenizer weaknesses:** Python's `re` does not treat combining marks as word characters, so words with nasal vowels like `ɔ̃` can be split (about 1% of words); capital `Ɖ` is often typed as `Ð`, which lowercases to a different character than `ɖ`; digits are deleted.
- **Data quality:** 39 lines of corrupted junk text (0.08% of the data).
- **Evaluation:** one split for the reported numbers (re-splitting with five seeds moved the trigram test perplexity by about 1.2%, one standard deviation, so differences of a few percent can be noise); perplexity only; text generation is unseeded and has not been rated by a native speaker.
- **Unigram sampler quirk:** it can emit the start marker `<s>` as if it were a word, because `<s>` is counted like any other token.
- **Small demo cell:** the last cell of Part 11 (comparing two word orders) does not support its own caption; do not treat it as a result.

## Status and next steps

- [x] Remove the 200 online Bible sentences so the data is fully local (test perplexity 2 to 3% lower; mostly within split noise, see guide Section 6, item 9)
- [ ] Widen the lambda grid, re-run, and update the results
- [ ] Fix the tokenizer (combining marks, `Ð`) and decide about the junk lines
- [ ] Seed the text generation
- [ ] Have a native speaker rate a sample of generated sentences
- [ ] Try Kneser-Ney smoothing
- [ ] Document data provenance and add a licence

## Licence

No licence has been chosen yet. The data files may carry their own terms.
