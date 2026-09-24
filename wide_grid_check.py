"""Re-runs the notebook's staged hyperparameter tuning with a wider lambda grid (DEV SET ONLY).

Why: in ewe_ngram_lm.ipynb the chosen interpolation weights for the trigram, 4-gram and 5-gram were all
0.5, the lowest value in the search grid. A best value on the edge of the range suggests the range is too
narrow. This script repeats the tuning with both grids and prints dev perplexity side by side.
It reuses the notebook's tokenizer, split (seed 42) and model, and never scores the test set.

Usage:  python wide_grid_check.py        (about a minute)
"""
import math
import random
import re
from collections import Counter

import pandas as pd

START, END, MAX_N = '<s>', '</s>', 5
NOTEBOOK_LAMBDAS = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
WIDE_LAMBDAS = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
K_GRID = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]


def load_raw():
    texts = list(pd.read_csv('data/EWE_ENGLISH.csv', index_col=0).dropna(subset=['EWE'])['EWE'].astype(str))
    x = pd.read_excel('data/waxal_transcriptions.xlsx', usecols=['Transcription'])['Transcription']
    x = x.dropna().str.strip().str.strip('\xa0')
    return texts + list(x[x.str.len() > 0])


def tokenize(text):
    """Same as preprocess_ewe in the notebook (without the <s> </s> wrapping)."""
    text = re.sub(r"([.,!?])", r" \1 ", text.lower())
    text = re.sub(r"[^\w\s.,!?]", " ", text)
    return re.sub(r"\s+", " ", re.sub(r"\d+", "", text)).strip().split()


class NgramLM:
    def __init__(self, train):
        self.counts = {1: Counter(t for s in train for t in s)}
        for n in range(2, MAX_N + 1):
            self.counts[n] = Counter(g for s in train for g in zip(*[s[i:] for i in range(n)]))
        self.total = sum(self.counts[1].values())
        self.vocab = set(self.counts[1]) | {'<UNK>'}
        self.k = 0.1
        self.lam = {2: 0.75, 3: 0.7, 4: 0.7, 5: 0.7}

    def p(self, ctx, w, n):
        if n == 1:
            return (self.counts[1].get(w, 0) + self.k) / (self.total + self.k * len(self.vocab))
        cc = self.counts[1].get(ctx[0], 0) if n == 2 else self.counts[n - 1].get(ctx, 0)
        mle = self.counts[n].get(ctx + (w,), 0) / cc if cc else 0
        return self.lam[n] * mle + (1 - self.lam[n]) * self.p(ctx[1:], w, n - 1)

    def perplexity(self, sents, n):
        nll, count = 0.0, 0
        for s in sents:
            s = [t if t in self.vocab else '<UNK>' for t in s]
            for i in range(1, len(s)):
                cl = min(n - 1, i)
                nll -= math.log(self.p(tuple(s[i - cl:i]), s[i], cl + 1))
                count += 1
        return math.exp(nll / count)

    def tune(self, dev, lambdas):
        """Staged grid search, exactly as in the notebook: k first, then lambda for n = 2..5 in order."""
        best_pp = {}
        pp, k = min((self._with(lambda: setattr(self, 'k', g), self.perplexity, dev, 1), g) for g in K_GRID)
        self.k, best_pp[1] = k, pp
        for n in range(2, MAX_N + 1):
            pp, lam = min((self._with(lambda: self.lam.__setitem__(n, g), self.perplexity, dev, n), g) for g in lambdas)
            self.lam[n], best_pp[n] = lam, pp
        return best_pp

    @staticmethod
    def _with(setter, fn, *args):
        setter()
        return fn(*args)


if __name__ == '__main__':
    raw = [s for s in load_raw() if tokenize(s)]
    random.seed(42)
    random.shuffle(raw)
    a, b = int(0.8 * len(raw)), int(0.9 * len(raw))
    wrap = lambda sents: [[START] + tokenize(s) + [END] for s in sents]
    train, dev = wrap(raw[:a]), wrap(raw[a:b])           # the test slice (raw[b:]) is never used here
    print(f"{len(raw):,} sentences: train {len(train):,} / dev {len(dev):,}   (test not touched)\n")

    results = {}
    for name, grid in [('notebook grid', NOTEBOOK_LAMBDAS), ('wide grid', WIDE_LAMBDAS)]:
        lm = NgramLM(train)
        pps = lm.tune(dev, grid)
        results[name] = (lm.k, dict(lm.lam), pps)

    print(f"{'order':<8}{'notebook grid':>26}{'wide grid':>26}")
    print(f"{'':<8}{'lambda  dev perplexity':>26}{'lambda  dev perplexity':>26}")
    for n in range(2, MAX_N + 1):
        row = [f"{results[g][1][n]:>10}  {results[g][2][n]:>12.1f}" for g in ('notebook grid', 'wide grid')]
        print(f"{n:<8}{row[0]:>26}{row[1]:>26}")
    print(f"\nk chosen: notebook grid {results['notebook grid'][0]}, wide grid {results['wide grid'][0]} "
          f"(unigram dev perplexity {results['wide grid'][2][1]:.1f})")
    print("If the best lambda is the smallest value in its grid, that grid is probably still too narrow.")
