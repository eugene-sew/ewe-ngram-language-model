# Building a Language Model for Ewe, From Scratch
### A guided journal for beginners, and a starting point for your technical report

This guide explains, step by step, what was built in `ewe_ngram_lm.ipynb`, why each decision was made, what went wrong along the way, and how to turn all of it into a report you can submit. It assumes **no background in machine learning or linguistics**. Every technical word is defined the first time it appears and again in the glossary (Section 3).

## 0. How to use this guide

| If you want to... | Read |
|---|---|
| Understand the project from zero | Sections 1, 2, 3, 4 in order |
| Retrace exactly what was done | Section 5 (with the notebook open next to it), then Section 6 |
| Write your report | Sections 7, 8, 9, and the checklist at the end of 9 |
| Test your own understanding | Section 10 |

**Conventions.** **Bold** marks a term being defined. Numbers come from a real run of the notebook (random seed 42, so they reproduce exactly). Anything I could not verify is marked *[verify]*. Please check those before putting them in a report.

**Status of the numbers.** Section 8.1 describes a tuning problem found while writing this guide. It means the headline numbers in Sections 5 and 7 are correct for the notebook *as it currently is*, but should be re-computed before you publish them. Read that section before you write your report.

---

## 1. The project on one page

**The goal.** Teach a computer to judge how "natural" a sentence of Ewe looks, and to write new Ewe-looking text, using only counting and arithmetic on a collection of real Ewe sentences.

**The idea.** A **language model** is a program that assigns a probability to sequences of words. If it has read enough real text, it learns which words tend to follow which. After the words `ame aɖewo`, the word `le` is likely and a random rare word is not. A model that has learned this can (a) score a sentence, and (b) generate one by repeatedly picking a likely next word.

**The method.** We used the oldest and simplest family of language models, **n-gram models**. They just count how often short word sequences occur in the training text, then turn counts into probabilities. We built five of them, from "no memory" (looks at one word alone) to "remembers the previous four words".

**Why this method for Ewe.** Ewe is a **low-resource language**: there is far less digital text than for English. Modern neural language models need huge amounts of text. Counting-based models work on small data, run in seconds on a laptop, and every number in them can be inspected, which makes them good for learning and for a first baseline.

**What the data looks like.** 47,965 lines of Ewe from three sources, about 1.5 million tokens after cleaning (Section 5, Part 2).

**What we found** (details and caveats in Sections 7 and 8):

| Model (how much context) | Test perplexity (lower is better) |
|---|---|
| Unigram (none) | 601.6 |
| Bigram (1 previous word) | 171.1 |
| **Trigram (2 previous words)** | **142.8** |
| 4-gram (3 previous words) | 190.0 |
| 5-gram (4 previous words) | 308.9 |

In plain words: giving the model one or two words of memory makes it much less "surprised" by real Ewe (601 down to 143). The notebook's tuning then made longer memory look worse, but Section 8.1 shows that is partly a flaw in how the settings were searched, not a fact about Ewe.

**Three lessons the project taught** (each explained later):
1. A model can look excellent and be broken. A "gibberish beats real Ewe" bug was caught only because we tested a nonsense sentence by hand (Section 6).
2. The order in which you clean, split and tune data decides whether your final number is honest (Section 5, Part 4).
3. Numbers from a search over settings must be checked for hitting the edge of the search range (Section 8.1).

---

## 2. Background

### 2.1 The Ewe language

Ewe (endonym *Eʋegbe*) is a language of West Africa in the Gbe group, which belongs to the Niger-Congo family. It is spoken mainly in the Volta Region of Ghana and in southern Togo. **Speaker numbers:** the notebook says "about 7 million people, mainly in Ghana, Togo and Benin". Common reference summaries I checked give roughly 5 to 5.5 million speakers, mainly in Ghana and Togo, so the notebook's figure and its mention of Benin should be checked against a source you can cite *[verify, and consider correcting Part 1 of the notebook]*.

Ewe is written in the Latin alphabet with extra letters. These matter for the project because the text cleaning must not destroy them:

| Letter | Approximate sound (as described in the notebook) |
|---|---|
| ɔ | open "o", as in "off" |
| ɛ | open "e", as in "bed" |
| ŋ | "ng", as in "sing" |
| ɖ | retroflex "d" |
| ƒ | bilabial "f" |

Ewe is also a tone language and uses accents and a tilde on vowels (for example the nasal vowel ɔ̃). Some of these are written as a base letter plus a separate **combining mark** in the file (see the glossary). This turned out to be a source of trouble (Section 8.3).

To give a feel for the vocabulary, here are the most frequent words in the corpus with approximate meanings from general knowledge *[verify with a speaker or dictionary before quoting]*: `le` (to be at/in), `eye` (and), `ne` (if/when), `ame` (person).

### 2.2 Why "low-resource" matters

A **low-resource language** has little written, digitised, cleaned text. For our purposes this has two consequences. First, our corpus is small (1.2 million training tokens; English models are trained on billions). Second, most words are rare: about half of all the different words in our data occur **exactly once**. Rare words are hard for any model, and handling them is the main technical challenge of this project.

### 2.3 The data

| Source | Rows | Notes |
|---|---|---|
| `EWE_ENGLISH.csv` | 28,614 | Ewe sentences paired with English translations. *[Record where you obtained it and its licence.]* |
| HuggingFace `ghananlpcommunity/ewe-bible-tts-200` | 200 | Ewe Bible sentences (text only). |
| Waxal image-caption / transcription spreadsheet | 19,151 | Ewe transcriptions, described in the notebook as University of Ghana (2023). *[Verify the source and cite it properly.]* |
| **Total** | **47,965** | |

Only the Ewe side is used. About a fifth of the lines (10,236 of 47,765 checked) begin with a number that looks like a Bible verse number, and many sentences mention names like Yehowa and Yesu Kristo, so **the corpus appears to contain a large share of religious text**. Results may not carry over to everyday conversation. This is a limitation to state in a report (Section 8.2).

---

## 3. Glossary

Read this once now; come back whenever a term is unfamiliar.

### Text and data
- **Corpus**: the collection of text you learn from. Ours is the 47,965 Ewe lines.
- **Token**: one unit of text after splitting. Here, usually a word, but punctuation marks like `,` and `.` are also tokens.
- **Word type vs. word token**: "the cat saw the dog" has 5 *tokens* but 4 *types* (`the` appears twice). **Vocabulary** = the set of types.
- **Tokenization / tokenizer**: the rules that split raw text into tokens. Our tokenizer is the function `preprocess_ewe`.
- **Normalization**: cleaning that makes equal things look equal, such as lowercasing so `Ame` and `ame` count as the same word.
- **Special tokens**: markers added by us. `<s>` = start of sentence, `</s>` = end of sentence, `<UNK>` = "unknown word". The start/end markers let the model learn how sentences begin and finish.
- **Unicode**: the standard that assigns a number to every character in every writing system.
- **Combining mark**: a character (like a tilde or accent) stored *separately* and drawn on top of the letter before it. `ɔ̃` is two characters: `ɔ` plus a combining tilde.
- **Hapax legomenon** (plural *hapax legomena*): a word that occurs exactly once in the corpus. In our data 21,811 of 42,329 word types (51.5%) are hapaxes.
- **Out-of-vocabulary (OOV) word**: a word in new text that never appeared in training.

### Counting and probability
- **n-gram**: a run of n consecutive tokens. **Unigram** = 1, **bigram** = 2, **trigram** = 3, then 4-gram and 5-gram. The sentence `ame le afi` contains the bigrams `ame le` and `le afi`.
- **Context** (or **history**): the words before the one we are predicting. In a trigram model the context is the previous two words.
- **Language model**: something that gives P(next word | context), the probability of the next word given the context, and therefore a probability for whole sentences.
- **Conditional probability**: the chance of one thing *given* that another already happened. P(cat | the) is the chance that `cat` comes next, given that the previous word was `the`.
- **Chain rule**: the probability of a whole sentence is the product of each word's probability given everything before it.
- **Markov assumption**: the simplification that only the last few words matter. An n-gram model assumes only the previous n-1 words matter.
- **Maximum likelihood estimate (MLE)**: the plain "count and divide" estimate: P(w | context) = count(context + w) / count(context). It is the best fit to the training data, and that is exactly its weakness (see *sparsity*).
- **Log probability**: the natural logarithm of a probability. Probabilities of long sentences are tiny numbers that a computer can no longer represent accurately, so we add logs instead of multiplying probabilities. Log probabilities are negative; closer to zero means more probable.

### Sparsity and smoothing
- **Sparsity** (sparse data): most possible word sequences never occur in any finite corpus. The longer the sequence, the more true this is. This is the central problem of n-gram models.
- **Zero-probability problem**: an MLE gives probability 0 to anything unseen, so one unseen pair makes a whole sentence "impossible".
- **Smoothing**: any method that moves a little probability from seen events to unseen ones so nothing is exactly zero.
- **Add-k smoothing** (**Laplace smoothing** when k = 1): pretend every possible word was seen k extra times.
- **Interpolation**: mix the estimate from a long context with estimates from shorter contexts, using weights.
- **Lambda (λ)**: our name for an interpolation weight, between 0 and 1. λ = 0.7 means "70% trust the longer context, 30% trust the shorter one".
- **Backoff**: use the shorter context only when the longer one has no data. (We use interpolation, which always mixes.)

### Evaluation
- **Perplexity**: the standard score for a language model. Lower is better. Intuitively, it is how many equally likely words the model is effectively choosing between at each step. Perplexity 143 means "about as uncertain as picking from 143 equally likely words". Formula in Section 4.
- **Intrinsic vs. extrinsic evaluation**: intrinsic = scoring the model itself (perplexity). Extrinsic = measuring how much it improves a real task such as speech recognition. We do only intrinsic.
- **Training / development (dev) / test sets**: three separate slices of the data. *Training* is where the model learns counts. *Dev* is where we choose settings. *Test* is used once at the end for the honest score.
- **Hyperparameter**: a setting chosen by the person, not learned from data. Ours: `k` and the λ values.
- **Grid search**: try every value on a fixed list and keep the best one.
- **Overfitting**: doing well on data you tuned or trained on but worse on new data. Tuning on the test set causes it.
- **Random seed**: a number that fixes the "random" shuffle so anyone re-running gets the same result.
- **Sampling** and **temperature**: to generate text we *sample* the next word at random, weighted by probability. Temperature reshapes those weights: low temperature = safe and repetitive, high = adventurous and messier.
- **Baseline**: a simple model you compare fancier ones against.

---

## 4. The core ideas, on a tiny example

We use English on purpose here so nothing is hidden by unfamiliar words. Every number below was computed by code.

**Toy corpus** (3 sentences, with markers added):

```
<s> the cat sat </s>
<s> the cat ran </s>
<s> the dog sat </s>
```

That is 15 tokens and 7 word types.

### 4.1 Counting and probabilities

Counts of words: `<s>`=3, `the`=3, `</s>`=3, `cat`=2, `sat`=2, `dog`=1, `ran`=1.
Counts of pairs: `the cat`=2, `the dog`=1, `cat sat`=1, `cat ran`=1, `dog sat`=1, `sat </s>`=2, `ran </s>`=1, `<s> the`=3.

The plain estimate divides pair count by the count of the first word:

- P(cat | the) = 2 / 3 = 0.667
- P(dog | the) = 1 / 3 = 0.333
- P(sat | dog) = 1 / 1 = 1.0

The probability of a sentence is a product. For `the cat sat`:
P(the | `<s>`) × P(cat | the) × P(sat | cat) × P(`</s>` | sat) = 1 × 0.667 × 0.5 × 1 = **0.333**.

This is the **chain rule with the Markov assumption**: each word depends only on the previous one (bigram model).

### 4.2 The zero problem

Now score a sentence we never saw: `the dog ran`. The pair `dog ran` never occurred, so P(ran | dog) = 0 / 1 = 0, and the whole product is **0**. A model that calls a perfectly sensible sentence "impossible" is useless, and its perplexity would be infinite.

### 4.3 First fix: add-k smoothing (and why it can backfire)

Pretend every word was seen `k` extra times. With k = 1 and 7 word types:

P(ran | dog) = (0 + 1) / (1 + 7) = 0.125. No longer zero. 

But look at what happened to a pair we *did* see: P(sat | dog) = (1 + 1) / (1 + 7) = **0.25**, down from 1.0. With so little data behind `dog`, the pretend counts drowned the real one. Real corpora make this far worse, since the vocabulary is tens of thousands of words but most contexts have only one or two observations. This is exactly what broke the first version of our trigram model (Section 6).

### 4.4 Better fix: interpolation

Instead of adding pretend counts everywhere, **mix** the bigram estimate with a smoothed one-word estimate, using λ = 0.7:

P(w | context) = 0.7 × (bigram MLE) + 0.3 × (smoothed unigram)

Here the smoothed unigram is add-1 on single words: P(ran) = (1 + 1) / (15 + 7) = 0.0909 and P(sat) = (2 + 1) / (15 + 7) = 0.1364.

- P(ran | dog) = 0.7 × 0 + 0.3 × 0.0909 = **0.0273**. Small but non-zero.
- P(sat | dog) = 0.7 × 1.0 + 0.3 × 0.1364 = **0.7409**. Still high, because the real evidence keeps most of its weight.

Unseen things get a little probability, seen things keep most of theirs. That is the entire idea behind our final model.

### 4.5 Scoring with perplexity

Take the logs of the word probabilities and average them:

$$\text{PP} = \exp\left(-\frac{1}{N}\sum_{i=1}^{N}\ln P(w_i \mid \text{context}_i)\right)$$

where N is the number of predictions made. Under the interpolated toy model of 4.4, `the cat sat` gets log-probability -2.181 over 4 predictions, so PP = exp(2.181 / 4) = **1.73**. The unseen `the dog ran` gets perplexity **3.97**: worse (higher) but finite, where the raw counts of 4.2 gave it probability zero.

**How to read perplexity:** if a model spreads its guess evenly over 100 words at every step, its perplexity is exactly 100. So perplexity is an "effective number of choices". Lower means the model narrows the choices down better. Caution: it is only comparable between models that use the same vocabulary and the same tokenization (Jurafsky & Martin make this point). We come back to this in Section 8.

### 4.6 Why we add logs

A 20-word sentence with probabilities around 0.001 per word has probability about 10⁻⁶⁰. Computers lose precision long before that. Logs turn the product into a sum: safe, fast, and mathematically equivalent.

---

## 5. The notebook, part by part

Open `ewe_ngram_lm.ipynb` next to this section. Each entry says what the part does, why, and what to look for.

### Part 1: Setup
Imports standard Python tools: `re` (pattern matching for cleaning), `math`, `random`, `collections.Counter` (counting), `pandas` (tables) and `matplotlib` (charts). Nothing here is specific to language modelling. **No NLTK or other NLP library is used**: counting, smoothing and perplexity are written by hand so every step is visible.

### Part 2: Load and explore the data
Loads the three sources (Section 2.3) into one table of 47,965 lines. Sentence length, counted in raw whitespace-separated words: mean 27.2, median 26, standard deviation 18.4, longest 416.

**Why explore first?** Always look at the data before modelling. The long tail (a 416-word "sentence") and the verse numbers we noticed later would both have been visible here.

### Part 3: Preprocessing (the tokenizer)
The function `preprocess_ewe` turns raw text into tokens:

```python
def preprocess_ewe(text):
    text = text.lower()                              # 1. lowercase
    text = re.sub(r"([.,!?])", r" \1 ", text)        # 2. make . , ! ? their own tokens
    text = re.sub(r"[^\w\s.,!?]", " ", text)         # 3. drop all other symbols
    text = re.sub(r"\d+", "", text)                  # 4. drop digits
    text = re.sub(r"\s+", " ", text).strip()         # 5. tidy spaces
    tokens = text.split()
    return ['<s>'] + tokens + ['</s>'] if tokens else []
```

Example from the notebook:

```
Original : Ne nyɔnu aɖe le evi dzim eye wo le kukum nɛ la,
Processed: ['<s>', 'ne', 'nyɔnu', 'aɖe', 'le', 'evi', 'dzim', 'eye', 'wo', 'le', 'kukum', 'nɛ', 'la', ',', '</s>']
```

**Decisions and why:**
- *Keep Ewe letters* (`ɔ ɛ ŋ ɖ ƒ`): they change meaning (`ko` and `kɔ` are different words). Python's `\w` matches these, so they survive.
- *Lowercase*: merges `Ame` and `ame`, which reduces sparsity. Cost: loses capitalisation information.
- *Keep `. , ! ?` as tokens*: so the model can learn where punctuation goes and generated text can punctuate itself. Everything else (quotes, brackets) is dropped as noise.
- *Drop digits*: numbers rarely help predict Ewe words.
- *Add `<s>` and `</s>`*: so the model learns typical sentence starts and ends.

After this step: 47,331 non-empty sentences and 1,531,742 tokens (markers and punctuation included), 42,331 distinct tokens.

**Known weaknesses of this tokenizer** are listed in Section 8.3. In short, it can split words that contain combining marks, and it does not treat `Ð` (capital eth) and `Ɖ` as the same letter.

### Part 4: Train / dev / test split
The sentences are shuffled (seed 42, so it is repeatable) and cut 80 / 10 / 10:

| Set | Sentences | Used for |
|---|---|---|
| Train | 37,864 | Counting n-grams |
| Dev | 4,733 | Choosing hyperparameters |
| Test | 4,734 | One final score |

**Why three sets, not two?** Suppose you try 50 different settings and keep the one with the best score. That best score is optimistic, because you effectively picked the setting that happened to suit that particular data. If you report it as your result, you fooled yourself. So the *dev* set is the sandbox for choosing, and the *test* set stays sealed until every choice is finished. This project touches the test set once.

**Why shuffle?** The files are ordered by source and topic. Without shuffling, "test" might be all Bible verses while "train" is all captions.

### Part 5: Building the n-gram models
For each order n from 1 to 5, count every n-gram in the training sentences. Results:

| n | Distinct n-grams in training | Per 100 token positions |
|---|---|---|
| 1 | 37,454 | 3 |
| 2 | 278,227 | 23 |
| 3 | 647,136 | 53 |
| 4 | 888,590 | 72 |
| 5 | 981,375 | 80 |

(Training has 1,226,363 tokens.)

**This table is the most important picture in the project.** Distinct unigrams are few because words repeat. But nearly every 5-word sequence is one-of-a-kind: 80 different 5-grams for every 100 positions. A model estimating "what follows this exact 4-word context" usually has seen that context zero or one times. That is **sparsity**, and it is why Part 6 exists.

Counts are also stored in a lookup keyed by context, so generation can instantly ask "what words ever followed `ame aɖewo`?" instead of scanning millions of entries.

### Part 6: Smoothing (the heart of the model)
The unigram level uses add-k. Every higher level uses **recursive interpolation**:

$$P_n(w \mid \text{context}) = \lambda_n \cdot P_{\text{MLE}}(w \mid \text{context}) + (1-\lambda_n) \cdot P_{n-1}(w \mid \text{shorter context})$$

Read it as a chain. The 5-gram estimate mixes its own count-based estimate with the 4-gram estimate, which mixes with the trigram estimate, and so on down to a smoothed unigram. Simplified code:

```python
def ngram_prob(context, word, n):
    if n == 1:
        return (count(word) + K) / (total_tokens + K * vocab_size)     # add-k floor
    mle = count(context + (word,)) / count(context)                   # 0 if context never seen
    return LAMBDA[n] * mle + (1 - LAMBDA[n]) * ngram_prob(context[1:], word, n - 1)
```

If the context was never seen (for the 5-gram model, that means a specific 4-word context), its `mle` is 0 and the estimate quietly falls through to the 4-gram model, then the trigram, and so on. **This is what makes higher-order models usable on sparse data.**

**Why recursive, with one λ per order?** The alternative is one flat mix of all orders with weights summing to 1. That needs a search over many weights that grows fast with n. The recursive form needs one number per order, so tuning cost grows only in a straight line.

Words never seen in training are mapped to `<UNK>`. It has no training count, so it receives only the small add-k floor probability. (An earlier version that gave `<UNK>` real training counts caused a serious bug; see Section 6, item 2.)

### Part 7: Scoring sentences
`score_sentence` adds up the log probabilities of every word after `<s>`. The first words of a sentence have less than n-1 words of context, so they use whatever context exists (the very first word only ever has `<s>`, so even the 5-gram model scores it like a bigram).

**The sanity check.** The notebook scores three sentences, dividing by length so they are comparable (average log-probability per word; higher, closer to zero, is better):

| Sentence | Unigram | Bigram | Trigram | 4-gram | 5-gram |
|---|---|---|---|---|---|
| `Ne nyɔnu aɖe le evi dzim` (real, from the data) | -5.93 | -4.50 | -3.96 | -3.55 | -2.79 |
| `le ame tɔ` (short phrase) | -4.25 | -4.99 | -5.61 | -6.21 | -6.51 |
| `zzz bbb qqq` (nonsense) | -13.11 | -14.50 | -15.40 | -16.01 | -16.31 |

Two readings. (1) The real sentence scores far better than nonsense at every order: the model has learned something. (2) The real sentence improves as context grows (-5.93 to -2.79) because it comes from the training data, and the long-context models simply remember it. I checked: every 2-, 3-, 4- and 5-gram of that phrase occurs in the training counts, and its source line is in the training set. That is **memorisation**, and it is why we never judge a model on sentences it was trained on. So this table shows the model has learned *something*, but says nothing yet about new text (Part 8 does). The short phrase `le ame tɔ` gets *worse* with more context: its bigrams were all seen in training, but the trigram `le ame tɔ` was not, so longer contexts have nothing to match.

### Part 7.5: Tuning hyperparameters on the dev set
The settings `k` and `λ₂ … λ₅` are chosen by **grid search** on the dev set, one at a time, lowest order first (so each order's fallback is already tuned):

| Setting | Values tried | Chosen | Dev perplexity |
|---|---|---|---|
| k (unigram smoothing) | 0.01, 0.05, 0.1, 0.3, 0.5, 1.0 | 1.0 | 594.0 |
| λ₂ (bigram) | 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95 | 0.75 | 169.6 |
| λ₃ (trigram) | same list | 0.5 | 143.1 |
| λ₄ (4-gram) | same list | 0.5 | 191.5 |
| λ₅ (5-gram) | same list | 0.5 | 312.3 |

**Look closely at the last three rows.** The chosen value, 0.5, is the *smallest* value on the list. When the best value sits at the edge of the search range, the true best is probably outside it. Section 8.1 shows this is exactly what happened.

### Part 8: Final evaluation on the test set
Now, once, the test set is scored (148,375 predictions):

| Model | Test perplexity |
|---|---|
| Unigram | 601.6 |
| Bigram | 171.1 |
| Trigram | **142.8** |
| 4-gram | 190.0 |
| 5-gram | 308.9 |

The notebook remarks that the trigram is 4.2 times "less confused" than the unigram (601.6 / 142.8). The dev and test numbers are close (for example 143.1 vs 142.8 for the trigram), which is a good sign that the tuning did not overfit the dev set.

### Part 9: Generating Ewe text
To generate: start with `<s>`; look up the candidate next words for the current context; keep the 50 most probable; pick one at random weighted by probability (so likely words are usually, not always, chosen); stop at `</s>` or a length limit.

Real output from one run (**not seeded, so yours will differ**):

| Model | Example |
|---|---|
| Unigram | `be ame me aɖe ke si ale nye . ɖe ,` |
| Bigram | `si le woƒe ʋu gã aɖe hã le afi si do awu` |
| Trigram | `elabena nu kae nèsusu ?` |
| 4-gram | `nyɔnu aɖewo le dɔ wɔm le mɔ dɔwɔƒe aɖe . wonye aɖaŋudɔwɔlawo` |
| 5-gram | `ŋutsu aɖe le dɔ wɔm tso ale yi ke woaza mɔɖaŋu sola` |

Read these as a model would score them: unigram output is word salad (no word knows its neighbour); bigram output is locally plausible but wanders; higher orders read more fluently mostly because they replay chunks of real training sentences. **Whether a sentence is grammatical Ewe cannot be decided by this code, and cannot be decided by someone who does not read Ewe.** For a report, have a speaker rate a sample.

The notebook also shows **temperature**: at 0.5 the bigram model repeats safe common phrases; at 2.0 it produces adventurous, messier text.

### Part 10: What did the model learn?
- Most frequent tokens: `.` (78,874), `le` (70,522), `,` (66,794), `ɖe` (38,804), `la` (30,092), `eye` (29,185). The 100 most common tokens cover 60.6% of all text.
- Words by frequency: 21,811 appear once, 11,801 appear 2 to 5 times, 6,853 appear 6 to 50 times, 1,864 appear more than 50 times. **A long tail of rare words**: this is what "low-resource" looks like in numbers.
- Most likely words after `le`: `wo` (3.3%), `afi` (2.4%), `nu` (2.1%). After `eye`: `ame` (4.4%), `wo` and `wole` (3.0% each). After `ne`: `ame` (4.3%), `.` (2.5%), `ɖe` (2.2%).

### Part 11: Interactive demo
`score_and_explain` prints the log-probability of every word, so you can see *where* a sentence surprises the model. For `Ame le mia si, ame le mia, enkoe nye yesu kristo` (bigram):

```
'<s>'  → 'ame'      -2.97      ','  → 'ame'     -3.37
'ame'  → 'le'       -4.24      ','  → 'enkoe'  -15.44   <-- very surprising
'le'   → 'mia'      -5.93      'enkoe' → 'nye' -6.30
...                            'kristo' → '</s>' -4.71
Total log-probability: -73.02      Perplexity of this sentence: 184.13
```

The single biggest cost is the rare word `enkoe` after a comma (-15.44): the model has almost never seen it. This is the kind of diagnosis a model that is a black box cannot give you.

**A caution about the last demo cell.** It compares `le ame tɔ` with the reordered `tɔ le ame` and says the higher score "sounds more natural". In the current run the *reordered* version scores slightly higher (-19.77 vs -20.01). Nothing here shows the first phrase is natural Ewe (it was chosen as a "simple phrase"), and a difference this small on three words is not evidence of anything. Do not use this cell as a result in a report.

### Part 12: Summary
Recaps what was built and learned.

---

## 6. The journey: problems we hit and how we solved them

This section is the "retrace my steps" part. Real projects are mostly debugging, and each problem below teaches a general habit.

**1. "Adding memory made the model worse."**
*Symptom:* unigram perplexity 794, bigram 690, trigram **4,800**. More context should help, not hurt by 6 times.
*Cause:* add-k smoothing added `k × vocabulary size` (about 0.1 × 40,000 = 4,000) to every denominator. A trigram context usually has only a handful of observations, so the pretend counts overwhelmed the real ones (the toy example in Section 4.3, at scale).
*Fix:* interpolation (Section 4.4). Trigram 4,800 to 225.
*Habit:* when a result contradicts common sense, suspect the code before the theory.

**2. The `<UNK>` trap: "gibberish beats real Ewe".**
*What we tried:* replace every once-only word (51% of the vocabulary) with `<UNK>` so the model learns about unknown words. Perplexity looked wonderful (trigram 150).
*Symptom:* the manual test showed `zzz bbb qqq` scoring *better* than a real Ewe sentence.
*Cause:* all unseen words become `<UNK>`, and `<UNK> <UNK> <UNK>` had become a very common pattern in training (rare words often sit next to other rare words), so gibberish matched it perfectly.
*Fix:* removed the replacement. Unseen words now fall back to the smoothed unigram floor.
*Habit:* **a good metric is not proof of a good model.** Always keep a qualitative test with a known right answer.

**3. Small bugs found by reading the code slowly.**
The unigram generator drew from words in the order they first appeared instead of by frequency; generation scanned hundreds of thousands of n-grams for every word (fixed with a lookup index); and the interactive demo silently printed perplexity 1.00 for models it did not implement.
*Habit:* read your own code as if a stranger wrote it.

**4. A bug from mixing two conventions.**
When the model was extended from 3-grams to 5-grams, bigram perplexity jumped to 1,203. Unigram counts were stored under plain words, all higher orders under tuples of words, and one lookup silently mismatched (always returning zero).
*Habit:* when one variant of a model fails and the others work, compare their data structures first.

**5. Stale and overwritten notebooks.**
Editing a notebook file while it is open in an editor does not update the editor, and saving from the editor can silently restore its older copy. Results shown on screen were from earlier code.
*Habit:* use **Restart kernel and Run All** before trusting any notebook output, and close a notebook before changing the file by other means.

**6. The dev set changes the story.** After splitting off a dev set and tuning, trigram perplexity improved (225 to 190 at that stage of the project), and tuning chose `k = 1.0`, contradicting the notebook's own text that called `k = 1` bad. The text was true for the old smoothing method and false for the new one, so it was rewritten.
*Habit:* text goes stale. Re-read your explanations after you change the code.

**7. Punctuation as tokens.** Keeping `. , ! ?` reduced per-token perplexity a lot at the time (189.5 to 134.9). Later analysis showed most of that was an artifact of counting: adding many easy-to-predict tokens lowers a *per-token* average by itself, without the model knowing more Ewe. Keeping punctuation is still reasonable, but the size of the "improvement" was overstated. This is why Section 8.4 warns against comparing perplexity across tokenizations.

**8. The tuning grid was too narrow.** Found while writing this guide (Section 8.1).

---

## 7. Results and how to interpret them honestly

The table in Section 5, Part 8 is the result. What you can and cannot say:

**You can say:**
- Adding one or two words of context reduces perplexity sharply relative to a unigram model (601.6 to 171.1 to 142.8 on held-out text).
- The models score real Ewe better than nonsense at every order (Part 7 sanity check).
- The dev and test scores agree closely, so tuning did not overfit the dev set.
- With this data size, most long n-grams occur once (Part 5), so high orders lean heavily on their lower-order fallbacks.

**You should not say (without more work):**
- "The trigram is the best order." See 8.1: it depends on how the higher orders are tuned.
- "The model generates fluent Ewe." Nobody has evaluated fluency.
- "Perplexity 142.8 is good." There is no reference point. It is only meaningful next to other models scored *the same way* on the *same data* (for example your own unigram baseline at 601.6). Do not compare it with numbers from papers that use other data, other tokenization or other vocabularies.
- "The model understands Ewe." It counts word sequences. It has no notion of meaning or grammar.

---

## 8. Limitations and threats to validity

A good report has this section. Reviewers trust work that states its own weaknesses. Below are the real ones, ranked by importance.

### 8.1 The tuning grid was too narrow (the most important one)
The λ values were searched over 0.5 to 0.95. For the trigram, 4-gram and 5-gram the winner was 0.5, the bottom of that range. I re-ran the staged tuning on the dev set only (test set untouched) with a wider grid of 0.01 to 0.95, using `wide_grid_check.py` in this repository, which reproduces the notebook's own numbers exactly where the grids overlap (k = 1.0, unigram dev perplexity 594.0):

| Order | Notebook grid: chosen λ, dev perplexity | Wide grid: chosen λ, dev perplexity |
|---|---|---|
| 3 | 0.5, 143.1 | 0.3, **136.5** |
| 4 | 0.5, 191.5 | 0.05, **133.9** |
| 5 | 0.5, 312.3 | 0.01, **133.8** |

Two things follow. **(a)** The notebook's headline that the 4-gram and 5-gram are much worse than the trigram (190 and 309 vs 143) is largely an artifact of the search range. With a fitting λ the 4-gram is slightly *better* than the trigram on dev data, and the 5-gram is no better than the 4-gram. **(b)** The 5-gram again chose the lowest value on the widened grid (0.01), so its value is still not pinned down, and the 4-gram's 0.05 is close to the bottom too.

What is still true: longer context gives little or no extra benefit with this much data. The sparsity story in Part 5 stands. What is *not* supported is the size of the penalty for going beyond trigrams. **Before reporting final numbers, extend the grid (for example down to 0.01), re-tune, and score the test set once.** After that, the summary text in Part 12 should be updated.

The general lesson, worth stating in a report: *if a hyperparameter search returns the boundary of its range, the range is wrong.*

### 8.2 Data limitations
- **Small and skewed corpus**: 1.2 million training tokens; a large share appears to be religious text (verse numbers on about 21% of lines). Conclusions may not transfer to conversational or news Ewe.
- **Data quality**: 39 lines are corrupted junk (control characters and escape sequences such as `\E7t\EEWr{`), some duplicated. They are a tiny fraction (0.08%) but add noise to the vocabulary and can be memorised by high-order models.
- **Mixed provenance and licences**: three sources of different origin. State where each came from and confirm you may use it *[verify]*.
- **Possible overlap**: duplicated or near-duplicate sentences across train and test would make scores look better than they are. This was not checked.

### 8.3 Tokenization limitations
The tokenizer is a hand-written function with known faults, found after the notebook was built:
- **Combining marks are mangled.** Python's built-in pattern matching does not count combining marks as word characters, so a word like `melɔ̃a` is split into `melɔ` and `a`, and `lɔlɔ̃` loses its final tilde. About 1% of words in the corpus contain such a mark. In later experiments (not included here) this made almost no difference to perplexity, but it does damage the generated words.
- **`Ð` versus `Ɖ`.** The data sometimes types capital `Ɖ` as `Ð` (eth). Lowercasing gives `ð`, a different character from `ɖ`, so one word gets two spellings (`ðe` and `ɖe`). It shows up in generated text.
- **Digits are deleted**, including Bible verse numbers.
- **Lowercasing** loses capitalisation.
- **Text is not Unicode-normalised**, so the same visible character can be stored in different ways.

Later experiments (not included in this repository) measured several of these effects by comparing other tokenizers.

### 8.4 Evaluation limitations
- **One split, one seed**: no confidence intervals. Differences of a few percent between models could be noise. To estimate noise, repeat with several seeds or bootstrap the test sentences.
- **Perplexity is not comparable across tokenizations.** A tokenizer that produces more, easier tokens gets a lower *per-token* score without modelling better. Only compare perplexities computed with the same tokenizer, or normalise by something fixed (for example bits per character of the original text).
- **Intrinsic only**: no task-based evaluation (for example spelling correction or next-word suggestion accuracy).
- **Generation is unseeded and unrated**: the samples change on every run and no speaker judged them.
- **Coarse settings search**: staged (one setting at a time), not joint, and on a short list of values.

### 8.5 Modelling limitations
n-gram models have no memory beyond n-1 words, no notion of meaning, and cannot use similarity between words (they treat `nyɔnu` and `ŋutsu` as unrelated). Standard improvements not tried here: Kneser-Ney smoothing (Section 11), subword or character models, and neural models.

---

## 9. Writing your technical report

### 9.1 Suggested structure

A typical layout for an 8 to 10 page report. Word budgets are rough.

| Section | Length | What to write | Draw from this guide |
|---|---|---|---|
| **Title and abstract** | 150-200 words | Problem, method, data size, headline result, one caveat | Section 1, 7 |
| **1. Introduction** | 1 page | Why Ewe language models matter (low-resource), what n-gram models are, what you contribute, roadmap | Sections 1, 2.2 |
| **2. Background and related work** | 1 page | n-gram models, smoothing (add-k, interpolation, Kneser-Ney), perplexity; cite the references in Section 11 | Sections 3, 4, 11 |
| **3. Data** | 1 page | Sources with citations and licences, sizes, preprocessing decisions, split, statistics table, data limitations | Sections 2.3, 5 (Parts 2-4), 8.2 |
| **4. Method** | 1.5 pages | Tokenization, counting, interpolation formula, hyperparameters, generation procedure | Sections 4, 5 (Parts 3, 5, 6, 9) |
| **5. Experimental setup** | 0.5 page | Train/dev/test protocol, tuning procedure, metric definition, software versions, seed | Section 5 (Parts 4, 7.5, 8), checklist 9.4 |
| **6. Results** | 1 page | Table of perplexity by order; dev vs test; sanity check; one figure | Sections 5 (Part 8), 7 |
| **7. Analysis and discussion** | 1-1.5 pages | Why more context helps then plateaus (sparsity table), the tuning-grid finding, error analysis on samples, what surprised you | Sections 5 (Parts 5, 9-11), 6, 8.1 |
| **8. Limitations** | 0.5 page | The honest list | Section 8 |
| **9. Conclusion and future work** | 0.5 page | What was learned; next steps (Kneser-Ney, better tokenizer, cleaning, native-speaker evaluation, neural models) | Sections 8, 11 |
| **References** | | Verified citations | Section 11 |
| **Appendix** | | Reproducibility details, extra tables, sample outputs | 9.4 |

### 9.2 A draft abstract you can adapt

Replace the bracketed numbers with your final re-tuned results.

> Ewe is a low-resource West African language with little digitised text. We build word-level n-gram language models (orders 1 to 5) from scratch on a corpus of about [47,000] Ewe sentences drawn from three sources. Sparsity is handled with add-k smoothing at the unigram level and recursive linear interpolation for higher orders, with hyperparameters chosen by grid search on a held-out development set. On a held-out test set, perplexity falls from [601.6] for a unigram model to [142.8] for a trigram model. We find that [higher orders give little additional benefit given the corpus size], and we show that the apparent penalty of higher orders depends strongly on the tuning range. We discuss limitations including a small and partly religious corpus, tokenization issues with Ewe combining marks, and the absence of native-speaker evaluation.

### 9.3 Formulas to include

Write these in your Method section (adapt the notation):

1. n-gram estimate: $P_{\text{MLE}}(w_i \mid w_{i-n+1}^{i-1}) = \dfrac{C(w_{i-n+1}^{i})}{C(w_{i-n+1}^{i-1})}$
2. Unigram add-k: $P(w) = \dfrac{C(w) + k}{N + k|V|}$
3. Recursive interpolation: $P_n(w \mid h) = \lambda_n P_{\text{MLE}}(w \mid h) + (1 - \lambda_n) P_{n-1}(w \mid h')$, where $h'$ is $h$ with its oldest word dropped
4. Sentence log-probability: $\ln P(s) = \sum_{i=1}^{N} \ln P(w_i \mid \text{context}_i)$
5. Perplexity: $\text{PP} = \exp\!\left(-\dfrac{1}{N}\ln P(s)\right)$, with N the number of predicted tokens (state that `<s>` is not predicted and `</s>` is)

### 9.4 Reproducibility checklist

Reviewers value this. Put it in an appendix.

- [ ] Python version and package versions (this project used pandas 3.0.5, matplotlib 3.11.2, numpy 2.5.3, openpyxl 3.1.5, Python 3.14; check `pip list`)
- [ ] Data sources, download dates, licences
- [ ] Number of raw lines, lines kept, final split sizes (47,965 / 47,331 / 37,864 / 4,733 / 4,734)
- [ ] Random seed (42) and how the shuffle was done
- [ ] Every hyperparameter grid *and* the chosen values
- [ ] The exact tokenization rules (paste `preprocess_ewe`)
- [ ] Definition of perplexity, including which tokens are predicted
- [ ] Statement that the test set was scored once, after all choices
- [ ] "Restart and Run All" was performed and results match
- [ ] The notebook needs internet access for the HuggingFace sentences; state this or save a local copy

### 9.5 Mistakes reviewers commonly flag

- Reporting only the best number, without the dev/test protocol.
- Comparing your perplexity with published numbers from different data.
- Claiming fluency or "understanding" without human evaluation.
- Tuning on the test set, even accidentally.
- A hyperparameter that sits on the edge of its search range (Section 8.1).
- Not saying what the tokenizer removed (digits, symbols, case).
- No limitations section.

### 9.6 Numbers you can quote, and where they come from

| Fact | Value | Source in the notebook |
|---|---|---|
| Raw lines / kept | 47,965 / 47,331 | Parts 2, 3 |
| Tokens / word types | 1,531,742 / 42,331 | Part 3 |
| Word types occurring once | 21,811 of 42,329 (51.5%) | Part 10 |
| Split | 37,864 / 4,733 / 4,734 | Part 4 |
| Distinct n-grams, n=1..5 | 37,454 / 278,227 / 647,136 / 888,590 / 981,375 | Part 5 |
| Chosen k, λ₂..λ₅ | 1.0, 0.75, 0.5, 0.5, 0.5 | Part 7.5 |
| Test perplexity, n=1..5 | 601.6 / 171.1 / 142.8 / 190.0 / 308.9 | Part 8 |
| Test predictions | 148,375 | Part 8 |
| Mean words per line | 27.2 | Part 2 |

**Before quoting the last two rows of results, re-tune with the wider grid (Section 8.1).**

---

## 10. Test your understanding

Try answering before reading the hint.

1. **Why can't we just use raw counts (MLE) as probabilities?** *(Unseen sequences get probability 0, so any sentence containing one is "impossible". Section 4.2.)*
2. **Why does the model use `<s>` and `</s>`?** *(So it can learn how sentences typically begin and end. Without them it would never learn where a sentence stops.)*
3. **Why do we keep a test set we never look at until the end?** *(Every choice made by looking at a score fits the model to that data. An untouched set gives an unbiased final estimate. Part 4.)*
4. **A colleague reports perplexity 90 on their English model and says yours (143) is worse. What is wrong with that comparison?** *(Different language, data, vocabulary and tokenization; perplexity is only comparable when those are the same. Section 7.)*
5. **The best interpolation weight came back as the smallest value in the grid. What should you do?** *(Widen the grid and re-run; the true best is probably outside it. Section 8.1.)*
6. **Why does a 5-gram model do so well on a sentence from the training data (-2.79 per word) but badly on unusual phrases?** *(Memorisation: long contexts match training text exactly but have nothing to match in new text. Part 7.)*
7. **A model scores nonsense better than real text. What do you check first?** *(How unseen words are handled; see the `<UNK>` trap, Section 6 item 2.)*
8. **Why add log-probabilities instead of multiplying probabilities?** *(Products of many small numbers underflow; logs turn them into stable sums. Section 4.6.)*

---

## 11. References and further reading

Details below were checked against the publisher or a library record unless marked *[verify]*. Add page numbers and DOIs in your own bibliography style.

**Textbook (best starting point for a beginner).**
- Jurafsky, D., & Martin, J. H. *Speech and Language Processing* (3rd ed. draft), Chapter 3, "N-gram Language Models". Free at https://web.stanford.edu/~jurafsky/slp3/ (covers n-grams, perplexity, add-k, interpolation, backoff, Kneser-Ney).

**Smoothing and interpolation.**
- Jelinek, F., & Mercer, R. L. (1980). Interpolated estimation of Markov source parameters from sparse data. In *Proceedings of the Workshop on Pattern Recognition in Practice* (Amsterdam), North-Holland, pp. 381-397. (Origin of linear interpolation of n-gram estimates.)
- Katz, S. M. (1987). Estimation of probabilities from sparse data for the language model component of a speech recognizer. *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 35(3), 400-401. (Backoff.)
- Kneser, R., & Ney, H. (1995). Improved backing-off for m-gram language modeling. In *Proceedings of ICASSP 1995*, pp. 181-184. (Kneser-Ney smoothing: the standard next improvement over what we built.)
- Chen, S. F., & Goodman, J. (1999). An empirical study of smoothing techniques for language modeling. *Computer Speech & Language*, 13(4), 359-394. (Earlier versions: ACL 1996; Harvard technical report TR-10-98.) The best single comparison of smoothing methods, and a good citation for why we smooth.

**Foundations.**
- Shannon, C. E. (1948). A mathematical theory of communication. *Bell System Technical Journal*, 27. (Introduced the idea of approximating language with n-gram statistics.) *[verify details]*

**Ewe language and data.**
- Ethnologue and the Wikipedia articles on the Ewe and Gbe languages give speaker counts and classification. Cite a specific, dated source *[verify]*.
- Cite the three datasets from their original providers *[verify]*.

---

## 12. How to run the notebook yourself

1. Use the project virtual environment (`venv/`). The notebook needs `pandas`, `matplotlib`, `openpyxl` (it installs `openpyxl` itself if missing).
2. Keep the `data/` folder (`EWE_ENGLISH.csv` and `waxal_transcriptions.xlsx`) next to the notebook.
3. Internet is needed once for the 200 HuggingFace sentences. If the download fails the notebook continues without them and your numbers will differ slightly.
4. Choose **Restart kernel and Run All**. It takes about half a minute.
5. Do not edit the notebook file from outside your editor while it is open.

### What to do before you submit
- [ ] Widen the λ grid and re-tune (8.1); update the results tables and Part 12 text.
- [ ] Decide whether to fix the tokenizer's combining-mark and `Ð` problems and re-run (8.3).
- [ ] Decide whether to remove the 39 junk lines (8.2). Say so in the report either way.
- [ ] Verify the speaker-count statement in Part 1 of the notebook and every *[verify]* in this guide.
- [ ] Seed the text generation so your reported samples are reproducible.
- [ ] Have a native speaker rate a sample of generated sentences, or state clearly that fluency was not assessed.
- [ ] Complete the reproducibility checklist (9.4).
