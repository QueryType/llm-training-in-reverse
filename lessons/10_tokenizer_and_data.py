# %% [markdown]
# # Lesson 10 — Where does the data come from?
#
# Part 2 trained on characters: the simplest tokenizer there is. It worked,
# but it's wasteful. A 128-character window holds about 25 words, so the
# model can't see far, and it spends most of its effort spelling.
#
# Every real LLM uses **subword tokens** made by **BPE** (byte-pair
# encoding), the algorithm we take apart here. Then we prepare the data for
# our own pretraining run (lesson 11): train a tokenizer, check the data, and
# turn 2 million stories into one long stream of token ids.
#
# Questions: **how does BPE decide what a token is? How big should the
# vocabulary be? And what can go wrong with the data before training even
# starts?**
#
# This is the first lesson of Part 3, where we rebuild the pipeline forwards:
# data (this lesson) → pretraining (11) → making it fit and fast (12) →
# choosing the model size (13) → turning it into a chat model (14). The data
# is TinyStories: 2.1 million short stories written by GPT-3.5 and GPT-4 in
# simple English.

# %%
import os, sys, collections, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd())
from common import *
import numpy as np
import matplotlib.pyplot as plt

seed_all(0)
R = {}
banner("Loading TinyStories")
with Timer() as t:
    train_stories = tinystories("train")
    val_stories = tinystories("validation")
print(f"  {len(train_stories):,} train stories, {len(val_stories):,} validation   ({t.s:.0f}s)")
print(f"  example: {train_stories[1][:200]!r}")

# %% [markdown]
# ## BPE from scratch
#
# Start with characters. Count every pair of neighbouring tokens. Merge the
# most frequent pair into a new token. Repeat. That's the whole algorithm.
# Each merge makes the text a bit shorter, and the vocabulary one token bigger.
#
# (Real tokenizers work on *bytes* and don't merge across spaces in the same
# way, but the idea is identical.)

# %%
def bpe_from_scratch(texts, n_merges):
    # words as tuples of symbols, with counts. A leading space stays attached to its word: " the"
    words = collections.Counter()
    for t in texts:
        for w in t.replace("\n", " \n ").split(" "):
            if w:
                words[tuple(" " + w)] += 1
    n_chars = sum(len(w) * c for w, c in words.items())
    merges, sizes = [], [n_chars]
    for _ in range(n_merges):
        pairs = collections.Counter()
        for w, c in words.items():
            for a, b in zip(w, w[1:]):
                pairs[a, b] += c
        (a, b), freq = pairs.most_common(1)[0]
        merges.append((a, b, freq))
        new = {}
        for w, c in words.items():
            out, i = [], 0
            while i < len(w):
                if i < len(w) - 1 and w[i] == a and w[i + 1] == b:
                    out.append(a + b); i += 2
                else:
                    out.append(w[i]); i += 1
            new[tuple(out)] = new.get(tuple(out), 0) + c
        words = new
        sizes.append(sum(len(w) * c for w, c in words.items()))
    return merges, sizes, n_chars


banner("BPE from scratch on 3,000 stories: the first merges")
with Timer() as t:
    merges, sizes, n_chars = bpe_from_scratch(train_stories[:3000], 300)
for i in list(range(12)) + [49, 99, 199, 299]:
    a, b, f = merges[i]
    print(f"  merge {i + 1:3}: {a!r:>10} + {b!r:<10} → {a + b!r:14} ({f:,} times)   text is now {sizes[i + 1] / n_chars:5.1%} of its length in chars")
print(f"  ({t.s:.0f}s in pure Python)")
R["scratch_merges"] = [[a, b, f] for a, b, f in merges]
R["scratch_sizes"] = [s / n_chars for s in sizes]

# %% [markdown]
# The first merges are the most common letter pairs: "he" (76,537 times),
# " t", " a", " s", " w", "nd". By merge 7 we have the first whole word,
# " the" (" t" + "he"), and by merge 10 " and". Later merges are rarer pieces
# and story words: merge 200 is " wanted" (1,730 times).
#
# Each merge helps less than the one before. The first one shortens the text
# by 2.9%; after 100 merges it's at 54% of its length in characters, after
# 300 merges at 40%. The merges that remain to be found are rarer and rarer
# pairs.
#
# Notice that the leading space is part of the token: " the" and "the" are
# different tokens. That's how almost every modern tokenizer works, and it's
# why tokens printed by `tokenizers` start with "Ġ" (its symbol for a space).
#
# ## The real thing, at several vocabulary sizes
#
# Hugging Face's `tokenizers` library does the same thing in Rust, on bytes,
# thousands of times faster. We reserve three **special tokens** now, because
# we'll need them later: `<|endoftext|>` between stories, and `<|im_start|>`
# / `<|im_end|>` for the chat format in lesson 14.

# %%
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]


def train_bpe(texts, vocab_size):
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIALS, show_progress=False,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tk.train_from_iterator(texts, trainer)
    return tk


sample = train_stories[:100_000]
probe_text = "\n".join(val_stories[:2000])
D_MODEL = 384                     # the width of lesson 11's model
banner("Vocabulary size vs compression (on 2,000 validation stories)")
vocab_rows = []
for vs in (512, 1024, 2048, 4096, 8192, 16384):
    with Timer() as t:
        tk = train_bpe(sample, vs)
    n = len(tk.encode(probe_text).ids)
    row = {"vocab": vs, "chars_per_token": len(probe_text) / n, "tokens": n, "emb_params": vs * D_MODEL}
    vocab_rows.append(row)
    print(f"  vocab {vs:6,}: {row['chars_per_token']:4.2f} chars/token   embedding table at width {D_MODEL}:"
          f" {row['emb_params'] / 1e6:5.2f}M params   (trained in {t.s:.0f}s)")
smol = AutoTokenizer.from_pretrained(BASE_ID)
n_smol = len(smol(probe_text).input_ids)
print(f"  SmolLM2's tokenizer (49,152, trained on the whole web): {len(probe_text) / n_smol:4.2f} chars/token,"
      f" embedding {49152 * D_MODEL / 1e6:.1f}M params")
R["vocab_sizes"], R["smol_chars_per_token"] = vocab_rows, len(probe_text) / n_smol

# %% [markdown]
# | vocab | 512 | 1,024 | 2,048 | 4,096 | 8,192 | 16,384 | SmolLM2 (49,152) |
# |---|---|---|---|---|---|---|---|
# | chars/token | 2.35 | 3.11 | 3.66 | 4.00 | 4.17 | 4.21 | 4.14 |
#
# Compression improves quickly up to a few thousand tokens and then barely
# moves: 8k → 16k gains less than 1%. TinyStories was generated from a small
# basic vocabulary (the TinyStories paper used about 1,500 words a young
# child knows), so a few thousand tokens cover nearly every word whole.
#
# SmolLM2's 49k-token vocabulary, trained on the whole web, compresses this
# text about as well as our 8k one (4.14 vs 4.17). Most of its tokens are
# for code, other languages and rare words that never appear here, and at
# lesson 11's width its embedding table alone (18.9M parameters) would be
# bigger than lesson 11's whole model.
#
# **We pick 4,096.** About 4 characters per token, so a 256-token window
# holds ~1,000 characters (more than a median story, 786 characters), and the
# embedding table stays small (1.6M parameters). Doubling to 8,192 would buy
# 4% shorter sequences for twice the table.

# %%
VOCAB = 4096
with Timer() as t:
    tk = train_bpe(train_stories[:200_000], VOCAB)
TOK_DIR = OUT / "tokenizer"
TOK_DIR.mkdir(parents=True, exist_ok=True)
tk.save(str(TOK_DIR / "tokenizer.json"))
print(f"  trained a {VOCAB}-token BPE on 200k stories in {t.s:.0f}s -> {(TOK_DIR / 'tokenizer.json').relative_to(ROOT)}")
example = "Once upon a time, there was a little girl named Lily. She loved to play outside."
enc = tk.encode(example)
print(f"  {example!r}\n  → {len(enc.ids)} tokens: {enc.tokens}")
print(f"  special token ids: {[tk.token_to_id(s) for s in SPECIALS]}")
R["example"] = {"text": example, "tokens": enc.tokens, "ids": enc.ids}
R["longest_tokens"] = sorted(tk.get_vocab(), key=len, reverse=True)[:20]

# %% [markdown]
# The example sentence is 19 tokens, one per word or punctuation mark: at
# this vocabulary size every word in it is a single token. The longest tokens
# are words like " uncomfortable", " compassionate" and " caterpillar":
# frequent enough in children's stories to earn their own entry.
#
# ## Look at the data before you train on it
#
# Two classic problems: **duplicates** (the model sees the same text many
# times and memorises it) and **contamination** (test text that also appears
# in the training set, so test loss looks better than it is).

# %%
banner("Data checks")
h = lambda s: hashlib.md5(s.encode()).hexdigest()
train_hashes = collections.Counter(h(s) for s in train_stories)
dups = sum(c - 1 for c in train_hashes.values() if c > 1)
most = train_hashes.most_common(1)[0][1]
val_in_train = sum(1 for s in val_stories if h(s) in train_hashes)
lens = np.array([len(s) for s in train_stories])
print(f"  exact duplicate stories in train: {dups:,} ({dups / len(train_stories):.2%}); the most repeated appears {most}×")
print(f"  validation stories that also appear in train: {val_in_train} of {len(val_stories):,}")
print(f"  story length (chars): median {np.median(lens):.0f}, 5% {np.percentile(lens, 5):.0f}, 95% {np.percentile(lens, 95):.0f}")
R["checks"] = {"dups": dups, "most_repeated": most, "val_in_train": val_in_train, "n_train": len(train_stories),
               "n_val": len(val_stories), "len_pct": [float(np.percentile(lens, q)) for q in (5, 25, 50, 75, 95)]}

# %% [markdown]
# Both problems are real here:
#
# * **320,241 exact duplicates** in the training set (15.1% of 2.1M
#   stories); no story appears more than 3 times, so it's many stories
#   duplicated once rather than a few repeated thousands of times.
# * **6,601 of the 21,990 validation stories (30%) also appear in train**,
#   word for word. A model that memorised them would score suspiciously well
#   on validation, and we'd never know why.
#
# Neither is a scandal: TinyStories is a generated dataset, and a generator
# asked for millions of short stories from a small vocabulary repeats itself.
# Web-scale datasets have the same problems at a bigger scale, which is why
# deduplication and decontamination are standard steps in every pretraining
# pipeline.
#
# We remove the duplicates and the contaminated validation stories, so our
# validation loss in lessons 11–14 measures generalisation, not memory.

# %%
seen, clean_train = set(), []
for s in train_stories:
    k = h(s)
    if k not in seen:
        seen.add(k); clean_train.append(s)
clean_val = [s for s in val_stories if h(s) not in train_hashes]
print(f"  after dedup: {len(clean_train):,} train stories, {len(clean_val):,} validation stories")

# %% [markdown]
# ## Packing: one long stream
#
# Stories have different lengths, but a training batch is a rectangle of
# (batch × context) tokens. The simple and standard answer: glue every story
# together with `<|endoftext|>` in between, and cut random windows from the
# stream (that's `get_batch` from Part 2). A window can start in the middle of
# one story and run into the next, and the model learns that
# `<|endoftext|>` means "forget the past, a new story starts".

# %%
EOT = tk.token_to_id("<|endoftext|>")
N_TRAIN = None if FULL else 1_000_000          # stories to tokenize; ~200M tokens is plenty for lesson 11


def tokenize_stream(stories, path, bs=10_000):
    total = 0
    with open(path, "wb") as f:
        for i in range(0, len(stories), bs):
            ids = []
            for e in tk.encode_batch(stories[i:i + bs]):
                ids += e.ids + [EOT]
            np.array(ids, dtype=np.uint16).tofile(f)
            total += len(ids)
    return total


banner("Tokenizing")
DATA.mkdir(parents=True, exist_ok=True)
with Timer() as t:
    n_tr = tokenize_stream(clean_train[:N_TRAIN], DATA / "ts4096_train.bin")
    n_va = tokenize_stream(clean_val, DATA / "ts4096_val.bin")
print(f"  train: {n_tr:,} tokens   val: {n_va:,} tokens   ({t.s:.0f}s) -> outputs/data/ts4096_*.bin  (uint16)")
R["stream"] = {"train_tokens": n_tr, "val_tokens": n_va, "train_stories": len(clean_train[:N_TRAIN])}

stream = np.fromfile(DATA / "ts4096_val.bin", dtype=np.uint16)
start = int(np.where(stream == EOT)[0][3]) - 40
window = stream[start:start + 64].tolist()
print("\n  a 64-token window from the stream ('|' marks token boundaries):")
print("  " + "|".join(tk.decode([i]) if i != EOT else "⟨EOT⟩" for i in window))
R["window"] = [tk.decode([i]) if i != EOT else "<|endoftext|>" for i in window]

# %% [markdown]
# 1M stories become 229M training tokens (about 229 per story) and the
# cleaned validation set 3.3M tokens. Stored as `uint16` (4,096 < 65,536),
# the training stream is 457 MB on disk, read with `np.memmap` so it never has
# to fit in memory. In the window above, "…helping others." is followed by
# `<|endoftext|>` and then "Once upon a time": the model sees story boundaries
# inside its windows and has to learn to start over.

# %%
fig, axs = plt.subplots(1, 2, figsize=(12, 3.4))
axs[0].plot(range(len(sizes)), [s / n_chars for s in sizes])
axs[0].set_xlabel("merges"); axs[0].set_ylabel("text length (vs characters)"); axs[0].set_title("BPE from scratch: each merge helps less")
axs[1].plot([r["vocab"] for r in vocab_rows], [r["chars_per_token"] for r in vocab_rows], "o-", label="our BPE")
axs[1].axhline(R["smol_chars_per_token"], ls="--", color="gray", label="SmolLM2 (49k)")
axs[1].set_xscale("log", base=2); axs[1].set_xlabel("vocabulary size"); axs[1].set_ylabel("chars per token"); axs[1].legend()
show_fig(fig, "10_tokenizer")

# %% [markdown]
# **Aha:**
# * A tokenizer is a compression scheme *learned from the data*: merge the most
#   common pair, repeat. Its vocabulary reflects what it was trained on.
# * A tokenizer for a narrow domain can be tiny. Vocabulary size is a trade:
#   shorter sequences vs a bigger embedding table and rarer tokens.
# * Check the data. Duplicates (15% here) and test-set leaks (30% of
#   validation here) are easy to find and easy to miss, and they quietly
#   inflate your results.
#
# ## What we know so far
#
# * Tokens are learned from the data: BPE merges the most common pair, again
#   and again, and stops at the vocabulary size you chose.
# * For narrow text, a small vocabulary compresses as well as a big one.
# * Real data needs cleaning before training, even when it looks clean.
#
# Next: we have 229M tokens, a tokenizer and everything from Part 2. **Can we
# pretrain a model ourselves, from random weights, in about ten minutes?**

# %%
save_result("10_tokenizer_and_data", R)

# %% [markdown]
# ## Try:
# * Train the tokenizer on only 1,000 stories. Which common words end up split?
# * Tokenize a sentence about computers or in French with our tokenizer. How many tokens?
# * Look at `R["longest_tokens"]`: what does the tokenizer think a "word" is?
# * Instead of packing, pad each story to the longest in the batch (like lesson 03).
#   How many of the tokens in a batch are padding?
