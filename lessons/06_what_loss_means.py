# %% [markdown]
# # Lesson 06 — What does the loss number mean?
#
# Every lesson so far printed a "loss": 2.19 → 1.18 in lesson 03, 11.0 at
# Pythia's step 0 in lesson 05. Training is nothing more than pushing that
# number down, so before we look at *how* it gets pushed (lessons 07–09) it's
# worth knowing exactly what it measures, and what a good value looks like.
#
#     loss = average over tokens of  −log P(the real next token)
#
# It's the model's **average surprise**. If the model gave the right token
# probability 1, surprise is 0. Probability 1/2 costs 0.69 nats (1 bit);
# 1/1000 costs 6.9 nats.
#
# Questions: **what loss does a model get with no knowledge at all? With only
# counting? How do you compare losses between models with different
# tokenizers? And how close are real models to "perfect"?**
#
# We use TinyStories (short, simple children's stories written by GPT-3.5
# and GPT-4), the dataset we'll pretrain on in Part 3: 60,000 training
# stories (53.5M characters) for counting, 2,000 validation stories (1.6M
# characters) for scoring. Its alphabet is 97 characters.
#
# The plan is a **ladder**: score many models on the same validation text,
# from "knows nothing" through "just counts" to real neural networks, all in
# one unit, so we can see where each kind of knowledge comes in.

# %%
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd())
from common import *
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

seed_all(0)
torch.set_grad_enabled(False)
device = pick_device()
ctok, train_ids, val_ids = char_data()
train_np, val_np = train_ids.numpy().astype(np.int64), val_ids.numpy().astype(np.int64)
V = len(ctok)
val_text = ctok.decode(val_ids.tolist()).replace("\n<end>\n", "\n\n")
print(f"device: {device}   train: {len(train_np):,} chars   val: {len(val_np):,} chars   vocab: {V} characters")
R = {"char_vocab": V}

# %% [markdown]
# ## Three units for the same thing
#
# * **nats per token**: what PyTorch's `cross_entropy` returns (natural log).
# * **bits per token** = nats / ln 2. One bit = one fair coin flip of uncertainty.
# * **perplexity** = e^loss: "as uncertain as picking uniformly among this many tokens".
#
# And one more, which is what makes models comparable:
#
# * **bits per character** = total bits / number of characters. A tokenizer
#   that makes fewer, bigger tokens has a higher loss *per token*, but it's
#   predicting more text per token. Per character, the comparison is fair.
#
# Why a logarithm at all? Because probabilities of a whole text *multiply*:
# P(story) = P(token 1) · P(token 2 | token 1) · … . Logs turn that product
# into a sum, and the average of the sum is the loss. So a loss of L nats per
# token means the model gives a 1,000-token text probability e^(−1000·L).
# Lower loss = the real text looked more likely to the model.

# %%
banner("One loss, four ways of writing it")
for nats in (10.8, 4.0, 2.0, 1.0):
    print(f"  {nats:5.2f} nats/token = {nats / math.log(2):5.2f} bits/token = perplexity {math.exp(nats):9,.1f}")

# %% [markdown]
# Notice the first row: 10.8 nats is a perplexity of about 49,000, which is
# roughly SmolLM2's vocabulary size (49,152). That's no coincidence, as the
# next section shows. And 1 nat (perplexity 2.7) means "as unsure as choosing
# between about three tokens".
#
# ## Rung 0: know nothing
#
# A model that knows nothing gives every token the same probability 1/V.
# Its loss is −log(1/V) = **log V**, whatever the text. That's why every
# freshly initialised model starts *near* ln(vocab size), usually a little
# above it: its random logits aren't perfectly flat, and any unevenness costs
# extra. Pythia measured 11.01 at step 0 in lesson 05 (ln 50,304 = 10.83);
# our own model in lesson 11 starts at 8.39 (ln 4,096 = 8.32).

# %%
ladder = []


def rung(name, bits_per_char, kind, train_bpc=None):
    ladder.append({"name": name, "bpc": bits_per_char, "kind": kind, "train_bpc": train_bpc})
    extra = f"   (on its own training text: {train_bpc:.3f})" if train_bpc is not None else ""
    print(f"  {name:40} {bits_per_char:6.3f} bits/char{extra}")


banner("The ladder, in bits per character (lower is better)")
rung("uniform over 97 characters", math.log2(V), "count")

# %% [markdown]
# ## Rungs 1–6: just count
#
# An **n-gram model** predicts the next character from counts of what followed
# the previous n−1 characters in the training text. No neural network, no
# gradient: a lookup table. n=1 knows only how common each character is. n=3
# knows that "th" is usually followed by "e".
#
# We give unseen combinations a tiny pseudo-count (add-k smoothing), otherwise
# a single surprise would cost infinite loss.

# %%
def ngram_bpc(n, k=0.01, eval_on=val_np, train=train_np):
    """bits/char of an n-gram character model with add-k smoothing."""
    if n == 1:
        counts = np.bincount(train, minlength=V) + k
        p = counts / counts.sum()
        return float(-np.log2(p[eval_on]).mean())
    def codes(a, m):                       # each window of m chars → one 64-bit integer
        c = np.zeros(len(a) - m + 1, dtype=np.int64)
        with np.errstate(over="ignore"):   # past 9 chars this wraps around: a hash, collisions negligible
            for j in range(m):
                c = c * V + a[j:len(a) - m + 1 + j]
        return c
    ctx_u, ctx_c = np.unique(codes(train[:-1], n - 1), return_counts=True)
    full_u, full_c = np.unique(codes(train, n), return_counts=True)
    e_full, e_ctx = codes(eval_on, n), codes(eval_on[:-1], n - 1)
    def lookup(u, c, q):
        i = np.searchsorted(u, q).clip(max=len(u) - 1)
        return np.where(u[i] == q, c[i], 0)
    num = lookup(full_u, full_c, e_full) + k
    den = lookup(ctx_u, ctx_c, e_ctx) + k * V
    return float(-np.log2(num / den).mean())


ngram_rows = []
for n in (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16):
    with Timer() as t:
        v = ngram_bpc(n)
        tr = ngram_bpc(n, eval_on=train_np[:len(val_np)])
    ngram_rows.append({"n": n, "val": v, "train": tr})
    rung(f"{n}-gram (counts of {n - 1} previous chars)", v, "count", tr)
R["ngrams"] = ngram_rows

# %% [markdown]
# Counting works surprisingly well, up to a point. Knowing only how common
# each letter is (n=1) already takes us from 6.60 to 4.40 bits per character.
# Each extra character of context helps: 3.31 (n=2), 2.44 (n=3), 1.56 (n=5).
# The best is **1.33 bits/char at n=7–8**: a lookup table of 6–7 previous
# characters.
#
# Then the two columns split. On the text it was counted from, the n-gram
# model keeps getting better, 1.02 at n=10 and 0.86 at n=16 (it's memorising).
# On new text it gets *worse*: 1.51 at n=10, 3.06 at n=16, worse than a
# 3-gram. A 15-character context is usually a phrase the table has never
# seen, so it has nothing to say and falls back on the smoothing pseudo-count.
#
# That's **overfitting** in its purest form, and it's the reason we always
# measure loss on held-out text. To do better, a model must **generalise**:
# it has to treat "the dog ran" and "the cat ran" as similar without having
# counted both. That's what the neural network buys us.
#
# ## Tokens instead of characters
#
# Same text, SmolLM2's tokenizer (49,152 tokens). Unigram and bigram token
# models, converted to bits per character so they fit on the same ladder.

# %%
stok = AutoTokenizer.from_pretrained(BASE_ID)
TRAIN_TEXT = ctok.decode(train_ids[:20_000_000].tolist()).replace("\n<end>\n", "\n\n")
tr_tok = np.array(stok(TRAIN_TEXT).input_ids, dtype=np.int64)
va_tok = np.array(stok(val_text).input_ids, dtype=np.int64)
chars_per_token = len(val_text) / len(va_tok)
VT = len(stok)
print(f"  SmolLM2 tokenizer: {chars_per_token:.2f} characters per token on TinyStories")


def tok_ngram_bits(n, k=0.01):
    if n == 0:
        return math.log2(VT)
    if n == 1:
        c = np.bincount(tr_tok, minlength=VT) + k
        return float(-np.log2((c / c.sum())[va_tok]).mean())
    ctx = np.bincount(tr_tok[:-1], minlength=VT)
    pair_u, pair_c = np.unique(tr_tok[:-1] * VT + tr_tok[1:], return_counts=True)
    q = va_tok[:-1] * VT + va_tok[1:]
    i = np.searchsorted(pair_u, q).clip(max=len(pair_u) - 1)
    num = np.where(pair_u[i] == q, pair_c[i], 0) + k
    return float(-np.log2(num / (ctx[va_tok[:-1]] + k * VT)).mean())


tok_rows = []
for n, name in ((0, "uniform over 49,152 tokens"), (1, "token unigram"), (2, "token bigram")):
    bpt = tok_ngram_bits(n)
    tok_rows.append({"n": n, "bits_per_token": bpt, "nats_per_token": bpt * math.log(2)})
    rung(f"{name} ({bpt * math.log(2):.2f} nats/token)", bpt / chars_per_token, "count")
R["token_ngrams"], R["chars_per_token"] = tok_rows, chars_per_token

# %% [markdown]
# Three things to see here:
#
# * The same "know nothing" model costs 10.80 nats **per token** but only
#   3.78 bits **per character**, better than uniform over characters (6.60).
#   The tokenizer itself carries knowledge: its 49k entries are common
#   chunks of English, so even a uniform guess over them can only spell
#   English-looking text. A token is about 4.1 characters here.
# * Token unigram (5.97 nats/token) = 2.09 bits/char, half the character
#   unigram's 4.40, with no context at all.
# * Token **bigram**, one previous token, gets 1.39 bits/char: about the same
#   as a 6-gram on characters (1.41). One token of context ≈ one word.
#
# Per token, these three look like 10.8, 6.0 and 4.0; per character, they
# sit between the character n-grams. That's why the per-character unit
# matters: the per-token numbers aren't comparable to anything else here.
#
# ## Real models on the same text
#
# Now neural networks. Pythia-70m (from lesson 05) at three checkpoints, and
# SmolLM2-135M. Neither was trained on TinyStories, and they use different
# tokenizers (Pythia: 50k tokens, SmolLM2: 49k), so bits per character is
# the only fair way to compare them.
#
# %%
def model_bpc(model, tokenizer, text=val_text, ctx=1024, max_tokens=200_000):
    ids = torch.tensor(tokenizer(text).input_ids[:max_tokens])
    n_chunks = len(ids) // ctx
    ids = ids[:n_chunks * ctx].view(n_chunks, ctx).to(device)
    nats, n = 0.0, 0
    for i in range(0, n_chunks, 8):
        x = ids[i:i + 8]
        logits = model(x).logits[:, :-1].float()
        nats += F.cross_entropy(logits.flatten(0, 1), x[:, 1:].flatten(), reduction="sum").item()
        n += x[:, 1:].numel()
    cpt = len(text[:len(tokenizer.decode(ids.flatten().tolist()))]) / ids.numel()   # chars per token in what we scored
    return (nats / n) / math.log(2) / cpt, nats / n


real = []
for name, mid, rev in (("Pythia-70m, step 1,000", "EleutherAI/pythia-70m", "step1000"),
                       ("Pythia-70m, step 8,000", "EleutherAI/pythia-70m", "step8000"),
                       ("Pythia-70m, final", "EleutherAI/pythia-70m", "step143000"),
                       ("SmolLM2-135M", BASE_ID, None)):
    kw = dict(revision=rev, use_safetensors=False) if rev else {}
    t_ = AutoTokenizer.from_pretrained(mid)
    m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32, **kw).to(device).eval()
    bpc, npt = model_bpc(m, t_)
    real.append({"name": name, "bpc": bpc, "nats_per_token": npt})
    rung(f"{name} ({npt:.2f} nats/token)", bpc, "model")
    del m
R["models"] = real

# %%
fig, ax = plt.subplots(figsize=(8, 5))
names = [r["name"] for r in ladder][::-1]
vals = [r["bpc"] for r in ladder][::-1]
cols = ["#b4530a" if r["kind"] == "model" else "#4a4fc4" for r in ladder][::-1]
ax.barh(names, vals, color=cols)
for i, v in enumerate(vals):
    ax.text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=8)
ax.set_xlabel("bits per character on TinyStories validation (lower is better)")
ax.set_title("The loss ladder: counting (blue) vs neural models (orange)")
show_fig(fig, "06_ladder")

# %% [markdown]
# The ladder, top to bottom:
#
# | model | bits/char |
# |---|---|
# | uniform over 97 characters | 6.60 |
# | best character n-gram (n=7–8) | 1.33 |
# | token bigram | 1.39 |
# | Pythia-70m, step 1,000 | 1.28 |
# | Pythia-70m, step 8,000 | 0.99 |
# | Pythia-70m, final (step 143,000) | 0.97 |
# | SmolLM2-135M | 0.66 |
#
# Pythia at step 1,000, the moment copying switched on in lesson 05, is only
# just better than the best lookup table. By step 8,000 it's well past it.
# The remaining 135,000 steps buy only 0.02 bits/char *on this text*: simple
# children's English is learned early; the rest of Pythia's training went
# into things TinyStories doesn't test (code, science, rare facts).
# SmolLM2, twice the size and trained on far more data (about 2T tokens vs
# Pythia's 300B), gets 0.66: a perplexity of about 1.6 per *character*.
#
# Per token, these models read 3.61, 2.79, 2.73 and 1.87 nats. Lesson 11
# trains our own model on TinyStories and places it on this ladder.
#
# ## Where is the loss?
#
# An average hides a lot. Look at SmolLM2's surprise on each token of one story.

# %%
story = val_text.split("\n\n")[3]
ids = encode(stok, story, device)
m = AutoModelForCausalLM.from_pretrained(BASE_ID, dtype=torch.float32).to(device).eval()
per_tok = F.cross_entropy(m(ids).logits[0, :-1], ids[0, 1:], reduction="none").tolist()
toks = [stok.decode([i]) for i in ids[0].tolist()]
banner("Per-token surprise (nats) in one story")
order = sorted(range(len(per_tok)), key=lambda i: -per_tok[i])
print("  hardest tokens:", ", ".join(f"{toks[i + 1]!r} {per_tok[i]:.1f}" for i in order[:10]))
print("  easiest tokens:", ", ".join(f"{toks[i + 1]!r} {per_tok[i]:.3f}" for i in order[-8:]))
print(f"  share of the story's total loss from its 10% hardest tokens: "
      f"{sum(per_tok[i] for i in order[:len(order) // 10]) / sum(per_tok):.0%}")
R["story"] = {"tokens": toks, "loss": per_tok}

# %% [markdown]
# This story averages 2.34 nats per token, above SmolLM2's 1.87 overall,
# because it's scored from its first word with no context. The median token
# costs 1.74 nats; only 5 of 70 tokens cost under 0.1.
#
# The expensive tokens are the ones nobody could predict: the name "Roxy"
# (the "oxy" after "R" costs 10.1 nats, the worst in the story), "tried"
# as the very first verb (9.7), "icy" (8.4), "named Billy" (6.0 and 5.5).
# These are choices the story *could* have made differently.
#
# Now watch the same words come back. That's lesson 05's copying at work:
#
# | token | 1st time | 2nd | 3rd | 4th |
# |---|---|---|---|---|
# | "oxy" (in Roxy) | 10.1 | 0.09 | 0.00 | 0.00 |
# | " R" (start of Roxy) | n/a | 7.35 | 1.57 | 0.32 |
# | " icy" | 8.4 | 4.4 | | |
# | " hill" | 2.24 | 0.06 | | |
# | " Billy" | 5.5 | 1.00 | | |
#
# Once a name is in the context, it's nearly free. What stays expensive is
# deciding *when* it comes back: " R" at the start of a sentence still cost
# 7.35 the first time.
#
# Overall, the 10% hardest tokens hold 32% of the story's loss. **There's a
# floor**: some of this surprise can never go away, because the story really
# could have gone another way (Billy could have been a dog). That
# irreducible part is the *entropy of the text itself*. Training pushes loss
# towards it, and never below.
#
# **Aha:**
# * Loss starts near log(vocab size), a little above it, for every model.
# * Counting gets you a long way, until it starts memorising instead of generalising.
# * Compare models in **bits per character** (or per byte), never in loss per
#   token across different tokenizers.
# * Most of the loss sits in a few hard tokens (new names, new events), and
#   part of it is irreducible. Repeats are nearly free.
#
# ## What we know so far
#
# * Loss = average −log P(next token). Lower = the real text looked more likely.
# * A model starts near ln(vocab) and has to beat counting to be worth anything:
#   1.33 bits/char is what a lookup table of the last 7 characters gets.
# * Neural models win because they generalise: Pythia at 0.97 and SmolLM2
#   at 0.66 on text neither was trained on.
#
# Next question: the loss is one number, and the model has millions of
# weights. **How does each weight know which way to move to lower it?**
# That's the gradient, and lesson 07 computes one by hand.

# %%
R["ladder"] = ladder
save_result("06_what_loss_means", R)

# %% [markdown]
# ## Try:
# * Change the smoothing `k` in `ngram_bpc` (1.0, 0.001). Which n is best now?
# * Count on less text (`train_np[:1_000_000]`). Overfitting starts at a smaller n. Why?
# * Score SmolLM2 on a text *unlike* TinyStories (code, Wikipedia). How many bits
#   per character now?
# * Compute the "copying" loss from lesson 05 in bits per character. What does
#   a model that copies perfectly cost?
