# %% [markdown]
# # Lesson 06 — What does the loss number mean?
#
# Every lesson so far printed a "loss": 2.19 → 1.18 in lesson 03, 10.8 at
# Pythia's step 0 in lesson 05. Training is nothing more than pushing that
# number down, so it's worth knowing exactly what it measures.
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
# We use TinyStories (short, simple children's stories), the dataset we'll
# pretrain on in Part 3.

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

# %%
banner("One loss, four ways of writing it")
for nats in (10.8, 4.0, 2.0, 1.0):
    print(f"  {nats:5.2f} nats/token = {nats / math.log(2):5.2f} bits/token = perplexity {math.exp(nats):9,.1f}")

# %% [markdown]
# ## Rung 0: know nothing
#
# A model that knows nothing gives every token the same probability 1/V.
# Its loss is −log(1/V) = **log V**, whatever the text. That's why every
# freshly initialised model starts at ln(vocab size): Pythia's 10.8 in lesson
# 05 is ln(50,304).

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
# Counting works surprisingly well, up to a point. Then two numbers split:
# on the text it was counted from, the n-gram model keeps getting better
# (it's memorising), while on new text it gets *worse*. Longer contexts have
# been seen fewer times, until most of them were never seen at all.
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
# A uniform guess over 49k tokens is *worse* per character than a uniform
# guess over 97 characters, but a token **bigram** beats a character 3-gram:
# each token already carries a whole word.
#
# ## Real models on the same text

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
# Most tokens are nearly free (the end of a word, "the" after "in", the
# period at the end of a sentence). A few tokens carry most of the loss:
# names, the first word of a new event, choices the story could have made
# differently. **There's a floor**: some of this surprise can never go away,
# because the story really could have gone another way. That irreducible
# part is the *entropy of the text itself*. Training pushes loss towards it,
# and never below.
#
# **Aha:**
# * Loss starts at log(vocab size), always.
# * Counting gets you a long way, until it starts memorising instead of generalising.
# * Compare models in **bits per character** (or per byte), never in loss per
#   token across different tokenizers.
# * Most of the loss sits in a few hard tokens, and part of it is irreducible.

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
