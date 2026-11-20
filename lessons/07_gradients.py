# %% [markdown]
# # Lesson 07 — What is a gradient, really?
#
# Every training loop in lessons 03 and 04 had the line `loss.backward()`.
# It fills in `p.grad` for every weight `p`: **how much the loss would change
# if this one weight were nudged up a tiny bit.** Then the optimizer moves
# each weight a little in the direction that lowers the loss.
#
# Lesson 06 told us what the loss measures. This lesson is about the one
# piece of information training gets back from it: a number per weight
# saying which way to move.
#
# Questions: **what exactly is in `p.grad`? Can we compute it ourselves and
# check PyTorch? And what does one step along it actually do?**
#
# The plan, from small to real:
# 1. one weight, where we can do the calculus on paper;
# 2. the softmax + cross-entropy gradient, the one every language model uses;
# 3. a tiny network with its backward pass written out by hand;
# 4. a real (tiny) Llama, checked weight by weight with finite differences;
# 5. one step along the gradient, with many step sizes;
# 6. what backward costs in time.
#
# Part 2 uses tiny models on the *characters* of TinyStories (lesson 06's
# data), so every experiment runs in seconds.

# %%
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd())
from common import *
import torch.nn.functional as F
import matplotlib.pyplot as plt

seed_all(0)
ctok, train_ids, val_ids = char_data()
V = len(ctok)
R = {}

# %% [markdown]
# ## One weight, by hand
#
# The smallest possible "model": predict y = w·x, loss = (w·x − y)².
# High-school calculus says d loss / d w = 2·x·(w·x − y). The **finite
# difference** says the same thing without calculus: nudge w up and down by
# a tiny ε and see how much the loss moves.

# %%
banner("d loss / d w, three ways")
x, y, w0 = 3.0, 6.0, 1.5
w = torch.tensor(w0, requires_grad=True)
loss = (w * x - y) ** 2
loss.backward()
eps = 1e-4
f = lambda w_: (w_ * x - y) ** 2
print(f"  calculus:          2·x·(w·x − y)      = {2 * x * (w0 * x - y):.6f}")
print(f"  finite difference: (f(w+ε)−f(w−ε))/2ε = {(f(w0 + eps) - f(w0 - eps)) / (2 * eps):.6f}")
print(f"  autograd:          w.grad             = {w.grad.item():.6f}")
print("  -> negative: increasing w lowers the loss. The true answer is w = 2.")

# %% [markdown]
# All three agree: −9.0. The sign is the useful part. Negative means "loss
# goes down if w goes up", so a step *against* the gradient moves w from 1.5
# towards 2. The size says how steep the slope is here, not how far to go.
#
# The finite difference is our ground truth for the rest of the lesson: it
# needs no calculus, only two forward passes, so it can check any gradient.
# It's far too slow to train with (two forward passes *per weight*), which is
# why backprop exists.
#
# ## The most important gradient in language modelling
#
# Our models end with `logits → softmax → −log P(correct token)`. The
# gradient of that loss with respect to the logits has a famously simple form:
#
#     d loss / d logits  =  softmax(logits) − one_hot(correct token)
#
# Read it as an instruction. For the correct token: *raise your score by
# (1 − p)*. For every other token: *lower your score by its probability p*.
# Tokens the model was confidently wrong about get pushed down hardest.

# %%
banner("Gradient of the loss w.r.t. 6 logits")
logits = torch.tensor([2.0, 1.0, 0.5, 0.1, -1.0, 3.0], requires_grad=True)
target = 0
loss = F.cross_entropy(logits[None], torch.tensor([target]))
loss.backward()
p = logits.softmax(-1).detach()
manual = p.clone(); manual[target] -= 1
print(f"  probabilities:     {[round(v, 3) for v in p.tolist()]}   (correct token = #0)")
print(f"  softmax − one_hot: {[round(v, 3) for v in manual.tolist()]}")
print(f"  autograd:          {[round(v, 3) for v in logits.grad.tolist()]}")
R["softmax_grad"] = {"logits": logits.tolist(), "p": p.tolist(), "grad": logits.grad.tolist(), "target": target}

# %% [markdown]
# The model gave the correct token #0 a probability of 0.22, so its gradient
# is 0.22 − 1 = −0.78: push it up hard. Token #5 had 0.60 (the model's
# favourite, and wrong), so it gets the biggest push down, +0.60. The
# unlikely tokens barely move. The gradients add up to zero: the push is only
# about *which token wins*, not about the overall level of the scores.
#
# This also explains lesson 03's loss masking. A token with label −100 is
# left out of the loss, so its row of `softmax − one_hot` is simply never
# computed and it sends no push at all.
#
# ## Backprop by hand: a tiny character model
#
# Now a real (tiny) network: previous character → embedding (16 numbers) →
# tanh hidden layer (64) → logits over 97 characters. A bigram model with a
# neural network inside. We write the forward pass, then the **backward pass
# by hand**, one chain-rule step per line, and compare with autograd.

# %%
g = torch.Generator().manual_seed(0)
E = torch.randn(V, 16, generator=g, dtype=torch.float64) * 0.5
W1 = torch.randn(16, 64, generator=g, dtype=torch.float64) * 0.3
b1 = torch.zeros(64, dtype=torch.float64)
W2 = torch.randn(64, V, generator=g, dtype=torch.float64) * 0.1
b2 = torch.zeros(V, dtype=torch.float64)
params = [E, W1, b1, W2, b2]
xb, yb = get_batch(train_ids, 32, 1, torch.Generator().manual_seed(1))
xb, yb = xb[:, 0], yb[:, 0]                               # 32 (previous char, next char) pairs
B = len(xb)

# forward, keeping every intermediate
emb = E[xb]                    # (B, 16)
pre = emb @ W1 + b1            # (B, 64)
hid = torch.tanh(pre)          # (B, 64)
logit = hid @ W2 + b2          # (B, V)
probs = logit.softmax(-1)
loss_manual = -probs[torch.arange(B), yb].log().mean()

# backward, by hand: each line is one chain-rule step, from the loss back to the inputs
d_logit = probs.clone(); d_logit[torch.arange(B), yb] -= 1; d_logit /= B    # softmax − one_hot, averaged
d_W2 = hid.T @ d_logit
d_b2 = d_logit.sum(0)
d_hid = d_logit @ W2.T
d_pre = d_hid * (1 - hid ** 2)                                               # tanh' = 1 − tanh²
d_W1 = emb.T @ d_pre
d_b1 = d_pre.sum(0)
d_emb = d_pre @ W1.T
d_E = torch.zeros_like(E).index_add_(0, xb, d_emb)                          # rows we looked up get the grads

# the same with autograd
for t_ in params:
    t_.requires_grad_(True)
loss_auto = F.cross_entropy(torch.tanh(E[xb] @ W1 + b1) @ W2 + b2, yb)
loss_auto.backward()

banner("Hand-written backward vs autograd (float64)")
print(f"  loss: ours {loss_manual.item():.10f}   autograd {loss_auto.item():.10f}")
hand_rows = []
for name, mine, t_ in zip(["E", "W1", "b1", "W2", "b2"], [d_E, d_W1, d_b1, d_W2, d_b2], params):
    diff = (mine - t_.grad).abs().max().item()
    hand_rows.append({"name": name, "shape": list(t_.shape), "max_diff": diff})
    print(f"  d{name:3} {str(tuple(t_.shape)):10} max |ours − autograd| = {diff:.2e}")
R["hand_backprop"] = hand_rows

# %% [markdown]
# Identical to the last digit (differences of 1e-17, the rounding error of
# float64). **Backprop isn't magic, it's bookkeeping**:
# keep every intermediate from the forward pass, then walk backwards
# multiplying by the local derivative of each step. Autograd writes those
# lines for you, for any model, including a transformer.
#
# Note what the backward pass needed: `hid`, `emb`, `probs`, the values from
# the forward pass. That's why training uses much more memory than
# inference: every activation has to be **kept** until backward (lesson 12).
#
# Look at `d_E` too. Only the embedding rows of characters that appeared in
# the batch get a gradient; every other row gets exactly zero. Rare
# characters are rarely in a batch, so their embeddings are rarely trained.
# Lesson 08 measures this on a real model.
#
# ## Checking a real transformer
#
# A tiny Llama (same architecture as SmolLM2, 2 layers, 64 wide). Pick a few
# individual weights anywhere in the model, nudge each one by ±ε, and compare
# the measured change in loss with autograd's `.grad`.

# %%
model = tiny_llama(V, d=64, layers=2, heads=4, ctx=64).double()
x, y = get_batch(train_ids, 8, 64, torch.Generator().manual_seed(2))

# One catch: Llama's RMSNorm always computes in float32 for numerical safety, even when
# the model is float64. That float32 rounding (≈1e-7) is far bigger than our nudges change
# the loss by, so the check would fail. For this check only, compute the norms in float64.
import types


def rmsnorm_f64(self, h):
    return self.weight * h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)


for mod in model.modules():
    if type(mod).__name__.endswith("RMSNorm"):
        mod.forward = types.MethodType(rmsnorm_f64, mod)


def lm_loss(m):
    return F.cross_entropy(m(input_ids=x).logits.flatten(0, 1), y.flatten())


loss = lm_loss(model)
loss.backward()
banner(f"Finite differences vs autograd on a {n_params(model):,}-parameter Llama")
fd_rows = []
gen = torch.Generator().manual_seed(3)
named = dict(model.named_parameters())
for name in ["model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight", "model.layers.0.mlp.up_proj.weight",
             "model.layers.1.self_attn.o_proj.weight", "model.layers.1.input_layernorm.weight", "model.norm.weight"]:
    p_ = named[name]
    idx = tuple(torch.randint(s, (1,), generator=gen).item() for s in p_.shape)
    if name.startswith("model.embed"):
        idx = (x[0, 5].item(), idx[1])                     # a row that's actually used
    eps = 1e-6
    with torch.no_grad():
        orig = p_[idx].item()
        p_[idx] = orig + eps; lp = lm_loss(model).item()
        p_[idx] = orig - eps; lm = lm_loss(model).item()
        p_[idx] = orig
    fd, ag = (lp - lm) / (2 * eps), p_.grad[idx].item()
    fd_rows.append({"name": name, "index": list(idx), "fd": fd, "autograd": ag})
    print(f"  {name:42} {str(idx):12} finite diff {fd:+.8f}   autograd {ag:+.8f}")
R["finite_diff"] = fd_rows

# %% [markdown]
# Every weight agrees with autograd to between 7 and 9 significant digits
# (relative differences of 3e-9 to 9e-7), from the embedding to the final
# norm. Autograd's chain rule through attention, RoPE, SwiGLU and RMSNorm is
# right, and our check is good enough to see it.
#
# The RMSNorm detail is worth remembering. A naive version of this check
# fails, because Llama's norm quietly computes in float32 inside a float64
# model, and float32 rounding is bigger than the change a 1e-6 nudge makes. Real training code is full of choices like this: where to
# keep extra precision so that rounding doesn't eat the signal (lesson 12
# comes back to number formats).
#
# ## What does one step do?
#
# The gradient says which way is downhill *right here*. Take one step of
# size η against it: `w ← w − η·grad`, and measure the loss again, on the
# same batch and on a different batch. Try many step sizes.

# %%
model = tiny_llama(V, d=128, layers=4, heads=4, ctx=128)
x, y = get_batch(train_ids, 32, 128, torch.Generator().manual_seed(4))
x2, y2 = get_batch(train_ids, 32, 128, torch.Generator().manual_seed(5))
lossf = lambda m, a, b: F.cross_entropy(m(input_ids=a).logits.flatten(0, 1), b.flatten()).item()
model.zero_grad()
F.cross_entropy(model(input_ids=x).logits.flatten(0, 1), y.flatten()).backward()
W0 = {n: p_.detach().clone() for n, p_ in model.named_parameters()}
G = {n: p_.grad.clone() for n, p_ in model.named_parameters()}
gnorm = torch.sqrt(sum((g_ ** 2).sum() for g_ in G.values())).item()
base_same, base_other = lossf(model, x, y), lossf(model, x2, y2)

banner("Loss after ONE step of size η along −grad")
print(f"  before: same batch {base_same:.3f}   other batch {base_other:.3f}   (‖grad‖ = {gnorm:.2f})")
step_rows = []
with torch.no_grad():
    for eta in (0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100):
        for n, p_ in model.named_parameters():
            p_.copy_(W0[n] - eta * G[n])
        ls, lo = lossf(model, x, y), lossf(model, x2, y2)
        lin = base_same - eta * gnorm ** 2                        # what a straight line would predict
        step_rows.append({"eta": eta, "same": ls, "other": lo, "linear": lin})
        print(f"  η = {eta:6}  same batch {ls:7.3f}   other batch {lo:7.3f}   (straight-line guess {lin:8.3f})")
R["one_step"] = {"before_same": base_same, "before_other": base_other, "grad_norm": gnorm, "rows": step_rows}

fig, ax = plt.subplots(figsize=(6.5, 3.5))
ax.plot([r["eta"] for r in step_rows], [r["same"] for r in step_rows], "o-", label="same batch")
ax.plot([r["eta"] for r in step_rows], [r["other"] for r in step_rows], "o-", label="a different batch")
ax.axhline(base_same, ls="--", color="gray")
ax.set_xscale("log"); ax.set_xlabel("step size η"); ax.set_ylabel("loss after one step"); ax.legend()
ax.set_title("One gradient step: small helps, big hurts")
show_fig(fig, "07_one_step")

# %% [markdown]
# The model starts at 4.52, close to ln 97 = 4.57 (a fresh model guesses
# almost uniformly, lesson 06). The "straight-line guess" is what the
# gradient promises: loss falls by η·‖grad‖² if the slope stayed the same
# all the way.
#
# | η | 0.01 | 0.03 | 0.1 | 0.3 | 1 | 3 | 10 |
# |---|---|---|---|---|---|---|---|
# | same batch | 4.27 | 4.11 | 3.95 | **3.72** | 4.33 | 9.30 | 25.8 |
# | other batch | 4.28 | 4.11 | 3.95 | **3.70** | 4.30 | 9.24 | 25.6 |
# | straight line | 4.20 | 3.57 | 1.36 | −4.9 | −27 | −90 | −311 |
#
# Even at η = 0.01 the real drop (0.24) is only three-quarters of the
# promise (0.32), and by η = 0.1 the promise is absurd. The slope changes as
# soon as you move. The best single step is around η = 0.3; at η = 1 it
# overshoots (still below the start, barely); at η = 3 the loss *doubles*.
#
# The surprise is the second row. The step was computed on one batch, yet the
# other batch improves just as much, even slightly more at η = 0.3. At this
# point the model knows nothing, so the first thing every gradient says is
# "spaces and 'e' are common, 'Q' is rare", and that's true of every batch.
# Later in training, the batch-specific part of each gradient gets larger
# (that's the noise that lesson 09's learning-rate decay deals with).
#
# **Aha:**
# * The gradient is only a *local* promise. It overstates the drop even for
#   small steps, and past some size the step overshoots and the loss goes
#   *up*. One step that's 10× too big can undo everything.
# * A step computed on one batch lowers the loss on a *different* batch too.
#   That's the whole reason training works: the batches share structure, so
#   a step for one is (mostly) a step for all.
# * Choosing η is the subject of lesson 09. Choosing a smarter direction than
#   plain −grad is lesson 08.
#
# ## How expensive is backward?

# %%
device = pick_device()
model = tiny_llama(V, d=256, layers=6, heads=8, ctx=256).to(device)
x, y = get_batch(train_ids, 32, 256, torch.Generator().manual_seed(6), device)


def timeit(fn, n=10):
    fn(); sync(device)
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    sync(device)
    return (time.perf_counter() - t0) / n


def fwd():
    with torch.no_grad():
        model(input_ids=x)


def fwd_bwd():
    F.cross_entropy(model(input_ids=x).logits.flatten(0, 1), y.flatten()).backward()


tf, tb = timeit(fwd), timeit(fwd_bwd)
banner(f"Forward vs forward+backward ({n_params(model) / 1e6:.1f}M params, 32×256 tokens, {device})")
print(f"  forward only        {tf * 1000:7.1f} ms")
print(f"  forward + backward  {tb * 1000:7.1f} ms   ({tb / tf:.1f}× the forward)")
print("  -> in theory backward costs 2 forwards (one for the weights' grads, one for the inputs'),")
print("     so a training step ≈ 3× a forward pass: the famous '6 × parameters × tokens' FLOPs (lesson 13).")
print("     Small models often measure more than 3×: fixed costs (launching GPU kernels, moving")
print("     activations around) don't shrink with the model the way the matrix maths does.")
R["timing"] = {"device": device, "fwd_ms": tf * 1000, "fwd_bwd_ms": tb * 1000, "params": n_params(model)}

# %% [markdown]
# On this run (MPS, a 4.7M-parameter model), forward + backward took 4.0×
# a forward pass, so backward alone cost 3×. In theory backward is 2× the
# forward (it computes gradients for both the weights and the inputs: two
# matrix products for every one in the forward), so a whole step is 3
# forward passes' worth of arithmetic. That's where lesson 13's
# "6 × parameters × tokens" comes from (2 for forward, 4 for backward, per
# parameter per token). Small models measure more than the theory, because
# fixed costs like launching GPU kernels don't shrink with the model.

# %%
save_result("07_gradients", R)

# %% [markdown]
# ## What we know so far
#
# * `p.grad` = how much the loss changes per unit nudge of `p`, measured
#   exactly by backprop and checkable with finite differences.
# * For a language model, it starts as `softmax − one_hot` at the output and
#   flows backwards through every layer by the chain rule.
# * Backward needs every activation from the forward pass. In theory it costs
#   about twice the forward; on a small model we measured 3×.
# * One step along −grad helps if it's small and hurts if it's big, and it
#   helps other batches too.
#
# Next question: plain `w − η·grad` is what we just did. Real training uses
# **Adam**. What does it do differently, and does it matter?
#
# ## Try:
# * In the hand-written backward, replace `tanh` with ReLU. Which line changes?
# * Make ε in the finite-difference check bigger (1e-2) and smaller (1e-10). Why
#   does it get worse in *both* directions?
# * Run the finite-difference check in float32 instead of float64. How many digits survive?
# * In "one step", use the gradient of the *other* batch. Is it still a good
#   direction for this batch?
