# %% [markdown]
# # Lesson 08 — What does `optimizer.step()` do?
#
# Lesson 07 gave us the gradient: which way is downhill for every weight.
# The **optimizer** decides how far to move each weight, and in exactly which
# direction. The simplest rule is plain gradient descent:
#
#     w ← w − lr · grad                              (SGD)
#
# Every LLM you've heard of is trained with something else: **AdamW**.
#
# Questions: **what does Adam do differently? Can we write it in a few lines
# and match PyTorch exactly? Why does it win, and what does it cost?**
#
# The plan: write four optimizers by hand and check them against
# `torch.optim`; race them on a 2D valley where we can watch every step; race
# them again on a real character model; then look inside the trained model at
# how big the gradients and steps actually are.

# %%
import os, sys, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd())
from common import *
import torch.nn.functional as F
import matplotlib.pyplot as plt

seed_all(0)
device = pick_device()
ctok, train_ids, val_ids = char_data()
V = len(ctok)
R = {}
print(f"device: {device}")

# %% [markdown]
# ## Four optimizers, by hand
#
# * **SGD**: step along −grad.
# * **Momentum**: keep a running average of past gradients (a velocity) and step
#   along that. Zig-zags cancel out; consistent directions build up speed.
# * **Adam**: momentum, *plus* a running average of each weight's squared
#   gradient. Divide by its square root, so every weight takes steps of about
#   the same size, whatever the scale of its gradient.
# * **AdamW**: Adam, plus weight decay applied directly to the weights
#   (shrink every weight a little each step), not mixed into the gradient.

# %%
class SGD:
    def __init__(self, params, lr, momentum=0.0):
        self.params, self.lr, self.mu = list(params), lr, momentum
        self.buf = [torch.zeros_like(p) for p in self.params] if momentum else None

    @torch.no_grad()
    def step(self):
        for i, p in enumerate(self.params):
            g = p.grad
            if self.mu:
                self.buf[i].mul_(self.mu).add_(g)          # velocity = μ·velocity + grad
                g = self.buf[i]
            p.sub_(self.lr * g)


class AdamW:
    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.params, self.lr, (self.b1, self.b2), self.eps, self.wd = list(params), lr, betas, eps, weight_decay
        self.m = [torch.zeros_like(p) for p in self.params]     # running mean of grads
        self.v = [torch.zeros_like(p) for p in self.params]     # running mean of grads²
        self.t = 0

    @torch.no_grad()
    def step(self):
        self.t += 1
        for p, m, v in zip(self.params, self.m, self.v):
            g = p.grad
            p.mul_(1 - self.lr * self.wd)                       # decoupled weight decay (the "W")
            m.mul_(self.b1).add_((1 - self.b1) * g)
            v.mul_(self.b2).add_((1 - self.b2) * g * g)
            m_hat = m / (1 - self.b1 ** self.t)                 # undo the bias of starting at 0
            v_hat = v / (1 - self.b2 ** self.t)
            p.sub_(self.lr * m_hat / (v_hat.sqrt() + self.eps))

    def zero_grad(self):
        for p in self.params:
            p.grad = None


SGD.zero_grad = AdamW.zero_grad

# %% [markdown]
# ## Do they match PyTorch?
#
# Train two copies of the same model on the same batches for 20 steps, one
# with ours and one with `torch.optim`, and compare every weight.

# %%
def lm_step(model, opt, x, y):
    loss = F.cross_entropy(model(input_ids=x).logits.flatten(0, 1), y.flatten())
    loss.backward()
    opt.step()
    opt.zero_grad()
    return loss.item()


banner("Our optimizers vs torch.optim, 20 steps")
match_rows = []
for name, ours_fn, ref_fn in (
    ("SGD", lambda p: SGD(p, lr=0.1), lambda p: torch.optim.SGD(p, lr=0.1)),
    ("SGD+momentum", lambda p: SGD(p, lr=0.05, momentum=0.9), lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9)),
    ("AdamW", lambda p: AdamW(p, lr=1e-3, weight_decay=0.1), lambda p: torch.optim.AdamW(p, lr=1e-3, weight_decay=0.1)),
):
    a = tiny_llama(V, d=64, layers=2, heads=4, ctx=64)
    b = copy.deepcopy(a)
    oa, ob = ours_fn(a.parameters()), ref_fn(b.parameters())
    g = torch.Generator().manual_seed(0)
    for _ in range(20):
        x, y = get_batch(train_ids, 16, 64, g)
        la, lb = lm_step(a, oa, x, y), lm_step(b, ob, x, y)
    diff = max((pa - pb).abs().max().item() for pa, pb in zip(a.parameters(), b.parameters()))
    match_rows.append({"name": name, "max_diff": diff, "loss_ours": la, "loss_torch": lb})
    print(f"  {name:13}  loss after 20 steps: ours {la:.5f}  torch {lb:.5f}   max |weight difference| {diff:.1e}")
R["match"] = match_rows

# %% [markdown]
# Same numbers, up to float32 rounding (the largest weight difference is
# 1.2e-7, one unit in the last place of a float32 near 1).
# `torch.optim.AdamW` is a faster, fused version of these ten lines. When a
# training recipe says "AdamW, β = (0.9, 0.95), weight decay 0.1", those are
# the three knobs in `__init__` above, nothing more.
#
# ## A race in two dimensions
#
# A loss surface shaped like a long, narrow valley: steep across, gentle
# along. That's typical for neural networks, where some directions are a
# thousand times steeper than others.

# %%
def valley(w):
    x, y = w[..., 0], w[..., 1]
    return 0.02 * x ** 2 + 5.0 * y ** 2          # 250× steeper across (y) than along (x)


def race(make_opt, steps=60, start=(-9.0, 2.0)):
    w = torch.tensor(start, requires_grad=True)
    opt = make_opt([w])
    path = [w.detach().tolist()]
    for _ in range(steps):
        valley(w).backward()
        opt.step(); opt.zero_grad()
        path.append(w.detach().tolist())
    return path


# SGD diverges for lr > 0.2 here (the steep direction overshoots), so 0.18 is as fast as it can go.
paths = {"SGD": race(lambda p: SGD(p, lr=0.18)),
         "SGD+momentum": race(lambda p: SGD(p, lr=0.04, momentum=0.9)),
         "Adam": race(lambda p: AdamW(p, lr=0.5))}
banner("Where is each optimizer after 60 steps? (minimum at 0, 0)")
for k, pth in paths.items():
    print(f"  {k:13} → ({pth[-1][0]:+.3f}, {pth[-1][1]:+.3f})   loss {valley(torch.tensor(pth[-1])).item():.4f}")
R["race2d"] = {"paths": paths, "fn": "0.02*x^2 + 5*y^2"}

fig, ax = plt.subplots(figsize=(8, 3.2))
gx, gy = torch.meshgrid(torch.linspace(-10, 2, 200), torch.linspace(-2.5, 2.5, 100), indexing="xy")
ax.contour(gx, gy, valley(torch.stack([gx, gy], -1)), levels=20, colors="lightgray", linewidths=0.6)
for k, pth in paths.items():
    p = torch.tensor(pth)
    ax.plot(p[:, 0], p[:, 1], ".-", label=k, ms=3)
ax.legend(); ax.set_title("60 steps on a narrow valley")
show_fig(fig, "08_race2d")

# %% [markdown]
# After 60 steps from (−9, 2):
#
# | optimizer | where it is | loss |
# |---|---|---|
# | SGD (lr 0.18) | (−5.83, 0.00) | 0.68 |
# | momentum (lr 0.04) | (−3.54, −0.04) | 0.26 |
# | Adam (lr 0.5) | (−0.34, +0.05) | 0.013 |
#
# SGD has one learning rate for both directions, and the steep direction
# sets the limit: above lr = 0.2 it overshoots across the valley and blows up.
# At lr 0.18 it bounces across the steep direction (y flips sign every step
# while shrinking), settles it, and then barely crawls along the gentle one:
# 3 units of progress in 60 steps. Momentum builds up speed along the valley,
# but it overshoots across it too and needs time to calm down. Adam divides
# each direction's gradient by its own typical size, so both directions move
# at about the same pace (its first steps are exactly 0.5 in x *and* y), and
# it gets close to the minimum in 60 steps. (Scale the valley by any factor:
# Adam's path barely changes, SGD's does.)
#
# ## The real race: a character model on TinyStories
#
# Same model, same batches, 600 steps. For a fair race, each optimizer gets
# its best learning rate from a small sweep.

# %%
STEPS = 1500 if FULL else 600


def train_with(make_opt, steps=STEPS, eval_every=50, seed=0):
    model = tiny_llama(V, d=128, layers=4, heads=4, ctx=128, seed=seed).to(device)
    opt = make_opt(model.parameters())
    g = torch.Generator().manual_seed(seed)
    curve = []
    for s in range(1, steps + 1):
        x, y = get_batch(train_ids, 32, 128, g, device)
        lm_step(model, opt, x, y)
        if s % eval_every == 0:
            curve.append((s, eval_loss(model, val_ids, 128, n_batches=8, device=device)))
    return model, opt, curve


banner(f"Each optimizer at 4 learning rates, {STEPS} steps each (final val loss)")
# Sweep at the full length: a learning rate that looks best after 200 steps can blow up by step 600.
sweep = {"SGD": [0.01, 0.03, 0.1, 0.3], "SGD+momentum": [0.003, 0.01, 0.03, 0.1], "Adam": [3e-4, 1e-3, 3e-3, 1e-2]}
make = {"SGD": lambda lr: (lambda p: SGD(p, lr)), "SGD+momentum": lambda lr: (lambda p: SGD(p, lr, momentum=0.9)),
        "Adam": lambda lr: (lambda p: AdamW(p, lr))}
best, sweep_rows, curves = {}, [], {}
for k, lrs in sweep.items():
    runs = []
    for lr in lrs:
        model_k, opt_k, c = train_with(make[k](lr))
        v = c[-1][1] if math.isfinite(c[-1][1]) else float("inf")
        runs.append((v, lr, c, model_k, opt_k))
        sweep_rows.append({"opt": k, "lr": lr, "val": v, "curve": c})
    v, best[k], curves[k], model_k, opt_k = min(runs, key=lambda r: r[0])
    print(f"  {k:13} " + "  ".join(f"lr {r[1]:g}: {r[0]:.3f}" for r in runs) + f"   → best {best[k]:g}")
    if k == "Adam":
        adam_model, adam_opt = model_k, opt_k
R["lr_sweep"] = sweep_rows
R["race"] = {"curves": curves, "best_lr": best, "steps": STEPS}

fig, ax = plt.subplots(figsize=(7, 3.6))
for k, c in curves.items():
    ax.plot(*zip(*c), label=f"{k} (lr {best[k]:g})")
ax.set_xlabel("step"); ax.set_ylabel("val loss (nats/char)"); ax.legend(); ax.set_title("Same model, same data")
show_fig(fig, "08_race")

# %% [markdown]
# | optimizer | lr tried | best lr | final val loss |
# |---|---|---|---|
# | SGD | 0.01, 0.03, 0.1, 0.3 | 0.03 | 2.39 |
# | SGD + momentum | 0.003, 0.01, 0.03, 0.1 | 0.01 | 2.05 |
# | Adam | 3e-4, 1e-3, 3e-3, 1e-2 | 1e-3 | **1.16** |
#
# Not close. Momentum after 600 steps is about where Adam was after 100
# (2.04). SGD at its *best* learning rate ends higher than that. And SGD's
# two larger learning rates both end at ~3.06 nats, the unigram level from
# lesson 06: it got stuck on "letter frequencies", the same failure lesson 09
# will see for Adam at too-high learning rates.
#
# The sweep matters for fairness. With one shared learning rate, any
# comparison mostly measures which optimizer that lr happens to suit.
#
# ## Why Adam wins
#
# Look at the gradients of the trained model: how big are they, tensor by
# tensor? And how big are the steps Adam actually takes?

# %%
x, y = get_batch(train_ids, 32, 128, torch.Generator().manual_seed(9), device)
F.cross_entropy(adam_model(input_ids=x).logits.flatten(0, 1), y.flatten()).backward()
banner("Gradient size vs Adam's step size")
grads, updates, scale_rows = [], [], []
t = adam_opt.t
for (name, p), m, v in zip(adam_model.named_parameters(), adam_opt.m, adam_opt.v):
    upd = best["Adam"] * (m / (1 - 0.9 ** t)) / ((v / (1 - 0.999 ** t)).sqrt() + 1e-8)
    grads.append(p.grad.abs().flatten().cpu()); updates.append(upd.abs().flatten().cpu())
    scale_rows.append({"name": name.replace("model.", ""), "grad": p.grad.abs().median().item(),
                       "update": upd.abs().median().item()})
G_all, U_all = torch.cat(grads), torch.cat(updates)
qs = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95])
gq, uq = torch.quantile(G_all[G_all > 0], qs), torch.quantile(U_all[U_all > 0], qs)
print("  across all individual weights:   5%       25%      50%      75%      95%")
print("  |gradient|                  " + " ".join(f"{v:.1e}" for v in gq.tolist()))
print("  |SGD step| (lr × grad)      " + " ".join(f"{v:.1e}" for v in (best["SGD"] * gq).tolist()))
print("  |Adam step|                 " + " ".join(f"{v:.1e}" for v in uq.tolist()))
print(f"\n  middle 90% of gradients spans a factor of {gq[-1] / gq[0]:.0f};"
      f" of Adam's steps, a factor of {uq[-1] / uq[0]:.0f}")
emb_g = adam_model.get_input_embeddings().weight.grad.abs().mean(1).cpu()
freq = torch.bincount(x.flatten().cpu(), minlength=V).float()
common, rare = freq.argsort(descending=True)[:5], [i for i in freq.argsort().tolist() if freq[i] > 0][:5]
print(f"  embedding rows: common chars {[ctok.chars[i] for i in common.tolist()]} mean |grad| {emb_g[common].mean():.1e},"
      f" rare chars {[ctok.chars[i] for i in rare]} {emb_g[rare].mean():.1e}")
R["scales"] = {"per_tensor": scale_rows, "quantiles": qs.tolist(), "grad_q": gq.tolist(), "adam_q": uq.tolist(),
               "sgd_lr": best["SGD"]}
R["emb_grad"] = {"common": [ctok.chars[i] for i in common.tolist()], "common_mean": emb_g[common].mean().item(),
                 "rare": [ctok.chars[i] for i in rare], "rare_mean": emb_g[rare].mean().item()}

# %% [markdown]
# Two things to read here.
#
# **Size.** SGD's learning rate is capped by its *largest* gradients: any
# higher and those weights overshoot (the sweep found the limit at 0.03). So
# the median weight moves lr × median gradient = 0.03 × 2.7e-4 ≈ 8e-6 per
# step. Adam's median step is 1.5e-4, about **18× further**, while the run
# stays stable. Adam's step is `lr · m̂/√v̂`, at most about `lr` (here 1e-3);
# the typical step is 0.15 × lr, because gradients that change sign from step
# to step partly cancel in the running mean m̂.
#
# **Spread.** Gradients vary a lot from weight to weight: the middle 90%
# spans a factor of 77. Adam's steps span 38. The embedding rows show why
# this matters. Rows of common characters (space, e, a, t, h) get a mean
# gradient of 6.6e-3, rows of rare ones (N, C, z, -, W) 3.4e-4: **19×
# smaller**, simply because they appear less often. With SGD, rare characters
# would learn 19× slower. Adam's division by √v̂ evens that out.
#
# That's why Adam's learning rate means roughly the same thing across very
# different models, and why it's the default for LLMs.
#
# ## What it costs: memory

# %%
banner("Memory per parameter, fp32 training")
n = n_params(adam_model)
for name, extra in (("SGD", 0), ("SGD+momentum", 1), ("Adam / AdamW", 2)):
    bpp = 4 * (2 + extra)
    print(f"  {name:13} weights + grads + {extra} state copies = {bpp:2} bytes/param"
          f"  → 135M model: {bpp * 134.5e6 / 1e9:4.1f} GB,  7B model: {bpp * 7e9 / 1e9:5.0f} GB")
state = sum(t_.numel() for t_ in adam_opt.m + adam_opt.v)
print(f"\n  our model: {n:,} params, Adam state: {state:,} numbers (= 2 × params)")
print("  (plus activations, which lesson 12 measures. Mixed precision and 8-bit optimizers shrink these.)")
R["memory"] = {"params": n, "adam_state": state}

# %% [markdown]
# **Aha:**
# * Optimizers are short. Ours match PyTorch to float32 rounding.
# * Adam's trick is per-weight step sizes. It fixes the "one learning rate
#   can't fit all directions" problem that cripples SGD on these models.
# * The price is 2 extra numbers per weight. For fp32 training that's
#   16 bytes per parameter before activations, which is why a 7B model
#   needs ~112 GB to train but only 14 GB to run in bf16.
#
# ## What we know so far
#
# * The optimizer turns gradients into steps. Plain SGD uses one step size for
#   all weights, and the steepest direction caps it.
# * Adam gives each weight its own scale (√v̂) and smooths the direction (m̂).
#   On our character model it beats SGD by 1.2 nats at the same step count.
# * It costs two extra numbers per weight.
#
# Next question: Adam still has one knob, `lr`, and the sweep showed a run
# can be ruined by a 3× change in it. **How sensitive is training to the
# learning rate, and why does every recipe change it over time?**

# %%
save_result("08_optimizer", R)

# %% [markdown]
# ## Try:
# * Set Adam's β2 to 0.9 or 0.99. Many LLM recipes use 0.95. Does it change the race?
# * Remove the bias correction (`m_hat = m`). What happens in the first steps?
# * Implement **sign descent** (`p -= lr * g.sign()`): Adam with β1 = β2 = 0 is almost
#   this. How close does it get?
# * Turn weight decay up to 1.0 in AdamW and train longer. Watch the weight norms.
