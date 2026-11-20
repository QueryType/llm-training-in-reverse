# %% [markdown]
# # Lesson 13 — Bigger model or more data?
#
# Lesson 11 spent ten minutes on a 12M-parameter model. Was that the right
# size? With a fixed amount of compute you can train a big model on few
# tokens or a small model on many. Training cost is about
#
#     C ≈ 6 × N × D        (N parameters, D training tokens; lesson 07)
#
# so for a fixed budget C, doubling N halves D. **Scaling laws** are the
# measured answer to "which split gives the lowest loss?". The famous one,
# *Chinchilla* (Hoffmann et al., 2022), found: grow N and D together, about
# **20 tokens per parameter**. Earlier work (Kaplan et al., 2020) had said:
# mostly grow N. The difference came largely from one detail, which we
# reproduce below: the learning-rate schedule must end where the run ends.
#
# Questions: **can we find the best model size for a budget ourselves? Do we
# get anything like 20 tokens per parameter?**

# %%
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd())
from common import *
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

seed_all(0)
device = pick_device()
tok = bpe_tokenizer()
train_data, val_data = bpe_data()
V = len(tok)
BATCH, CTX = 16, 256                       # 4,096 tokens per step
TOK_PER_STEP = BATCH * CTX
R = {"batch": BATCH, "ctx": CTX}
print(f"device: {device}")

# %% [markdown]
# ## A family of models
#
# Same architecture, eight sizes. N counts *all* parameters: with tied
# embeddings, the output layer (width × 4,096) is real compute on every
# token, and for the smallest models it's most of the model.

# %%
SIZES = [(32, 2, 2), (48, 2, 2), (64, 2, 2), (96, 3, 3), (128, 4, 4), (192, 4, 4), (256, 6, 4), (384, 6, 6)]
family = []
for d, L, H in SIZES:
    n = n_params(tiny_llama(V, d=d, layers=L, heads=H, ctx=CTX))
    family.append({"d": d, "layers": L, "heads": H, "params": n})
    print(f"  width {d:3}, {L} layers: {n / 1e6:6.2f}M params")
R["family"] = family

# %% [markdown]
# ## The runs
#
# For each budget, train every size for exactly C / (6N) tokens, with its own
# warmup + cosine schedule ending at the last step. Sizes that would get fewer
# than 60 steps or more than 6,000 are skipped (too few steps to learn, or too long
# for a laptop). Learning rate: larger for small models, a common rule of
# thumb (lr ∝ 1/√width).

# %%
BUDGETS = [1e13, 4e13] + ([1.6e14] if FULL else [])


def lr_for(d):
    return 3e-3 * math.sqrt(64 / d) if d > 64 else 3e-3


def train_budget(cfg, budget, seed=0):
    steps = int(budget / (6 * cfg["params"] * TOK_PER_STEP))
    lr, warm = lr_for(cfg["d"]), max(10, steps // 20)
    m = tiny_llama(V, d=cfg["d"], layers=cfg["layers"], heads=cfg["heads"], ctx=CTX, seed=seed).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    g = torch.Generator().manual_seed(seed)
    for s in range(steps):
        f = min(1, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / steps)))
        for grp in opt.param_groups:
            grp["lr"] = lr * f
        x, y = get_batch(train_data, BATCH, CTX, g, device)
        F.cross_entropy(m(input_ids=x).logits.flatten(0, 1), y.flatten()).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
    return eval_loss(m, val_data, CTX, batch=32, n_batches=20, device=device), steps


runs = []
for C in BUDGETS:
    banner(f"Budget C = {C:.1e} FLOPs")
    for cfg in family:
        steps = int(C / (6 * cfg["params"] * TOK_PER_STEP))
        if not 60 <= steps <= 6000:
            continue
        with Timer() as t:
            loss, steps = train_budget(cfg, C)
        D = steps * TOK_PER_STEP
        runs.append({"budget": C, "params": cfg["params"], "d": cfg["d"], "tokens": D, "steps": steps,
                     "tokens_per_param": D / cfg["params"], "loss": loss, "seconds": t.s})
        print(f"  N = {cfg['params'] / 1e6:5.2f}M  D = {D / 1e6:6.1f}M tokens ({D / cfg['params']:6.0f} tokens/param)"
              f"  {steps:5} steps   val loss {loss:.3f}   ({t.s:.0f}s)")
R["runs"] = runs

# %% [markdown]
# Thirteen runs, 4.4 minutes in total. At the smaller budget (1e13 FLOPs):
#
# | N | 0.16M | 0.25M | 0.36M | 0.73M | 1.33M | 2.56M | 5.72M |
# |---|---|---|---|---|---|---|---|
# | tokens/param | 67 | 26 | 13 | 3 | 1 | 0.3 | 0.1 |
# | val loss | 3.24 | 3.14 | **3.14** | 3.36 | 3.70 | 4.21 | 5.49 |
#
# and at 4× the budget (4e13):
#
# | N | 0.36M | 0.73M | 1.33M | 2.56M | 5.72M | 12.2M |
# |---|---|---|---|---|---|---|
# | tokens/param | 52 | 13 | 4 | 1 | 0.2 | 0.03 |
# | val loss | 2.74 | **2.63** | 2.77 | 3.03 | 4.01 | 4.59 |
#
# Both are U-shapes. With the same FLOPs, the 5.7M model is far worse than a
# model 16× smaller: it only got 71 steps. And the bottom moves right as the
# budget grows: the best measured size doubles from 0.36M to 0.73M.
#
# ## Find the bottom of each curve
#
# For each budget, loss vs log(N) is roughly a parabola (an "IsoFLOP curve").
# Fit one and read off the size at its minimum.

# %%
banner("Best model size per budget")
optima = []
for C in BUDGETS:
    rs = [r for r in runs if r["budget"] == C]
    x = np.log10([r["params"] for r in rs])
    yv = np.array([r["loss"] for r in rs])
    a, b, c = np.polyfit(x, yv, 2)
    n_opt = 10 ** (-b / (2 * a)) if a > 0 else rs[int(np.argmin(yv))]["params"]
    d_opt = C / (6 * n_opt)
    optima.append({"budget": C, "n_opt": n_opt, "d_opt": d_opt, "ratio": d_opt / n_opt, "fit": [a, b, c],
                   "best_measured": min(rs, key=lambda r: r["loss"])["params"]})
    print(f"  C = {C:.1e}: best N ≈ {n_opt / 1e6:.2f}M params, D ≈ {d_opt / 1e6:.1f}M tokens"
          f"  → {d_opt / n_opt:.0f} tokens per parameter   (best measured run: {optima[-1]['best_measured'] / 1e6:.2f}M)")
R["optima"] = optima
if len(optima) >= 2:
    slope = np.polyfit(np.log10([o["budget"] for o in optima]), np.log10([o["n_opt"] for o in optima]), 1)[0]
    print(f"\n  N_opt grows like C^{slope:.2f}   (Chinchilla: C^0.5, i.e. N and D grow equally; Kaplan: C^0.73)")
    R["slope"] = slope

# %%
fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
for C, o in zip(BUDGETS, optima):
    rs = [r for r in runs if r["budget"] == C]
    xs = np.log10([r["params"] for r in rs])
    line = axs[0].plot([r["params"] for r in rs], [r["loss"] for r in rs], "o", label=f"C = {C:.0e}")[0]
    xx = np.linspace(xs.min(), xs.max(), 50)
    axs[0].plot(10 ** xx, np.polyval(o["fit"], xx), "--", color=line.get_color(), lw=1)
axs[0].set_xscale("log"); axs[0].set_xlabel("parameters N"); axs[0].set_ylabel("val loss")
axs[0].set_title("IsoFLOP curves: same compute, different sizes"); axs[0].legend()
axs[1].scatter([r["budget"] * (1 + 0.05 * i) for i, r in enumerate(runs)], [r["loss"] for r in runs],
               c=np.log10([r["params"] for r in runs]), cmap="viridis")
axs[1].set_xscale("log"); axs[1].set_xlabel("compute C (FLOPs)"); axs[1].set_ylabel("val loss")
axs[1].set_title("More compute, lower loss (colour = size)")
show_fig(fig, "13_scaling")

# %% [markdown]
# The fitted optima: **0.32M parameters at 1e13** (16 tokens per parameter)
# and **0.58M at 4e13** (20 tokens per parameter). With two budgets, the
# best size grows like C^0.42, between Chinchilla's C^0.5 and Kaplan's
# C^0.73... and with two points, "grows like" is a generous description.
#
# One honest caveat about the right-hand side of the curves: a 5.7M model on
# a 1e13 budget gets only 71 steps, part of them warmup. Some of its high loss
# is "too few optimizer steps", not just "too few tokens per parameter". Real
# scaling studies use much larger budgets, where even the biggest models get
# thousands of steps.
#
# ## The schedule detail that changed the answer
#
# Kaplan et al. trained models with a long schedule and read off the loss at
# intermediate points, so mid-run losses were measured with the learning rate
# still high. Lesson 09 showed that the final drop comes from the decay itself.
# Take one size at the 4e13 budget and compare: a schedule sized for this
# budget, vs a schedule sized for 4× more steps, stopped at the budget.

# %%
cfg = min((c for c in family), key=lambda c: abs(c["params"] - optima[-1]["n_opt"]))
C = BUDGETS[1]
steps = int(C / (6 * cfg["params"] * TOK_PER_STEP))
banner(f"N = {cfg['params'] / 1e6:.2f}M at C = {C:.0e}: matched vs too-long schedule")


def run_sched(total_steps, stop):
    lr, warm = lr_for(cfg["d"]), max(10, total_steps // 20)
    m = tiny_llama(V, d=cfg["d"], layers=cfg["layers"], heads=cfg["heads"], ctx=CTX).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    g = torch.Generator().manual_seed(0)
    for s in range(stop):
        f = min(1, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / total_steps)))
        for grp in opt.param_groups:
            grp["lr"] = lr * f
        x, y = get_batch(train_data, BATCH, CTX, g, device)
        F.cross_entropy(m(input_ids=x).logits.flatten(0, 1), y.flatten()).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
    return eval_loss(m, val_data, CTX, batch=32, n_batches=20, device=device)


matched, too_long = run_sched(steps, steps), run_sched(4 * steps, steps)
print(f"  schedule ends at the budget:        {matched:.3f}")
print(f"  schedule sized for 4×, stopped early: {too_long:.3f}   (+{too_long - matched:.3f})")
R["schedule_test"] = {"params": cfg["params"], "budget": C, "steps": steps, "matched": matched, "too_long": too_long}

# %% [markdown]
# Same model (0.73M), same 2,242 steps, same data. The only difference is
# whether the cosine schedule had reached its end: 2.629 when it had, 2.695
# when it was sized for 4× the steps and still at a high learning rate.
# **+0.066 nats**, just from the schedule. In Kaplan-style measurements, every
# run except the longest one is caught mid-schedule like this, and the
# penalty is largest for the short runs of big models. That made big models
# look better than they were.
#
# **Aha:**
# * For a fixed compute budget there is a best model size: too small can't
#   absorb the data, too big doesn't see enough of it. Near the bottom it's
#   forgiving (2× off costs ~0.1–0.2 nats here); 10× off costs a nat or more.
# * The best size grows with the budget, and the tokens-per-parameter ratio at
#   the optimum lands in Chinchilla's neighbourhood (~20), with models
#   10,000× smaller than theirs on a single narrow dataset. Treat it with care:
#   two budgets, one seed, and parabola fits on lopsided curves. But the method
#   is exactly the one used to plan real training runs.
# * Comparing models fairly needs schedules that end where the run ends. A
#   run caught mid-schedule looks worse than it is, and that bias pushes you
#   towards bigger models.
#
# And lesson 11? Its 12M model trained on 22M tokens: 1.8 tokens per
# parameter, far to the "too big" side by this measure. Its compute,
# 6 × 12.2M × 22M ≈ 1.6e15 FLOPs, would by ~20 tokens/param suggest a model of
# about 4M parameters trained on ~70M tokens. (On our hardware small models
# run fewer FLOP/s, lesson 12, so "same compute" isn't quite "same ten minutes".) (Real labs often
# go the other way on purpose, training *past* the optimum, because a smaller
# model that's been trained longer is cheaper to run afterwards.)
#
# ## What we know so far
#
# * Compute is the budget: C ≈ 6ND, and every split of it between N and D is
#   a different model.
# * The best split can be measured with a handful of small runs (IsoFLOP
#   curves), and here it lands near Chinchilla's ~20 tokens per parameter.
# * Fair comparisons need schedules that end where the run ends.
#
# Our model from lesson 11 writes stories, but it's a *base* model, like
# SmolLM2-135M in lesson 01: it continues text. The last step closes the
# circle: **can we turn it into a chat model with lesson 03's SFT?**

# %%
save_result("13_scaling_laws", R)

# %% [markdown]
# ## Try:
# * `FULL=1` adds a third budget (about 25 more minutes). Does the slope of N_opt vs C settle?
# * Use only non-embedding parameters for N (Kaplan's choice). Does the optimum move?
# * Train lesson 11 again at the size this lesson recommends for its compute
#   (≈ 6 × 12.2M × 22M ≈ 1.6e15 FLOPs). Is the loss lower?
# * Repeat one point with 3 seeds. How much of the curve is noise?
