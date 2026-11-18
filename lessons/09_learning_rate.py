# %% [markdown]
# # Lesson 09 — Why does the learning rate matter so much?
#
# Lesson 07: one gradient step helps if it's small and hurts if it's big.
# Lesson 08: Adam makes every weight move about `lr` per step. So `lr` is the
# single most important number in a training run. Every real recipe also
# *changes* it during training: a warmup at the start, a decay at the end.
#
# Questions: **how sensitive is training to the learning rate? What does
# "diverging" look like? Why warm up, why decay, and what does gradient
# clipping rescue?**
#
# Same setup as lesson 08: a 0.8M-parameter character model on TinyStories,
# AdamW, 600 steps per run (12 seconds on an M4 Pro).

# %%
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd())
from common import *
import torch.nn.functional as F
import matplotlib.pyplot as plt

seed_all(0)
device = pick_device()
ctok, train_ids, val_ids = char_data()
V = len(ctok)
STEPS = 1500 if FULL else 600
R = {"steps": STEPS}
print(f"device: {device}")


def run(lr_at, steps=STEPS, clip=None, seed=0, eval_every=50, d=128, layers=4):
    """Train one model. lr_at(step) gives the learning rate. Returns a log dict."""
    model = tiny_llama(V, d=d, layers=layers, heads=4, ctx=128, seed=seed).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr_at(0), weight_decay=0.0)
    g = torch.Generator().manual_seed(seed)
    log = {"train": [], "gnorm": [], "lr": [], "val_step": [], "val": []}
    for s in range(steps):
        for grp in opt.param_groups:
            grp["lr"] = lr_at(s)
        x, y = get_batch(train_ids, 32, 128, g, device)
        loss = F.cross_entropy(model(input_ids=x).logits.flatten(0, 1), y.flatten())
        loss.backward()
        # clip_grad_norm_ returns the norm *before* clipping; max_norm=inf just measures it
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), clip or float("inf")).item()
        opt.step(); opt.zero_grad()
        lv = loss.item()
        log["train"].append(lv if math.isfinite(lv) else None)
        log["gnorm"].append(gn if math.isfinite(gn) else None)
        log["lr"].append(lr_at(s))
        if (s + 1) % eval_every == 0:
            v = eval_loss(model, val_ids, 128, n_batches=8, device=device)
            log["val_step"].append(s + 1); log["val"].append(v if math.isfinite(v) else None)
    log["final"] = log["val"][-1]
    return log


def const(lr):
    return lambda s: lr

# %% [markdown]
# ## The sweep
#
# The same model at seven constant learning rates, each 3× apart.

# %%
banner(f"Constant learning rate, {STEPS} steps (final val loss, nats/char)")
LRS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
sweep = {}
for lr in LRS:
    with Timer() as t:
        sweep[lr] = run(const(lr))
    f = sweep[lr]["final"]
    spikes = sum(1 for a, b in zip(sweep[lr]["train"], sweep[lr]["train"][1:]) if a and b and b > a + 0.5)
    print(f"  lr {lr:7g}  final val {f if f else float('nan'):6.3f}   loss jumps > 0.5 nats: {spikes:3}   ({t.s:.0f}s)")
R["sweep"] = {str(k): v for k, v in sweep.items()}
best_lr = min(LRS, key=lambda lr: sweep[lr]["final"] or 99)
print(f"  -> best constant lr: {best_lr:g}")

fig, axs = plt.subplots(1, 2, figsize=(12, 3.6))
axs[0].plot(LRS, [sweep[lr]["final"] or float("nan") for lr in LRS], "o-")
axs[0].set_xscale("log"); axs[0].set_xlabel("learning rate"); axs[0].set_ylabel("final val loss")
axs[0].set_title("Too cold, just right, too hot")
for lr in LRS:
    axs[1].plot(sweep[lr]["train"], label=f"{lr:g}", lw=0.8)
axs[1].set_ylim(0.9, 4.8); axs[1].set_xlabel("step"); axs[1].set_ylabel("train loss"); axs[1].legend(fontsize=7, ncol=2)
show_fig(fig, "09_sweep")

# %% [markdown]
# A U-shape, with a fairly narrow bottom. Too small: the model learns, but
# slowly. Too big: the loss goes down at first, then gets stuck high or
# jumps around (each "jump" is a step that overshot, lesson 07's picture
# happening for real). Within the good range, a 3× change in lr changes the
# final loss a little; outside it, a 3× change ruins the run.
#
# ## Warmup: why start slow?
#
# At step 0, Adam knows nothing about the gradients' typical size (its
# running averages start at zero), and the random model's gradients are
# unusually large and chaotic. A full-size step right away can push the
# model somewhere it never recovers from. **Warmup** ramps the lr up from ~0
# over the first steps.

# %%
HOT = best_lr * 3          # a learning rate that's a bit too hot without help
WARM = 60


def warmup(lr, n=WARM):
    return lambda s: lr * min(1.0, (s + 1) / n)


banner(f"lr {HOT:g}: no warmup vs {WARM}-step warmup (3 seeds each)")
warm_rows = []
for seed in range(3):
    a, b = run(const(HOT), seed=seed), run(warmup(HOT), seed=seed)
    warm_rows.append({"seed": seed, "none": a, "warmup": b})
    print(f"  seed {seed}:  no warmup {a['final']:.3f}   warmup {b['final']:.3f}"
          f"   (loss at step 50: {a['train'][49]:.2f} vs {b['train'][49]:.2f})")
R["warmup"] = {"lr": HOT, "steps": WARM, "runs": warm_rows}

# %% [markdown]
# ## Decay: why end slow?
#
# Late in training, the noise of each batch's gradient matters more than
# its direction: a big step mostly jumps around the minimum. Lowering the lr
# lets the model settle into it. Four schedules, same peak lr, same warmup:
#
# * **constant**: never decays
# * **cosine**: smooth decay to 10% of peak (the classic LLM schedule)
# * **linear**: straight line down to 10%
# * **WSD** (warmup-stable-decay): constant, then a quick decay over the last 20%.
#   Used by many recent models, because you can stop or extend training at any
#   point before the decay.

# %%
PEAK = best_lr * 2


def cosine(lr, steps=STEPS, warm=WARM, floor=0.1):
    return lambda s: lr * min(1, (s + 1) / warm) * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * s / steps)))


def linear(lr, steps=STEPS, warm=WARM, floor=0.1):
    return lambda s: lr * min(1, (s + 1) / warm) * (1 - (1 - floor) * s / steps)


def wsd(lr, steps=STEPS, warm=WARM, decay_frac=0.2, floor=0.1):
    d0 = int(steps * (1 - decay_frac))
    return lambda s: lr * min(1, (s + 1) / warm) * (1 if s < d0 else 1 - (1 - floor) * (s - d0) / (steps - d0))


banner(f"Schedules at peak lr {PEAK:g}")
sched = {}
for name, fn in (("constant", warmup(PEAK)), ("cosine", cosine(PEAK)), ("linear", linear(PEAK)), ("WSD", wsd(PEAK))):
    sched[name] = run(fn)
    l = sched[name]
    at = min(range(len(l["val_step"])), key=lambda i: abs(l["val_step"][i] - 0.8 * STEPS))
    print(f"  {name:9} val at 80% of training {l['val'][at]:.3f}   final {l['final']:.3f}")
R["schedules"] = {"peak": PEAK, "runs": sched}

fig, axs = plt.subplots(1, 2, figsize=(12, 3.4))
for name, l in sched.items():
    axs[0].plot(l["lr"], label=name)
    axs[1].plot(l["val_step"], l["val"], "o-", ms=3, label=name)
axs[0].set_title("learning rate schedule"); axs[0].legend()
axs[1].set_title("val loss"); axs[1].set_ylim(None, sched["constant"]["val"][2])
show_fig(fig, "09_schedules")

# %% [markdown]
# Watch WSD: it trails cosine for most of the run (its lr is higher), then
# catches up or overtakes in the last 20%, when it decays. **The loss drop at
# the end comes from the decay itself**, not from extra steps. That's why a
# run's "loss so far" before the decay is a pessimistic estimate of where it
# will finish.
#
# ## Gradient clipping: a seat belt
#
# Sometimes one batch produces a huge gradient (an unusual sequence, or the
# model wandering into a steep region). One huge step can undo hours of
# training. **Clipping** rescales the whole gradient whenever its norm exceeds
# a limit (1.0 is the common choice), so no single step can be enormous.

# %%
TOO_HOT = best_lr * 10
banner(f"lr {TOO_HOT:g} (10× the best), with and without clipping at 1.0")
clip_runs = {"no clipping": run(warmup(TOO_HOT)), "clip at 1.0": run(warmup(TOO_HOT), clip=1.0)}
for name, l in clip_runs.items():
    g_ = [v for v in l["gnorm"] if v]
    print(f"  {name:12} final val {l['final'] or float('nan'):.3f}   grad norm: median {sorted(g_)[len(g_) // 2]:.2f}, max {max(g_):.1f}")
R["clipping"] = {"lr": TOO_HOT, "runs": clip_runs}

fig, axs = plt.subplots(1, 2, figsize=(12, 3.2))
for name, l in clip_runs.items():
    axs[0].plot(l["train"], lw=0.8, label=name)
    axs[1].plot(l["gnorm"], lw=0.8, label=name)
axs[0].set_title("train loss"); axs[1].set_title("gradient norm (before clipping)"); axs[1].set_yscale("log")
axs[0].legend()
show_fig(fig, "09_clipping")

# %% [markdown]
# **Aha:**
# * The learning rate has a narrow sweet spot, and outside it training fails
#   in characteristic ways: too slow, or spiky, or stuck.
# * Warmup protects the fragile first steps; decay lets the model settle at
#   the end. The final drop of a schedule is real and comes from the decay.
# * Clipping caps the damage of a bad batch. It's cheap insurance, which is
#   why almost every LLM recipe uses it (lessons 03 and 04 did too).
#
# Part 3 uses everything from Part 2: AdamW, warmup + cosine decay, clipping
# at 1.0, and an lr found the same way (a small sweep).

# %%
save_result("09_learning_rate", R)

# %% [markdown]
# ## Try:
# * Double the model width (`d=256`). Does the best learning rate move? (Bigger
#   models usually want a *smaller* lr. Fixing that is the idea behind muP.)
# * Double the batch size. Does the best lr move?
# * Find the largest lr that works *with* warmup and clipping. How much higher
#   than without?
# * Make WSD decay over the last 50% instead of 20%. Better or worse?
