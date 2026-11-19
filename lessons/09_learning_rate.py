# %% [markdown]
# # Lesson 09 — Why does the learning rate matter so much?
#
# Lesson 07: one gradient step helps if it's small and hurts if it's big.
# Lesson 08: Adam's step for every weight is a fraction of `lr`, whatever
# the gradient's scale. So `lr` directly sets how far the whole model moves
# per step, and it's the single most important number in a training run.
# Every real recipe also *changes* it during training: a warmup at the
# start, a decay at the end.
#
# Questions: **how sensitive is training to the learning rate? What does
# "diverging" look like? Why warm up, why decay, and what does gradient
# clipping rescue?**
#
# Same setup as lesson 08: a 0.8M-parameter character model on TinyStories,
# AdamW, 600 steps per run (12 seconds on an M4 Pro). Every run starts from
# the same weights and sees the same batches, so the only difference between
# two runs is the thing we changed.
#
# A reference point from lesson 06: a model that knows only how common each
# character is (a unigram) gets 3.05 nats per character. Anything near 3 has
# learned letter frequencies and little else.

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
# | lr | 1e-4 | 3e-4 | **1e-3** | 3e-3 | 1e-2 | 3e-2 | 1e-1 |
# |---|---|---|---|---|---|---|---|
# | final val | 1.61 | 1.25 | **1.16** | 1.34 | 2.17 | 2.58 | 2.99 |
#
# A U-shape, with a fairly narrow bottom at 1e-3. **Too small** (1e-4): the
# model learns steadily, but after 600 steps it's only where the best run was
# between steps 150 and 200. **Too big**: the loss drops for a few dozen steps, then
# *stops*. The 3e-2 run sits at 2.6 and the 1e-1 run at 3.0, right at the
# unigram level: they learned letter frequencies and got stuck there.
#
# What you might expect a too-hot run to look like is spikes: a loss that
# jumps up after each bad step. That barely happened here (only the 1e-1 run
# had any jumps over 0.5 nats, three of them). Instead the steps keep
# kicking the model out of anything finer than letter counts, and its
# gradients shrink (median gradient norm 0.22 at the end of the 3e-2 run vs
# 0.81 for 1e-3): the run is stuck on a plateau, not exploding. A stuck run
# is easy to mistake for a "converged" one; the sweep is how you tell.
#
# Around the bottom, a 3× change costs 0.09–0.18 nats. Two steps (9×) away
# it costs 0.45 nats on the cold side and 1.0 on the hot side: too hot is
# much worse than too cold.
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
# | seed | 0 | 1 | 2 |
# |---|---|---|---|
# | no warmup | 1.32 | 1.32 | 1.39 |
# | 60-step warmup | **1.22** | **1.23** | **1.24** |
#
# Warmup wins on all three seeds, by 0.09–0.15 nats, and it makes the result
# more repeatable. The interesting part is *where* the difference comes from.
# At step 50 the two runs have about the same training loss (2.7–2.9 either
# way). The damage of a hot start isn't visible as a loss blow-up; it shows
# up later, as a model that ends worse. The first full-size steps, taken
# while Adam's statistics are still empty, pushed the model to a worse
# region it never fully left.
#
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
# | schedule | val at step 300 | val at step 500 | final (600) |
# |---|---|---|---|
# | constant | 1.356 | 1.199 | 1.167 |
# | cosine | 1.338 | 1.176 | 1.143 |
# | linear | 1.360 | 1.186 | 1.139 |
# | WSD | 1.356 | 1.193 | **1.102** |
#
# Watch WSD: until step 480 it *is* the constant run (same lr, same
# numbers), so it trails cosine for most of the training. Then in the last
# 120 steps it decays, and it drops from 1.193 at step 500 to 1.102, past
# everything else. **The loss drop at the end comes from the decay itself**,
# not from extra steps: the constant run, with the same steps, ends at 1.167.
# That's why a run's "loss so far" before the decay is a pessimistic estimate
# of where it will finish.
#
# All three decaying schedules beat constant. Which decay is best depends on
# the run length and isn't a big effect here (cosine vs linear: 0.004).
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
# At 10× the best lr (1e-2, with warmup), both runs hit one huge gradient
# early on, 60–90× their median (35.6 at step 20 without clipping, 42.5 at
# step 15 with it). In the clipped run, 36 of 600 steps had a norm above
# 1.0 and got scaled down.
#
# |  | final val | median grad norm | largest |
# |---|---|---|---|
# | no clipping | 1.80 | 0.59 | 35.6 |
# | clip at 1.0 | **1.38** | 0.48 | 42.5 |
#
# Clipping takes 0.42 nats off the final loss. It doesn't make the hot lr
# as good as the right one (1.16), but it makes it survivable. One likely
# reason the unclipped spike hurts for so long: Adam keeps a running average
# of squared gradients with β2 = 0.999, which remembers roughly the last
# 1,000 steps. One giant gradient inflates that average and shrinks every
# later step for a long time.
#
# **Aha:**
# * The learning rate has a narrow sweet spot, and outside it training fails
#   in characteristic ways: too slow, or stuck on a plateau. Being stuck
#   doesn't look dramatic, which is why you sweep.
# * Warmup protects the fragile first steps (3/3 seeds, ~0.1 nats); decay
#   lets the model settle at the end. The final drop of a schedule is real
#   and comes from the decay (WSD: 1.19 → 1.10 in the last 100 steps).
# * Clipping caps the damage of a bad batch. It's cheap insurance, which is
#   why almost every LLM recipe uses it (lessons 03 and 04 did too).
#
# ## What we know so far (Part 2)
#
# * Loss is average surprise; a model starts at ln(vocab) and has to beat
#   counting (lesson 06).
# * The gradient says which way each weight should move, exactly and cheaply
#   (lesson 07).
# * Adam turns the gradient into evenly sized steps; that's why it beats SGD
#   by more than a nat in the same number of steps (lesson 08).
# * The step size, and how it changes over time, decides whether those steps
#   add up to a good model (this lesson).
#
# Part 3 uses everything from Part 2: AdamW, warmup + cosine decay, clipping
# at 1.0, and an lr found the same way (a small sweep). What's still missing
# is the *data*: so far we've used single characters. Lesson 10 builds a real
# tokenizer and a training set, and asks what's in it.

# %%
save_result("09_learning_rate", R)

# %% [markdown]
# ## Try:
# * Double the model width (`d=256`). Does the best learning rate move? (Bigger
#   models usually want a *smaller* lr. Fixing that is the idea behind muP.)
# * Double the batch size. Does the best lr move?
# * Set Adam's `betas=(0.9, 0.95)` (a common LLM choice) in the unclipped hot
#   run. If the β2 memory explanation is right, the spike should hurt less.
# * Find the largest lr that works *with* warmup and clipping. How much higher
#   than without?
# * Make WSD decay over the last 50% instead of 20%. Better or worse?
