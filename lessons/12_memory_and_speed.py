# %% [markdown]
# # Lesson 12 — What limits training?
#
# Lesson 11's model trained at a few tens of thousands of tokens per second
# and fit easily in memory. Scale it up and two walls appear: **memory** (the
# model, its gradients, the optimizer state and the activations must all fit
# on the device) and **speed** (how many tokens per second the hardware can
# push through).
#
# Questions: **where does the memory go? What can you trade for it? And how
# close does our training loop get to what the hardware can do?**
#
# Numbers here depend a lot on your device. On CUDA we can measure memory
# exactly; on Apple Silicon (MPS) we use Metal's allocator counters; on CPU
# we skip the memory measurements.

# %%
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd())
from common import *
import torch.nn.functional as F
import matplotlib.pyplot as plt

seed_all(0)
device = pick_device()
tok = bpe_tokenizer()
train_data, _ = bpe_data()
V = len(tok)
D, LAYERS, HEADS, CTX = 384, 6, 6, 256            # lesson 11's model
R = {"device": device}
print(f"device: {device}")


def mem_mb():
    """Memory currently held by tensors on the device, in MB (None on CPU)."""
    if device == "cuda":
        return torch.cuda.memory_allocated() / 2 ** 20
    if device == "mps":
        return torch.mps.current_allocated_memory() / 2 ** 20
    return None


def reset():
    import gc
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    elif device == "mps":
        torch.mps.empty_cache()

# %% [markdown]
# ## The budget, on paper
#
# For every parameter, fp32 training with AdamW keeps four numbers: the
# weight, its gradient, and Adam's two running averages (lesson 08). That's
# **16 bytes per parameter**, before a single activation.
#
# *Mixed precision* (bf16 compute, fp32 master weights) doesn't shrink this
# part: it adds a bf16 copy of the weights (2 bytes) and keeps the fp32 state.
# What it shrinks is the **activations** and the time.

# %%
banner("Memory for weights + grads + AdamW state (no activations)")
size_rows = []
for name, n in (("our model", 12.2e6), ("SmolLM2-135M", 134.5e6), ("1B", 1e9), ("7B", 7e9), ("70B", 70e9)):
    row = {"name": name, "params": n, "infer_bf16_gb": 2 * n / 1e9, "train_fp32_gb": 16 * n / 1e9, "train_mixed_gb": 18 * n / 1e9}
    size_rows.append(row)
    print(f"  {name:13} run in bf16: {row['infer_bf16_gb']:8.2f} GB    train (fp32 AdamW): {row['train_fp32_gb']:8.1f} GB"
          f"    train (mixed): {row['train_mixed_gb']:8.1f} GB")
print("  -> a 7B model runs on a laptop; training it needs ~112 GB before activations, i.e. several big GPUs.")
R["sizes"] = size_rows

# %% [markdown]
# ## The budget, measured
#
# Build lesson 11's model, then watch the memory counter through one training step.

# %%
def fresh(ckpt=False):
    m = tiny_llama(V, d=D, layers=LAYERS, heads=HEADS, ctx=CTX).to(device)
    if ckpt:
        m.gradient_checkpointing_enable()
        m.config.use_cache = False
    return m


def step_memory(batch, ckpt=False):
    """Memory at each stage of one AdamW step (MB above the empty device)."""
    reset()
    base = mem_mb()
    m = fresh(ckpt)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    x, y = get_batch(train_data, batch, CTX, torch.Generator().manual_seed(0), device)
    r = {"weights": mem_mb() - base}
    loss = F.cross_entropy(m(input_ids=x).logits.flatten(0, 1), y.flatten())
    sync(device); r["after_forward"] = mem_mb() - base          # weights + saved activations
    loss.backward()
    sync(device); r["after_backward"] = mem_mb() - base         # activations freed, grads allocated
    opt.step()
    sync(device); r["after_step"] = mem_mb() - base             # + Adam's m and v
    r["activations"] = r["after_forward"] - r["weights"]
    del m, opt, x, y, loss
    return r


if mem_mb() is None:
    print("  (CPU: no device memory counter; skipping the measured parts)")
    R["stages"] = None
else:
    banner("One training step, batch 32 × 256 tokens (MB)")
    st = step_memory(32)
    n = n_params(fresh())
    print(f"  weights only                 {st['weights']:7.1f}   (expected {4 * n / 2 ** 20:.1f} = 4 bytes × {n:,})")
    print(f"  after forward                {st['after_forward']:7.1f}   (activations kept for backward: {st['activations']:.1f})")
    print(f"  after backward               {st['after_backward']:7.1f}   (activations freed, gradients added)")
    print(f"  after optimizer step         {st['after_step']:7.1f}   (+ Adam's two running averages)")
    R["stages"] = st

# %% [markdown]
# Even for this small model, **activations dominate**: every layer keeps its
# inputs, attention outputs and MLP intermediate values until the backward
# pass needs them (lesson 07). They grow with batch size and context length;
# the weights don't.

# %%
if R["stages"] is not None:
    banner("Activation memory vs batch size (MB)")
    act_rows = []
    for b in (4, 8, 16, 32, 64):
        a = step_memory(b)["activations"]
        c = step_memory(b, ckpt=True)["activations"]
        act_rows.append({"batch": b, "activations": a, "checkpointed": c})
        print(f"  batch {b:3}: {a:8.1f}   with activation checkpointing: {c:7.1f}")
    R["activations"] = act_rows

# %% [markdown]
# **Activation checkpointing** keeps only each layer's *input* and recomputes
# the rest during backward. Memory drops a lot, at the price of an extra
# forward pass (we time it below). It's the standard trick for fitting longer
# contexts or bigger batches.
#
# ## Speed: how big should a batch be?

# %%
def tokens_per_sec(batch, ckpt=False, amp=None, n=8):
    m = fresh(ckpt)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(0)

    def step():
        x, y = get_batch(train_data, batch, CTX, g, device)
        with torch.autocast(device_type=device, dtype=amp, enabled=amp is not None):
            loss = F.cross_entropy(m(input_ids=x).logits.flatten(0, 1).float(), y.flatten())
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    for _ in range(2):
        step()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(n):
        step()
    sync(device)
    dt = (time.perf_counter() - t0) / n
    del m, opt
    reset()
    return batch * CTX / dt, dt


banner(f"Throughput vs batch size ({device})")
speed_rows = []
for b in (1, 2, 4, 8, 16, 32, 64):
    tps, dt = tokens_per_sec(b)
    speed_rows.append({"batch": b, "tokens_per_s": tps, "step_s": dt})
    print(f"  batch {b:3} ({b * CTX:6,} tokens/step): {tps / 1e3:7.1f}k tokens/s   {dt * 1000:7.1f} ms/step")
R["speed"] = speed_rows

# %% [markdown]
# Small batches waste the hardware: each step has fixed costs (launching
# hundreds of GPU kernels, the optimizer touching every weight), and a GPU
# needs a lot of parallel work to be busy. Throughput rises with batch size,
# then flattens once the device is full.
#
# ## Gradient accumulation: a big batch in small pieces
#
# If a batch of 32 doesn't fit, run 4 batches of 8, add up their gradients,
# and step once. Mathematically it's the same step. Check:

# %%
banner("4 × 8 accumulated vs 1 × 32 at once")
m1 = fresh()
m2 = fresh()
m2.load_state_dict(m1.state_dict())
x, y = get_batch(train_data, 32, CTX, torch.Generator().manual_seed(3), device)
F.cross_entropy(m1(input_ids=x).logits.flatten(0, 1), y.flatten()).backward()
for i in range(4):
    xs, ys = x[i * 8:(i + 1) * 8], y[i * 8:(i + 1) * 8]
    (F.cross_entropy(m2(input_ids=xs).logits.flatten(0, 1), ys.flatten()) / 4).backward()   # /4: average, not sum
diff = max((a.grad - b.grad).abs().max().item() for a, b in zip(m1.parameters(), m2.parameters()))
gmax = max(a.grad.abs().max().item() for a in m1.parameters())
print(f"  largest gradient {gmax:.2e}, largest difference {diff:.2e}  -> the same, up to float rounding")
R["accum"] = {"max_diff": diff, "max_grad": gmax}
del m1, m2

# %% [markdown]
# ## Trading speed for memory, and precision for speed

# %%
banner("Time per step, batch 32")
base_tps, base_dt = tokens_per_sec(32)
ck_tps, ck_dt = tokens_per_sec(32, ckpt=True)
rows = [{"name": "fp32", "tokens_per_s": base_tps}, {"name": "fp32 + activation checkpointing", "tokens_per_s": ck_tps}]
print(f"  fp32                            {base_tps / 1e3:7.1f}k tokens/s")
print(f"  fp32 + activation checkpointing {ck_tps / 1e3:7.1f}k tokens/s   ({ck_dt / base_dt:.2f}× slower)")
for name, dt_ in (("bf16 autocast", torch.bfloat16), ("fp16 autocast", torch.float16)):
    try:
        tps, _ = tokens_per_sec(32, amp=dt_)
        rows.append({"name": name, "tokens_per_s": tps})
        print(f"  {name:31} {tps / 1e3:7.1f}k tokens/s   ({tps / base_tps:.2f}× fp32)")
    except Exception as e:      # some devices don't support every dtype
        print(f"  {name:31} not supported here ({type(e).__name__})")
R["precision"] = rows

# %% [markdown]
# On NVIDIA GPUs, bf16 runs on tensor cores and is typically 2–3× faster than
# fp32 for models this shape (more for bigger models). On Apple Silicon the
# gain is small or none: MPS doesn't have the same fast low-precision
# matrix units. Mixed precision is a big win on the hardware it was designed for.
#
# ## How close are we to the hardware's limit?
#
# Training costs about **6 × parameters × tokens** FLOPs (lesson 07: one
# forward, two for backward; 2 FLOPs per weight per token each). Divide by
# time to get achieved FLOP/s, and compare with the device's peak: that ratio
# is **MFU**, model FLOPs utilisation.

# %%
n = n_params(fresh(), embeddings=True)
best = max(speed_rows + [{"tokens_per_s": r["tokens_per_s"]} for r in rows], key=lambda r: r["tokens_per_s"])["tokens_per_s"]
achieved = 6 * n * best
peak = float(os.environ.get("PEAK_TFLOPS", "0")) * 1e12
banner("Model FLOPs utilisation")
print(f"  6 × {n / 1e6:.1f}M params × {best / 1e3:.0f}k tokens/s = {achieved / 1e12:.2f} TFLOP/s achieved")
if peak:
    print(f"  your device's peak ({peak / 1e12:g} TFLOP/s): MFU = {achieved / peak:.0%}")
else:
    print("  set PEAK_TFLOPS=<your device's peak> to get MFU. Rough peaks: M4 Pro GPU ~9 (fp32),"
          " RTX 4090 ~83 (fp32) / ~165 (bf16), A100 ~19.5 (fp32) / 312 (bf16), H100 ~989 (bf16)")
print("  Large training runs reach 30-50% MFU. Tiny models reach much less: fixed costs dominate.")
R["mfu"] = {"params": n, "tokens_per_s": best, "achieved_tflops": achieved / 1e12}

# %%
fig, axs = plt.subplots(1, 2, figsize=(12, 3.4))
axs[0].plot([r["batch"] for r in speed_rows], [r["tokens_per_s"] / 1e3 for r in speed_rows], "o-")
axs[0].set_xscale("log", base=2); axs[0].set_xlabel("batch size"); axs[0].set_ylabel("k tokens/s"); axs[0].set_title(f"Throughput ({device})")
if R.get("activations"):
    axs[1].plot([r["batch"] for r in R["activations"]], [r["activations"] for r in R["activations"]], "o-", label="normal")
    axs[1].plot([r["batch"] for r in R["activations"]], [r["checkpointed"] for r in R["activations"]], "o-", label="checkpointed")
    axs[1].set_xlabel("batch size"); axs[1].set_ylabel("activation memory (MB)"); axs[1].legend(); axs[1].set_title("Activations")
show_fig(fig, "12_memory_speed")

# %% [markdown]
# **Aha:**
# * Training memory = weights + grads + optimizer state (16 bytes/param with
#   fp32 AdamW) **+ activations**, which grow with batch × context and often dominate.
# * Every fix is a trade. Gradient accumulation: same maths, more time.
#   Activation checkpointing: less memory, ~one more forward pass. Mixed
#   precision: less memory and more speed, on hardware that supports it.
# * Small models can't keep a GPU busy. That's the reason big models are
#   trained with big batches, and it's what lesson 13's "compute budget" is really
#   about: FLOPs, not steps.

# %%
save_result("12_memory_and_speed", R)

# %% [markdown]
# ## Try:
# * Double the context to 512 (`CTX = 512`). How do activations and speed change?
# * On a CUDA machine: `torch.compile(model)` before the speed test. How much faster?
# * Use `torch.optim.AdamW(..., fused=True)` (CUDA). What does fusing save?
# * Estimate: what's the largest model you could *train* on your machine with
#   fp32 AdamW, batch 8 and checkpointing?
