# %% [markdown]
# # Lesson 11 — Pretrain your own model
#
# Everything comes together. We start from random weights, the state of
# Pythia at step 0 in lesson 05, and train a Llama-style model on lesson 10's
# token stream, using Part 2's recipe:
#
# * loss: next-token cross-entropy (lesson 06)
# * optimizer: AdamW (lesson 08)
# * learning rate: a quick sweep, then warmup + cosine decay, clipping at 1.0 (lesson 09)
#
# About 10 minutes on an M4 Pro (MPS) or a modest NVIDIA GPU.
#
# Questions: **can a 12M-parameter model learn to write stories in 10
# minutes? Does it go through the same stages Pythia did? And how does it
# compare with SmolLM2-135M on this kind of text?**

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
EOT = tok.convert_tokens_to_ids("<|endoftext|>")
R = {}
print(f"device: {device}   train stream: {len(train_data):,} tokens   val stream: {len(val_data):,}")

# %% [markdown]
# ## The model
#
# Same architecture as SmolLM2 (and Llama): 6 layers, width 384, 6 heads,
# a 256-token window. Built with `transformers`' own `LlamaForCausalLM`, so
# every tool from *Transformers, Inside Out* works on it.

# %%
D, LAYERS, HEADS, CTX = 384, 6, 6, 256
BATCH = 32                                    # 32 × 256 = 8,192 tokens per step
STEPS = int(os.environ.get("STEPS", 8000 if FULL else 2700))
model = tiny_llama(len(tok), d=D, layers=LAYERS, heads=HEADS, ctx=CTX).to(device)
n_all, n_emb = n_params(model), model.get_input_embeddings().weight.numel()
banner("Our model")
print(f"  {n_all:,} parameters: {n_emb:,} in the (tied) embedding table, {n_all - n_emb:,} in the 6 blocks")
print(f"  training: {STEPS:,} steps × {BATCH * CTX:,} tokens = {STEPS * BATCH * CTX / 1e6:.0f}M tokens"
      f" ({STEPS * BATCH * CTX / len(train_data):.0%} of the stream, so every window is new)")
R["config"] = {"d": D, "layers": LAYERS, "heads": HEADS, "ctx": CTX, "batch": BATCH, "steps": STEPS,
               "params": n_all, "emb_params": n_emb, "device": device}

# %% [markdown]
# ## A quick learning-rate check
#
# Lesson 09 found 1e-3 for a model 15× smaller. Bigger models often want a
# smaller lr, so we try three values for 150 steps each (about a minute in total).

# %%
def make_opt(m, lr):
    # The usual LLM recipe: weight decay on matrices only, not on norms; β2 = 0.95
    decay = [p for n, p in m.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in m.named_parameters() if p.dim() < 2]
    return torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                             lr=lr, betas=(0.9, 0.95))


def train_steps(m, opt, n, lr_at, gen, log=None, start=0, every=None, on_eval=None):
    for s in range(start, start + n):
        for grp in opt.param_groups:
            grp["lr"] = lr_at(s)
        x, y = get_batch(train_data, BATCH, CTX, gen, device)
        loss = F.cross_entropy(m(input_ids=x).logits.flatten(0, 1), y.flatten())
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        if log is not None and (s + 1) % 10 == 0:
            log["step"].append(s + 1); log["train"].append(loss.item()); log["gnorm"].append(gn.item()); log["lr"].append(lr_at(s))
        if on_eval and every and ((s + 1) in every):
            on_eval(s + 1)


banner("Learning-rate check: 150 steps each (val loss)")
lr_check = {}
for lr in (3e-4, 1e-3, 3e-3):
    m = tiny_llama(len(tok), d=D, layers=LAYERS, heads=HEADS, ctx=CTX).to(device)
    o = make_opt(m, lr)
    with Timer() as t:
        train_steps(m, o, 150, lambda s, lr=lr: lr * min(1, (s + 1) / 50), torch.Generator().manual_seed(1))
    lr_check[lr] = eval_loss(m, val_data, CTX, n_batches=10, device=device)
    print(f"  lr {lr:g}: {lr_check[lr]:.3f}   ({t.s:.0f}s)")
    del m, o
LR = min(lr_check, key=lr_check.get)
print(f"  -> using peak lr {LR:g}")
R["lr_check"] = {str(k): v for k, v in lr_check.items()}

# %% [markdown]
# ## Probes, like lesson 05
#
# While it trains we stop every so often and run lesson 05's probes on our own
# model: val loss, a story from the same prompt, and the copy test (a val
# passage shown twice: is the second copy cheaper?).

# %%
PROMPT = "Once upon a time, there was a little"
prompt_ids = tok(PROMPT).input_ids
vals = np.asarray(val_data[:200_000]).astype(np.int64)
passages = torch.tensor(np.stack([vals[i * 5003:i * 5003 + 64] for i in range(16)]))
COPY = torch.cat([passages, passages], 1).to(device)
chars_per_token = 4.0                                # lesson 10 measured 4.00 chars/token for our tokenizer


@torch.no_grad()
def probes(m, step):
    m.eval()
    l = F.cross_entropy(m(input_ids=COPY).logits[:, :-1].transpose(1, 2), COPY[:, 1:], reduction="none")
    m.train()
    r = {"step": step, "tokens": step * BATCH * CTX, "val": eval_loss(m, val_data, CTX, n_batches=20, device=device),
         "copy_first": l[:, :63].mean().item(), "copy_second": l[:, 64:].mean().item(),
         "sample": tok.decode(sample_text(m, prompt_ids, 80, temperature=0.8, device=device, seed=0, stop_id=EOT))}
    return r


PROBE_AT = sorted({0, 25, 50, 100, 200, 300, 500, 800, 1200, 1700, 2200, STEPS} | set(range(500, STEPS + 1, 500)))
PROBE_AT = [p for p in PROBE_AT if p <= STEPS]

# %% [markdown]
# ## Train
#
# Warmup for 100 steps, cosine decay to 10% of the peak, clipping at 1.0.

# %%
WARM = 100


def lr_at(s):
    return LR * min(1, (s + 1) / WARM) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s / STEPS, 1))))


seed_all(0)
model = tiny_llama(len(tok), d=D, layers=LAYERS, heads=HEADS, ctx=CTX).to(device)
opt = make_opt(model, LR)
gen = torch.Generator().manual_seed(0)
log = {"step": [], "train": [], "gnorm": [], "lr": []}
tape = [probes(model, 0)]
t0 = time.perf_counter()


def on_eval(step):
    tape.append(probes(model, step))
    r = tape[-1]
    print(f"  step {step:5}  val {r['val']:.3f} ({bits_per_char(r['val'], chars_per_token):.3f} bits/char)"
          f"  copy {r['copy_first']:.2f}→{r['copy_second']:.2f}   {time.perf_counter() - t0:5.0f}s   {r['sample'][:90]!r}")


banner(f"Pretraining: {STEPS} steps, peak lr {LR:g}")
on_eval_step0 = tape[0]
print(f"  step     0  val {on_eval_step0['val']:.3f}  (ln 4096 = {math.log(4096):.3f})   {on_eval_step0['sample'][:90]!r}")
train_steps(model, opt, STEPS, lr_at, gen, log=log, every=set(PROBE_AT), on_eval=on_eval)
train_s = time.perf_counter() - t0
print(f"\n  done in {train_s / 60:.1f} min ({STEPS * BATCH * CTX / train_s / 1e3:.0f}k tokens/s)")
R["log"], R["tape"], R["train_seconds"] = log, tape, train_s

# %%
CKPT.mkdir(parents=True, exist_ok=True)
model.save_pretrained(CKPT / "11_pretrain")
tok.save_pretrained(CKPT / "11_pretrain")
print(f"  saved -> {(CKPT / '11_pretrain').relative_to(ROOT)}   (lessons 12-14 and the MLX addendum use it)")

# %%
fig, axs = plt.subplots(1, 2, figsize=(12, 3.6))
axs[0].plot(log["step"], log["train"], alpha=0.4, label="train")
axs[0].plot([r["step"] for r in tape], [r["val"] for r in tape], "o-", label="val")
axs[0].set_ylim(1, 4); axs[0].set_xlabel("step"); axs[0].set_ylabel("loss (nats/token)"); axs[0].legend()
axs[0].set_title("Our 12M model on TinyStories")
axs[1].plot([r["step"] for r in tape], [r["copy_first"] for r in tape], "o-", label="1st time")
axs[1].plot([r["step"] for r in tape], [r["copy_second"] for r in tape], "o-", label="2nd time (copyable)")
axs[1].set_xscale("symlog", linthresh=50); axs[1].set_xlabel("step"); axs[1].set_title("copying a passage"); axs[1].legend()
show_fig(fig, "11_pretrain")

# %% [markdown]
# ## Stories
#
# Sampling at temperature 0.8 from a few different openings.

# %%
banner("Stories from our model")
stories = {}
for i, p in enumerate(["Once upon a time, there was a little", "Tom and his dog went to the park.",
                       "The dragon was sad because", "Lily found a box under her bed."]):
    out = tok.decode(sample_text(model, tok(p).input_ids, 200, temperature=0.8, device=device, seed=i, stop_id=EOT))
    stories[p] = out.replace("<|endoftext|>", "").strip()
    print(f"\n  [{p}]{stories[p]}")
R["stories"] = stories

# %% [markdown]
# ## How good is it, really?
#
# Lesson 06 put every model on one ladder in bits per character on TinyStories
# validation text. Where does ours land?

# %%
final_bpc = bits_per_char(tape[-1]["val"], chars_per_token)
banner("Bits per character on TinyStories validation (lesson 06's ladder)")
for name, v in (("char 8-gram (lesson 06)", 1.331), ("Pythia-70m final (lesson 06)", 0.968),
                ("SmolLM2-135M (lesson 06)", 0.657), (f"ours, {n_all / 1e6:.0f}M params, {train_s / 60:.0f} min", final_bpc)):
    print(f"  {name:34} {v:.3f}")
print("  (lesson 06's numbers are from its own run; see outputs/results/06_what_loss_means.json)")
R["final_bpc"] = final_bpc

# %% [markdown]
# **Aha:**
# * A model 11× smaller than SmolLM2, trained for about 10 minutes on a laptop,
#   writes simple, mostly coherent stories with dialogue and a plot. On
#   TinyStories it comes close to SmolLM2-135M in bits per character, a model
#   trained on trillions of tokens of everything. Specialists get far on their
#   own turf (lesson 03 found the same for chat).
# * Its samples went through Pythia's stages: frequent words, then word
#   salad with real words, then local phrases, then grammar, then story
#   structure ("One day, ...").
# * One stage is missing: the copy test barely improves (the second copy is
#   only a little cheaper than the first). TinyStories almost never repeats a
#   passage, so there is little reward for learning to copy, while web text
#   (Pythia's data) is full of repetition. **What a model learns depends on what
#   its data rewards.**
#
# Ten minutes is short: the loss was still falling. Lesson 12 asks what limits
# the speed; lesson 13 asks whether a different size would have done better
# with the same compute.

# %%
save_result("11_pretrain", R)

# %% [markdown]
# ## Try:
# * `STEPS=8000 python lessons/11_pretrain.py` (about 30 minutes). How much better
#   do the stories get?
# * Sample at temperature 0.3 and 1.2. Where does it repeat itself, where does it fall apart?
# * Open the model with *Transformers, Inside Out*: `MODEL_ID=outputs/checkpoints/11_pretrain`
#   works for the logit-lens and attention lessons (06, 07). Find your model's induction heads.
# * Remove warmup or clipping and run again. Does a 12M model need them?
