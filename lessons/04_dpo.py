# %% [markdown]
# # Lesson 04 — Teach a preference, not an answer (DPO)
#
# SFT (lesson 03) copies demonstrations: "here is the answer, write this".
# Often you don't have perfect answers, but you can easily say **which of two
# answers is better**. That's preference tuning. The production recipe used to
# be RLHF (train a reward model, then run PPO). **DPO** (Direct Preference
# Optimization, 2023) gets the same effect with one loss function and no
# reward model.
#
# Our toy preference, chosen because we can measure it: **longer, more
# detailed answers are better.** We'll sample two answers per prompt from our
# own SFT model (lesson 03), call the longer one "chosen" and the shorter one
# "rejected", and train. No human labels, no new data.
#
# Questions: **does the model learn the preference without ever seeing a
# "correct" answer? What does it cost? And what happens if we pull too hard?**

# %%
import os, sys, copy, random, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd())
from common import *
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datasets import load_dataset

seed_all(0)
SFT = CKPT / "03_sft"
if not SFT.exists():
    sys.exit("Run lessons/03_sft.py first: this lesson starts from its checkpoint.")
tok, policy, device = load(str(SFT))
tok.pad_token = "<|endoftext|>"
IM_END = tok.convert_tokens_to_ids("<|im_end|>")
ds = load_dataset("HuggingFaceTB/everyday-conversations-llama3.1-2k")
# the 2nd user turn is the real request (the 1st is usually just "Hi!")
train_prompts = list(dict.fromkeys(c[2]["content"] for c in ds["train_sft"]["messages"] if len(c) > 2))
test_prompts = list(dict.fromkeys(c[2]["content"] for c in ds["test_sft"]["messages"] if len(c) > 2))
print(f"device: {device}   prompts: {len(train_prompts)} train, {len(test_prompts)} test")

# %% [markdown]
# ## Step 1: make preference pairs
#
# Sample two answers for each prompt (temperature 0.9), and keep the pairs
# where the lengths differ clearly. Batched generation needs **left** padding,
# so every prompt ends right where the answer starts.

# %%
@torch.no_grad()
def sample(model, prompts, n=1, max_new=120, greedy=False, bs=64):
    """Returns a list (len(prompts)*n) of (answer_text, n_tokens, stopped_by_itself)."""
    model.eval()
    tok.padding_side = "left"
    out = []
    for i in range(0, len(prompts), bs):
        enc = tok([our_chat_prompt(tok, p) for p in prompts[i:i + bs]], return_tensors="pt", padding=True).to(device)
        kw = dict(do_sample=False, repetition_penalty=1.1) if greedy else dict(do_sample=True, temperature=0.9, top_p=0.95)
        gen = model.generate(**enc, **kw, max_new_tokens=max_new, num_return_sequences=n,
                             eos_token_id=IM_END, pad_token_id=0)[:, enc.input_ids.shape[1]:]
        for g in gen.tolist():
            stopped = IM_END in g
            g = g[:g.index(IM_END)] if stopped else [t for t in g if t != 0]
            out.append((tok.decode(g), len(g), stopped))
    tok.padding_side = "right"
    model.train()
    return out


N_PROMPTS = len(train_prompts) if FULL else 1024
banner(f"Sampling 2 answers for {N_PROMPTS} prompts")
with Timer() as t:
    samples = sample(policy, train_prompts[:N_PROMPTS], n=2)
pairs = []
for i, p in enumerate(train_prompts[:N_PROMPTS]):
    (a, la, sa), (b, lb, sb) = samples[2 * i], samples[2 * i + 1]
    if not (sa and sb) or abs(la - lb) < 8:      # need both finished and clearly different
        continue
    chosen, rejected = (a, b) if la > lb else (b, a)
    pairs.append({"prompt": p, "chosen": chosen, "rejected": rejected, "len_c": max(la, lb), "len_r": min(la, lb)})
print(f"  {t.s:.0f}s, kept {len(pairs)} pairs (both answers finished, lengths differ by ≥ 8 tokens)")
ex = pairs[0]
print(f"\n  prompt:   {ex['prompt']}\n  chosen:   {ex['chosen']!r}  ({ex['len_c']} tokens)"
      f"\n  rejected: {ex['rejected']!r}  ({ex['len_r']} tokens)")

# %% [markdown]
# ## Step 2: score an answer
#
# DPO needs one number per answer: how likely is this whole answer under a
# model? That's the **sum of the log-probabilities** of its tokens (the
# prompt tokens don't count, same masking as lesson 03).

# %%
def batch_ids(rows, key):
    return pad_batch([chat_ids(tok, [{"role": "user", "content": r["prompt"]},
                                     {"role": "assistant", "content": r[key]}]) for r in rows], device)


def seq_logp(model, x, y, att):
    """log π(answer | prompt): sum over the answer's tokens."""
    logits = model(input_ids=x, attention_mask=att).logits[:, :-1]
    tgt = y[:, 1:]
    lp = torch.gather(logits.log_softmax(-1), 2, tgt.clamp(min=0)[..., None])[..., 0]
    return (lp * (tgt != -100)).sum(-1)

# %% [markdown]
# ## Step 3: the DPO loss
#
# Keep a frozen copy of the starting model: the **reference**. For each answer,
# the *implicit reward* is how much more likely the policy finds it than the
# reference does:
#
#     r(answer) = β · (log π(answer) − log π_ref(answer))
#
# The loss pushes the chosen answer's reward above the rejected one's:
#
#     loss = −log σ( r(chosen) − r(rejected) )
#
# β is the leash. Small β: the loss stays high until the policy moves far from
# the reference. Large β: small moves are enough. That's the whole algorithm.

# %%
def dpo_loss(pc, pr, rc, rr, beta):
    """p* = policy log-probs, r* = reference log-probs, c = chosen, r = rejected."""
    reward_c, reward_r = beta * (pc - rc), beta * (pr - rr)
    loss = -F.logsigmoid(reward_c - reward_r).mean()
    return loss, reward_c.detach(), reward_r.detach()

# %% [markdown]
# The reference never changes, so we score every pair with it once, up front.

# %%
@torch.no_grad()
def ref_scores(model, rows, bs=16):
    model.eval()
    out = []
    for i in range(0, len(rows), bs):
        b = rows[i:i + bs]
        out.append(torch.stack([seq_logp(model, *batch_ids(b, "chosen")), seq_logp(model, *batch_ids(b, "rejected"))], 1))
    model.train()
    return torch.cat(out)


with Timer() as t:
    REF = ref_scores(policy, pairs)
print(f"  reference scores for {len(pairs)} pairs in {t.s:.0f}s")

# %% [markdown]
# ## What to measure
#
# * **length** of greedy answers on held-out prompts (the thing we asked for)
# * **stops by itself**: share of answers that end with `<|im_end|>` within 200 tokens
# * **repetition**: share of repeated 3-grams (a sign of a degenerate model)
# * **SFT test loss** from lesson 03 (does it still fit normal conversations?)

# %%
test_data = [chat_ids(tok, c) for c in ds["test_sft"]["messages"]]
EVAL_PROMPTS = test_prompts[:48]


@torch.no_grad()
def sft_test_loss(model, bs=16):
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(test_data), bs):
        x, y, att = pad_batch(test_data[i:i + bs], device)
        logits = model(input_ids=x, attention_mask=att).logits
        k = (y[:, 1:] != -100).sum().item()
        tot += F.cross_entropy(logits[:, :-1].flatten(0, 1), y[:, 1:].flatten(), ignore_index=-100).item() * k
        n += k
    model.train()
    return tot / n


def rep3(text):
    w = text.split()
    grams = list(zip(w, w[1:], w[2:]))
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def evaluate(model):
    outs = sample(model, EVAL_PROMPTS, greedy=True, max_new=200)
    return {"len": statistics.mean(o[1] for o in outs), "stopped": statistics.mean(o[2] for o in outs),
            "rep": statistics.mean(rep3(o[0]) for o in outs), "test_loss": sft_test_loss(model),
            "examples": [o[0] for o in outs[:6]]}


reference = copy.deepcopy(policy).eval()
for p in reference.parameters():
    p.requires_grad_(False)
before = evaluate(policy)
print(f"  before DPO: length {before['len']:.0f} tokens, stops {before['stopped']:.0%}, "
      f"repeated 3-grams {before['rep']:.1%}, SFT test loss {before['test_loss']:.3f}")

# %% [markdown]
# ## Step 4: train
#
# One run with a sensible leash (β = 0.1), then the same data with a very
# loose one (β = 0.01) and a higher learning rate, to see what over-optimising
# looks like.

# %%
def train_dpo(beta, lr, epochs, label, eval_every=40, bs=8):
    model = copy.deepcopy(reference).train()
    for p in model.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    steps = epochs * len(pairs) // bs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 10))
    rng = random.Random(0)
    order = []
    log = {k: [] for k in ("step", "loss", "acc", "margin", "dc", "dr")}
    evals = [{"step": 0, **{k: v for k, v in before.items() if k != "examples"}}]
    banner(f"DPO {label}: β={beta}, lr={lr}, {steps} steps × {bs} pairs")
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        if len(order) < bs:
            order += rng.sample(range(len(pairs)), len(pairs))
        idx = [order.pop() for _ in range(bs)]
        rows = [pairs[i] for i in idx]
        pc, pr = seq_logp(model, *batch_ids(rows, "chosen")), seq_logp(model, *batch_ids(rows, "rejected"))
        rc, rr = REF[idx, 0], REF[idx, 1]
        loss, wc, wr = dpo_loss(pc, pr, rc, rr, beta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad()
        log["step"].append(step); log["loss"].append(loss.item())
        log["acc"].append((wc > wr).float().mean().item()); log["margin"].append((wc - wr).mean().item())
        log["dc"].append((pc - rc).mean().item()); log["dr"].append((pr - rr).mean().item())
        if step % eval_every == 0 or step == steps:
            e = evaluate(model)
            evals.append({"step": step, **{k: v for k, v in e.items() if k != "examples"}})
            k = slice(-eval_every, None)
            print(f"  step {step:4}  loss {statistics.mean(log['loss'][k]):.3f}  acc {statistics.mean(log['acc'][k]):.0%}"
                  f"  Δlogp chosen {statistics.mean(log['dc'][k]):+6.1f} rejected {statistics.mean(log['dr'][k]):+6.1f}"
                  f" | length {e['len']:5.1f}  stops {e['stopped']:4.0%}  rep {e['rep']:.1%}  test loss {e['test_loss']:.3f}"
                  f"  ({time.perf_counter() - t0:.0f}s)")
    return model, log, evals, e["examples"]


EPOCHS = 2 if FULL else 1
good, log_good, ev_good, ex_good = train_dpo(beta=0.1, lr=5e-6, epochs=EPOCHS, label="(sensible)")

# %% [markdown]
# Read the columns from left to right:
# * **loss / acc**: on the training pairs, the policy quickly ranks the longer
#   answer higher.
# * **Δlogp chosen / rejected**: how much more (or less) likely each answer has
#   become than under the reference. Watch the signs. DPO only cares about the
#   *gap*, and the cheapest way to open a gap is often to make *both* answers
#   less likely, the rejected one much more so.
# * **length**: the behaviour we asked for, on prompts it never trained on.

# %%
loose, log_loose, ev_loose, ex_loose = train_dpo(beta=0.01, lr=2e-5, epochs=EPOCHS * 2, label="(pulled hard)")

# %%
banner("Same held-out prompts, greedy answers")
for i in range(3):
    print(f"\n  Q: {EVAL_PROMPTS[i]}")
    print(f"  SFT (start)  : {before['examples'][i]!r}")
    print(f"  DPO β=0.1    : {ex_good[i]!r}")
    print(f"  DPO β=0.01   : {ex_loose[i][:400]!r}")

# %% [markdown]
# Greedy answers only show the single most likely path. Sampling (what chat
# apps actually do) shows the whole distribution the model has become.

# %%
banner("Sampled answers (temperature 0.9), 48 held-out prompts × 2")
sampled = {}
for name, m in (("SFT (start)", reference), ("DPO β=0.1", good), ("DPO β=0.01", loose)):
    seed_all(1)
    outs = sample(m, EVAL_PROMPTS, n=2, max_new=200)
    sampled[name] = {"len": statistics.mean(o[1] for o in outs), "stopped": statistics.mean(o[2] for o in outs),
                     "rep": statistics.mean(rep3(o[0]) for o in outs), "examples": [o[0] for o in outs[:4]]}
    print(f"  {name:12} length {sampled[name]['len']:5.1f}  stops {sampled[name]['stopped']:4.0%}"
          f"  repeated 3-grams {sampled[name]['rep']:.1%}")
    print(f"               e.g. {outs[0][0][:300]!r}")

# %%
fig, axs = plt.subplots(1, 3, figsize=(13, 3.4))
for ev, lab in ((ev_good, "β=0.1, lr 5e-6"), (ev_loose, "β=0.01, lr 2e-5")):
    s = [e["step"] for e in ev]
    axs[0].plot(s, [e["len"] for e in ev], "o-", label=lab)
    axs[2].plot(s, [e["test_loss"] for e in ev], "o-", label=lab)
for lg, lab in ((log_good, "β=0.1"), (log_loose, "β=0.01")):
    axs[1].plot(lg["step"], lg["dc"], label=f"{lab} chosen")
    axs[1].plot(lg["step"], lg["dr"], "--", label=f"{lab} rejected")
axs[0].set_title("greedy answer length (tokens)"); axs[1].set_title("log π − log π_ref on training pairs")
axs[2].set_title("SFT test loss (lesson 03's test set)")
for a in axs:
    a.set_xlabel("step")
axs[0].legend(); axs[1].legend(fontsize=7)
show_fig(fig, "04_dpo")

# %% [markdown]
# **Aha:** the greedy answers of the loose-leash model look fine, even good.
# The measurements say otherwise: its SFT test loss has tripled, and its
# *chosen* answers, the ones it was told to prefer, are now far less likely
# than before training. DPO only rewards the **gap** between chosen and
# rejected, and the cheapest way to open a gap is to push everything down,
# the rejected answers fastest. The model drifts away from normal language
# while still winning on the training pairs. That is **over-optimisation**:
# it rarely shows in a handful of greedy samples, and clearly shows in
# held-out measurements.
#
# β and the reference model are the leash that keeps the policy close to a
# model that was already sensible. Real preference data has the same length
# trap (raters tend to prefer longer answers), which is one reason chat
# models are so wordy.

# %%
save_result("04_dpo", {
    "n_pairs": len(pairs), "pair_examples": pairs[:6],
    "len_hist": {"chosen": [p["len_c"] for p in pairs], "rejected": [p["len_r"] for p in pairs]},
    "before": before, "eval_prompts": EVAL_PROMPTS[:6], "sampled": sampled,
    "runs": [{"beta": 0.1, "lr": 5e-6, "log": log_good, "evals": ev_good, "examples": ex_good},
             {"beta": 0.01, "lr": 2e-5, "log": log_loose, "evals": ev_loose, "examples": ex_loose}],
})

# %% [markdown]
# ## Try:
# * Flip the preference: make the *shorter* answer "chosen". How short does it get
#   before it starts answering with nothing?
# * Use a different, more useful preference: answers that end with a question
#   back to the user, or that mention a word from the prompt. Anything you can
#   check with a line of Python can be a preference.
# * Set β = 1.0. The loss is satisfied almost immediately. Does behaviour change at all?
# * Drop the reference (`rc = rr = 0`). What's left is "push chosen up, push
#   rejected down" without a leash. How quickly does it break?
# * The library version: `trl.DPOTrainer` does exactly this, with many more options.
