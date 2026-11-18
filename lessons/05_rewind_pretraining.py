# %% [markdown]
# # Lesson 05 — Rewind pretraining
#
# Lessons 01–04 undid post-training. Everything left, the grammar, the facts,
# the ability to continue any text, came from **pretraining**: next-token
# prediction on a huge pile of text, from random weights.
#
# We can't rewind SmolLM2's pretraining: only the final weights were
# published. But EleutherAI's **Pythia** models were released with *154
# checkpoints* saved during training. We take Pythia-70m (70M parameters,
# trained on 300B tokens of "The Pile") and load it at 11 moments of its life.
#
# Questions: **what does a model know at step 0, 128, 1,000, 143,000? In what
# order do skills appear? And does anything happen suddenly?**

# %%
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd())
from common import *
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datasets import load_dataset

seed_all(0)
torch.set_grad_enabled(False)
device = pick_device()
PYTHIA = "EleutherAI/pythia-70m"
STEPS = [0, 128, 256, 512, 1000, 2000, 4000, 8000, 16000, 64000, 143000]
TOKENS_PER_STEP = 1024 * 2048            # Pythia's batch: 1,024 sequences of 2,048 tokens
tok = AutoTokenizer.from_pretrained(PYTHIA)


def load_step(step):
    # use_safetensors=False: the .bin files are fp16 (166 MB); the safetensors are fp32 (282 MB)
    return AutoModelForCausalLM.from_pretrained(PYTHIA, revision=f"step{step}", dtype=torch.float32,
                                                use_safetensors=False).to(device).eval()


print(f"device: {device}   checkpoints: {STEPS}")
print(f"  the last checkpoint has seen {143000 * TOKENS_PER_STEP / 1e9:.0f}B tokens")

# %% [markdown]
# ## The probes
#
# We run the same tests on every checkpoint:
#
# 1. **Loss** on held-out text (WikiText-2), in nats per token.
# 2. **Completions**: greedy continuations of three prompts.
# 3. **Facts**: probability of the right next token for ten simple facts, from
#    very common ("The United States of → America") to rarer ones.
# 4. **Copying** (in-context learning): a passage shown twice. A model that can
#    look back and copy gets the second copy almost free. *Transformers, Inside
#    Out* lesson 07 found the "induction heads" that do this. We test it on real
#    text, and on random tokens (which can *only* be predicted by copying).
# 5. **Variety**: how many *different* tokens does it ever predict as its top choice?

# %%
wiki = "\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
wiki_ids = tok(wiki).input_ids
WIKI = torch.tensor(wiki_ids[:64 * 512], device=device).view(64, 512)

PROMPTS = ["The capital of France is", "Once upon a time, there was a", "def fibonacci(n):"]
FACTS = [("The United States of", " America"), ("one, two, three,", " four"), ("January, February,", " March"),
         ("Monday, Tuesday,", " Wednesday"), ("Water is made of hydrogen and", " oxygen"), ("Paris is the capital of", " France"),
         ("The sun rises in the", " east"), ("The Eiffel Tower is in", " Paris"), ("The capital of France is", " Paris"),
         ("Barack", " Obama")]
FACT_IDS = [tok(a).input_ids[0] for _, a in FACTS]
nat = torch.tensor([wiki_ids[i * 997:i * 997 + 64] for i in range(16)])               # 16 real passages
COPY_NAT = torch.cat([nat, nat], 1).to(device)
rnd = torch.randint(1000, 20000, (16, 64), generator=torch.Generator().manual_seed(0))  # 16 random-token passages
COPY_RND = torch.cat([rnd, rnd], 1).to(device)


def copy_losses(model, X):
    l = F.cross_entropy(model(X).logits[:, :-1].transpose(1, 2), X[:, 1:], reduction="none")
    return l[:, :63].mean().item(), l[:, 64:].mean().item()        # first copy, second copy


def probe(model):
    r = {}
    losses = [F.cross_entropy(model(WIKI[i:i + 8]).logits[:, :-1].flatten(0, 1), WIKI[i:i + 8, 1:].flatten()).item()
              for i in range(0, 64, 8)]
    r["loss"] = sum(losses) / len(losses)
    r["completions"] = {p: tok.decode(model.generate(encode(tok, p, device), max_new_tokens=20, do_sample=False,
                                                     pad_token_id=0)[0, len(tok(p).input_ids):]) for p in PROMPTS}
    r["facts"] = [model(encode(tok, p, device)).logits[0, -1].softmax(-1)[i].item() for (p, _), i in zip(FACTS, FACT_IDS)]
    r["copy_nat"], r["copy_rnd"] = copy_losses(model, COPY_NAT), copy_losses(model, COPY_RND)
    top1 = model(WIKI[:8]).logits.argmax(-1).flatten()
    vals, counts = top1.unique(return_counts=True)
    r["distinct_top1"] = len(vals)
    r["most_predicted"] = [[tok.decode([v]), c / len(top1)] for c, v in
                           sorted(zip(counts.tolist(), vals.tolist()), reverse=True)[:3]]
    return r

# %%
banner("Rewinding Pythia-70m")
results = []
for step in STEPS:
    with Timer() as t:
        model = load_step(step)
        r = probe(model)
    r["step"], r["tokens"] = step, step * TOKENS_PER_STEP
    results.append(r)
    del model
    print(f"\n  step {step:>6}  ({r['tokens'] / 1e9:6.2f}B tokens)  loss {r['loss']:5.2f}"
          f"  copy text: {r['copy_nat'][0]:5.2f} → {r['copy_nat'][1]:5.2f}"
          f"  copy random: {r['copy_rnd'][0]:5.2f} → {r['copy_rnd'][1]:5.2f}"
          f"  distinct top-1: {r['distinct_top1']:4}  most predicted: {r['most_predicted'][0][0]!r} {r['most_predicted'][0][1]:.0%}")
    print("      facts: " + "  ".join(f"{a.strip()} {p:.0%}" for (_, a), p in zip(FACTS, r["facts"])))
    for p, c in r["completions"].items():
        print(f"      {p!r:34} → {c!r}")

# %% [markdown]
# Read it top to bottom; that's the order the skills arrive in:
#
# * **step 0**: random weights. Loss ≈ ln(50,304) ≈ 10.8, the loss of a uniform
#   guess over the vocabulary (lesson 06 explains why). Completions are random tokens.
# * **~128 steps**: it has learned *which tokens are common*, and nothing else.
#   Its top prediction is " the" a quarter of the time; it writes "the the the".
#   That's a *unigram* model.
# * **256–512 steps**: short phrases ("a little", "the first of"), repeated in loops.
# * **512 → 1,000 steps**: copying switches on. The second copy of a passage
#   suddenly becomes cheap. Grammar follows quickly.
# * **after that**: facts, roughly in order of how often they appear in text.
#   "United States of America" is there by step 1,000. "one, two, three, four"
#   and "hydrogen and oxygen" creep up over the rest of training. "Paris" and
#   "Obama" never really arrive: a 70M model is too small to store them.
#   And knowledge *flickers*: "America" is at 79% at step 4,000, 5% at step
#   8,000, and back to 83% at step 16,000. A checkpoint is a snapshot of a
#   process that is still moving.
#
# ## The picture

# %%
tokens = [max(r["tokens"], TOKENS_PER_STEP / 4) for r in results]    # step 0 drawn at the left edge
fig, axs = plt.subplots(1, 3, figsize=(14, 3.6))
axs[0].plot(tokens, [r["loss"] for r in results], "o-")
axs[0].axhline(math.log(len(tok)), ls="--", color="gray")
axs[0].text(tokens[1], math.log(len(tok)) - 0.6, "uniform guess  ln(V)", color="gray", fontsize=8)
axs[0].set_title("loss on WikiText-2 (nats/token)")
axs[1].plot(tokens, [r["copy_nat"][0] for r in results], "o-", label="real text, 1st time")
axs[1].plot(tokens, [r["copy_nat"][1] for r in results], "o-", label="real text, 2nd time (copyable)")
axs[1].set_title("copying from context"); axs[1].legend(fontsize=8)
for (_, a), i in zip(FACTS[:5], range(5)):
    axs[2].plot(tokens, [r["facts"][i] for r in results], "o-", label=a.strip(), ms=3)
axs[2].set_title("P(right answer) for simple facts"); axs[2].legend(fontsize=7)
for a in axs:
    a.set_xscale("log"); a.set_xlabel("tokens seen")
show_fig(fig, "05_rewind")

# %% [markdown]
# **Aha 1:** skills arrive in a fixed order. Frequency first, then local
# phrases, then copying from context, then grammar and the most common facts,
# then (slowly) rarer facts. Fluency comes long before knowledge.
#
# **Aha 2:** copying doesn't fade in smoothly. It appears between two
# neighbouring checkpoints (steps 512 and 1,000). This is the *induction head
# phase change* (Olsson et al., 2022).
#
# **Aha 3:** most of the loss drop happens almost immediately. By step 1,000
# (0.7% of training) the loss has fallen about 85% of the way from 11.0 to its
# final value. The remaining 99% of the compute buys the last nat, and
# that's where the knowledge lives.
#
# **A surprise:** look at the *random-token* copy column. It works best around
# steps 2,000–4,000 and then gets **worse**, and by the end the model is
# confidently wrong about random tokens (loss far above the uniform 10.8).
# Real-text copying stays fine. Small models are known to show odd behaviour
# late in training (one line of work: Godey et al., 2024, "Why do small
# language models underperform?"). Don't assume "more training" improves every
# ability; measure the ones you care about.

# %%
save_result("05_rewind_pretraining", {"model": PYTHIA, "tokens_per_step": TOKENS_PER_STEP, "vocab": len(tok),
                                       "prompts": PROMPTS, "facts": FACTS, "checkpoints": results})

# %% [markdown]
# ## Try:
# * Add more checkpoints around the copying jump (`STEPS = [512, 640, 768, 896, 1000, 1500, 2000]`;
#   Pythia has every 1,000 steps plus 1, 2, 4 … 512). Where exactly does it happen?
# * Use a bigger Pythia (`pythia-160m`, `pythia-410m`). Does the jump come earlier or
#   later in tokens? Do facts arrive sooner?
# * Probe other facts or skills: arithmetic ("2 + 2 ="), translation, code.
#   Which ones never arrive in a 70M model?
# * Combine with *Transformers, Inside Out* lesson 07: find the induction heads in
#   the step-1000 and step-4000 checkpoints and see them form.
