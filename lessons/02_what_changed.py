# %% [markdown]
# # Lesson 02 — What did post-training change?
#
# Lesson 01: same architecture, very different behaviour. So the difference
# lives entirely in the numbers. We have both sets of weights, so we can
# subtract them:
#
#     Δ = W_instruct − W_base
#
# Questions: **How big is Δ? Where is it? Is it a big rewrite or a small,
# precise edit? And did the model forget anything?**

# %%
import os, sys, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd())
from common import *
import matplotlib.pyplot as plt

seed_all()
torch.set_grad_enabled(False)
tok, base, device = load(BASE_ID)
_, inst, _ = load(INSTRUCT_ID)
ctok = chat_tokenizer()

W_base = {n: p.detach().clone() for n, p in base.named_parameters()}
W_inst = {n: p.detach() for n, p in inst.named_parameters()}
results = {}

# %% [markdown]
# ## How big is the change?
#
# For each weight matrix, compare the size of the change with the size of the
# matrix: `‖Δ‖ / ‖W_base‖`. (‖·‖ is the square root of the sum of squares:
# the "length" of all the numbers.)

# %%
banner("Relative change, whole model")
flat_b = torch.cat([w.flatten() for w in W_base.values()])
flat_d = torch.cat([(W_inst[n] - w).flatten() for n, w in W_base.items()])
rel_global = (flat_d.norm() / flat_b.norm()).item()
print(f"  ‖Δ‖ / ‖W_base‖ = {rel_global:.1%}")
print(f"  share of weights that moved by less than 0.001: {(flat_d.abs() < 1e-3).float().mean():.1%}")
print(f"  typical |weight| {flat_b.abs().median():.4f}   typical |change| {flat_d.abs().median():.4f}")
results["rel_global"] = rel_global

# %%
banner("Where did it change? (relative change per matrix, by layer)")
KINDS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
         "input_layernorm", "post_attention_layernorm"]
L = base.config.num_hidden_layers
grid = torch.zeros(len(KINDS), L)
for n, w in W_base.items():
    if ".layers." not in n:
        continue
    layer = int(n.split(".layers.")[1].split(".")[0])
    kind = next(k for k in KINDS if k in n)
    grid[KINDS.index(kind), layer] = ((W_inst[n] - w).norm() / w.norm()).item()
for n in ("model.embed_tokens.weight", "model.norm.weight"):
    print(f"  {n:28} {((W_inst[n] - W_base[n]).norm() / W_base[n].norm()).item():6.1%}")
top = sorted(((grid[i, j].item(), KINDS[i], j) for i in range(len(KINDS)) for j in range(L)), reverse=True)[:5]
print("  biggest changes:", ", ".join(f"layer {j} {k} {v:.0%}" for v, k, j in top))
print(f"  average by layer: first 3 layers {grid[:, :3].mean():.1%}, the rest {grid[:, 3:].mean():.1%}")

fig, ax = plt.subplots(figsize=(11, 3.4))
im = ax.imshow(grid, aspect="auto", cmap="viridis")
ax.set_yticks(range(len(KINDS)), KINDS, fontsize=8)
ax.set_xlabel("layer")
ax.set_title("How much each matrix changed, base → instruct  (‖Δ‖/‖W‖)")
fig.colorbar(im, ax=ax, format=lambda x, _: f"{x:.0%}")
show_fig(fig, "02_delta_map")
results["delta_grid"] = {"kinds": KINDS, "grid": [[round(v, 4) for v in r] for r in grid.tolist()]}

# %% [markdown]
# Around 9% overall. The change is spread across every layer, strongest at
# the very start (layer 0) where tokens first get read. Is 9% a lot? It
# depends on the **direction** of the change, not only its size. Test: make a
# *random* change of exactly the same size in each matrix and compare.

# %%
TEXT = ("The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. "
        "It is named after the engineer Gustave Eiffel, whose company designed and built the tower "
        "from 1887 to 1889. Locally nicknamed \"La dame de fer\", it was constructed as the "
        "centerpiece of the 1889 World's Fair.")
Q = "What is the capital of France?"


def perplexity(m, text=TEXT):
    """exp(average loss) on encyclopedia text: lower = the model 'knows' this kind of text."""
    x = encode(tok, text, device)
    return m(x, labels=x).loss.exp().item()


def answer(m, n=30):
    return complete(tok, m, chat_prompt(ctok, Q), n)


mix = copy.deepcopy(base)


def set_weights(fn):
    for n, p in mix.named_parameters():
        p.copy_(fn(n))


banner("Same size of change, different direction")
print(f"  {'model':26} {'perplexity':>10}   answer to {Q!r}")
print(f"  {'base':26} {perplexity(base):10.2f}   {answer(base)!r}")
print(f"  {'instruct (trained Δ)':26} {perplexity(inst):10.2f}   {answer(inst)!r}")
rand_rows = []
for scale in (1.0, 0.5, 0.25):
    g = torch.Generator(device="cpu").manual_seed(0)

    def rand_delta(n):
        d = W_inst[n] - W_base[n]
        r = torch.randn(d.shape, generator=g).to(device)
        return W_base[n] + r * (d.norm() / r.norm()) * scale
    set_weights(rand_delta)
    ppl = perplexity(mix)
    rand_rows.append({"scale": scale, "ppl": ppl})
    print(f"  {f'base + random Δ ×{scale}':26} {ppl:10.2f}   {answer(mix)!r}")
results["random"] = rand_rows
results["ppl_base"], results["ppl_inst"] = perplexity(base), perplexity(inst)

# %% [markdown]
# A random change of the same size **nearly doubles the perplexity** and still
# doesn't chat. The trained change hardly moves the perplexity on
# encyclopedia text (the facts survived) *and* it adds a whole new behaviour.
#
# **Aha:** post-training is a small, very precisely aimed edit. Nearly all the
# knowledge was already in the base model.
#
# ## Dial it in: how much of Δ do we need?
#
#     W(α) = W_base + α · Δ          α = 0 → base,  α = 1 → instruct

# %%
banner("Walking from base (α=0) to instruct (α=1)")
im_end = tok.convert_tokens_to_ids("<|im_end|>")
alpha_rows = []
for a in (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.25):
    set_weights(lambda n: W_base[n] + a * (W_inst[n] - W_base[n]))
    x = encode(tok, chat_prompt(ctok, Q), device)
    first = mix(x).logits[0, -1].softmax(-1)
    ans = answer(mix)
    alpha_rows.append({"alpha": a, "ppl": perplexity(mix), "answer": ans})
    print(f"  α={a:4.2f}  ppl {alpha_rows[-1]['ppl']:5.2f}  {ans!r}")
results["alpha"] = alpha_rows

# %% [markdown]
# Somewhere between 10% and 30% of the way, it switches from "lost" to
# "assistant". The rest of the way mostly polishes the style. You can even
# go *past* 1.0 (extrapolate), which is a real trick ("model arithmetic");
# check whether it helps or hurts here.
#
# ## Which tokens changed?
#
# Let the instruct model write answers. Then, at every position, ask the
# **base** model: what rank would you have given that same token?
# Rank 0 means "the base model would have picked it too".

# %%
banner("Instruct's tokens, ranked by the base model")
PROMPTS = ["What is the capital of France?", "Give me three tips for sleeping better.",
           "Write a haiku about the sea.", "How do I boil an egg?", "Why is the sky blue?",
           "Can you help me plan a birthday party?", "What is photosynthesis?"]
all_ranks, examples = [], []
for p in PROMPTS:
    prompt = chat_prompt(ctok, p)
    reply = complete(tok, inst, prompt, 60)
    full = encode(tok, prompt + reply, device)
    n0 = encode(tok, prompt, device).shape[1]
    logits_b = base(full).logits[0, n0 - 1:-1]       # base model's guesses for each answer token
    target = full[0, n0:]
    ranks = (logits_b > logits_b.gather(1, target[:, None])).sum(1).tolist()
    all_ranks += ranks
    examples.append({"prompt": p, "tokens": [tok.decode([t]) for t in target.tolist()], "ranks": ranks})
    moved = [f"{tok.decode([t])!r}:{r}" for t, r in zip(target.tolist(), ranks) if r >= 5][:8]
    print(f"  {p[:38]:38}  tokens the base would NOT have picked (rank ≥5): {' '.join(moved)}")

R = torch.tensor(all_ranks)
share = {"rank 0": (R == 0).float().mean().item(), "rank 1-2": ((R >= 1) & (R <= 2)).float().mean().item(),
         "rank 3-9": ((R >= 3) & (R <= 9)).float().mean().item(), "rank ≥10": (R >= 10).float().mean().item()}
print()
for k, v in share.items():
    print(f"  {k:9} {v:6.1%} {'█' * int(v * 60)}")
results["rank_share"], results["rank_examples"] = share, examples

fig, ax = plt.subplots(figsize=(6, 3))
ax.bar(list(share), list(share.values()), color=["#0b7a68", "#5aa897", "#b4b4b4", "#b4530a"])
ax.set_ylabel("share of the instruct model's tokens")
ax.set_title("How the BASE model ranks the INSTRUCT model's choices")
show_fig(fig, "02_token_ranks")

# %% [markdown]
# About half of the instruct model's tokens are the base model's top choice
# too, and most of the rest are in its top three. The tokens that changed
# rank are mostly **style and format**: the first word of the answer
# ("Here", "Absolutely"), markdown ("**", "1"), and words that set the tone.
# Some content words move too (" Paris" is only rank 22 for the base model,
# because the chat template confuses it so much).
#
# **Aha:** this is the *superficial alignment hypothesis* (LIMA, 2023; URIAL,
# 2023). Post-training mostly teaches *how* to talk, while *what* the model knows
# comes from pretraining. So how do you teach that "how"? Lesson 03: write the
# training loop that does it.

# %%
save_result("02_what_changed", results)

# %% [markdown]
# ## Try:
# * Which matrices matter most? Copy Δ into only the first 3 layers
#   (`set_weights(lambda n: W_inst[n] if '.layers.0.' in n ... else W_base[n])`). Does it chat?
# * Copy Δ for only attention (`q/k/v/o_proj`) or only the MLP. Which one carries the "chat-ness"?
# * Measure perplexity on a *chat* reply instead of encyclopedia text. Which model wins now?
# * Look at the per-token ranks in `outputs/results/02_what_changed.json`: which
#   *kinds* of words get re-ranked most?
