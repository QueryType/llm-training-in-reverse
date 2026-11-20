# %% [markdown]
# # Lesson 02 — What did post-training change?
#
# Lesson 01: same architecture, very different behaviour. So the difference
# lives entirely in the numbers. We have both sets of weights, so we can
# subtract them:
#
#     Δ = W_instruct − W_base
#
# This is a luxury. Normally you only get the finished chat model. SmolLM2
# publishes both ends of post-training, so Δ is *exactly* what SFT and
# preference tuning did, summed up into one set of numbers per weight.
#
# Questions: **How big is Δ? Where is it? Is it a big rewrite or a small,
# precise edit? And did the model forget anything?**
#
# We'll answer them with four experiments, all without any training:
# 1. measure Δ: total size, and size per matrix and layer;
# 2. replace Δ with **random** noise of the same size, to see if size alone matters;
# 3. add only a **fraction** α of Δ, to see how much of it you need;
# 4. ask the base model to **rate the instruct model's words**, token by token.

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
#
# Why relative? A raw difference of 0.01 means little on its own: it's huge
# for a weight that is 0.001 and noise for one that is 10. Dividing by ‖W‖
# says "how far did this matrix move, as a fraction of its own size?". We
# also look at the *typical* single weight and the typical single change
# (medians, so a few outliers don't dominate).

# %%
banner("Relative change, whole model")
flat_b = torch.cat([w.flatten() for w in W_base.values()])
flat_d = torch.cat([(W_inst[n] - w).flatten() for n, w in W_base.items()])
rel_global = (flat_d.norm() / flat_b.norm()).item()
print(f"  ‖Δ‖ / ‖W_base‖ = {rel_global:.1%}")
print(f"  share of weights that moved by less than 0.001: {(flat_d.abs() < 1e-3).float().mean():.1%}")
print(f"  typical |weight| {flat_b.abs().median():.4f}   typical |change| {flat_d.abs().median():.4f}")
results["rel_global"] = rel_global
results["share_tiny"] = (flat_d.abs() < 1e-3).float().mean().item()
results["median_w"], results["median_d"] = flat_b.abs().median().item(), flat_d.abs().median().item()

# %% [markdown]
# 8.6%. And it's **dense**: only 7.3% of the weights moved by less than 0.001.
# A typical weight is about 0.107 and a typical change about 0.010, so almost
# every one of the 134.5M numbers was nudged by roughly a tenth of its size.
# Post-training didn't find "the chat neurons" and rewrite them; it moved
# everything a little.
#
# Now break it down by matrix and layer. (`q/k/v/o_proj` are the attention
# projections, `gate/up/down_proj` the MLP, and the two layernorms are the
# per-channel scales applied before each of them.)

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
    results[n] = ((W_inst[n] - W_base[n]).norm() / W_base[n].norm()).item()
    print(f"  {n:28} {results[n]:6.1%}")
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
# Two patterns stand out in the map:
#
# * **Layer 0 moved most**: its `v_proj` changed by 33%, its input layernorm
#   by 32%, its `o_proj` by 23%. This is where the model first reads the
#   tokens. The chat template is full of tokens the base model barely used
#   (`<|im_start|>`, `<|im_end|>`, the word `assistant` right after them), and
#   the first layer is where their meaning gets set up. The token embeddings
#   moved by 11.7%, more than the average matrix, for the same reason.
# * **Queries and keys grow with depth**: `q_proj` and `k_proj` move ~4–7% in
#   the first layers and ~11–12% in the last ones. Attention in the late
#   layers decides *which earlier tokens to use* for the next word; a new role
#   ("answer the user") plausibly needs new choices there. (Plausibly: this
#   map shows *where* things changed, not *why*.)
#
# The final norm barely moved (0.7%).
#
# Is 8.6% a lot? It depends on the **direction** of the change, not only its
# size. Test: make a *random* change of exactly the same size in each matrix
# and compare. We measure two things for each model: **perplexity** on a short
# encyclopedia paragraph (lower = the text looks more familiar; perplexity is
# exp of the average loss, and lesson 06 explains it properly) and its answer
# to the France question inside the chat template. (If you did *Transformers,
# Inside Out*: its lesson 11 measured SmolLM2's perplexity as 19.9 on a
# different text. Perplexity depends on the text as much as on the model, so
# only compare numbers measured on the same text.)

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
# A random change of the same size **nearly doubles the perplexity**
# (5.87 → 10.74) and still doesn't chat; it starts writing quiz questions.
# Even a quarter of that random change costs as much perplexity as the whole
# trained one. The trained change barely moves the perplexity on encyclopedia
# text (5.87 → 6.01: the knowledge survived) *and* it adds a whole new
# behaviour.
#
# In a space of 134.5 million dimensions, almost every direction is useless
# or harmful. Training found one of the very few that help.
#
# **Aha:** post-training is a small, very precisely aimed edit. Nearly all the
# knowledge was already in the base model.
#
# ## Dial it in: how much of Δ do we need?
#
#     W(α) = W_base + α · Δ          α = 0 → base,  α = 1 → instruct
#
# If chatting needed the full edit, the answers should stay broken until α is
# close to 1. If it's one main direction plus fine-tuning on top, they
# should switch early.

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
# At α=0.1 the model is half-way: it mentions France and Paris but rambles
# ("It's now calledParis"). At **α=0.2 it already answers**: "The capital
# city of France is Paris." A fifth of the edit is enough to switch from
# "lost" to "assistant". The rest of the way changes the style (its facts
# get worse at 0.2: Paris "in the south-east corner of Europe") and the
# phrasing slowly becomes the instruct model's.
#
# Look at the perplexity column too. It **drops** below the base model's
# between α=0.1 and 0.75 (best 5.61 vs 5.87), then rises to 6.01 at α=1.
# Part of Δ makes the model better at ordinary text, and the last stretch
# trades a little of that for chat style. Some forgetting does happen; it's
# just small. (Remember this when lesson 04 pulls harder.)
#
# Going *past* 1.0 (extrapolation, a real trick sometimes called "model
# arithmetic") doesn't help here: at α=1.25 perplexity climbs to 6.69 and the
# answer becomes "The capital of France. It's not an easy task to find".
#
# ## Which tokens changed?
#
# Let the instruct model write answers. Then, at every position, ask the
# **base** model: what rank would you have given that same token?
# Rank 0 means "the base model would have picked it too".
#
# This turns "what changed?" into a question about words, not weights. If
# post-training rewrote the model's knowledge, the base model would rank the
# instruct model's tokens badly everywhere. If it mostly changed the style,
# the bad ranks should cluster on a few kinds of tokens.

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
# 52.5% of the instruct model's tokens are the base model's top choice too,
# and 74.5% are in its top three. Only 10.9% are ranked 10th or worse. And the
# tokens that changed rank most are mostly **style and format**:
#
# * the first word of the answer: "Absol(utely)" is rank 1,189 for the base
#   model, "Bo(iling)" 247, "Here" 28;
# * markdown: the "1" that starts a numbered list (ranks 140–2,749) and "**"
#   for bold (up to 295);
# * words that set a tone: "angry beast", "whisper secrets" in the haiku.
#
# Once the instruct model has written "Here's my advice...\n\n1. **", the
# base model happily predicts the rest itself. The big re-rankings happen at
# the *start* of an answer and at *formatting* points; in between, the base
# model mostly agrees.
#
# Some content words move too (" Paris" is only rank 22 for the base model).
# That isn't missing knowledge: the base model answered Paris to the plain
# question in lesson 01. It's the chat template confusing it so much that it
# has no idea a capital is being asked for.
#
# **Aha:** this is the *superficial alignment hypothesis* (LIMA, 2023; URIAL,
# 2023). Post-training mostly teaches *how* to talk, while *what* the model
# knows comes from pretraining. Three measurements agree: the edit is small and
# aimed (random noise of the same size breaks things), a fifth of it is enough
# to switch roles, and most of the new tokens are ones the base model would
# have chosen anyway.
#
# So how do you teach that "how"? Lesson 03: write the training loop that does it.

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
