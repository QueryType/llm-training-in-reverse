# %% [markdown]
# # Lesson 14 — Make *your* model chat
#
# The tape has run all the way back and forward again. In lesson 03 we took
# someone else's base model and made it an assistant with SFT. Now we do the
# same to **our own** model from lesson 11, which was born in this course
# from random weights.
#
# Our model only knows children's stories, so its "chat" will be a story
# writer that takes instructions: *"Write a story that uses the words dog,
# ball and happy, and has dialogue."* The data comes from TinyStoriesInstruct,
# where every story comes with the words, features and summary it was written
# from. That gives us something lesson 03 didn't have: **a way to check
# whether the instruction was followed.**
#
# Questions: **can a 12M model learn to follow instructions? How much of
# lesson 03's recipe carries over unchanged?**

# %%
import os, sys, random, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd())
from common import *
import torch.nn.functional as F
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

seed_all(0)
device = pick_device()
PRE = CKPT / "11_pretrain"
if not PRE.exists():
    sys.exit("Run lessons/11_pretrain.py first: this lesson starts from its model.")
tok = bpe_tokenizer()
model = AutoModelForCausalLM.from_pretrained(PRE, dtype=torch.float32).to(device)
CTX = model.config.max_position_embeddings
IM_END = tok.convert_tokens_to_ids("<|im_end|>")
print(f"device: {device}   our model: {n_params(model) / 1e6:.1f}M params, context {CTX}")
print(f"  '<|im_start|>' and '<|im_end|>' are single tokens: {tok('<|im_start|>user<|im_end|>').input_ids}")
R = {}

# %% [markdown]
# ## The data: stories with their instructions
#
# Each TinyStoriesInstruct record looks like:
#
#     Words: dog, ball, happy
#     Features: Dialogue
#     Summary: ...
#     Story: ...
#
# We turn each into a one-turn conversation. The user asks for a story with
# some of these constraints; the assistant's reply is the story.

# %%
path = hf_hub_download("roneneldan/TinyStoriesInstruct", "TinyStories-Instruct-valid.txt", repo_type="dataset")
raw = open(path, encoding="utf-8").read().split("<|endoftext|>")


def parse(block):
    r = {}
    m = re.search(r"Story:\s*(.*)", block, re.S)
    if not m:
        return None
    story = re.split(r"\n(?:Summary|Features|Words|Random sentence):", m.group(1))[0]   # fields can follow the story
    r["story"] = clean_story(story)
    head = block[:m.start()]
    for key in ("Words", "Features", "Summary"):
        mm = re.search(rf"{key}:\s*(.*)", head)
        if mm:
            r[key.lower()] = clean_story(mm.group(1))
    return r if r["story"] else None


records = [r for r in (parse(b) for b in raw) if r and "words" in r]
rng = random.Random(0)


def instruction(r, rng):
    words = [w.strip() for w in r["words"].split(",") if w.strip()]
    feats = [f.strip() for f in r.get("features", "").split(",") if f.strip()]
    ask = f"Write a story that uses the words {', '.join(words)}."
    if "Dialogue" in feats and rng.random() < 0.7:
        ask += " Include dialogue."
    if "BadEnding" in feats and rng.random() < 0.7:
        ask += " Give it a sad ending."
    if "summary" in r and rng.random() < 0.3:
        ask = f"Write a story about this: {r['summary']} " + ask
    return ask, words


convs = []
for r in records:
    ask, words = instruction(r, rng)
    msgs = [{"role": "user", "content": ask}, {"role": "assistant", "content": r["story"]}]
    ids, labels = chat_ids(tok, msgs)
    if len(ids) <= CTX:                              # our model's window is 256 tokens
        convs.append({"ask": ask, "words": words, "ids": ids, "labels": labels,
                      "dialogue": "Include dialogue." in ask, "sad": "sad ending" in ask})
rng.shuffle(convs)
test, train = convs[:300], convs[300:]
print(f"  {len(records):,} records with word lists, {len(convs):,} fit in {CTX} tokens → {len(train):,} train, {len(test)} test")
print(f"\n  example instruction: {train[0]['ask']}")
print(f"  example story: {tok.decode([t for t in train[0]['labels'] if t != -100])[:200]!r}...")
R["n_train"], R["n_test"] = len(train), len(test)

# %% [markdown]
# ## How to grade a story
#
# * **words**: share of requested words that appear in the story (any form: "jump" counts "jumped")
# * **dialogue**: when asked, does the story contain quoted speech?
# * **stops**: does the model end its turn with `<|im_end|>` by itself?

# %%
def grade(m, items, n=120, max_new=220):
    m.eval()
    scores = {"words": [], "dialogue": [], "stops": [], "examples": []}
    for it in items[:n]:
        prompt = tok(our_chat_prompt(tok, it["ask"])).input_ids
        out = sample_text(m, prompt, max_new, temperature=0.7, device=device, seed=len(scores["words"]), stop_id=IM_END)
        stopped = IM_END in out
        story = tok.decode([t for t in out if t != IM_END], skip_special_tokens=True).lower()
        hits = [bool(re.search(rf"\b{re.escape(w.lower()[:max(4, len(w) - 2)])}", story)) for w in it["words"]]
        scores["words"].append(sum(hits) / len(hits))
        scores["stops"].append(stopped)
        if it["dialogue"]:
            scores["dialogue"].append('"' in story)
        if len(scores["examples"]) < 5:
            scores["examples"].append({"ask": it["ask"], "story": story, "hits": hits, "words": it["words"]})
    m.train()
    avg = lambda v: sum(v) / max(len(v), 1)
    return {"words": avg(scores["words"]), "dialogue": avg(scores["dialogue"]), "stops": avg(scores["stops"]),
            "n": len(scores["words"]), "n_dialogue": len(scores["dialogue"]), "examples": scores["examples"]}


# A base rate: how often do the requested words appear in a *random* story from the data?
base_rate = sum(sum(bool(re.search(rf"\b{re.escape(w.lower()[:max(4, len(w) - 2)])}",
                                   tok.decode(train[i + 1]["ids"]).lower())) for w in it["words"]) / len(it["words"])
                for i, it in enumerate(test[:120])) / 120
before = grade(model, test)
banner("Before SFT: our pretrained model, given the chat prompt")
print(f"  requested words used: {before['words']:.0%}   (a random story from the data uses {base_rate:.0%})")
print(f"  dialogue when asked:  {before['dialogue']:.0%}   stops by itself: {before['stops']:.0%}")
print(f"  e.g. {before['examples'][0]['ask']!r}\n       → {before['examples'][0]['story'][:200]!r}")
R["base_rate"], R["before"] = base_rate, before

# %% [markdown]
# 10,953 conversations fit in the 256-token window; 300 are held out for
# testing. (We use the *validation* file of TinyStoriesInstruct: it's small,
# and lesson 11 never trained on it, so none of these stories are memorised.)
#
# Before SFT: requested words used **7%** (a random story from the data
# scores 4%, so that's barely above chance), stops by itself **0%**.
#
# Before SFT, the model has never seen `<|im_start|>` or `<|im_end|>`: they
# never appeared in lesson 11's token stream. So it ignores the request,
# writes *a* story (sometimes starting with the moral, sometimes several in a
# row), and never ends its turn. The dialogue number is high already because
# most TinyStories contain dialogue; it can't tell us much either way.
#
# ## SFT, exactly as in lesson 03
#
# Same `chat_ids` masking, same loss, same loop. Only the model and data changed.

# %%
STEPS = 1500 if FULL else 600
BATCH, LR, WARMUP = 16, 5e-4, 30
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0, betas=(0.9, 0.95))
log = {"step": [], "train": [], "test_step": [], "test": []}


@torch.no_grad()
def test_loss(m):
    m.eval()
    x, y, att = pad_batch([(c["ids"], c["labels"]) for c in test], device)
    loss = F.cross_entropy(m(input_ids=x, attention_mask=att).logits[:, :-1].flatten(0, 1), y[:, 1:].flatten(), ignore_index=-100)
    m.train()
    return loss.item()


log["test_step"].append(0); log["test"].append(test_loss(model))
banner(f"SFT: {STEPS} steps × {BATCH} conversations, lr {LR}")
order, t0 = [], time.perf_counter()
for step in range(1, STEPS + 1):
    if len(order) < BATCH:
        order += rng.sample(range(len(train)), len(train))
    x, y, att = pad_batch([(train[i]["ids"], train[i]["labels"]) for i in (order.pop() for _ in range(BATCH))], device)
    for grp in opt.param_groups:
        grp["lr"] = LR * min(1, step / WARMUP) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / STEPS)))
    loss = F.cross_entropy(model(input_ids=x, attention_mask=att).logits[:, :-1].flatten(0, 1), y[:, 1:].flatten(),
                           ignore_index=-100)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    log["step"].append(step); log["train"].append(loss.item())
    if step % 100 == 0 or step == STEPS:
        log["test_step"].append(step); log["test"].append(test_loss(model))
        print(f"  step {step:4}  train {sum(log['train'][-100:]) / len(log['train'][-100:]):.3f}"
              f"  test {log['test'][-1]:.3f}  ({time.perf_counter() - t0:.0f}s)")
R["log"] = log

# %% [markdown]
# Test loss on the story tokens: 2.01 → 1.80 after 100 steps → 1.63 after
# 600. Like lesson 03, most of the drop comes early: the format is learned
# first. 600 × 16 = 9,600 conversations, less than one pass over the data.

# %%
after = grade(model, test)
banner("After SFT")
for k, label in (("words", "requested words used"), ("dialogue", "dialogue when asked"), ("stops", "stops by itself")):
    print(f"  {label:22} {before[k]:5.0%} → {after[k]:5.0%}")
print(f"  (words and stops: {after['n']} requests; dialogue: only the {after['n_dialogue']} that asked for it)")
R["after"] = after

# %%
banner("Your model, taking requests")
asks = ["Write a story that uses the words cat, rain and umbrella.",
        "Write a story that uses the words dragon, cake and friend. Include dialogue.",
        "Write a story that uses the words boat, storm and brave. Give it a sad ending."]
custom = {}
for i, a in enumerate(asks):
    out = sample_text(model, tok(our_chat_prompt(tok, a)).input_ids, 220, temperature=0.7, device=device, seed=i, stop_id=IM_END)
    custom[a] = tok.decode([t for t in out if t != IM_END], skip_special_tokens=True).strip()
    print(f"\n  > {a}\n  {custom[a]}")
R["custom"] = custom

CKPT.mkdir(parents=True, exist_ok=True)
model.save_pretrained(CKPT / "14_chat")
tok.save_pretrained(CKPT / "14_chat")
print(f"\n  saved -> {(CKPT / '14_chat').relative_to(ROOT)}")

# %%
fig, axs = plt.subplots(1, 2, figsize=(11, 3.4))
axs[0].plot(log["step"], log["train"], alpha=0.35, label="train")
axs[0].plot(log["test_step"], log["test"], "o-", label="test")
axs[0].set_xlabel("step"); axs[0].set_ylabel("loss on story tokens"); axs[0].legend(); axs[0].set_title("SFT on our own model")
ks = ["words", "dialogue", "stops"]
xb = range(len(ks))
axs[1].bar([i - 0.2 for i in xb], [before[k] for k in ks], 0.4, label="before")
axs[1].bar([i + 0.2 for i in xb], [after[k] for k in ks], 0.4, label="after")
axs[1].set_xticks(list(xb), ["words used", "dialogue", "stops"]); axs[1].set_ylim(0, 1); axs[1].legend()
axs[1].set_title("Following instructions")
show_fig(fig, "14_chat")

# %% [markdown]
# | | before SFT | after SFT | measured on |
# |---|---|---|---|
# | requested words used | 7% | **35%** | 120 requests |
# | stops by itself | 0% | **95%** | 120 requests |
# | dialogue when asked | 73% | 80% | 30 requests |
#
# Read the three custom requests above with the words in mind. "Cat, rain
# and umbrella" gets a cat and rain, no umbrella. "Dragon, cake and friend"
# gets a lovely cake story, with no dragon. "Boat, storm and brave" gets a
# brave boy, no boat, no storm, and no sad ending. Every story is fluent,
# ends cleanly and is about *one* of the things asked for.
#
# One pattern worth checking: in all five saved test examples, the *first*
# requested word is the one that's missing (act, stir, regret, guess, get).
# Five examples prove nothing, but it's the kind of thing a real evaluation
# would break down (see **Try**).
#
# **Aha:**
# * The recipe from lesson 03 works unchanged on a model we built ourselves.
#   SFT doesn't care where the base model came from.
# * Our model now does something its pretraining never asked for. It ends its
#   turn almost every time, and it uses the requested words 5× more often
#   than before, though far from always: a 12M model with 12 minutes of
#   pretraining and 600 steps of SFT follows instructions *partly*. Like
#   lesson 03, SFT taught **format and role**; the story-writing came from pretraining.
# * The dialogue number is the shakiest one. It's measured on only the few
#   dozen requests that asked for dialogue, most stories contain dialogue
#   anyway, and in our runs it moved by up to ±15 points from one run to the
#   next (once up, once down). With that few samples, a change that size is
#   noise. Before you believe a metric moved, check how many samples it rests on.
# * Having a checkable instruction (words, dialogue) turns "seems better" into
#   a number. That's the core of every instruction-following benchmark, and
#   it's what you'd need to try lesson 04's DPO here too.
#
# ## The whole course, in one table
#
# | lesson | question | answer, measured |
# |---|---|---|
# | 01 | what does post-training change? | format and role, not size: same 134.5M weights |
# | 02 | how much did the weights move? | 8.6%, everywhere, in a direction that matters |
# | 03 | where does "answering" come from? | 200 steps of SFT on 2k chats |
# | 04 | and preferences? | DPO learns what the labels measure (length), at a hidden cost |
# | 05 | where does knowledge come from? | pretraining, in a fixed order; copying switches on suddenly |
# | 06 | what does the loss mean? | average surprise; compare in bits per character |
# | 07 | what's a gradient? | softmax − one_hot, chained back; checkable by hand |
# | 08 | why Adam? | per-weight step sizes; >1 nat better than SGD here |
# | 09 | why does lr matter? | narrow sweet spot; warmup, decay, clipping |
# | 10 | where does data come from? | BPE + dedup + decontamination |
# | 11 | can we pretrain? | 12M params, 12 min, 0.69 bits/char |
# | 12 | what limits training? | activations (98% of memory here), and hardware |
# | 13 | how big a model? | ~20 tokens per parameter for a fixed budget |
# | 14 | can ours chat? | yes, partly: stops 95%, uses 35% of asked words |
#
# The whole pipeline, end to end, is now yours: data → tokenizer → pretraining → SFT.

# %%
save_result("14_make_it_chat", R)

# %% [markdown]
# ## Try:
# * Add lesson 04's DPO: sample two stories per request, prefer the one that uses
#   more of the requested words. Does the words score go up further?
# * Add a new instruction type the data supports: "Use this sentence: …" (the
#   records have a "Random sentence" field).
# * Grade more requests (`grade(model, test, n=300)`) and split the words score
#   by position in the list. Is the first word really missed more often?
# * Ask for things outside the training data ("Write a poem", "What is 2+2?").
#   What does a 12M story model do?
# * Look inside it with *Transformers, Inside Out*: `MODEL_ID=outputs/checkpoints/14_chat`
#   and the logit lens (lesson 06). When does it "decide" to use a requested word?
