# %% [markdown]
# # Lesson 03 — Do SFT yourself
#
# Lesson 02 showed that the chat model is the base model plus a small, aimed
# edit. The first tool for making that edit is **supervised fine-tuning
# (SFT)**: show the model example conversations and train it to predict the
# assistant's words.
#
# That's the whole idea. It's the same "predict the next token" training as
# pretraining, with two changes:
# 1. the data is **conversations** in the chat template, and
# 2. we only **grade the assistant's tokens** (loss masking).
#
# "Supervised" just means every example comes with the right answer: a
# human (or a bigger model) wrote what the assistant should say, and we train
# the model to say it. There's no reward, no judge, no comparison between
# answers; that comes in lesson 04.
#
# Question: **how much data and time does it take to turn our base model into
# an assistant?** We'll use 2,260 short conversations (`everyday-conversations`,
# a dataset from the SmolLM2 team) and about 2 minutes of training.
#
# This is the first lesson that *trains* anything, so it's also our first
# look at the training loop that every later lesson takes apart: forward pass,
# loss, backward pass, optimizer step. Here we use it as a black box from
# PyTorch. Lessons 06–09 open each piece.

# %%
import os, sys, random, inspect
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd())
from common import *
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datasets import load_dataset

seed_all(0)
tok, model, device = load(BASE_ID, train=True)
ds = load_dataset("HuggingFaceTB/everyday-conversations-llama3.1-2k")
train_convs = list(ds["train_sft"]["messages"])
test_convs = list(ds["test_sft"]["messages"])
print(f"device: {device}   train: {len(train_convs)} conversations   test: {len(test_convs)}")

# %% [markdown]
# ## The data
#
# Each example is a list of messages, alternating user and assistant, about
# 7.7 messages long on average. The topics are everyday small talk: greetings,
# budgeting, cooking, travel. No code, no maths, no poems.

# %%
banner("One training conversation")
for m in train_convs[0][:4]:
    print(f"  {m['role']:>9}: {m['content'][:90]}")

# %% [markdown]
# ## From conversation to (ids, labels)
#
# We write the template ourselves: one line per message. Then we make a
# second list, `labels`, with the same length. Wherever it says **-100**,
# PyTorch's loss ignores that position. That's all "loss masking" is.
#
# Note what *is* graded: the assistant's words **and** the `<|im_end|>` after
# them. That one token is how the model learns to stop (lesson 01). The
# header `<|im_start|>assistant\n` is not graded: at inference time we write
# it ourselves, so the model never needs to produce it.
#
# Also note that we don't use a system prompt. SmolLM2-Instruct's template
# adds one by default; our models will be trained without it, so we prompt
# them without it too (`our_chat_prompt`). A model only knows the format it
# was trained on.

# %%
print(inspect.getsource(chat_ids))

# %%
banner("Which tokens get graded?   (^ = learned, . = ignored)")
ids, labels = chat_ids(tok, train_convs[0][:2])
toks = [tok.decode([i]).replace("\n", "↵") for i in ids]
line1, line2 = "", ""
for t, l in zip(toks, labels):
    w = max(len(t), 1)
    line1 += t.ljust(w) + " "
    line2 += ("^" if l != -100 else ".") * w + " "
for a in range(0, len(line1), 100):
    print("  " + line1[a:a + 100] + "\n  " + line2[a:a + 100])

all_data = [chat_ids(tok, c) for c in train_convs]
n_tok = sum(len(i) for i, _ in all_data)
n_learn = sum(sum(l != -100 for l in lab) for _, lab in all_data)
print(f"\n  whole dataset: {n_tok:,} tokens, of which {n_learn:,} ({n_learn / n_tok:.0%}) are graded")

# %% [markdown]
# 57% of the tokens are graded: assistant replies are longer than user
# messages in this dataset, so the masked part is less than half.
#
# ## The loss, written out
#
# The model sees tokens `0..n-1` and predicts tokens `1..n`. So the logits at
# position *t* are compared with the token at position *t+1*: shift by one.
#
# For each graded position, the loss is **cross-entropy**: minus the log of
# the probability the model gave to the real next token. Give it 100% and
# the loss is 0; give it 1% and the loss is 4.6. The loss of a batch is the
# average over all graded positions. (Lesson 06 is about what this number
# means; for now, lower is better.)

# %%
def sft_loss(model, x, y, att):
    logits = model(input_ids=x, attention_mask=att).logits
    return F.cross_entropy(logits[:, :-1].flatten(0, 1),     # predictions for positions 1..n
                           y[:, 1:].flatten(),                  # the real tokens there
                           ignore_index=-100)                   # skip the masked ones

x, y, att = pad_batch(all_data[:4], device)
with torch.no_grad():
    ours = sft_loss(model, x, y, att).item()
    lib = model(input_ids=x, attention_mask=att, labels=y).loss.item()
print(f"  our loss {ours:.5f}   transformers' loss {lib:.5f}   (the library does the same shift)")

# %% [markdown]
# ## Before training: a baseline
#
# The **test loss** on assistant tokens of 119 held-out conversations, and a
# few answers. For comparison, we also score Hugging Face's own
# SmolLM2-135M-Instruct (trained on hundreds of thousands of varied
# conversations, including ones like these, then preference-tuned).
#
# The test conversations are never used for training. If the model only
# memorised the training set, its train loss would fall and its test loss
# wouldn't. Watching both is the simplest check that it's learning something
# general.
#
# Half the prompts look like the training data; the other half (a fact, a
# poem) don't. That tells us what SFT transfers and what it doesn't.

# %%
test_data = [chat_ids(tok, c) for c in test_convs]


@torch.no_grad()
def test_loss(m, data=test_data, bs=16):
    m.eval()
    tot, n = 0.0, 0
    for i in range(0, len(data), bs):
        x, y, att = pad_batch(data[i:i + bs], device)
        k = (y[:, 1:] != -100).sum().item()
        tot += sft_loss(m, x, y, att).item() * k
        n += k
    m.train()
    return tot / n


PROMPTS = ["Hi! Can you help me find a good book to read?",     # like the training data
           "What should I cook for dinner tonight?",
           "What is the capital of France?",                    # not like the training data
           "Write a haiku about the sea."]


def show_answers(m, label):
    m.eval()
    print(f"\n  --- {label} ---")
    out = {}
    for p in PROMPTS:
        out[p] = complete(tok, m, our_chat_prompt(tok, p), 45)
        print(f"  Q: {p}\n     {out[p]!r}")
    m.train()
    return out


loss_before = test_loss(model)
_, inst, _ = load(INSTRUCT_ID)
loss_inst = test_loss(inst)
del inst
print(f"  test loss: base {loss_before:.3f}   SmolLM2-Instruct {loss_inst:.3f}")
answers_before = show_answers(model, "base model, before SFT")

# %% [markdown]
# Before training, the base model's test loss is 2.19 and its answers are
# what lesson 01 showed: it writes the *user's* next message ("what is the
# best way to get rid of a bad smell in my house"), or a quiz. Even inside
# the template, it doesn't know which side of the conversation it's on.
#
# ## The training loop
#
# The full recipe in ~20 lines: shuffle, batch, forward, loss, backward, step.
# The learning rate warms up for a few steps, then decays (lesson 09 explains
# why). We look at the test loss every 50 steps.
#
# Three lines do all the work, and each gets its own lesson later:
# * `loss = sft_loss(...)`: how wrong are we? (lesson 06)
# * `loss.backward()`: for every weight, which direction lowers the loss? (lesson 07)
# * `opt.step()`: move each weight a little in that direction. AdamW scales
#   each step per weight (lesson 08), by a learning rate that changes over
#   time (lesson 09).
#
# `clip_grad_norm_` caps the size of one step in case a batch gives a huge
# gradient (also lesson 09).

# %%
STEPS = 2 * len(all_data) // 8 if FULL else 200      # FULL=1: two epochs (~6 min on an M4 Pro)
BATCH, LR, WARMUP = 8, 1e-4, 20

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1, (s + 1) / WARMUP) * 0.5 * (1 + math.cos(math.pi * min(s / STEPS, 1))))

banner(f"SFT: {STEPS} steps × {BATCH} conversations, lr {LR}")
order, log = [], {"step": [], "train": [], "test_step": [0], "test": [loss_before]}
rng = random.Random(0)
seed_all(0)
t0 = time.perf_counter()
for step in range(1, STEPS + 1):
    if len(order) < BATCH:                                   # new epoch: reshuffle
        order += rng.sample(range(len(all_data)), len(all_data))
    batch = [all_data[order.pop()] for _ in range(BATCH)]
    x, y, att = pad_batch(batch, device)

    loss = sft_loss(model, x, y, att)                        # 1. how wrong are we?
    loss.backward()                                          # 2. which way should each weight move?
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step(); opt.zero_grad()                # 3. move them a little

    log["step"].append(step); log["train"].append(loss.item())
    if step % 50 == 0 or step == STEPS:
        log["test_step"].append(step); log["test"].append(test_loss(model))
        print(f"  step {step:4}  train loss {sum(log['train'][-50:]) / len(log['train'][-50:]):.3f}"
              f"  test loss {log['test'][-1]:.3f}  ({time.perf_counter() - t0:.0f}s)")
train_s = time.perf_counter() - t0

# %%
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.plot(log["step"], log["train"], alpha=0.35, label="train (each batch)")
ax.plot(log["test_step"], log["test"], "o-", label="test (held-out)")
ax.axhline(loss_inst, ls="--", color="gray", label="SmolLM2-Instruct on test")
ax.set_xlabel("step"); ax.set_ylabel("loss on assistant tokens"); ax.legend()
ax.set_title("SFT on 2k everyday conversations")
show_fig(fig, "03_sft_loss")

# %%
answers_after = show_answers(model, f"after {STEPS} steps of SFT ({train_s:.0f}s)")

# %% [markdown]
# Two minutes (133 s on an M4 Pro), and the base model talks like an
# assistant: short, friendly replies that **stop by themselves**. Go through
# the four answers:
#
# * **Book**: "Hello! How can I help you today?" Polite, the right role, but
#   no book. That is word for word the assistant's first reply in **2,248 of
#   the 2,260** training conversations: they almost all start with a greeting
#   and that exact reply. SFT copies the *habits* of its data, good and bad.
# * **Dinner**: a pasta salad suggestion. On topic and in the right voice.
# * **France**: "The capital city of France, Paris", with the Louvre and Notre
#   Dame. None of the 2,260 conversations taught it that; the fact was in the
#   base model all along, and SFT gave it a way to say it.
# * **Haiku**: fails. There are no poems in 2k everyday conversations, so it
#   falls back to chit-chat and asks *you* a question about the ocean.
#
# Note that 200 steps × 8 is 1,600 conversations: the model hasn't even
# seen the whole training set once. And most of the drop happens in the first
# 50 steps (test loss 2.19 → 1.26); the next 150 only take it to 1.18. Learning
# a format is fast; it's the first thing a model picks up.
#
# Look at the loss plot, too: on *this* test set, our 2-minute model beats
# SmolLM2-Instruct (1.18 vs 1.20). Theirs was trained on far more varied
# conversations; ours specialised. A low loss on your own test set only
# tells you about data like your test set (the haiku is the proof).
#
# **Aha:** SFT doesn't add knowledge, it adds **format and role**. A couple of
# thousand examples and two minutes are enough for that, because (lesson 02)
# the knowledge was already there. What it does add, it copies from the
# data: the style, the openers, and the gaps.
#
# ## What did masking do?
#
# We never graded the user's tokens. So the model should be good at writing
# assistant turns, and no better than before at writing *user* turns.

# %%
banner("Loss on the USER's tokens (never trained)  vs  assistant tokens (trained)")
user_data = [(i, [t if l == -100 and t not in (1, 2) else -100 for t, l in zip(i, lab)])
             for i, lab in test_data]
_, fresh, _ = load(BASE_ID)
u_before, u_after = test_loss(fresh, user_data), test_loss(model, user_data)
a_after = test_loss(model)
print(f"  user tokens:      before {u_before:.3f}   after {u_after:.3f}")
print(f"  assistant tokens: before {loss_before:.3f}   after {a_after:.3f}")
del fresh

# %% [markdown]
# The assistant loss nearly halved (2.19 → 1.18). The user loss **didn't
# move** (2.357 → 2.346). Every bit of the training signal went into the
# assistant's side of the conversation: the model learned to *answer* users,
# not to *imitate* them. That's what masking buys you. Without it (try below),
# part of every step is spent learning to write questions that the model will
# never need to write.
#
# It's also a nice sanity check that masking works. If a bug let user tokens
# into the loss, the user loss would have dropped too.
#
# ## What we know so far
#
# * SFT = next-token training on conversations, grading only the assistant's
#   tokens (labels = -100 elsewhere), including the `<|im_end|>` that ends a turn.
# * 200 steps on 2k conversations turns a base model into a passable
#   assistant. The format is learned in the first few dozen steps.
# * It copies its data: it answers what the data covers and stumbles on what it
#   doesn't.
#
# SFT can only teach "say *this*". It can't say "this answer is *better* than
# that one". We'll save the model; lesson 04 starts from it and teaches
# preferences.

# %%
CKPT.mkdir(parents=True, exist_ok=True)
model.save_pretrained(CKPT / "03_sft")
tok.save_pretrained(CKPT / "03_sft")
print(f"  saved -> {(CKPT / '03_sft').relative_to(ROOT)}  (lesson 04 starts from here)")
save_result("03_sft", {"steps": STEPS, "batch": BATCH, "lr": LR, "seconds": train_s, "device": device,
                       "log": log, "loss_inst": loss_inst, "n_tokens": n_tok, "n_graded": n_learn,
                       "user_loss": [u_before, u_after], "assistant_loss": [loss_before, a_after],
                       "answers_before": answers_before, "answers_after": answers_after,
                       "mask_example": {"tokens": [tok.decode([i]) for i in chat_ids(tok, train_convs[0][:4])[0]],
                                        "graded": [l != -100 for l in chat_ids(tok, train_convs[0][:4])[1]]}})

# %% [markdown]
# ## Try:
# * Train with `train_on="all"` (no masking) in `chat_ids`. The train loss
#   now also counts user tokens. Is the *assistant* test loss better or worse?
# * Change `LR` to `1e-3` and to `1e-5`. Which is too hot, which too cold?
#   (Lesson 09 is all about this.)
# * Use only 200 conversations (`all_data = all_data[:200]`) and train for the
#   same number of steps. Watch the test loss go back up: **overfitting**.
# * Ask the model something the dataset never covers (code, maths). What
#   does a model trained on 2k chit-chat conversations do?
# * The same with a library: `trl`'s `SFTTrainer(model, train_dataset=ds["train_sft"])`
#   does this loop, plus packing and many options.
