# %% [markdown]
# # Lesson 01 — Two models, one body
#
# We'll learn LLM training backwards. Start with the finished product, a chat
# model, and peel off one training stage at a time until we reach a single
# gradient step. Then we build the whole pipeline again from nothing.
#
# The finished product: **SmolLM2-135M-Instruct**, a small chat model.
# One step back: **SmolLM2-135M**, the *base* model it was made from. Same
# architecture, same size, same tokenizer. The only difference is a few more
# rounds of training (lessons 02–04).
#
# ### The map, read backwards
#
# A modern chat model is made in roughly this order:
#
#     random weights ──pretraining──▶ base model ──SFT──▶ ──preference tuning──▶ chat model
#                     lessons 05–13               lesson 03     lesson 04 (DPO)      lessons 01–02
#
# **Pretraining** is by far the biggest stage: the model reads a huge pile of
# text (SmolLM2-135M saw about 2 trillion tokens) and learns one thing,
# *predict the next token*. Everything it knows comes from here.
# **Post-training** (SFT, then preference tuning) is tiny by comparison and
# turns that text-predictor into something that answers you.
#
# Courses usually go left to right, so you spend ten lessons on gradients
# before you see anything that talks. We go right to left. Each lesson asks
# "where did *this* come from?" and the answer is the next lesson. By lesson
# 07 we're inside a single gradient step; from lesson 10 on we build the whole
# pipeline ourselves, ending with a chat model we trained from random numbers
# (lesson 14).
#
# ### Today
#
# Question: **what does that extra training actually change?** We won't look
# inside yet. We treat both models as black boxes and only compare what goes
# in and what comes out. Three things to watch for:
# 1. Are the two models built the same way?
# 2. Do they react differently to the same text?
# 3. What exactly is the "chat format" each of them expects?

# %%
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd())
from common import *

seed_all()
torch.set_grad_enabled(False)          # no training in this lesson
tok, base, device = load(BASE_ID)
_, inst, _ = load(INSTRUCT_ID)
ctok = chat_tokenizer()
print(f"device: {device}")
R = {"answers": {}}          # what we save for the interactive page

# %% [markdown]
# ## Are they really the same shape?
#
# If post-training added layers or changed the architecture, the difference
# would be easy to explain. Let's check: count parameters, compare every
# tensor's shape and diff the two config files.

# %%
banner("Same body?")
n_base = sum(p.numel() for p in base.parameters())
n_inst = sum(p.numel() for p in inst.parameters())
print(f"  parameters:  base {n_base:,}   instruct {n_inst:,}")
same_shapes = all(a.shape == b.shape for a, b in zip(base.parameters(), inst.parameters()))
print(f"  every tensor has the same shape: {same_shapes}")
diff = {k: (v, inst.config.to_dict().get(k)) for k, v in base.config.to_dict().items()
        if inst.config.to_dict().get(k) != v and not k.startswith(("_", "transformers"))}
print(f"  config differences: {diff}")
print("  -> only the special-token ids differ. Everything else is the numbers inside the weights.")

# %% [markdown]
# Same 134,515,008 numbers arranged the same way; the configs differ only in
# which token ids count as "beginning" and "end". So whatever post-training
# did, it did by **changing the values** of existing weights. Keep that in
# mind: it means we can compare the two models number by number (lesson 02).
#
# ## Same question, plain text
#
# Ask both models a question as plain text, the way you'd type it into a
# search box. We use **greedy decoding** (always take the most likely next
# token) so that every run gives the same output and differences come from
# the models, not from the dice.

# %%
QUESTIONS = ["What is the capital of France?",
             "Give me three tips for sleeping better.",
             "Write a haiku about the sea."]

banner("Plain text in, greedy decoding")
for q in QUESTIONS:
    a = R["answers"].setdefault(q, {})
    a["base_plain"], a["inst_plain"] = complete(tok, base, q, 40), complete(tok, inst, q, 40)
    print(f"\n  Q: {q}")
    print(f"  base     -> {a['base_plain']!r}")
    print(f"  instruct -> {a['inst_plain']!r}")

# %% [markdown]
# The base model treats the question as the start of a document and keeps
# writing that document. For France it happens to continue with an answer
# (plus a made-up population and a Seine "near the border with Belgium").
# For sleep tips it writes a numbered list, because lists often follow such a
# line on the web. For the haiku it writes `10) Write an essay on...`: to a
# base model, "Write a haiku about the sea." looks like item 9 on a homework
# sheet, so the most likely next text is item 10.
#
# None of this is a bug. A base model was trained to answer one question,
# *"what text comes next on a typical web page?"*, and it answers it well.
# Being helpful was never part of the job.
#
# The instruct model returns **nothing** at all: an empty string. Why? Look
# at what it predicts as the very first token:

# %%
banner("Instruct model: next token after a plain question")
probs = inst(encode(tok, QUESTIONS[0], device)).logits[0, -1].softmax(-1)
show_top_k(probs, tok, 5)
top = torch.topk(probs, 5)
R["inst_plain_next"] = [[tok.decode([i]), p] for p, i in zip(top.values.tolist(), top.indices.tolist())]
print("  -> it predicts '<|im_end|>' (end of turn). It has only ever seen questions")
print("     inside a chat template, so to it, a bare question is a finished message.")

# %% [markdown]
# 73% of its probability goes to `<|im_end|>`, the token that closes a turn.
# Post-training showed it thousands of conversations, and in every one a
# user's message was followed by `<|im_end|>`. A question with nothing
# around it looks like a user message that just ended, so it closes it.
# The model isn't refusing; it's waiting for the template.
#
# ## The chat template is just text
#
# A "chat" is a single string with special tokens marking who is speaking.
# The API's list of `{"role": ..., "content": ...}` messages is turned into
# this string before the model sees anything. The model never sees JSON or
# roles, only this:

# %%
banner("What the model actually reads")
prompt = chat_prompt(ctok, QUESTIONS[0])
print(prompt)
ids = encode(tok, prompt, device)[0]
print(f"  {len(ids)} tokens: {[tok.decode([i]) for i in ids.tolist()]}")
R["template_tokens"] = [tok.decode([i]) for i in ids.tolist()]
print(f"\n  <|im_start|> = token {tok.convert_tokens_to_ids('<|im_start|>')},"
      f"  <|im_end|> = token {tok.convert_tokens_to_ids('<|im_end|>')}"
      "  (both already exist in the BASE vocabulary)")

# %% [markdown]
# Notice the system prompt: "You are a helpful AI assistant named SmolLM,
# trained by Hugging Face" is 16 tokens of ordinary text that the template
# adds unless you supply your own. "Who am I" is part of the prompt, not
# something stored in a special place.
#
# The two special tokens already exist in the base model's vocabulary; the
# base model just never learned what they mean. Changing what a model does
# doesn't require new tokens, only new *habits* around existing ones.

# %%
banner("Same question, inside the chat template")
for q in QUESTIONS:
    p = chat_prompt(ctok, q)
    a = R["answers"][q]
    a["base_chat"], a["inst_chat"] = complete(tok, base, p, 40), complete(tok, inst, p, 40)
    print(f"\n  Q: {q}")
    print(f"  base     -> {a['base_chat']!r}")
    print(f"  instruct -> {a['inst_chat']!r}")

# %% [markdown]
# Now the instruct model answers, and the base model falls apart:
# a cave animal with no legs for "capital of France", run-together words for
# the sleep question. It has rarely seen `<|im_start|>assistant` in its
# training data, so after it the base model has no idea what text is likely.
# Unusual input leads to unusual output.
#
# "Answers" is generous for the haiku, though: the instruct model writes a
# run-on sentence about waves, not three lines of 5-7-5. Post-training gave it
# the role, not every skill. At 135M parameters poetry is beyond it, and
# lesson 03's own SFT runs into the same wall.
#
# Put the two experiments side by side:
#
# | | plain question | chat template |
# |---|---|---|
# | **base** | continues the document (sometimes answers) | lost |
# | **instruct** | ends the turn, says nothing | answers |
#
# Each model works only in the format it was trained on.
#
# **Aha:** the two models have the same architecture and the same size. The
# instruct model learned a *format* (the template) and a *role* (answer the
# question, then stop). It didn't need a bigger brain to do that. Did it also
# learn new facts? Did it lose any? Lesson 02 opens up the weights to find out.

# %% [markdown]
# ## Where does it stop?
#
# A chat model has to know when its turn is over. Without that, every answer
# would run on until the length limit, and the app would have to guess where
# to cut. Let's count how many tokens each model writes before it emits
# `<|im_end|>` by itself. (We add a mild `repetition_penalty` so that greedy
# decoding doesn't get stuck in a loop and hide the result.)

# %%
banner("Who knows when to stop? (up to 120 new tokens)")
im_end = tok.convert_tokens_to_ids("<|im_end|>")
for name, m in (("base", base), ("instruct", inst)):
    lens = []
    for q in QUESTIONS:
        x = encode(tok, chat_prompt(ctok, q), device)
        out = m.generate(x, max_new_tokens=120, do_sample=False, eos_token_id=im_end,
                         pad_token_id=0, repetition_penalty=1.2)[0, x.shape[1]:]
        stopped = out[-1].item() == im_end
        lens.append(f"{len(out)}{'' if stopped else '+ (never stopped)'}")
    print(f"  {name:8}: {', '.join(lens)}")
    R.setdefault("stop_lengths", {})[name] = lens
save_result("01_two_models", R)

# %% [markdown]
# The base model never stops: it fills all 120 tokens every time, because in
# its training data `<|im_end|>` almost never appears. The instruct model
# stops after 94 and 78 tokens on two of the three questions and runs out of
# room on the sleep tips (a long list). Stopping is a *learned* behaviour, and
# a fragile one in a model this small. Lesson 03 teaches it in 200 steps (the
# result stops by itself on 100% of held-out prompts, counted in lesson 04),
# and lesson 14 measures it on our own model.
#
# ## What we know so far
#
# * Same architecture, same 134.5M weights, same tokenizer. Only the values differ.
# * The base model continues documents. The instruct model answers turns.
# * A "chat" is a plain string with `<|im_start|>` / `<|im_end|>` markers,
#   and each model only behaves well in its own format.
# * Knowing when to stop is part of what post-training teaches.
#
# Next question: since only the values of the weights differ, **how much did
# they change, and where?**
#
# ## Try:
# * Put the plain question in a *document-like* frame for the base model:
#   `"Q: What is the capital of France?\nA:"`. A base model can be prompted into
#   answering. This is called **few-shot / format prompting**, and it's how
#   people used GPT-3 before chat models existed.
# * Change the system prompt in `chat_prompt(ctok, q, system="You are a pirate.")`.
#   Does a 135M model listen?
# * Try with sampling: `complete(tok, inst, p, 60, greedy=False, temperature=0.7)`.
# * Try a bigger pair: `BASE_ID=Qwen/Qwen2.5-0.5B INSTRUCT_ID=Qwen/Qwen2.5-0.5B-Instruct`.
#   Its template uses the same `<|im_start|>` tokens.
