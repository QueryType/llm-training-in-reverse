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
# Question: **what does that extra training actually change?** Let's look from
# the outside first.

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
# ## Same question, plain text
#
# Ask both models a question as plain text, the way you'd type it into a search box.

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
# The base model treats the question as the start of a document. Sometimes it
# answers, sometimes it writes the *next question* on the worksheet (look at
# the haiku one: it starts listing more homework).
#
# The instruct model returns **nothing**. Why? Look at what it predicts first:

# %%
banner("Instruct model: next token after a plain question")
probs = inst(encode(tok, QUESTIONS[0], device)).logits[0, -1].softmax(-1)
show_top_k(probs, tok, 5)
top = torch.topk(probs, 5)
R["inst_plain_next"] = [[tok.decode([i]), p] for p, i in zip(top.values.tolist(), top.indices.tolist())]
print("  -> it predicts '<|im_end|>' (end of turn). It has only ever seen questions")
print("     inside a chat template, so to it, a bare question is a finished message.")

# %% [markdown]
# ## The chat template is just text
#
# A "chat" is a single string with special tokens marking who is speaking.
# The model never sees JSON or roles, only this:

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
# Now the instruct model answers properly, and the base model falls apart.
# It has rarely seen `<|im_start|>assistant` in its training data, so it has
# no idea what comes next.
#
# **Aha:** the two models have the same architecture and the same size. The
# instruct model learned a *format* (the template) and a *role* (answer the
# question, then stop). Did it also learn new facts? Did it lose any? Lesson 02
# opens up the weights to find out.

# %% [markdown]
# ## Where does it stop?
#
# A chat model has to know when its turn is over. Let's count how many
# tokens each model writes before it emits `<|im_end|>` by itself.

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
