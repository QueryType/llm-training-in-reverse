"""Shared helpers for every lesson. Nothing clever here: loading, seeding, plotting, timing."""
import os
import sys
import math
import time
import random
import warnings
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()

# Part 1 rewinds a finished model: the base model and its chat ("Instruct") version.
# Swap them with e.g.  BASE_ID=Qwen/Qwen2.5-0.5B INSTRUCT_ID=Qwen/Qwen2.5-0.5B-Instruct
BASE_ID = os.environ.get("BASE_ID", "HuggingFaceTB/SmolLM2-135M")
INSTRUCT_ID = os.environ.get("INSTRUCT_ID", "HuggingFaceTB/SmolLM2-135M-Instruct")

# FULL=1 runs the longer versions of the training runs (more steps, more data).
FULL = os.environ.get("FULL", "0") == "1"

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
CKPT = OUT / "checkpoints"          # trained models we make ourselves (lessons 03, 04, 11, 14)

IN_NOTEBOOK = "ipykernel" in sys.modules
if not IN_NOTEBOOK:
    import matplotlib
    matplotlib.use("Agg")  # scripts: save figures to files, don't pop up windows


def pick_device():
    # Force a device with:  DEVICE=cpu python lessons/01_two_models.py
    if os.environ.get("DEVICE"):
        return os.environ["DEVICE"]
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"  # Apple Silicon GPU
    return "cpu"


def seed_all(seed=0):
    """Same seed, same run (on the same device; different devices round differently)."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load(model_id=BASE_ID, device=None, dtype=torch.float32, train=False):
    """Returns (tokenizer, model, device).

    float32 on purpose: the models are tiny, and full precision keeps our hand-written
    training code comparable across devices. Lesson 12 shows what bf16 buys you.
    """
    device = device or pick_device()
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)
    model.train(train)
    return tok, model, device


def chat_tokenizer():
    """The base tokenizer has no chat template, but it has the same 49,152 tokens,
    including <|im_start|> and <|im_end|>. We borrow the Instruct model's template."""
    return AutoTokenizer.from_pretrained(INSTRUCT_ID)


def encode(tok, text, device):
    return tok(text, return_tensors="pt").input_ids.to(device)


@torch.no_grad()
def complete(tok, model, text, max_new_tokens=40, greedy=True, stop_at_im_end=True, **kw):
    """Continue `text` and return only the new part (as a string)."""
    ids = encode(tok, text, model.device)
    gen = dict(max_new_tokens=max_new_tokens, do_sample=not greedy, pad_token_id=tok.pad_token_id or 0)
    if stop_at_im_end:
        gen["eos_token_id"] = [tok.convert_tokens_to_ids("<|im_end|>"), 0]
    if greedy:
        gen["repetition_penalty"] = kw.pop("repetition_penalty", 1.2)
    out = model.generate(ids, **gen, **kw)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def chat_prompt(ctok, user, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    return ctok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def chat_ids(tok, messages, train_on="assistant"):
    """Lesson 03's chat format: token ids + labels. Labels are -100 ("don't learn this")
    everywhere except the assistant's words (train_on="assistant") or everywhere (train_on="all")."""
    ids, labels = [], []
    for m in messages:
        head = tok(f"<|im_start|>{m['role']}\n").input_ids
        body = tok(f"{m['content']}<|im_end|>\n").input_ids
        learn = train_on == "all" or m["role"] == "assistant"
        ids += head + body
        labels += [-100] * len(head) + (body if learn else [-100] * len(body))
    return ids, labels


def our_chat_prompt(tok, user):
    """The prompt our own SFT models (lessons 03, 04, 14) were trained to answer. No system prompt."""
    return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"


def pad_batch(rows, device, pad_id=0):
    """rows: list of (ids, labels). Returns input_ids, labels, attention_mask, right-padded."""
    L = max(len(i) for i, _ in rows)
    x = torch.tensor([i + [pad_id] * (L - len(i)) for i, _ in rows], device=device)
    y = torch.tensor([l + [-100] * (L - len(l)) for _, l in rows], device=device)
    att = torch.tensor([[1] * len(i) + [0] * (L - len(i)) for i, _ in rows], device=device)
    return x, y, att


def show_fig(fig, name):
    """Save a figure to outputs/ (and display it when running as a notebook)."""
    import matplotlib.pyplot as plt
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"  [figure saved -> {path.relative_to(ROOT)}]")
    plt.show() if IN_NOTEBOOK else plt.close(fig)


def save_result(name, data):
    """Store a lesson's numbers as JSON in outputs/results/. site/build_data.py turns
    them into the interactive page, so the page always shows *your* runs."""
    import json
    d = OUT / "results"
    d.mkdir(exist_ok=True)
    (d / f"{name}.json").write_text(json.dumps(data, indent=1))
    print(f"  [results saved -> outputs/results/{name}.json]")


def banner(text):
    print("\n" + "=" * 70 + f"\n{text}\n" + "=" * 70)


def show_top_k(probs, tok, k=10):
    """Print the k most likely next tokens from a 1-D probability vector."""
    top = torch.topk(probs, k)
    for p, i in zip(top.values.tolist(), top.indices.tolist()):
        bar = "█" * int(p * 50)
        print(f"  {p:6.2%}  {tok.decode([i])!r:18} {bar}")


def sync(device):
    """Wait for the GPU to finish, so timings are honest."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.s = time.perf_counter() - self.t0
