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


# ---------------------------------------------------------------------------
# Part 2 and 3: tiny models trained from scratch on TinyStories
# ---------------------------------------------------------------------------
DATA = OUT / "data"                  # tokenized datasets we build (cached between lessons)
EOS_CHAR = "\x03"                    # end-of-story marker for the character tokenizer


def clean_story(text):
    """TinyStories is almost pure ASCII. Straighten curly quotes, drop the rare junk characters."""
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    return text.encode("ascii", "ignore").decode().strip()


def tinystories(split="train", n=None):
    """List of cleaned stories. The dataset (~1.8 GB) downloads on first use."""
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories", split=split)
    texts = ds["text"] if n is None else ds.select(range(min(n, len(ds))))["text"]
    return [t for t in (clean_story(x) for x in texts) if t]


class CharTokenizer:
    """The simplest tokenizer there is: one token per character (lessons 06-09)."""
    def __init__(self):
        self.chars = [EOS_CHAR, "\n"] + [chr(c) for c in range(32, 127)]
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.eos_id = 0

    def __len__(self):
        return len(self.chars)

    def encode(self, text):
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids).replace(EOS_CHAR, "\n<end>\n")


def char_data(n_train=60_000, n_val=2_000):
    """(tokenizer, train_ids, val_ids) as uint8 tensors: stories joined with an end-of-story char.
    Cached in outputs/data/ after the first call. 60k stories ≈ 50M characters."""
    tokz = CharTokenizer()
    path = DATA / f"chars_{n_train}_{n_val}.pt"
    if path.exists():
        d = torch.load(path)
        return tokz, d["train"], d["val"]
    DATA.mkdir(parents=True, exist_ok=True)
    def build(texts):
        return torch.tensor(tokz.encode(EOS_CHAR.join(texts) + EOS_CHAR), dtype=torch.uint8)
    train, val = build(tinystories("train", n_train)), build(tinystories("validation", n_val))
    torch.save({"train": train, "val": val}, path)
    return tokz, train, val


def get_batch(data, batch, ctx, gen=None, device="cpu"):
    """Random windows of ctx+1 tokens: x = first ctx, y = the same shifted by one.
    `data` is a 1-D tensor or numpy array (e.g. a memory-mapped token file)."""
    ix = torch.randint(len(data) - ctx - 1, (batch,), generator=gen)
    rows = [torch.as_tensor(data[i:i + ctx + 1].astype("int64")) if not torch.is_tensor(data)
            else data[i:i + ctx + 1] for i in ix.tolist()]
    w = torch.stack(rows).long()
    return w[:, :-1].to(device), w[:, 1:].to(device)


def tiny_llama(vocab, d=128, layers=4, heads=4, ctx=256, ffn=None, seed=0):
    """A Llama-style model (same design as SmolLM2), just small. Built with transformers'
    own LlamaForCausalLM, so everything from 'Transformers, Inside Out' works on it."""
    from transformers import LlamaConfig, LlamaForCausalLM
    ffn = ffn or 32 * round(8 / 3 * d / 32)          # SwiGLU's usual 8/3 ratio, rounded
    cfg = LlamaConfig(vocab_size=vocab, hidden_size=d, intermediate_size=ffn, num_hidden_layers=layers,
                      num_attention_heads=heads, num_key_value_heads=heads, max_position_embeddings=ctx,
                      tie_word_embeddings=True, rms_norm_eps=1e-5, rope_theta=10000.0,
                      bos_token_id=None, eos_token_id=None, pad_token_id=None)
    torch.manual_seed(seed)
    return LlamaForCausalLM(cfg)


def n_params(model, embeddings=True):
    n = sum(p.numel() for p in model.parameters())
    if not embeddings:
        n -= model.get_input_embeddings().weight.numel()
    return n


@torch.no_grad()
def eval_loss(model, data, ctx, batch=32, n_batches=20, device="cpu", seed=1234):
    """Average next-token loss (nats) on fixed random windows of `data`."""
    was = model.training
    model.eval()
    g = torch.Generator().manual_seed(seed)
    tot = 0.0
    for _ in range(n_batches):
        x, y = get_batch(data, batch, ctx, g, device)
        logits = model(input_ids=x).logits
        tot += torch.nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten()).item()
    model.train(was)
    return tot / n_batches


@torch.no_grad()
def sample_text(model, prompt_ids, n=200, temperature=0.8, top_k=None, device="cpu", seed=0, stop_id=None):
    """Plain sampling loop for our own models (no KV cache: they're tiny)."""
    model.eval()
    g = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.tensor([prompt_ids], device=device)
    ctx = model.config.max_position_embeddings
    for _ in range(n):
        logits = model(input_ids=ids[:, -ctx:]).logits[0, -1].float().cpu() / max(temperature, 1e-5)
        if top_k:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[-1]] = -float("inf")
        nxt = torch.multinomial(logits.softmax(-1), 1, generator=g).item()
        ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], 1)
        if stop_id is not None and nxt == stop_id:
            break
    model.train()
    return ids[0, len(prompt_ids):].tolist()


def bpe_tokenizer():
    """Lesson 10's 4,096-token BPE as a regular transformers tokenizer."""
    from transformers import PreTrainedTokenizerFast
    path = OUT / "tokenizer" / "tokenizer.json"
    if not path.exists():
        sys.exit("Run lessons/10_tokenizer_and_data.py first: it trains the tokenizer.")
    return PreTrainedTokenizerFast(tokenizer_file=str(path), eos_token="<|endoftext|>", pad_token="<|endoftext|>",
                                   additional_special_tokens=["<|im_start|>", "<|im_end|>"])


def bpe_data():
    """(train, val) token streams from lesson 10, memory-mapped (they never fully load into RAM)."""
    import numpy as np
    tr, va = DATA / "ts4096_train.bin", DATA / "ts4096_val.bin"
    if not tr.exists():
        sys.exit("Run lessons/10_tokenizer_and_data.py first: it builds the token streams.")
    return np.memmap(tr, dtype=np.uint16, mode="r"), np.memmap(va, dtype=np.uint16, mode="r")


def bits_per_char(nats_per_token, chars_per_token):
    return nats_per_token / math.log(2) / chars_per_token
