# %% [markdown]
# # Addendum — The same training loop in MLX (Apple Silicon)
#
# Everything in this course runs on PyTorch, which supports CUDA, Apple
# Silicon (MPS) and CPU. On a Mac there's a second option: **MLX**, Apple's
# own array framework, designed around the unified memory of M-series chips.
#
# This addendum rebuilds lesson 11's model in MLX in ~50 lines, checks that
# it computes *exactly the same function* by loading lesson 11's trained
# weights, and then trains it from scratch to compare speed.
#
#     pip install mlx          # Apple Silicon only
#     python lessons/mlx_addendum.py
#
# Questions: **how different is the code? Do the two frameworks agree to the
# last digit? Which one trains faster on your Mac?**

# %%
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd())
from common import *
import numpy as np
try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
except ImportError:
    sys.exit("MLX isn't installed (pip install mlx). It needs an Apple Silicon Mac.")

PRE = CKPT / "11_pretrain"
if not PRE.exists():
    sys.exit("Run lessons/11_pretrain.py first.")
cfg = json.loads((PRE / "config.json").read_text())
tok = bpe_tokenizer()
train_data, val_data = bpe_data()
R = {"device": str(mx.default_device())}
print(f"MLX device: {mx.default_device()}   model config from lesson 11: d={cfg['hidden_size']}, "
      f"{cfg['num_hidden_layers']} layers, {cfg['num_attention_heads']} heads")

# %% [markdown]
# ## The model, in MLX
#
# Line for line the same Llama block as `transformers` uses: RMSNorm,
# attention with rotary position embeddings, a SwiGLU MLP, residual
# connections, tied input/output embeddings. MLX's `nn.RoPE(traditional=False)`
# uses the same "rotate half" layout as Hugging Face's Llama.

# %%
class Attention(nn.Module):
    def __init__(self, d, heads, theta):
        super().__init__()
        self.h, self.hd = heads, d // heads
        self.q_proj, self.k_proj = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        self.v_proj, self.o_proj = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        self.rope = nn.RoPE(self.hd, traditional=False, base=theta)

    def __call__(self, x):
        B, T, D = x.shape
        q, k, v = (p(x).reshape(B, T, self.h, self.hd).transpose(0, 2, 1, 3) for p in (self.q_proj, self.k_proj, self.v_proj))
        q, k = self.rope(q), self.rope(k)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.hd ** -0.5, mask="causal")
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(B, T, D))


class MLP(nn.Module):
    def __init__(self, d, ffn):
        super().__init__()
        self.gate_proj, self.up_proj = nn.Linear(d, ffn, bias=False), nn.Linear(d, ffn, bias=False)
        self.down_proj = nn.Linear(ffn, d, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        d = c["hidden_size"]
        self.input_layernorm = nn.RMSNorm(d, eps=c["rms_norm_eps"])
        self.self_attn = Attention(d, c["num_attention_heads"], c.get("rope_theta", 10000.0))
        self.post_attention_layernorm = nn.RMSNorm(d, eps=c["rms_norm_eps"])
        self.mlp = MLP(d, c["intermediate_size"])

    def __call__(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))


class Llama(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.embed_tokens = nn.Embedding(c["vocab_size"], c["hidden_size"])
        self.layers = [Block(c) for _ in range(c["num_hidden_layers"])]
        self.norm = nn.RMSNorm(c["hidden_size"], eps=c["rms_norm_eps"])

    def __call__(self, ids):
        x = self.embed_tokens(ids)
        for layer in self.layers:
            x = layer(x)
        return self.embed_tokens.as_linear(self.norm(x))          # tied output layer

# %% [markdown]
# ## Same weights, same numbers?
#
# Load lesson 11's PyTorch weights into the MLX model (the parameter names
# match after dropping the `model.` prefix) and compare the logits on the
# same input.

# %%
from safetensors.numpy import load_file
rope_cfg = cfg.get("rope_parameters") or {}
cfg["rope_theta"] = cfg.get("rope_theta") or rope_cfg.get("rope_theta", 10000.0)
mlx_model = Llama(cfg)
w = load_file(str(PRE / "model.safetensors"))
weights = [(k.removeprefix("model."), mx.array(v)) for k, v in w.items() if k != "lm_head.weight"]
mlx_model.load_weights(weights)

x = np.asarray(val_data[:128]).astype(np.int64)[None]
pt_model = AutoModelForCausalLM.from_pretrained(PRE, dtype=torch.float32).eval()
with torch.no_grad():
    pt_logits = pt_model(torch.from_numpy(x)).logits.numpy()
mx_logits = np.array(mlx_model(mx.array(x)))
diff = np.abs(pt_logits - mx_logits).max()
agree = (pt_logits.argmax(-1) == mx_logits.argmax(-1)).mean()
banner("Lesson 11's model: PyTorch (CPU) vs MLX (GPU)")
print(f"  max |logit difference| = {diff:.2e}   (logits are around ±{np.abs(pt_logits).mean():.1f})")
print(f"  same top-1 prediction at {agree:.1%} of positions")
R["port"] = {"max_diff": float(diff), "top1_agree": float(agree)}

# %% [markdown]
# Same function, to float32 rounding. The architecture is the whole model:
# the weights move between frameworks as plain arrays.
#
# ## Train from scratch in MLX
#
# Lesson 11's recipe: AdamW (β2 = 0.95, weight decay 0.1), warmup + cosine,
# clipping at 1.0, batch 32 × 256. MLX is *lazy*: operations build a graph that
# runs when you call `mx.eval`. `mx.compile` fuses the whole training step.

# %%
pt = json.loads((OUT / "results" / "11_pretrain.json").read_text())
SCHED = pt["config"]["steps"]                        # lesson 11's schedule length (2,700)
STEPS = int(os.environ.get("STEPS", SCHED if FULL else 600))
BATCH, CTX, WARM = 32, cfg["max_position_embeddings"], 100
LR = float(min(pt["lr_check"], key=pt["lr_check"].get))
# Same schedule as lesson 11 (sized for its full run), so losses at the same step are comparable
# even when we stop early. A schedule sized for 600 steps would decay sooner and look better (lesson 09).

mx.random.seed(0)
model = Llama(cfg)
schedule = optim.join_schedules([optim.linear_schedule(1e-7, LR, WARM),
                                 optim.cosine_decay(LR, SCHED - WARM, end=LR * 0.1)], [WARM])
opt = optim.AdamW(learning_rate=schedule, betas=[0.9, 0.95], weight_decay=0.1)


def loss_fn(m, x, y):
    return nn.losses.cross_entropy(m(x), y, reduction="mean")


from functools import partial
state = [model.state, opt.state]                 # what the compiled step reads and updates


@partial(mx.compile, inputs=state, outputs=state)
def step(x, y):
    loss, grads = nn.value_and_grad(model, loss_fn)(model, x, y)
    grads, _ = optim.clip_grad_norm(grads, 1.0)
    opt.update(model, grads)
    return loss


def batch(gen):
    xb, yb = get_batch(train_data, BATCH, CTX, gen)
    return mx.array(xb.numpy()), mx.array(yb.numpy())


def val_loss(m, n=20):
    g = torch.Generator().manual_seed(1234)
    tot = 0.0
    for _ in range(n):
        xb, yb = get_batch(val_data, 32, CTX, g)
        tot += loss_fn(m, mx.array(xb.numpy()), mx.array(yb.numpy())).item()
    return tot / n


banner(f"MLX training: {STEPS} steps × {BATCH * CTX:,} tokens")
gen = torch.Generator().manual_seed(0)
curve = []
t0 = time.perf_counter()
for s in range(1, STEPS + 1):
    xb, yb = batch(gen)
    loss = step(xb, yb)
    mx.eval(state, loss)
    if s % 100 == 0 or s == STEPS:
        v = val_loss(model)
        curve.append((s, v))
        print(f"  step {s:5}  train {loss.item():.3f}  val {v:.3f}   ({time.perf_counter() - t0:.0f}s)")
secs = time.perf_counter() - t0
mlx_tps = STEPS * BATCH * CTX / secs
R["mlx"] = {"steps": STEPS, "seconds": secs, "tokens_per_s": mlx_tps, "curve": curve, "lr": LR}

# %%
# Lesson 11's own timing includes its probes, so compare with lesson 12's clean measurement (batch 32)
l12 = OUT / "results" / "12_memory_and_speed.json"
pt_tps = next(r["tokens_per_s"] for r in json.loads(l12.read_text())["speed"] if r["batch"] == 32) if l12.exists() \
    else pt["config"]["steps"] * BATCH * CTX / pt["train_seconds"]
banner("Same model, same recipe, same Mac")
print(f"  PyTorch ({pt['config']['device']}, lesson 12's measurement): {pt_tps / 1e3:6.1f}k tokens/s")
print(f"  MLX     ({mx.default_device()}):                       {mlx_tps / 1e3:6.1f}k tokens/s   ({mlx_tps / pt_tps:.2f}×)")
at = [r for r in pt["tape"] if r["step"] <= STEPS]
if at:
    print(f"  val loss at step {at[-1]['step']}: PyTorch {at[-1]['val']:.3f}, MLX "
          f"{min(curve, key=lambda c: abs(c[0] - at[-1]['step']))[1]:.3f}")
R["pytorch_tokens_per_s"] = pt_tps

# %%
ids = tok("Once upon a time, there was a little").input_ids
for _ in range(80):
    logits = model(mx.array([ids[-CTX:]]))[0, -1]
    ids.append(mx.random.categorical(logits / 0.8).item())
print("\n  MLX model says:", tok.decode(ids))
save_result("mlx_addendum", R)

# %% [markdown]
# **Aha:**
# * The model is the same few equations in any framework. Porting is mostly
#   renaming: `nn.Linear` is `nn.Linear`, `F.scaled_dot_product_attention` is
#   `mx.fast.scaled_dot_product_attention`, and the weights are just arrays.
# * MLX differs in *how* it runs: lazy evaluation and `mx.compile` let it fuse
#   the whole step, which matters most for small models like ours, where fixed
#   per-operation costs dominate (lesson 12).
#
# * The MLX run reached a slightly *lower* loss at the same step with the same
#   schedule. The likely reason is initialisation: MLX's `nn.Linear` and
#   `nn.Embedding` start from their own default random values, while
#   Hugging Face's Llama draws every weight from N(0, 0.02). Same architecture,
#   same recipe, different starting point, different run. (Check it below.)
#
# ## Try:
# * Initialise the MLX model like Hugging Face (every weight from
#   `mx.random.normal(shape) * 0.02`, norms at 1) and train again. Does the gap close?
# * Train in bf16: `model.set_dtype(mx.bfloat16)`. Faster? Does the loss suffer?
# * Port the trained MLX model back: `np.array(...)` each weight and load it into
#   `LlamaForCausalLM`. Then run lesson 14 on it.
# * Use `mlx-lm` (`pip install mlx-lm`): it has LoRA fine-tuning for real models, the
#   tool you'd use next for SmolLM2 or bigger on a Mac.
