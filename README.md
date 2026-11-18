# LLM Training, In Reverse

**Learn how language models are trained by rewinding a real one.**

Training tutorials usually start at the bottom: a gradient, then an optimizer,
then a loop, then a tiny model. Hours later, maybe, something talks back.

This course runs the tape backwards. **Start from a finished chat model and
undo its training one stage at a time**, back to a single weight update. Then
build the whole pipeline again yourself: a tokenizer, a model pretrained from
scratch, and a chat model of your own. Every lesson answers a question the
previous one raised, and everything runs on a laptop.

```
 PART 1  rewind a finished model
   chat model ──► what changed? ──► do SFT ──► do DPO ──► rewind pretraining
       01              02             03          04             05
                                                                  │
 PART 2  one training step, opened up                             ▼
           loss ──► gradients ──► optimizer ──► learning rate
            06         07            08             09
                                                     │
 PART 3  rebuild it from scratch                     ▼
   tokenizer+data ──► pretrain your own ──► memory/speed ──► scaling laws ──► make it chat
         10                 11                  12                13              14
                                                                                   │
 ADDENDUM  the same training loop in MLX, for Apple Silicon  ◄─────────────────────┘
```

Companion course: [Transformers, Inside Out](https://querytype.github.io/transformers-inside-out/)
opens up the same model family from the *inference* side. You don't need it first.

## The lessons

| # | Question | What you do | Aha moment |
|---|----------|-------------|-----------|
| 01 | What makes a chat model different? | Ask SmolLM2-135M base and Instruct the same questions, plain and in the chat template | Same architecture and size. The instruct model returns *nothing* for a plain question (73% on `<|im_end|>`); the base model falls apart inside the template |
| 02 | What did post-training change? | Subtract the weights; add a random edit of the same size; blend base → instruct; rank instruct's tokens under the base model | The edit is 8.6% of the weights' size. A random 8.6% nearly doubles perplexity; the trained one keeps the knowledge. 20% of the edit already answers. 52% of instruct's tokens are the base model's top pick anyway |
| 03 | Can I do SFT myself? | Hand-written SFT loop with loss masking on 2,260 everyday conversations | 2 minutes: test loss 2.19 → 1.18 (SmolLM2-Instruct: 1.21 on the same test). Loss on the user's tokens doesn't move at all |
| 04 | How do preferences work? | DPO in ~30 lines | *coming next* |
| 05 | What did pretraining look like along the way? | Pythia-70m checkpoints, step 0 → 143k | *coming next* |
| 06 | What does the loss number mean? | Uniform, unigram, bigram baselines; bits per token | *coming next* |
| 07 | What is a gradient, really? | Autograd and finite-difference checks | *coming next* |
| 08 | What does `optimizer.step()` do? | SGD and AdamW by hand, matched to `torch.optim` | *coming next* |
| 09 | Why does the learning rate matter so much? | LR sweep, warmup, cosine, divergence, clipping | *coming next* |
| 10 | Where does the data come from? | Train a BPE tokenizer on TinyStories, pack sequences | *coming next* |
| 11 | Can I pretrain my own? | A small Llama-style model from scratch, ~10 minutes | *coming next* |
| 12 | What limits training? | Memory budget, batch vs. accumulation, bf16, throughput | *coming next* |
| 13 | Bigger model or more data? | Mini scaling laws at fixed compute | *coming next* |
| 14 | Can I make *my* model chat? | SFT on your own pretrained model | *coming next* |
| + | MLX addendum | The pretraining loop again, in MLX, for Apple Silicon | *coming next* |

Each lesson is a plain `.py` file split into `# %%` cells. It runs as a script
*or* as an interactive notebook (VS Code / Cursor / PyCharm "Run Cell", or
`jupytext` to convert to `.ipynb`). Figures go to `outputs/`.

## The interactive explainer (`site/`)

`site/index.html` explains every lesson in plain language, with widgets
drawn from **your own runs**. Lessons save their numbers to `outputs/results/`
when they run (the training runs are too slow to repeat just for the page):

```bash
bash run_all.sh                  # runs every lesson, then site/build_data.py
open site/index.html             # works offline, no server needed
```

## Setup

```bash
conda create -n llm_training_learning python=3.11 -y
conda activate llm_training_learning
pip install -r requirements.txt
```

The two SmolLM2 models (~270 MB each) and the datasets download automatically
on first use into the Hugging Face cache.

```bash
python lessons/01_two_models.py       # one lesson
bash run_all.sh                       # all of them
FULL=1 bash run_all.sh                # longer training runs (more steps, more data)
```

### Hardware

| | Part 1 & 2 | Part 3 pretraining (lesson 11) |
|---|---|---|
| Apple Silicon (MPS) | a few minutes per lesson | ~10 min on an M4 Pro |
| NVIDIA (CUDA) | a few minutes per lesson | ~10 min or less on a modest card |
| CPU only | works, slower for lessons 03/04 | possible, but expect an hour+ |

CUDA, MPS and CPU are detected automatically (CUDA first). To force one:
`DEVICE=cpu python lessons/03_sft.py`. On a CPU-only Linux machine, install the
CPU build of PyTorch first: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.

Training runs are seeded, so the same device gives the same result. Different
devices round differently, so your exact losses may differ in the second or
third decimal. The lessons talk about this where it matters.

### Use a different model pair (Part 1)

```bash
BASE_ID=Qwen/Qwen2.5-0.5B INSTRUCT_ID=Qwen/Qwen2.5-0.5B-Instruct python lessons/02_what_changed.py
```

## How to get the most out of it

1. **Predict before you run.** Each lesson asks a question. Guess the answer first.
2. **Do the "Try:" at the bottom of every lesson.** That's where it sticks.
3. **Break the training on purpose.** Wrong learning rates, no masking, tiny
   datasets. Watching it fail teaches more than watching it work.

## Project layout

```
lessons/
  common.py              shared helpers: loading, devices, chat format, plotting, saving results
  01_two_models.py  02_what_changed.py  03_sft.py  ...
site/
  index.html  style.css  js/      the interactive explainer
  build_data.py                   collects outputs/results/*.json into site/data/results.js
outputs/                          figures, results, and the checkpoints of models you train
requirements.txt
run_all.sh
```

## License

[MIT](LICENSE). The models and datasets the lessons download come with their
own licenses (SmolLM2 and Pythia: Apache-2.0; everyday-conversations: Apache-2.0;
TinyStories: CDLA-Sharing-1.0).
