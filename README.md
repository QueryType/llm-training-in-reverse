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
| 04 | How do preferences work? | DPO in ~30 lines, on 499 self-generated "longer is better" pairs | It learns exactly what the label measures: answers grow 33 → 59 tokens. Pulled hard, the held-out loss triples while every sample still reads fine: over-optimisation is silent |
| 05 | What did pretraining look like along the way? | Pythia-70m at 11 checkpoints, step 0 → 143k: loss, completions, facts, copying | Skills arrive in order: token frequencies ("the the the") → phrases → copying → grammar → facts. Copying switches on between steps 512 and 1,000 (6.5 → 2.0 nats). Facts flicker |
| 06 | What does the loss number mean? | A ladder in bits per character: uniform, character n-grams, token n-grams, Pythia, SmolLM2 | Counting reaches 1.33 bits/char, then memorises (n=16: 0.86 on its own text, 3.06 on new). SmolLM2: 0.66. Repeated names are nearly free |
| 07 | What is a gradient, really? | softmax − one_hot, a hand-written backward pass, finite differences on a Llama, one step at many sizes | Hand backprop = autograd to 1e-17. One step helps a *different* batch just as much. Too big a step (η = 3) doubles the loss |
| 08 | What does `optimizer.step()` do? | SGD, momentum and AdamW by hand, matched to `torch.optim`; a 2D valley race; a real race | Ours match to 1e-7. Same 600 steps: SGD 2.39, momentum 2.05, Adam 1.16. Adam's median step is 18× SGD's; rare characters' gradients are 19× smaller |
| 09 | Why does the learning rate matter so much? | LR sweep, warmup (3 seeds), four schedules, clipping | A narrow U at 1e-3. Too hot doesn't explode, it gets stuck at the letter-frequency level. Warmup wins 3/3; WSD's decay gives the best final loss; clipping rescues a 10× lr (1.80 → 1.38) |
| 10 | Where does the data come from? | BPE from scratch, then a 4,096-token BPE; dedup and decontamination; packing | 4k tokens compress TinyStories about as well as SmolLM2's 49k. 15% of training stories are duplicates, and 30% of validation stories are also in train |
| 11 | Can I pretrain my own? | A 12M-parameter Llama from random weights, 2,700 steps, with lesson 05's probes | 12 minutes on a laptop: coherent stories, 0.69 bits/char (Pythia-70m 0.97, SmolLM2 0.66). Same learning order as Pythia; copying barely appears, because the data never rewards it |
| 12 | What limits training? | Memory counter through one step; checkpointing; throughput vs batch; accumulation; bf16 | Activations are 41× the weights (98% of the peak). Checkpointing: 8× less, 1.31× slower. Throughput flat from batch 8; bf16 gives nothing on MPS |
| 13 | Bigger model or more data? | IsoFLOP curves: 8 sizes at 2 compute budgets; a mid-schedule test | Best size ≈ 16 and 20 tokens per parameter (Chinchilla: ~20). A schedule caught mid-way costs +0.07 nats, the bias that favoured big models |
| 14 | Can I make *my* model chat? | Lesson 03's SFT, unchanged, on lesson 11's model with TinyStoriesInstruct requests | Stops by itself 0% → 95%; uses the requested words 7% → 35%. Follows instructions *partly*. The dialogue metric rests on 30 samples: noise |
| + | MLX addendum | Lesson 11's model in ~50 lines of MLX; load the PyTorch weights; train again | Same function (logits within 2.5e-5). 1.23× faster than PyTorch on MPS. Lower loss at the same step, probably from a different initialisation (left as a Try) |

The numbers above come from one run on an M4 Pro; yours will differ a little (see "Hardware").
Each lesson is a plain `.py` file split into `# %%` cells. It runs as a script
*or* as an interactive notebook (VS Code / Cursor / PyCharm "Run Cell", or
`jupytext` to convert to `.ipynb`). Figures go to `outputs/`.

## The interactive explainer (`site/`)

**Read it online: https://querytype.github.io/llm-training-in-reverse/** (no setup).

`site/index.html` explains every lesson in plain language, with widgets
drawn from **your own runs**. Lessons save their numbers to `outputs/results/`
when they run (the training runs are too slow to repeat just for the page):

```bash
bash run_all.sh                  # runs every lesson, then site/build_data.py
open site/index.html             # works offline, no server needed
```

Maintainers: `bash site/publish_pages.sh` copies the built page to the
`gh-pages` branch, which GitHub Pages serves. `main` stays source-only.

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
| Apple Silicon (MPS) | a few minutes per lesson | ~12 min on an M4 Pro |
| NVIDIA (CUDA) | a few minutes per lesson | ~10 min or less on a modest card |
| CPU only | works, slower for lessons 03/04 | possible, but expect an hour+ |

The whole course (`bash run_all.sh`) takes about an hour on an M4 Pro: lesson 11's
pretraining is 12 minutes of it, lesson 13's thirteen scaling runs about 5, the MLX
addendum 2. On CUDA, lesson 12 also shows what bf16 buys (on MPS it buys nothing).

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
  index.html  style.css           the interactive explainer
  js/ui.js                        shared widgets (bars, charts, chips)
  js/part1.js part2.js part3.js   one file of widgets per part
  build_data.py                   collects outputs/results/*.json into site/data/results.js
  publish_pages.sh                publishes the built site to GitHub Pages
outputs/                          figures, results, and the checkpoints of models you train
requirements.txt
run_all.sh
```

## License

[MIT](LICENSE). The models and datasets the lessons download come with their
own licenses (SmolLM2 and Pythia: Apache-2.0; everyday-conversations: Apache-2.0;
TinyStories and TinyStoriesInstruct: CDLA-Sharing-1.0; WikiText: CC BY-SA).
