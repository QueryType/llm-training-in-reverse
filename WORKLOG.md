# Worklog: LLM Training, In Reverse

Handoff notes between working sessions. Not part of the course itself; delete before publishing.

## How we work (agreed with the user)
- **Code and results first, then prose and site.** All lesson code is done and verified; the
  remaining work is the writing pass.
- **Every claim must match a measured number.** Read numbers from `outputs/results/*.json`
  (or by re-running a lesson). Never hand-type a number into the site: widgets read
  `site/data/results.js` (built by `python site/build_data.py`).
- Chapters should be **fuller** than the first draft of 01–03, which the user found "a little short".
- Style follows `../transformers_learning` (lessons as `# %%` scripts with Question → experiment →
  **Aha** → **Try:**; site = lab-notebook page, same CSS family, indigo accent).
- Commit locally with the attribution line; **don't push / no GitHub yet** (user will do GitHub later).
- CUDA must keep working (user will test on RunPod). MLX is an addendum only.

## Environment
- Conda env `llm_training_learning` (python 3.11, torch 2.15, transformers 5.23, datasets, mlx).
  Run: `/Volumes/d/conda/conda_envs/llm_training_learning/bin/python lessons/NN_x.py`
- HF cache: `/Volumes/d/cache/hf_cache`. Pythia-70m checkpoints, TinyStories, TinyStoriesInstruct
  valid file, SmolLM2-135M (+Instruct), wikitext-2 are cached.
- Full course: `bash run_all.sh` ≈ 55 min on M4 Pro (lesson 11 ≈ 12 min). Use
  `PYTHONUNBUFFERED=1` when logging to a file, or nothing appears until exit.
- Headless screenshots for checking the site: playwright's chromium at
  `~/Library/Caches/ms-playwright/chromium_headless_shell-1255/.../chrome-headless-shell`
  (`--headless --window-size=1400,9000 --virtual-time-budget=4000 --screenshot=...`).

## Done
- [x] Plan agreed: 14 lessons + MLX addendum, reverse order, Pythia-70m, ~10 min pretraining
- [x] `lessons/common.py` (devices CUDA→MPS→CPU, chat format, char + BPE data, tiny_llama, save_result)
- [x] Lessons 01–14 + `lessons/mlx_addendum.py` written, run, numbers checked against prose comments
- [x] Full clean `run_all.sh` passed (commit 574b939); results in `outputs/results/`
- [x] Site skeleton: `site/index.html`, `style.css`, `js/ui.js` (+ `xy()` line chart), `js/part1.js`
      with working widgets for lessons 01–03 (checked at desktop and phone width)
- [x] `site/build_data.py` (keys "01".."14", "mlx")

## To do (writing pass)
- [x] **Lessons 01–03 `.py`**: expand the markdown cells (more explanation, intuition, links between lessons)
- [x] **Lessons 04–14 + addendum `.py`**: review/expand markdown; make sure every stated number
      matches the latest results JSON (see "Key results" below; re-check them)
      - [x] 04, 05 done (2026-11-18)   - [x] 06–09 done (2026-11-19)   - [x] 10–14, addendum (2026-11-19)
- [x] **Site Part 1**: expand 01–03 sections; write 04 (DPO) and 05 (Pythia) sections + widgets
      (checked desktop + 390px); user approved (2026-11-19)
- [x] **Site Part 2**: 06–09 sections + widgets in `site/js/part2.js` (checked desktop + 390px)
- [x] **Site Part 3**: 10–14 sections + MLX section in `site/js/part3.js` (checked desktop + 390px)
- [x] Update hero journey one-liners with measured facts; remove "soon" classes in the TOC (MLX added to TOC)
- [x] **README**: lesson table with measured ahas, hardware timings, layout, licenses
- [x] Final: rebuild site data, screenshot check (desktop + 390px), every data-n key bound, commit (2026-11-19)
- [ ] Before publishing: delete this WORKLOG.md; push to GitHub (user will do this)

### Widget ideas (from the original plan; data already saved)
- 04: chosen/rejected pair viewer; Δlogp chosen vs rejected curves for both runs; length/test-loss evals
- 05: checkpoint scrubber (completions, facts, copy losses, top-1 variety) over log-tokens axis
- 06: the bits/char ladder (counting vs models); n-gram train vs val U-curve; per-token surprise heat-text
- 07: softmax−one_hot bars; one-step η sweep (same vs other batch); finite-diff table
- 08: 2D optimizer race animation (paths saved); real race curves; gradient vs Adam step quantiles; memory table
- 09: lr sweep U-curve + curves; warmup seeds; schedule shapes + val; clipping loss/gnorm
- 10: BPE merges stepper (first 300 merges + compression); vocab size vs chars/token; dup/contamination stats; packed window
- 11: training "tape" scrubber (sample text + val + copy at each probe step); final stories; ladder placement
- 12: memory stages bar; activations vs batch (normal/checkpointed); throughput vs batch; memory calculator (params × bytes)
- 13: IsoFLOP curves with fitted parabolas; tokens/param at optimum; schedule test
- 14: before/after instruction following; request → story examples with word hits highlighted
- Also planned: a live in-browser tiny model (char bigram) trainer with lr/batch knobs (Part 2)

## Key results (from the verified run; re-read the JSON before quoting)
- 01: Instruct on a plain question → 72.6% `<|im_end|>`; base with chat template → nonsense
- 02: ‖Δ‖/‖W‖ = 8.6% (layer 0 v_proj 33%); random same-size Δ → ppl 5.87→10.74; α=0.2 already answers;
  52.5% of instruct tokens are base's top-1
- 03: SFT 200 steps, 133 s, test 2.19→1.18 (SmolLM2-Instruct 1.205); user-token loss 2.357→2.346 (unchanged)
- 04: 499 self-generated pairs ("longer is better"); β=0.1: length 33→59, test loss 1.18→1.29;
  "pulled hard" (β=0.01, lr 2e-5, 2 epochs): test loss →3.76, Δlogp chosen −36 / rejected −133, samples still look fine
- 05: step 128 = unigram ("the the the", 40 distinct top-1); copying switches on 512→1000 (text 6.07→1.99);
  facts by frequency and they flicker ("America" 79%→5%→83%); random-token copying degrades late (final 23.9 nats)
- 06 (bits/char on TinyStories val): uniform 6.60; char n-gram best 1.33 at n=7–8, n=16: val 3.06 / train 0.86;
  token bigram 1.39; Pythia-70m 0.97; SmolLM2 0.66; 10% hardest tokens hold 32% of a story's loss
- 07: hand backprop = autograd to 1e-17; finite diffs match to 8 digits after making RMSNorm float64
  (Llama's RMSNorm upcasts to float32!); fwd+bwd = 4.1× fwd on MPS (theory 3×)
- 08: ours = torch.optim to ~1e-7; 2D valley: Adam 0.013, momentum 0.26, SGD 0.68; real race (best lr each):
  SGD 2.38, momentum 1.97, Adam 1.16; common-char embedding grads ~20× rare ones
- 09: best constant lr 1e-3 (U-curve); warmup better on 3/3 seeds (~1.34→1.23); WSD trails at 80%, ends best (1.10);
  clipping at 10× lr: 1.80→1.38; too-hot runs get *stuck* (2.58, 2.99 ≈ unigram 3.05), not spiky
- 07 correction: the one-step gain on a *different* batch is as large as on the same batch (not smaller)
- 10: TinyStories train 15.1% exact duplicates; 6,601 of 21,990 val stories also in train (removed);
  4096-vocab BPE: 4.00 chars/token vs SmolLM2's 49k: 4.14; 229M train tokens
- 11: 12.2M params, 2,700 steps × 8,192 tokens, 12 min, val 1.922 nats = 0.693 bits/char (SmolLM2 0.657: close, not better);
  copy test barely improves (TinyStories rarely repeats passages); stories coherent
- 12: activations 1,940 MB vs weights 47 MB at batch 32; checkpointing ~8× less, 1.31× slower; throughput flat from batch 8;
  bf16/fp16 no gain on MPS; grad accumulation diff 4e-7; ~2.7 TFLOP/s
- 13: optimum ≈16 (1e13) and ≈20 (4e13) tokens/param; N_opt ∝ C^0.42; schedule-too-long penalty +0.066
- 14: words used 7%→35%, stops 0%→95%; dialogue metric on only 30 prompts is noise (moved ±15 between runs)
- MLX: same function (max logit diff 2.7e-5, 100% top-1); 1.16–1.23× faster than PyTorch MPS; lower loss at same step,
  *hypothesis* = different init (left as a Try)

## Site conventions (added in the writing pass)
- Numbers in prose: `<span data-n="05.c1k">2.0</span>`; the lesson's JS block fills `N["05.c1k"]` from results,
  the text inside is only a fallback. Add the key in the widget JS when you quote a new number.
- Half-width charts: pass `width: 460` to `xy()`/`chart()` or the text renders tiny.
- Bar labels that are prose (not tokens): use `labelNode: [document.createTextNode(...)]`, else spaces show as `·`.
- Lesson 02 now also saves share_tiny, median_w/median_d, embed/norm relative change (re-run done).
- Lesson 08 now also saves `emb_grad` (common vs rare char embedding grads, 19×); re-run 2026-11-19
  (momentum's best lr moved 0.03→0.01 on the re-run, MPS isn't bit-deterministic; prose reads the JSON).
- Each part has its own JS file with its own `N` map and data-n fill at the end.

## Gotchas found along the way
- Base tokenizer has no chat template; `chat_tokenizer()` borrows Instruct's. Our own SFT models use `our_chat_prompt` (no system prompt).
- Pythia: load with `use_safetensors=False` (fp16 .bin, half the download).
- Lesson 04's first draft predicted "rambling" for the loose run; reality was silent drift. Don't pre-write ahas.
- Lesson 05's P(Paris) is ~0 for a 70M model; use the FACTS list instead.
- MPS: bf16 autocast gives no speedup; throughput ~37k tok/s for the 12M model.
