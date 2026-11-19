// Part 3 widgets (lessons 10-14 + MLX addendum). All data comes from window.RESULTS (site/build_data.py).
(function () {
  const { $, h, chip, bars, seg, xy } = UI;
  const R = window.RESULTS;
  const N = {};                                    // numbers quoted in the prose (data-n)
  const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
  const smooth = (a, k) => a.map((_, i) => mean(a.slice(Math.max(0, i - k + 1), i + 1)));
  const f1 = (v) => v.toFixed(1), f2 = (v) => v.toFixed(2), f3 = (v) => v.toFixed(3), f0 = (v) => Math.round(v).toLocaleString();
  const ex = (v, d = 1) => v.toExponential(d).replace("e-", "e−");
  const C = { a: "var(--accent)", hot: "var(--hot)", good: "var(--good)", bad: "var(--bad)", mute: "var(--muted)" };
  const txt = (t) => [document.createTextNode(t)];
  const stat = (b, t) => h("div", { class: "stat" }, h("b", { text: b }), h("span", { text: t }));
  const nearest = (arr, x) => arr.reduce((a, b) => (Math.abs(b - x) < Math.abs(a - x) ? b : a));

  function needs(key, ids, lesson) {
    if (R[key]) return true;
    for (const id of ids) {
      const el = $(id);
      if (el) el.replaceChildren(h("div", { class: "pending", text: `Run lessons/${lesson} and then python site/build_data.py to fill this in.` }));
    }
    return false;
  }

  // ---------------------------------------------------------------- 10
  if (needs("10", ["w10-tok", "w10-vocab", "w10-checks", "w10-window"], "10_tokenizer_and_data.py")) {
    const D = R["10"];
    const merges = D.scratch_merges, sizes = D.scratch_sizes;
    const text = D.example.text;
    const words = text.replace(/\n/g, " \n ").split(" ").filter(Boolean).map((w) => [..." " + w]);
    const tokenize = (k) => {
      let ws = words.map((w) => w.slice());
      for (let i = 0; i < k; i++) {
        const [a, b] = merges[i];
        ws = ws.map((w) => {
          const out = [];
          for (let j = 0; j < w.length; j++) {
            if (j < w.length - 1 && w[j] === a && w[j + 1] === b) { out.push(a + b); j++; } else out.push(w[j]);
          }
          return out;
        });
      }
      return ws.flat();
    };
    const kIn = $("w10-k");
    kIn.max = merges.length;
    let curveSvg;
    const show = (k) => {
      const toks = tokenize(k);
      const fresh = new Set(k ? [merges[k - 1][0] + merges[k - 1][1]] : []);
      $("w10-tok").replaceChildren(...toks.map((t) => chip(t, { cls: fresh.has(t) ? "hot" : "" })));
      $("w10-k-out").textContent = String(k);
      const lo = Math.max(0, k - 6);
      $("w10-merges").replaceChildren(h("p", { class: "lbl muted", style: "font:600 .75rem var(--font-mono);text-transform:uppercase;letter-spacing:.05em;margin:0", text: k ? "latest merges" : "no merges yet: characters" }),
        ...merges.slice(lo, k).map(([a, b, f], j) => h("div", { class: "tokrow" },
          h("span", { class: "muted", style: "font:500 .8rem var(--font-mono);min-width:2.6rem", text: "#" + (lo + j + 1) }),
          chip(a), h("span", { text: "+" }), chip(b), h("span", { text: "→" }), chip(a + b, { cls: lo + j === k - 1 ? "hot" : "" }),
          h("span", { class: "muted", style: "font:500 .8rem var(--font-mono)", text: `${f.toLocaleString()}×` }))).reverse());
      $("w10-cap").textContent = `After ${k} merges, the 3,000 stories are ${Math.round(sizes[k] * 100)}% of their length in characters, and this sentence is ${toks.length} tokens for ${text.length} characters.`;
      if (curveSvg) curveSvg.setCursor(k);
    };
    curveSvg = xy($("w10-curve"), [{ name: "length", x: sizes.map((_, i) => i), y: sizes.map((v) => v * 100), color: C.a, width: 2.4 }],
      { width: 460, height: 230, xmin: 0, xmax: merges.length, ymin: 30, ymax: 100, xticks: [0, 50, 100, 150, 200, 250, 300], yticks: [40, 60, 80, 100], yfmt: (v) => v + "%",
        xtitle: "merges", ytitle: "text length", onX: (x) => { const k = Math.round(x); kIn.value = k; show(k); return k; } });
    kIn.addEventListener("input", () => show(+kIn.value));
    show(+kIn.value);

    const vs = D.vocab_sizes;
    xy($("w10-vocab"), [{ name: "our BPE", x: vs.map((r) => r.vocab), y: vs.map((r) => r.chars_per_token), color: C.a, dots: true, width: 2.4 }],
      { width: 460, height: 250, xlog: true, xmin: 400, xmax: 20000, ymin: 2, ymax: 4.5, xticks: vs.map((r) => r.vocab), xfmt: (v) => (v >= 1024 ? v / 1024 + "k" : v), yticks: [2, 2.5, 3, 3.5, 4, 4.5],
        xtitle: "vocabulary size (log)", ytitle: "characters per token", hlines: [{ y: D.smol_chars_per_token, color: C.hot }],
        tip: (se, i) => `${vs[i].vocab.toLocaleString()} tokens: ${f2(vs[i].chars_per_token)} chars/token, embedding ${f2(vs[i].emb_params / 1e6)}M params` });
    $("w10-example").replaceChildren(...D.example.tokens.map((t) => chip(t.replace(/Ġ/g, " "))));
    const v4k = vs.find((r) => r.vocab === 4096);

    const ck = D.checks;
    $("w10-checks").replaceChildren(
      stat(ck.n_train.toLocaleString(), "training stories"),
      stat(ck.dups.toLocaleString(), `exact duplicates (${(ck.dups / ck.n_train * 100).toFixed(1)}%); none appears more than ${ck.most_repeated}×`),
      stat(`${ck.val_in_train.toLocaleString()} / ${ck.n_val.toLocaleString()}`, `validation stories also in train (${Math.round(ck.val_in_train / ck.n_val * 100)}%)`),
      stat(f0(ck.len_pct[2]), "median story length, characters"));

    const st = D.stream;
    $("w10-streamhint").textContent = `${(st.train_tokens / 1e6).toFixed(0)}M training tokens from ${(st.train_stories / 1e6).toFixed(0)}M stories, stored as uint16`;
    $("w10-window").replaceChildren(...D.window.map((t) => chip(t, { cls: t === "<|endoftext|>" ? "special" : "" })));
    if (v4k) N["10.cpt"] = f2(v4k.chars_per_token);
    N["10.dup"] = Math.round(ck.dups / ck.n_train * 100) + "%";
  }

  // ---------------------------------------------------------------- 11
  if (needs("11", ["w11-chart", "w11-sample", "w11-story", "w11-ladder"], "11_pretrain.py")) {
    const D = R["11"], cfg = D.config, tape = D.tape;
    const cpt = (R["10"] && (R["10"].vocab_sizes.find((r) => r.vocab === 4096) || {}).chars_per_token) || 4.0;
    const bpc = (nats) => nats / Math.LN2 / cpt;
    const iIn = $("w11-i");
    iIn.max = tape.length - 1;
    let svg;
    const showI = (i) => {
      const r = tape[i];
      $("w11-i-out").textContent = `step ${r.step.toLocaleString()}`;
      $("w11-when").textContent = `${(r.tokens / 1e6).toFixed(1)}M tokens seen`;
      $("w11-sample").textContent = r.sample.trim();
      $("w11-stats").replaceChildren(stat(f2(r.val), `val loss, nats/token (${f2(bpc(r.val))} bits/char)`),
        stat(`${f2(r.copy_first)} → ${f2(r.copy_second)}`, "copy test: 1st → 2nd time"));
      if (svg) svg.setCursor(r.step);
    };
    const steps = tape.map((r) => r.step);
    svg = xy($("w11-chart"), [
      { name: "train", x: D.log.step, y: D.log.train, color: C.a, width: 1.2, opacity: 0.4 },
      { name: "val", x: steps, y: tape.map((r) => r.val), color: C.a, dots: true, width: 2.2 },
    ], { height: 240, xmin: 0, xmax: cfg.steps, ymin: 1.5, ymax: 8.5, xticks: [0, 500, 1000, 1500, 2000, 2500, cfg.steps].filter((v, i, a) => a.indexOf(v) === i && (v === cfg.steps || cfg.steps - v > 150)),
      yticks: [2, 4, 6, 8], xtitle: "step", ytitle: "loss (nats/token)", hlines: [{ y: Math.log(4096), label: "ln 4096: uniform guess" }],
      tip: (se, i) => `step ${se.x[i]}: ${f3(se.y[i])}`,
      onX: (x) => { const s = nearest(steps, x), i = steps.indexOf(s); iIn.value = i; showI(i); return s; } });
    iIn.addEventListener("input", () => showI(+iIn.value));
    showI(0);

    const st = D.stories, keys = Object.keys(st);
    const showS = (i) => { $("w11-story").textContent = keys[i].trim() + " " + st[keys[i]]; };
    seg($("w11-storysel"), keys.map((k, i) => ({ label: k.length > 30 ? k.slice(0, 28) + "…" : k, value: i })), 0, showS);
    showS(0);

    const rows = [];
    if (R["06"]) {
      const L = R["06"].ladder, ng = R["06"].ngrams.reduce((a, b) => (b.val < a.val ? b : a));
      rows.push({ labelNode: txt(`char ${ng.n}-gram (best counting)`), value: ng.val, text: f2(ng.val) });
      for (const m of R["06"].models.filter((m) => /final|SmolLM2/.test(m.name))) rows.push({ labelNode: txt(m.name), value: m.bpc, text: f2(m.bpc) });
    }
    rows.push({ labelNode: txt(`ours, ${f1(cfg.params / 1e6)}M, ${Math.round(D.train_seconds / 60)} min`), value: D.final_bpc, text: f2(D.final_bpc), cls: "hot" });
    rows.sort((a, b) => b.value - a.value);
    bars($("w11-ladder"), rows, Math.max(...rows.map((r) => r.value)));

    const last = tape[tape.length - 1], first = tape[0];
    const t500 = tape.find((r) => r.step === 500);
    Object.assign(N, { "11.params": f1(cfg.params / 1e6) + "M", "11.c1": f2(last.copy_first), "11.c2": f2(last.copy_second),
      "11.min": String(Math.round(D.train_seconds / 60)), "11.bpc": f2(D.final_bpc) });
    if (t500) N["11.frac"] = Math.round((first.val - t500.val) / (first.val - last.val) * 100) + "%";
  }

  // ---------------------------------------------------------------- 12
  if (needs("12", ["w12-stages", "w12-act", "w12-speed", "w12-prec", "w12-sizes"], "12_memory_and_speed.py")) {
    const D = R["12"];
    if (D.stages) {
      const s = D.stages;
      bars($("w12-stages"), [
        { labelNode: txt("weights"), value: s.weights, text: f0(s.weights) },
        { labelNode: txt("after forward"), value: s.after_forward, text: f0(s.after_forward), cls: "hot" },
        { labelNode: txt("after backward"), value: s.after_backward, text: f0(s.after_backward) },
        { labelNode: txt("after optimizer step"), value: s.after_step, text: f0(s.after_step) }], s.after_forward);
      $("w12-stagecap").textContent = `Forward keeps ${f0(s.activations)} MB of activations for backward. Backward frees them and adds gradients (${f0(s.after_backward - s.weights)} MB); the optimizer step adds Adam's m and v (${f0(s.after_step - s.after_backward)} MB). Weights + grads + state = ${f0(s.after_step)} MB, 4 × the weights.`;
      const A = D.activations;
      xy($("w12-act"), [
        { name: "normal", x: A.map((r) => r.batch), y: A.map((r) => r.activations), color: C.hot, dots: true },
        { name: "activation checkpointing", x: A.map((r) => r.batch), y: A.map((r) => r.checkpointed), color: C.a, dots: true }],
        { width: 460, height: 250, xlog: true, xmin: A[0].batch, xmax: A[A.length - 1].batch, ymin: 0, ymax: Math.ceil(Math.max(...A.map((r) => r.activations)) / 1000) * 1000,
          xticks: A.map((r) => r.batch), yticks: [0, 1000, 2000, 3000, 4000], yfmt: (v) => (v ? v / 1000 + " GB" : "0"), xtitle: "batch size (× 256 tokens)", legend: true,
          tip: (se, i) => `batch ${se.x[i]} · ${se.name}: ${f0(se.y[i])} MB` });
      const a32 = A.find((r) => r.batch === 32);
      N["12.actx"] = String(Math.round(s.activations / s.weights));
      if (a32) N["12.ck"] = f1(a32.activations / a32.checkpointed);
    } else $("w12-stages").replaceChildren(h("div", { class: "pending", text: "Memory counters aren't available on CPU." }));

    const S = D.speed;
    $("w12-dev").textContent = `${D.device}, context 256`;
    xy($("w12-speed"), [{ name: "tokens/s", x: S.map((r) => r.batch), y: S.map((r) => r.tokens_per_s / 1e3), color: C.a, dots: true, width: 2.4 }],
      { width: 460, height: 250, xlog: true, xmin: 1, xmax: S[S.length - 1].batch, ymin: 0, ymax: Math.ceil(Math.max(...S.map((r) => r.tokens_per_s)) / 1e4) * 10,
        xticks: S.map((r) => r.batch), yticks: [0, 10, 20, 30, 40], xtitle: "batch size", ytitle: "k tokens/s", tip: (se, i) => `batch ${se.x[i]}: ${f1(se.y[i])}k tokens/s` });
    const P = D.precision, base = P[0].tokens_per_s;
    bars($("w12-prec"), P.map((r, i) => ({ labelNode: txt(r.name.replace("activation ", "")), value: r.tokens_per_s, text: (r.tokens_per_s / 1e3).toFixed(1) + "k", cls: i === 0 ? "hot" : "" })), Math.max(...P.map((r) => r.tokens_per_s)));
    $("w12-stats").replaceChildren(stat(f1(D.mfu.achieved_tflops), "TFLOP/s of useful training work (6 × params × tokens/s)"),
      stat(ex(D.accum.max_diff), "4×8 accumulated vs 1×32: largest gradient difference"));
    $("w12-sizes").replaceChildren(
      h("thead", {}, h("tr", {}, ["model", "run (bf16)", "train (fp32 AdamW)", "train (mixed)"].map((c) => h("th", { text: c })))),
      h("tbody", {}, D.sizes.map((r) => h("tr", {}, [r.name, r.infer_bf16_gb, r.train_fp32_gb, r.train_mixed_gb].map((v, j) =>
        h("td", { text: j === 0 ? v : v < 1 ? f0(v * 1000) + " MB" : (v < 10 ? f1(v) : f0(v)) + " GB" }))))));
    const ck = P.find((r) => /checkpoint/.test(r.name));
    Object.assign(N, { "12.acc": ex(D.accum.max_diff), "12.tflops": f1(D.mfu.achieved_tflops) });
    if (ck) N["12.ckslow"] = f2(base / ck.tokens_per_s);
  }

  // ---------------------------------------------------------------- 13
  if (needs("13", ["w13-iso", "w13-sched"], "13_scaling_laws.py")) {
    const D = R["13"];
    const budgets = [...new Set(D.runs.map((r) => r.budget))].sort((a, b) => a - b);
    const cols = [C.a, C.hot, C.good];
    const series = [];
    budgets.forEach((B, bi) => {
      const rs = D.runs.filter((r) => r.budget === B), o = D.optima[bi];
      series.push({ name: `C = ${B.toExponential(0).replace("+", "")} FLOPs`, x: rs.map((r) => r.params), y: rs.map((r) => r.loss), color: cols[bi], dots: true, width: 0 });
      const lx = rs.map((r) => Math.log10(r.params)), lo = Math.min(...lx), hi = Math.max(...lx);
      const xs = Array.from({ length: 40 }, (_, i) => lo + (hi - lo) * i / 39);
      series.push({ name: "", x: xs.map((v) => 10 ** v), y: xs.map((v) => o.fit[0] * v * v + o.fit[1] * v + o.fit[2]), color: cols[bi], dash: "5 4", width: 1.4 });
    });
    xy($("w13-iso"), series, { height: 300, xlog: true, xmin: 1.2e5, xmax: 1.5e7, ymin: 2.4, ymax: 5.8, xticks: [1e5 * 2, 5e5, 1e6, 2e6, 5e6, 1e7],
      xfmt: (v) => (v >= 1e6 ? v / 1e6 + "M" : v / 1e3 + "k"), yticks: [2.5, 3, 3.5, 4, 4.5, 5, 5.5], xtitle: "parameters N (log)", ytitle: "val loss", legend: true,
      vlines: D.optima.map((o, i) => ({ x: o.n_opt, color: cols[i] })),
      tip: (se, i) => { const r = D.runs.find((q) => q.params === se.x[i] && Math.abs(q.loss - se.y[i]) < 1e-9); return r ? `${f2(r.params / 1e6)}M params · ${r.steps} steps · ${r.tokens_per_param.toFixed(1)} tokens/param · loss ${f3(r.loss)}` : ""; } });
    $("w13-opt").replaceChildren(...D.optima.map((o) => stat(`${f2(o.n_opt / 1e6)}M`, `best size at C = ${o.budget.toExponential(0).replace("+", "")}: ${Math.round(o.ratio)} tokens/param`)),
      ...(D.slope != null ? [stat("C^" + f2(D.slope), "how the best size grows (Chinchilla: C^0.5)")] : []));
    const s = D.schedule_test;
    bars($("w13-sched"), [{ labelNode: txt("ends at budget"), value: s.matched, text: f3(s.matched), cls: "hot" },
      { labelNode: txt("sized for 4×"), value: s.too_long, text: f3(s.too_long) }], s.too_long);
    Object.assign(N, { "13.r1": String(Math.round(D.optima[0].ratio)), "13.r2": String(Math.round(D.optima[D.optima.length - 1].ratio)) });
    if (R["11"]) { const c = R["11"].config; N["13.l11"] = f1(c.steps * c.batch * c.ctx / c.params); }
  }

  // ---------------------------------------------------------------- 14
  if (needs("14", ["w14-scores", "w14-loss", "w14-before", "w14-custom"], "14_make_it_chat.py")) {
    const D = R["14"], B = D.before, A = D.after;
    const items = [["words used", "words", B.n], ["stops", "stops", B.n], ["dialogue", "dialogue", B.n_dialogue]];
    bars($("w14-scores"), items.flatMap(([l, k]) => [
      { labelNode: txt(`${l} · before`), value: B[k], text: Math.round(B[k] * 100) + "%" },
      { labelNode: txt(`${l} · after`), value: A[k], text: Math.round(A[k] * 100) + "%", cls: "hot" }]), 1);
    $("w14-scorecap").textContent = `Words and stops: ${A.n} held-out requests. Dialogue: only the ${A.n_dialogue} that asked for it. A random story from the data uses ${Math.round(D.base_rate * 100)}% of the requested words.`;
    const L = D.log;
    xy($("w14-loss"), [{ name: "train (smoothed)", x: L.step, y: smooth(L.train, 25), color: C.a, width: 1.4, opacity: 0.6 },
      { name: "held-out test", x: L.test_step, y: L.test, color: C.hot, dots: true, width: 2.4 }],
      { width: 460, height: 250, xmin: 0, xmax: L.step[L.step.length - 1], ymin: 1.4, ymax: 2.2, xticks: [0, 100, 200, 300, 400, 500, 600], yticks: [1.4, 1.6, 1.8, 2, 2.2], yfmt: f1,
        xtitle: "SFT step", legend: true, tip: (se, i) => `step ${se.x[i]}: ${f3(se.y[i])}` });

    const marked = (story, words) => {
      const res = words.map((w) => new RegExp("\\b" + w.toLowerCase().slice(0, Math.max(4, w.length - 2)).replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + "\\w*", "g"));
      const spans = [];
      res.forEach((re) => { for (const m of story.matchAll(re)) spans.push([m.index, m.index + m[0].length]); });
      spans.sort((a, b) => a[0] - b[0]);
      const out = []; let at = 0;
      for (const [a, b] of spans) { if (a < at) continue; out.push(story.slice(at, a), h("mark", { class: "hitword", text: story.slice(a, b) })); at = b; }
      out.push(story.slice(at));
      return out;
    };
    const showE = (i) => {
      const eb = B.examples[i], ea = A.examples[i];
      $("w14-ask").replaceChildren(document.createTextNode("“" + ea.ask + "”  "), ...ea.words.map((w, j) =>
        h("span", { class: "tok" + (ea.hits[j] ? " hot" : ""), style: "margin-left:.3rem", title: ea.hits[j] ? "used after SFT" : "missing after SFT", text: w })));
      $("w14-before").replaceChildren(...marked(eb.story, eb.words));
      $("w14-after").replaceChildren(...marked(ea.story, ea.words));
    };
    seg($("w14-sel"), A.examples.map((e, i) => ({ label: e.words.join(", "), value: i })), 0, showE);
    showE(0);

    const cu = D.custom, ck = Object.keys(cu);
    const wordsOf = (a) => (/uses the words (.*?)\./.exec(a) || [, ""])[1].replace(" and ", ", ").split(",").map((w) => w.trim()).filter(Boolean);
    const showC = (i) => $("w14-custom").replaceChildren(document.createTextNode("> " + ck[i] + "\n\n"), ...marked(cu[ck[i]], wordsOf(ck[i])));
    seg($("w14-csel"), ck.map((a, i) => ({ label: wordsOf(a).join(", "), value: i })), 0, showC);
    showC(0);
    N["14.ntrain"] = D.n_train.toLocaleString();
  }

  // ---------------------------------------------------------------- MLX
  if (needs("mlx", ["wmlx-curve"], "mlx_addendum.py")) {
    const D = R["mlx"], M = D.mlx;
    const series = [{ name: "MLX", x: M.curve.map((c) => c[0]), y: M.curve.map((c) => c[1]), color: C.hot, dots: true, width: 2.4 }];
    if (R["11"]) {
      const t = R["11"].tape.filter((r) => r.step > 0 && r.step <= M.steps);
      series.unshift({ name: "PyTorch (lesson 11)", x: t.map((r) => r.step), y: t.map((r) => r.val), color: C.a, dots: true, width: 2.4 });
    }
    xy($("wmlx-curve"), series, { height: 250, xmin: 0, xmax: M.steps, ymin: 2, ymax: 6, xticks: [0, 100, 200, 300, 400, 500, 600].filter((v) => v <= M.steps), yticks: [2, 3, 4, 5, 6],
      xtitle: "step", ytitle: "val loss", legend: true, tip: (se, i) => `${se.name} · step ${se.x[i]}: ${f3(se.y[i])}` });
    const sp = M.tokens_per_s / D.pytorch_tokens_per_s;
    $("wmlx-stats").replaceChildren(stat(ex(D.port.max_diff), "largest logit difference, same weights"),
      stat(Math.round(D.port.top1_agree * 100) + "%", "same top-1 prediction"),
      stat(`${(M.tokens_per_s / 1e3).toFixed(1)}k vs ${(D.pytorch_tokens_per_s / 1e3).toFixed(1)}k`, "tokens/s: MLX vs PyTorch (MPS)"),
      stat(f2(sp) + "×", "MLX speed-up"));
    Object.assign(N, { "mlx.diff": ex(D.port.max_diff), "mlx.speed": f2(sp) });
  }

  document.querySelectorAll("[data-n]").forEach((el) => { if (N[el.dataset.n] != null) el.textContent = N[el.dataset.n]; });
})();
