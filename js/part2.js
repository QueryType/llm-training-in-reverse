// Part 2 widgets (lessons 06-09). All data comes from window.RESULTS (site/build_data.py).
(function () {
  const { $, h, s, chip, bars, seg, bindTip, xy, chart, svgRoot } = UI;
  const R = window.RESULTS;
  const N = {};                                    // numbers quoted in the prose (data-n)
  const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
  const smooth = (a, k) => a.map((_, i) => { const w = a.slice(Math.max(0, i - k + 1), i + 1).filter((v) => v != null); return w.length ? mean(w) : null; });
  const f1 = (v) => v.toFixed(1), f2 = (v) => v.toFixed(2), f3 = (v) => v.toFixed(3);
  const ex = (v, d = 1) => v.toExponential(d).replace("e-", "e−");
  const lrf = (v) => (v >= 0.01 ? String(v) : v.toExponential(0).replace("e-", "e−"));
  const C = { a: "var(--accent)", hot: "var(--hot)", good: "var(--good)", bad: "var(--bad)", mute: "var(--muted)" };
  const txt = (t) => [document.createTextNode(t)];
  const stat = (b, t) => h("div", { class: "stat" }, h("b", { text: b }), h("span", { text: t }));
  const table = (el, head, rows) => el.replaceChildren(
    h("thead", {}, h("tr", {}, head.map((c) => h("th", { text: c })))),
    h("tbody", {}, rows.map((r) => h("tr", {}, r.map((c) => h("td", { text: c }))))));

  function needs(key, ids, lesson) {
    if (R[key]) return true;
    for (const id of ids) {
      const el = $(id);
      if (el) el.replaceChildren(h("div", { class: "pending", text: `Run lessons/${lesson} and then python site/build_data.py to fill this in.` }));
    }
    return false;
  }

  // ---------------------------------------------------------------- 06
  if (needs("06", ["w06-units", "w06-ladder", "w06-ngram", "w06-story"], "06_what_loss_means.py")) {
    const D = R["06"];
    const cpt = D.chars_per_token;

    // unit converter
    const slider = $("w06-nats");
    const showUnits = (L) => {
      $("w06-units").replaceChildren(
        stat(f2(L), "nats per token"),
        stat(f2(L / Math.LN2), "bits per token"),
        stat(Math.exp(L) >= 100 ? Math.round(Math.exp(L)).toLocaleString() : f2(Math.exp(L)), "perplexity (e^loss)"),
        stat(f2(L / Math.LN2 / cpt), `bits per character, at SmolLM2's ${f2(cpt)} chars/token`));
      $("w06-nats-out").textContent = f2(L) + " nats";
    };
    const presets = [
      ["uniform, 49k tokens", D.token_ngrams[0].nats_per_token], ["token unigram", D.token_ngrams[1].nats_per_token],
      ["token bigram", D.token_ngrams[2].nats_per_token], ["Pythia-70m final", D.models[2].nats_per_token], ["SmolLM2-135M", D.models[3].nats_per_token]];
    const setP = seg($("w06-presets"), presets.map(([n], i) => ({ label: n, value: i })), 4, (i) => { slider.value = Math.round(presets[i][1] * 100); showUnits(presets[i][1]); });
    slider.addEventListener("input", () => { setP(-1); showUnits(+slider.value / 100); });
    slider.value = Math.round(presets[4][1] * 100);
    showUnits(presets[4][1]);

    // ladder
    const ng = D.ngrams, bestNg = ng.reduce((a, b) => (b.val < a.val ? b : a));
    const short = (r) => r.name.replace(/ \(counts of .*\)/, "").replace(/ \([\d.]+ nats\/token\)/, "").replace(/^uniform over ([\d,]+) (characters|tokens)$/, (m, n, u) => `uniform · ${n.length > 3 ? "49k" : n} ${u === "tokens" ? "tokens" : "chars"}`);
    const pick = D.ladder.filter((r) => r.kind === "model" || /uniform|token/.test(r.name) || /^(1|3|5|16)-gram/.test(r.name) || r.name.startsWith(bestNg.n + "-gram"));
    bars($("w06-ladder"), pick.map((r) => ({
      labelNode: txt(/gram/.test(r.name) && !/token/.test(r.name) ? "char " + short(r) + (r.name.startsWith(bestNg.n + "-gram") ? " (best)" : "") : short(r)),
      value: r.bpc, text: f2(r.bpc), cls: r.kind === "model" ? "hot" : "" })), D.ladder[0].bpc);

    const pyF = D.models[2], smol = D.models[3];
    xy($("w06-ngram"), [
      { name: "new text (validation)", x: ng.map((r) => r.n), y: ng.map((r) => r.val), color: C.a, dots: true, width: 2.4 },
      { name: "the text it counted (train)", x: ng.map((r) => r.n), y: ng.map((r) => r.train), color: C.mute, dots: true, dash: "5 4" },
    ], { width: 460, height: 260, xmin: 1, xmax: 16, ymin: 0, ymax: 5, xticks: [1, 2, 4, 6, 8, 10, 12, 16], yticks: [0, 1, 2, 3, 4, 5], xtitle: "n (characters of context + 1)", ytitle: "bits/char",
      hlines: [{ y: pyF.bpc, color: C.hot }, { y: smol.bpc, color: C.hot }], legend: true,
      tip: (se, i) => `${ng[i].n}-gram · ${se.name}: ${f3(se.y[i])} bits/char` });
    const n16 = ng[ng.length - 1];
    $("w06-ngramcap").textContent = `Best on new text: ${bestNg.n}-gram, ${f2(bestNg.val)}. At n = ${n16.n}: ${f2(n16.train)} on its own text, ${f2(n16.val)} on new text. The gap is memorisation. Amber dashed lines: Pythia-70m (${f2(pyF.bpc)}) and SmolLM2 (${f2(smol.bpc)}) on new text.`;

    // per-token surprise
    const st = D.story, L = st.loss;
    const nodes = st.tokens.map((t, i) => {
      if (i === 0) return chip(t, { title: "first token: not scored" });
      const l = L[i - 1], f = Math.min(1, l / 8);
      const c = chip(t);
      c.style.background = `color-mix(in srgb, var(--hot) ${Math.round(f * 88)}%, var(--surface))`;
      c.style.borderColor = "transparent";
      if (f > 0.55) c.style.color = "var(--surface)";
      bindTip(c, () => `${JSON.stringify(t)}: ${f2(l)} nats · P = ${Math.exp(-l) < 0.001 ? Math.exp(-l).toExponential(1) : (Math.exp(-l) * 100).toFixed(1) + "%"}`);
      return c;
    });
    $("w06-story").replaceChildren(...nodes);
    const sorted = [...L].sort((a, b) => b - a), tot = L.reduce((a, b) => a + b, 0);
    const top10 = sorted.slice(0, Math.floor(L.length / 10)).reduce((a, b) => a + b, 0) / tot;
    const med = (sorted[Math.floor((L.length - 1) / 2)] + sorted[Math.ceil((L.length - 1) / 2)]) / 2;
    $("w06-storystats").replaceChildren(
      stat(f2(tot / L.length), "nats per token on this story (scored from its first word)"),
      stat(f2(med), "median token"),
      stat(Math.round(top10 * 100) + "%", "of the loss in the 10% hardest tokens"),
      stat(String(L.filter((v) => v < 0.1).length), `of ${L.length} tokens cost under 0.1 nats`));

    const byName = (re) => D.ladder.find((r) => re.test(r.name));
    Object.assign(N, { "06.uni": f2(D.ladder[0].bpc), "06.best": f2(bestNg.val), "06.n16v": f2(n16.val), "06.cpt": f1(cpt),
      "06.bigram": f2(byName(/token bigram/).bpc), "06.py1k": f2(D.models[0].bpc), "06.smol": f2(smol.bpc), "06.hard": Math.round(top10 * 100) + "%" });
  }

  // ---------------------------------------------------------------- 07
  if (needs("07", ["w07-softmax", "w07-hand", "w07-fd", "w07-step"], "07_gradients.py")) {
    const D = R["07"];
    const sg = D.softmax_grad, P = sg.p;
    const showT = (t) => {
      const g = P.map((p, i) => p - (i === t ? 1 : 0));
      chart($("w07-softmax"), { n: P.length, height: 230, ymin: -1, ymax: 1, yticks: [-1, -0.5, 0, 0.5, 1], yfmt: (v) => (v > 0 ? "+" : "") + v,
        xlabel: (i) => `#${i}${i === t ? " ✓" : ""} · logit ${f1(sg.logits[i])}`, legend: true,
        series: [{ name: "probability p", values: P, color: C.a }, { name: "gradient = p − one_hot", values: g, colorAt: (i, v) => (v < 0 ? C.good : C.bad), color: C.bad }],
        tip: (i, se) => `#${i} · ${se.name}: ${(se.name.startsWith("prob") ? P[i] : g[i]).toFixed(3)}` });
      const others = P.map((p, i) => [p, i]).filter(([, i]) => i !== t).sort((a, b) => b[0] - a[0]);
      $("w07-softmaxcap").textContent = `The right answer #${t} had p = ${f2(P[t])}, so its gradient is ${f2(P[t] - 1)}: a step against the gradient raises its score by the most. ` +
        `The biggest push down goes to #${others[0][1]}, the model's ${others[0][0] === Math.max(...P) ? "favourite" : "most likely wrong answer"} (p = ${f2(others[0][0])}). A step against the gradient raises the green scores and lowers the red ones; the gradients sum to zero.`;
    };
    seg($("w07-target"), P.map((_, i) => ({ label: `#${i} is right`, value: i })), sg.target, showT);
    showT(sg.target);

    table($("w07-hand"), ["tensor", "shape", "max |ours − autograd|"], D.hand_backprop.map((r) => ["d" + r.name, r.shape.join(" × "), ex(r.max_diff)]));
    const digits = (r) => -Math.log10(Math.abs(r.fd - r.autograd) / Math.abs(r.autograd));
    table($("w07-fd"), ["weight", "finite diff", "autograd", "digits"], D.finite_diff.map((r) => [
      r.name.replace("model.", "").replace("layers.", "L").replace(".weight", "").replace("self_attn.", "").replace("mlp.", ""),
      r.fd.toExponential(5), r.autograd.toExponential(5), f1(digits(r))]));
    $("w07-fdhint").textContent = "±1e-6 nudges, float64";

    const os = D.one_step, rows = os.rows;
    const lin = rows.filter((r) => r.linear >= 1);
    const best = rows.reduce((a, b) => (b.same < a.same ? b : a));
    const showEta = (r) => $("w07-stepstats").replaceChildren(
      stat("η = " + r.eta, "step size"), stat(f2(r.same), "same batch"), stat(f2(r.other), "a different batch"),
      stat(r.linear >= -999 ? f1(r.linear) : "", "what the gradient promised"));
    const svg = xy($("w07-step"), [
      { name: "same batch", x: rows.map((r) => r.eta), y: rows.map((r) => r.same), color: C.a, dots: true, width: 5 },
      { name: "a different batch", x: rows.map((r) => r.eta), y: rows.map((r) => r.other), color: C.hot, width: 1.6 },
      { name: "straight-line promise", x: lin.map((r) => r.eta), y: lin.map((r) => r.linear), color: C.mute, dash: "5 4" },
    ], { height: 280, xlog: true, ylog: true, xmin: 0.01, xmax: 100, ymin: 1, ymax: 500, xticks: rows.map((r) => r.eta), yticks: [1, 2, 4, 10, 30, 100, 400],
      xtitle: "step size η (log)", ytitle: "loss after one step (log)", legend: true,
      hlines: [{ y: os.before_same, label: "before: " + f2(os.before_same) }],
      tip: (se, i) => `η = ${se.x[i]} · ${se.name}: ${f2(se.y[i])}`,
      onX: (x) => { const r = rows.reduce((a, b) => (Math.abs(Math.log(b.eta / x)) < Math.abs(Math.log(a.eta / x)) ? b : a)); showEta(r); return r.eta; } });
    svg.setCursor(best.eta); showEta(best);

    const tm = D.timing;
    Object.assign(N, { "07.besteta": String(best.eta), "07.ratio": f1(tm.fwd_bwd_ms / tm.fwd_ms) });
  }

  // ---------------------------------------------------------------- 08
  if (needs("08", ["w08-valley", "w08-race", "w08-quant", "w08-emb", "w08-mem"], "08_optimizer.py")) {
    const D = R["08"];
    const cols = { SGD: C.bad, "SGD+momentum": C.good, Adam: C.a };

    // 2D race
    const m = /([\d.]+)\*x\^2 \+ ([\d.]+)\*y\^2/.exec(D.race2d.fn);
    const A = +m[1], B = +m[2], fval = (p) => A * p[0] ** 2 + B * p[1] ** 2;
    const paths = D.race2d.paths, T = paths.SGD.length - 1;
    const W = 760, H = 280, L0 = 34, R0 = 10, T0 = 10, B0 = 26, xr = [-10, 1.5], yr = [-2.5, 2.5];
    const X = (x) => L0 + (W - L0 - R0) * (x - xr[0]) / (xr[1] - xr[0]);
    const Y = (y) => T0 + (H - T0 - B0) * (1 - (y - yr[0]) / (yr[1] - yr[0]));
    const svgV = svgRoot(W, H, "Optimizer paths on a narrow valley");
    const clip = s("clipPath", { id: "w08-clip" }, s("defs", {}, svgV));
    s("rect", { x: L0, y: T0, width: W - L0 - R0, height: H - T0 - B0 }, clip);
    const g = s("g", { "clip-path": "url(#w08-clip)" }, svgV);
    for (const c of [0.02, 0.08, 0.2, 0.4, 0.7, 1.1, 1.6, 2.2, 3.5, 6, 10, 16]) {
      const rx = Math.sqrt(c / A), ry = Math.sqrt(c / B);
      s("ellipse", { cx: X(0), cy: Y(0), rx: X(rx) - X(0), ry: Y(0) - Y(ry), fill: "none", stroke: "var(--line)", "stroke-width": 1 }, g);
    }
    s("line", { x1: L0, x2: W - R0, y1: H - B0, y2: H - B0, class: "axis" }, svgV);
    for (const t of [-10, -8, -6, -4, -2, 0]) s("text", { x: X(t), y: H - B0 + 14, "text-anchor": "middle", text: t }, svgV);
    for (const t of [-2, 0, 2]) s("text", { x: L0 - 6, y: Y(t) + 3.5, "text-anchor": "end", text: t }, svgV);
    s("text", { x: X(0) + 6, y: Y(0) - 6, text: "min", style: "fill:var(--muted)" }, svgV);
    const lines = {}, dots = {};
    for (const k of Object.keys(paths)) {
      lines[k] = s("polyline", { fill: "none", stroke: cols[k], "stroke-width": 1.8, "stroke-linejoin": "round", "stroke-opacity": 0.9 }, g);
      dots[k] = s("circle", { r: 5, fill: cols[k], stroke: "var(--surface)", "stroke-width": 1.5 }, g);
    }
    const lg = h("div", { class: "seg", style: "font:500 .78rem var(--font-mono);gap:1rem;color:var(--muted)" },
      Object.keys(paths).map((k) => h("span", {}, h("span", { style: `display:inline-block;width:.75rem;height:.75rem;border-radius:2px;background:${cols[k]};margin-right:.35rem;vertical-align:-1px` }), k)));
    $("w08-valley").replaceChildren(svgV, lg);
    const tIn = $("w08-t");
    const drawT = (t) => {
      for (const [k, p] of Object.entries(paths)) {
        lines[k].setAttribute("points", p.slice(0, t + 1).map((q) => `${X(q[0]).toFixed(1)},${Y(q[1]).toFixed(1)}`).join(" "));
        dots[k].setAttribute("cx", X(p[t][0])); dots[k].setAttribute("cy", Y(p[t][1]));
      }
      $("w08-t-out").textContent = String(t);
      $("w08-valleystats").replaceChildren(...Object.entries(paths).map(([k, p]) =>
        stat(fval(p[t]) < 0.1 ? f3(fval(p[t])) : f2(fval(p[t])), `${k} loss · at (${f2(p[t][0])}, ${f2(p[t][1])})`)));
    };
    tIn.max = T; tIn.value = T;
    tIn.addEventListener("input", () => { stop(); drawT(+tIn.value); });
    let timer = null;
    const stop = () => { if (timer) { clearInterval(timer); timer = null; $("w08-play").textContent = "Play"; } };
    $("w08-play").addEventListener("click", () => {
      if (timer) return stop();
      let t = +tIn.value >= T ? 0 : +tIn.value;
      $("w08-play").textContent = "Pause";
      timer = setInterval(() => { tIn.value = t; drawT(t); if (++t > T) stop(); }, 90);
    });
    drawT(T);

    // real race
    const uni = R["06"] ? R["06"].ngrams[0].val * Math.LN2 : null;
    const race = D.race, best = race.best_lr;
    const toS = (c, name, color, o = {}) => ({ name, x: c.map((p) => p[0]), y: c.map((p) => p[1]), color, dots: o.dots, width: o.width ?? 2.2, dash: o.dash, opacity: o.opacity });
    const sweepCol = [C.mute, C.good, C.a, C.bad];
    const showRace = (v) => {
      let series, cap;
      if (v === "best") {
        series = Object.entries(race.curves).map(([k, c]) => toS(c, `${k} (lr ${lrf(best[k])})`, cols[k], { dots: true }));
        const fin = Object.fromEntries(Object.entries(race.curves).map(([k, c]) => [k, c[c.length - 1][1]]));
        cap = `After ${race.steps} steps at each optimizer's best learning rate: SGD ${f2(fin.SGD)}, momentum ${f2(fin["SGD+momentum"])}, Adam ${f2(fin.Adam)}.`;
      } else {
        const runs = D.lr_sweep.filter((r) => r.opt === v);
        series = runs.map((r, i) => toS(r.curve, `lr ${lrf(r.lr)}: ${isFinite(r.val) ? f2(r.val) : "diverged"}`, sweepCol[i], { width: r.lr === best[v] ? 3 : 1.8 }));
        cap = `${v}, four learning rates; the thick line is the best (lr ${lrf(best[v])}).` + (v === "SGD" ? " The two biggest end near the letter-frequency line: stuck, not exploding." : "");
      }
      xy($("w08-race"), series, { height: 270, xmin: 0, xmax: race.steps, ymin: 1, ymax: 3.5, xticks: [0, 100, 200, 300, 400, 500, 600], yticks: [1, 1.5, 2, 2.5, 3, 3.5],
        xtitle: "step", ytitle: "val loss", legend: true, hlines: uni ? [{ y: uni, label: "letter frequencies only (lesson 06)" }] : [],
        tip: (se, i) => `${se.name} · step ${se.x[i]}: ${f3(se.y[i])}` });
      $("w08-racecap").textContent = cap;
    };
    seg($("w08-racesel"), [{ label: "best of each", value: "best" }, { label: "SGD sweep", value: "SGD" }, { label: "momentum sweep", value: "SGD+momentum" }, { label: "Adam sweep", value: "Adam" }], "best", showRace);
    showRace("best");

    // step sizes
    const sc = D.scales, q = sc.quantiles.map((v) => Math.round(v * 100) + "%");
    const sgdQ = sc.grad_q.map((v) => v * sc.sgd_lr);
    table($("w08-quant"), ["", ...q, "95/5"], [
      ["gradient", ...sc.grad_q.map((v) => ex(v)), Math.round(sc.grad_q[4] / sc.grad_q[0]) + "×"],
      ["SGD step", ...sgdQ.map((v) => ex(v)), Math.round(sgdQ[4] / sgdQ[0]) + "×"],
      ["Adam step", ...sc.adam_q.map((v) => ex(v)), Math.round(sc.adam_q[4] / sc.adam_q[0]) + "×"]]);
    const ratio = sc.adam_q[2] / sgdQ[2];
    $("w08-quantcap").textContent = `The median Adam step is ${Math.round(ratio)}× the median SGD step, and Adam's steps are spread over a ${Math.round(sc.adam_q[4] / sc.adam_q[0])}× range instead of ${Math.round(sc.grad_q[4] / sc.grad_q[0])}×. Absolute values; SGD at lr ${sc.sgd_lr}, Adam at lr ${lrf(best.Adam)}, gradients from the trained model.`;

    let embRatio = null;
    if (D.emb_grad) {
      const e = D.emb_grad;
      embRatio = e.common_mean / e.rare_mean;
      bars($("w08-emb"), [
        { labelNode: txt("common: " + e.common.map((c) => (c === " " ? "space" : c)).join(" ")), value: e.common_mean, text: ex(e.common_mean), cls: "hot" },
        { labelNode: txt("rare: " + e.rare.join(" ")), value: e.rare_mean, text: ex(e.rare_mean) }], e.common_mean);
      $("w08-embcap").textContent = `${Math.round(embRatio)}× apart, just because common characters appear in more positions of the batch. SGD would train the rare rows ${Math.round(embRatio)}× slower; Adam's √v̂ evens it out.`;
    } else $("w08-emb").replaceChildren(h("div", { class: "pending", text: "Re-run lessons/08_optimizer.py to fill this in." }));

    // memory calculator
    const sizes = [[D.memory.params, "our model"], [134.5e6, "135M (SmolLM2)"], [360e6, "360M"], [1.7e9, "1.7B"], [7e9, "7B"], [13e9, "13B"], [70e9, "70B"], [405e9, "405B"]];
    const gb = (b) => (b < 1e9 ? (b / 1e6).toFixed(b < 1e7 ? 1 : 0) + " MB" : b < 1e12 ? (b / 1e9).toFixed(b < 1e10 ? 1 : 0) + " GB" : (b / 1e12).toFixed(1) + " TB");
    const nIn = $("w08-n");
    const showMem = () => {
      const [n, lab] = sizes[+nIn.value];
      $("w08-n-out").textContent = lab;
      const items = [["inference, bf16 (2 B)", 2], ["SGD (8 B)", 8], ["momentum (12 B)", 12], ["AdamW (16 B)", 16]];
      bars($("w08-mem"), items.map(([l, b], i) => ({ labelNode: txt(l), value: n * b, text: gb(n * b), cls: i === 3 ? "hot" : "" })), n * 16);
    };
    nIn.addEventListener("input", showMem); showMem();

    const fin = (k) => race.curves[k][race.curves[k].length - 1][1];
    Object.assign(N, { "08.match": ex(Math.max(...D.match.map((r) => r.max_diff))), "08.adam": f2(fin("Adam")), "08.mom": f2(fin("SGD+momentum")), "08.sgd": f2(fin("SGD")),
      "08.ratio": String(Math.round(ratio)) });
    if (embRatio) N["08.emb"] = String(Math.round(embRatio));
  }

  // ---------------------------------------------------------------- 09
  if (needs("09", ["w09-u", "w09-curves", "w09-warm", "w09-lr", "w09-sched", "w09-cliploss", "w09-clipg"], "09_learning_rate.py")) {
    const D = R["09"];
    const uni = R["06"] ? R["06"].ngrams[0].val * Math.LN2 : null;
    const uniLine = uni ? [{ y: uni, label: "letter frequencies only" }] : [];
    const lrs = Object.keys(D.sweep).map(Number).sort((a, b) => a - b);
    const run = (lr) => D.sweep[String(lr)];
    const bestLr = lrs.reduce((a, b) => ((run(b).final ?? 99) < (run(a).final ?? 99) ? b : a));
    const steps = D.steps, xt = [0, 100, 200, 300, 400, 500, 600].filter((v) => v <= steps);
    const pal = ["#8a93a6", "#6f7fd0", C.a, C.good, "#d08a2e", C.hot, C.bad];

    let sel = bestLr;
    const drawU = () => xy($("w09-u"), [{ name: "final val", x: lrs, y: lrs.map((lr) => run(lr).final), color: C.a, dots: true, width: 2.4 }],
      { width: 460, height: 260, xlog: true, xmin: lrs[0], xmax: lrs[lrs.length - 1], ymin: 1, ymax: 3.2, xticks: lrs, xfmt: lrf, yticks: [1, 1.5, 2, 2.5, 3],
        xtitle: "learning rate (log)", ytitle: "final val loss", hlines: uniLine, tip: (se, i) => `lr ${lrf(se.x[i])}: ${f3(se.y[i])}`,
        onX: (x) => { const lr = lrs.reduce((a, b) => (Math.abs(Math.log(b / x)) < Math.abs(Math.log(a / x)) ? b : a)); if (lr !== sel) { sel = lr; drawC(); } return lr; } });
    const drawC = () => {
      const series = lrs.map((lr, i) => ({ name: lrf(lr), x: run(lr).val_step, y: run(lr).val, color: pal[i], width: lr === sel ? 3.2 : 1.4, opacity: lr === sel ? 1 : 0.45 }));
      xy($("w09-curves"), series, { width: 460, height: 260, xmin: 0, xmax: steps, ymin: 1, ymax: 3.5, xticks: xt, yticks: [1, 1.5, 2, 2.5, 3, 3.5],
        xtitle: "step", ytitle: "val loss", hlines: uniLine, legend: true, tip: (se, i) => `lr ${se.name} · step ${se.x[i]}: ${f3(se.y[i])}` });
      $("w09-curvehint").textContent = `highlighted: lr ${lrf(sel)} → ${f2(run(sel).final)} · hover the left chart`;
      const u = $("w09-u").querySelector("svg"); if (u && u.setCursor) u.setCursor(sel);
    };
    drawU(); drawC();

    // warmup
    const W = D.warmup;
    $("w09-warmhint").textContent = `lr ${lrf(W.lr)} (3× the best), ${W.steps}-step warmup`;
    const showSeed = (i) => {
      const r = W.runs[i], xs = r.none.train.map((_, j) => j + 1);
      xy($("w09-warm"), [{ name: "no warmup", x: xs, y: smooth(r.none.train, 20), color: C.a }, { name: "warmup", x: xs, y: smooth(r.warmup.train, 20), color: C.hot }],
        { width: 460, height: 240, xmin: 0, xmax: steps, ymin: 1, ymax: 4.6, xticks: xt, yticks: [1, 2, 3, 4], xtitle: "step", ytitle: "train loss (smoothed)", legend: true,
          vlines: [{ x: W.steps, label: "warmup ends" }] });
      bars($("w09-warmbars"), W.runs.flatMap((rr) => [
        { labelNode: txt(`seed ${rr.seed} · no warmup`), value: rr.none.final, text: f3(rr.none.final), cls: "" },
        { labelNode: txt(`seed ${rr.seed} · warmup`), value: rr.warmup.final, text: f3(rr.warmup.final), cls: "hot" }]), 1.5);
      [...$("w09-warmbars").children].forEach((b, j) => { b.style.opacity = Math.floor(j / 2) === i ? 1 : 0.45; });
    };
    seg($("w09-seed"), W.runs.map((r, i) => ({ label: `seed ${r.seed}`, value: i })), 0, showSeed);
    showSeed(0);

    // schedules
    const S = D.schedules.runs, sc = { constant: C.mute, cosine: C.a, linear: C.good, WSD: C.hot };
    xy($("w09-lr"), Object.entries(S).map(([k, l]) => ({ name: k, x: l.lr.map((_, j) => j), y: l.lr, color: sc[k], width: k === "WSD" ? 2.6 : 1.8 })),
      { width: 460, height: 240, xmin: 0, xmax: steps, ymin: 0, ymax: D.schedules.peak * 1.05, xticks: xt, yticks: [0, D.schedules.peak / 2, D.schedules.peak], yfmt: (v) => (v ? v.toExponential(0).replace("e-", "e−") : "0"),
        xtitle: "step", legend: true });
    const tail = (l) => l.val_step.map((x, j) => [x, l.val[j]]).filter(([x]) => x >= steps / 2);
    xy($("w09-sched"), Object.entries(S).map(([k, l]) => { const t = tail(l); return { name: `${k} ${f3(l.final)}`, x: t.map((p) => p[0]), y: t.map((p) => p[1]), color: sc[k], dots: true, width: k === "WSD" ? 2.6 : 1.8 }; }),
      { width: 460, height: 240, xmin: steps / 2, xmax: steps, ymin: 1.08, ymax: 1.38, xticks: xt.filter((v) => v >= steps / 2), yticks: [1.1, 1.2, 1.3], xtitle: "step", ytitle: "val loss", legend: true,
        vlines: [{ x: steps * 0.8, label: "WSD decays" }], tip: (se, i) => `${se.name.split(" ")[0]} · step ${se.x[i]}: ${f3(se.y[i])}` });

    // clipping
    const CL = D.clipping.runs, ccol = { "no clipping": C.bad, "clip at 1.0": C.a };
    const xs = CL["no clipping"].train.map((_, j) => j + 1);
    xy($("w09-cliploss"), Object.entries(CL).map(([k, l]) => ({ name: `${k}: final ${f2(l.final)}`, x: xs, y: smooth(l.train, 20), color: ccol[k] })),
      { width: 460, height: 240, xmin: 0, xmax: steps, ymin: 1, ymax: 4.6, xticks: xt, yticks: [1, 2, 3, 4], xtitle: "step", legend: true });
    xy($("w09-clipg"), Object.entries(CL).map(([k, l]) => ({ name: k, x: xs, y: l.gnorm, color: ccol[k], width: 1.1, opacity: 0.85 })),
      { width: 460, height: 240, xmin: 0, xmax: steps, ylog: true, ymin: 0.05, ymax: 60, xticks: xt, yticks: [0.1, 1, 10], xtitle: "step", legend: true,
        hlines: [{ y: 1, label: "clip limit 1.0" }] });

    const sw = S.WSD, at500 = sw.val[sw.val_step.indexOf(500)];
    const lrsHot = lrs.slice(-2);
    Object.assign(N, { "09.best": lrf(bestLr), "09.hot1": f2(run(lrsHot[0]).final), "09.hot2": f2(run(lrsHot[1]).final),
      "09.wsd": f2(sw.final), "09.const": f2(S.constant.final), "09.clip0": f2(CL["no clipping"].final), "09.clip1": f2(CL["clip at 1.0"].final) });
    if (at500 != null) N["09.wsd500"] = f2(at500);
  }

  document.querySelectorAll("[data-n]").forEach((el) => { if (N[el.dataset.n] != null) el.textContent = N[el.dataset.n]; });
})();
