// Part 1 widgets (lessons 01-05). All data comes from window.RESULTS (site/build_data.py).
(function () {
  const { $, h, chip, bars, seg, bindTip, heat, xy, chart, pct, range } = UI;
  const R = window.RESULTS;
  // Numbers quoted in the prose: <span data-n="key">fallback</span> gets N[key] from this run's results.
  const N = {};
  const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
  const smooth = (a, k) => a.map((_, i) => mean(a.slice(Math.max(0, i - k + 1), i + 1)));
  const f1 = (v) => v.toFixed(1), f2 = (v) => v.toFixed(2), f0 = (v) => Math.round(v).toLocaleString();
  const C = { a: "var(--accent)", hot: "var(--hot)", good: "var(--good)", bad: "var(--bad)", mute: "var(--muted)" };

  function needs(key, ids, lesson) {
    if (R[key]) return true;
    for (const id of ids) {
      const el = $(id);
      if (el) el.replaceChildren(h("div", { class: "pending", text: `Run lessons/${lesson} and then python site/build_data.py to fill this in.` }));
    }
    return false;
  }
  const outText = (t) => (t && t.trim() ? t.trim() : null);
  function outBox(el, text) {
    const t = outText(text);
    el.replaceChildren(t ? document.createTextNode(t) : h("span", { class: "empty", text: "(nothing: first token was <|im_end|>)" }));
  }

  // ---------------------------------------------------------------- 01
  if (needs("01", ["w01-grid", "w01-next", "w01-template"], "01_two_models.py")) {
    const D = R["01"];
    const qs = Object.keys(D.answers);
    const grid = $("w01-grid");
    const show = (qi) => {
      const a = D.answers[qs[qi]];
      grid.replaceChildren(
        h("span", {}), h("span", { class: "hd", text: "plain text in" }), h("span", { class: "hd", text: "chat template in" }),
        ...[["base", "base"], ["instruct", "inst"]].flatMap(([lab, k]) => {
          const b1 = h("div", { class: "out-box", "data-k": lab + " · plain text" }), b2 = h("div", { class: "out-box", "data-k": lab + " · chat template" });
          outBox(b1, a[k + "_plain"]); outBox(b2, a[k + "_chat"]);
          return [h("span", { class: "rowhd", text: lab }), b1, b2];
        })
      );
    };
    seg($("w01-q"), qs.map((q, i) => ({ label: q, value: i })), 0, show);
    show(0);

    bars($("w01-next"), D.inst_plain_next.map(([t, p], i) => ({ label: t, value: p, cls: i === 0 ? "hot" : "" })), 1);
    const tr = $("w01-template");
    tr.replaceChildren(...D.template_tokens.map((t) => chip(t, { cls: /^<\|.*\|>$/.test(t) ? "special" : "" })));
    $("w01-ntok").textContent = `${D.template_tokens.length} tokens`;

    // who stops? values like "94" or "120+ (never stopped)"
    const SL = D.stop_lengths;
    const rows = [];
    for (const [name, lens] of Object.entries(SL)) lens.forEach((v, i) => {
      const n = parseInt(v), never = /never/.test(v);
      rows.push({ labelNode: [document.createTextNode(`${name} · ${qs[i].split(" ").slice(-2).join(" ")}`)], value: n, text: never ? `${n}+ never` : `${n} ✓`, cls: never ? "" : "hot" });
    });
    const box = h("div", { class: "bars wide" });
    $("w01-stop").replaceChildren(box);
    bars(box, rows, 120);
  }

  // ---------------------------------------------------------------- 02
  if (needs("02", ["w02-heat", "w02-dir", "w02-ans", "w02-tokens"], "02_what_changed.py")) {
    const D = R["02"];
    $("w02-global").textContent = `whole model: ‖Δ‖/‖W‖ = ${pct(D.rel_global)}`;
    N["02.rel"] = pct(D.rel_global);
    if (D.share_tiny != null) Object.assign(N, { "02.tiny": pct(D.share_tiny), "02.medw": D.median_w.toFixed(3), "02.medd": D.median_d.toFixed(3),
      "02.embed": pct(D["model.embed_tokens.weight"]) });
    const K = D.delta_grid.kinds, G = D.delta_grid.grid;
    N["02.v0"] = (G[K.indexOf("v_proj")][0] * 100).toFixed(0) + "%";
    N["02.n0"] = (G[K.indexOf("input_layernorm")][0] * 100).toFixed(0) + "%";
    N["02.ppl_rand"] = f2(D.random[0].ppl); N["02.ppl_base"] = f2(D.ppl_base); N["02.ppl_inst"] = f2(D.ppl_inst);
    const r0 = D.rank_share["rank 0"]; N["02.r0"] = pct(r0);
    const NICE = { input_layernorm: "norm before attn", post_attention_layernorm: "norm before MLP" };
    const g = D.delta_grid.grid;
    const mx = Math.max(...g.flat());
    $("w02-heat").replaceChildren(heat(g.map((r) => r.map((v) => v / mx)), {
      cell: 22, cellH: 18, left: 150, top: 22, label: "relative change per matrix and layer",
      rowLabel: (r) => NICE[D.delta_grid.kinds[r]] || D.delta_grid.kinds[r], colLabel: (c) => (c % 5 === 0 ? c : null),
      tip: (r, c) => `layer ${c} · ${D.delta_grid.kinds[r]}\nchanged by ${pct(g[r][c])}`,
    }));

    const dir = [
      { labelNode: [document.createTextNode("base")], value: D.ppl_base, text: D.ppl_base.toFixed(2) },
      { labelNode: [document.createTextNode("instruct (Δ)")], value: D.ppl_inst, text: D.ppl_inst.toFixed(2), cls: "hot" },
      ...D.random.map((r) => ({ labelNode: [document.createTextNode(`random Δ ×${r.scale}`)], value: r.ppl, text: r.ppl.toFixed(2) })),
    ];
    bars($("w02-dir"), dir);

    const A = D.alpha;
    const upd = range("w02-alpha", (i) => A[i].alpha.toFixed(2), (i) => {
      outBox($("w02-ans"), A[i].answer);
      $("w02-ppl").textContent = `perplexity ${A[i].ppl.toFixed(2)}` + (A[i].alpha === 0 ? " · base" : A[i].alpha === 1 ? " · instruct" : "");
    });
    $("w02-alpha").max = A.length - 1;
    upd();
    const ppls = A.map((a) => a.ppl);
    const drawA = (i) => xy($("w02-alpha-ppl"), [{ name: null, x: A.map((a) => a.alpha), y: ppls, color: C.a, dots: true }], {
      width: 460, height: 170, left: 40, bottom: 30, ymin: Math.floor(Math.min(...ppls) * 2) / 2, ymax: Math.ceil(Math.max(...ppls) * 2) / 2,
      yticks: [5.5, 6, 6.5, 7].filter((t) => t >= Math.min(...ppls) - 0.5 && t <= Math.max(...ppls) + 0.5),
      xticks: [0, 0.5, 1], xtitle: "α", hlines: [{ y: D.ppl_base, label: "base", color: C.mute }],
      vlines: [{ x: A[i].alpha, color: C.hot }], tip: (se, j) => `α = ${A[j].alpha}: perplexity ${f2(ppls[j])}`,
    });
    drawA(0);
    $("w02-alpha").addEventListener("input", () => drawA(+$("w02-alpha").value));

    const ex = D.rank_examples;
    const cls = (r) => (r === 0 ? "r0" : r <= 2 ? "r1" : r <= 9 ? "r2" : "r3");
    const showEx = (i) => {
      const e = ex[i];
      $("w02-tokens").replaceChildren(...e.tokens.map((t, j) => {
        const c = chip(t, { cls: cls(e.ranks[j]) });
        bindTip(c, () => `${JSON.stringify(t)}\nbase model's rank: ${e.ranks[j]}${e.ranks[j] === 0 ? " (its top choice)" : ""}`);
        return c;
      }));
    };
    seg($("w02-prompts"), ex.map((e, i) => ({ label: e.prompt, value: i })), 0, showEx);
    showEx(0);
    const sh = D.rank_share;
    bars($("w02-share"), Object.entries(sh).map(([k, v], i) => ({ label: k, value: v, cls: i === 3 ? "hot" : "" })), 1);
  }

  // ---------------------------------------------------------------- 03
  if (needs("03", ["w03-mask", "w03-curve", "w03-before", "w03-maskeffect"], "03_sft.py")) {
    const D = R["03"];
    const m = D.mask_example;
    $("w03-mask").replaceChildren(...m.tokens.map((t, i) => chip(t, { cls: (/^<\|.*\|>$/.test(t) ? "special " : "") + (m.graded[i] ? "graded" : "ignored") })));
    $("w03-maskstat").textContent = `whole dataset: ${(D.n_graded / D.n_tokens * 100).toFixed(0)}% of ${D.n_tokens.toLocaleString()} tokens graded`;

    // smooth the per-batch train loss a little so the curve is readable
    const L = D.log, k = 10;
    const sm = L.train.map((_, i) => { const a = L.train.slice(Math.max(0, i - k + 1), i + 1); return a.reduce((x, y) => x + y, 0) / a.length; });
    const ymax = Math.ceil(Math.max(...L.train.slice(0, 5), ...L.test) * 2) / 2;
    xy($("w03-curve"), [
      { name: null, x: L.step, y: L.train, color: "var(--accent)", opacity: 0.25, width: 1 },
      { name: "train (smoothed)", x: L.step, y: sm, color: "var(--accent)" },
      { name: "test (held-out)", x: L.test_step, y: L.test, color: "var(--hot)", dots: true },
    ], {
      height: 260, ymin: 1, ymax, yticks: [1, 1.5, 2, 2.5].filter((t) => t <= ymax), xticks: L.test_step,
      xtitle: "step", ytitle: "loss", legend: true,
      hlines: [{ y: D.loss_inst, color: "var(--muted)" }],
      tip: (se, i) => `step ${se.x[i]}\n${se.name}: ${se.y[i].toFixed(3)}`,
    });
    $("w03-runinfo").textContent = `${D.steps} steps × ${D.batch} conversations · lr ${D.lr} · ${D.device}`;
    N["03.t50"] = f2(L.test[1]); N["03.t0"] = f2(L.test[0]); N["03.tend"] = f2(L.test[L.test.length - 1]);
    $("w03-stats").replaceChildren(
      ...[[`${D.seconds.toFixed(0)} s`, "training time"],
        [L.test[0].toFixed(2) + " → " + L.test[L.test.length - 1].toFixed(2), "test loss"],
        [(D.steps * D.batch).toLocaleString(), "conversations seen"],
        [D.loss_inst.toFixed(2), "SmolLM2-Instruct on the same test (dashed line)"]].map(([b, s_]) => h("div", { class: "stat" }, h("b", { text: b }), h("span", { text: s_ }))));

    const ps = Object.keys(D.answers_after);
    const showP = (i) => { outBox($("w03-before"), D.answers_before[ps[i]]); outBox($("w03-after"), D.answers_after[ps[i]]); };
    seg($("w03-prompts"), ps.map((p, i) => ({ label: p, value: i })), 0, showP);
    showP(0);

    const mx = Math.max(...D.user_loss, ...D.assistant_loss);
    bars($("w03-maskeffect"), [
      { labelNode: [document.createTextNode("assistant · before")], value: D.assistant_loss[0], text: D.assistant_loss[0].toFixed(2) },
      { labelNode: [document.createTextNode("assistant · after")], value: D.assistant_loss[1], text: D.assistant_loss[1].toFixed(2), cls: "hot" },
      { labelNode: [document.createTextNode("user · before")], value: D.user_loss[0], text: D.user_loss[0].toFixed(2) },
      { labelNode: [document.createTextNode("user · after")], value: D.user_loss[1], text: D.user_loss[1].toFixed(2) },
    ], mx);
  }

  // ---------------------------------------------------------------- 04
  if (needs("04", ["w04-chosen", "w04-hist", "w04-dlogp", "w04-len", "w04-loss", "w04-three", "w04-table"], "04_dpo.py")) {
    const D = R["04"];
    const [good, loose] = D.runs;
    const lc = D.len_hist.chosen, lr = D.len_hist.rejected;
    $("w04-npairs").textContent = `${D.n_pairs} pairs, self-generated`;
    const showPair = (i) => {
      const p = D.pair_examples[i];
      $("w04-prompt").textContent = "“" + p.prompt + "”";
      $("w04-chosen").textContent = p.chosen; $("w04-rejected").textContent = p.rejected;
      $("w04-clab").textContent = `chosen · ${p.len_c} tokens`; $("w04-rlab").textContent = `rejected · ${p.len_r} tokens`;
    };
    seg($("w04-pairsel"), D.pair_examples.map((p, i) => ({ label: `pair ${i + 1}`, value: i })), 0, showPair);
    showPair(0);

    // histogram, 10-token bins
    const bw = 10, nb = Math.ceil(Math.max(...lc, ...lr) / bw);
    const cnt = (a) => { const c = Array(nb).fill(0); a.forEach((v) => c[Math.min(nb - 1, Math.floor(v / bw))]++); return c; };
    const hc = cnt(lc), hr = cnt(lr), top = Math.max(...hc, ...hr);
    chart($("w04-hist"), { n: nb, width: 460, height: 240, series: [{ name: "rejected (shorter)", values: hr, color: C.bad }, { name: "chosen (longer)", values: hc, color: C.good }],
      ymax: top, yticks: [0, Math.round(top / 2), top], xlabel: (i) => i * bw, xevery: 2, xtitle: "answer length (tokens)", legend: true,
      tip: (i, se) => `${i * bw}–${i * bw + bw - 1} tokens\n${se.name}: ${se.values[i]} answers` });
    $("w04-histcap").textContent = `Chosen answers average ${f1(mean(lc))} tokens, rejected ${f1(mean(lr))}. From 1,024 prompts, ${D.n_pairs} gave a usable pair.`;

    // Δlogp curves
    const dl = [];
    for (const [run, lab, w] of [[good, "β=0.1", 2.2], [loose, "β=0.01", 2.2]]) {
      const lg = run.log;
      dl.push({ name: `${lab} chosen`, x: lg.step, y: smooth(lg.dc, 8), color: lab === "β=0.1" ? C.good : C.a, width: w });
      dl.push({ name: `${lab} rejected`, x: lg.step, y: smooth(lg.dr, 8), color: lab === "β=0.1" ? C.good : C.a, width: w, dash: "6 4" });
    }
    const ymn = Math.min(...dl.flatMap((d) => d.y));
    xy($("w04-dlogp"), dl, { height: 260, ymin: Math.floor(ymn / 50) * 50, ymax: 5, yticks: [0, -50, -100, -150, -200].filter((t) => t >= ymn - 50),
      xticks: [0, 20, 40, 60, 80, 100, 120], xmin: 0, xtitle: "training step", ytitle: "Δ log-prob (nats)", legend: true,
      hlines: [{ y: 0, color: C.mute, label: "reference" }] });

    // evals over steps
    const ev = (run, k) => ({ x: run.evals.map((e) => e.step), y: run.evals.map((e) => e[k]) });
    const lenS = [{ name: "β = 0.1", ...ev(good, "len"), color: C.good, dots: true }, { name: "β = 0.01, pulled hard", ...ev(loose, "len"), color: C.a, dots: true }];
    const lmax = Math.max(...lenS.flatMap((s_) => s_.y));
    xy($("w04-len"), lenS, { width: 460, height: 260, ymin: 0, ymax: Math.ceil(lmax / 20) * 20, yticks: [0, 20, 40, 60, 80].filter((t) => t <= lmax + 20), xticks: [0, 40, 80, 120],
      xmin: 0, xtitle: "step", ytitle: "tokens", legend: true, tip: (se, i) => `${se.name}\nstep ${se.x[i]}: ${f1(se.y[i])} tokens` });
    const lossS = [{ name: "β = 0.1", ...ev(good, "test_loss"), color: C.good, dots: true }, { name: "β = 0.01, pulled hard", ...ev(loose, "test_loss"), color: C.a, dots: true }];
    const tmax = Math.max(...lossS.flatMap((s_) => s_.y));
    xy($("w04-loss"), lossS, { width: 460, height: 260, ymin: 0, ymax: Math.ceil(tmax), yticks: [0, 1, 2, 3, 4, 5].filter((t) => t <= Math.ceil(tmax)), xticks: [0, 40, 80, 120],
      xmin: 0, xtitle: "step", ytitle: "loss", legend: true, hlines: [{ y: R["03"] ? R["03"].log.test[0] : 2.19, label: "base model, before SFT", color: C.mute }],
      tip: (se, i) => `${se.name}\nstep ${se.x[i]}: test loss ${f2(se.y[i])}` });

    // three greedy answers
    const showE = (i) => $("w04-three").replaceChildren(
      ...[["SFT (start)", D.before.examples[i]], ["DPO β = 0.1", good.examples[i]], ["DPO β = 0.01, pulled hard", loose.examples[i]]].map(([lab, t]) =>
        h("div", {}, h("p", { class: "lbl", style: "font:600 .72rem var(--font-mono);color:var(--muted);letter-spacing:.02em;margin-bottom:.35rem", text: `${lab} · ${t.split(/\s+/).length} words` }), h("div", { class: "out-box", style: "font-size:.85rem", text: t }))));
    seg($("w04-evalsel"), D.eval_prompts.map((p, i) => ({ label: p.length > 44 ? p.slice(0, 42) + "…" : p, value: i })), 0, showE);
    showE(0);

    // scoreboard
    const lastG = good.evals[good.evals.length - 1], lastL = loose.evals[loose.evals.length - 1];
    const tail = (lg, k) => mean(lg[k].slice(-40));
    const S = D.sampled, sk = Object.keys(S);
    const rowsT = [
      ["greedy length (tokens)", f1(D.before.len), f1(lastG.len), f1(lastL.len)],
      ["sampled length (tokens)", f1(S[sk[0]].len), f1(S[sk[1]].len), f1(S[sk[2]].len)],
      ["stops by itself (sampled)", pct(S[sk[0]].stopped, 0), pct(S[sk[1]].stopped, 0), pct(S[sk[2]].stopped, 0)],
      ["repeated 3-grams (sampled)", pct(S[sk[0]].rep), pct(S[sk[1]].rep), pct(S[sk[2]].rep)],
      ["lesson 03 test loss", f2(D.before.test_loss), f2(lastG.test_loss), f2(lastL.test_loss)],
      ["Δ log-prob chosen / rejected", "0 / 0", `${f1(tail(good.log, "dc"))} / ${f1(tail(good.log, "dr"))}`, `${f1(tail(loose.log, "dc"))} / ${f1(tail(loose.log, "dr"))}`],
      ["accuracy on training pairs", "50%", pct(tail(good.log, "acc"), 0), pct(tail(loose.log, "acc"), 0)],
    ];
    $("w04-table").replaceChildren(
      h("thead", {}, h("tr", {}, ...["", "SFT start", `β = ${good.beta}`, `β = ${loose.beta}, pulled hard`].map((t) => h("th", { text: t })))),
      h("tbody", {}, ...rowsT.map((r) => h("tr", {}, ...r.map((c, j) => h(j ? "td" : "th", { text: c, style: j === 3 && /loss|Δ/.test(r[0]) ? "color:var(--bad);font-weight:600" : null }))))));
    Object.assign(N, { "04.len0": f0(D.before.len), "04.len1": f0(lastG.len), "04.tl0": f2(D.before.test_loss), "04.tl1": f2(lastG.test_loss), "04.tl2": f2(lastL.test_loss) });
  }

  // ---------------------------------------------------------------- 05
  if (needs("05", ["w05-losschart", "w05-comps", "w05-facts", "w05-copy", "w05-variety"], "05_rewind_pretraining.py")) {
    const D = R["05"], CK = D.checkpoints, TPS = D.tokens_per_step;
    const tx = CK.map((c) => Math.max(c.tokens, TPS / 4));                 // step 0 drawn at the left edge
    const fmtT = (t) => (t >= 1e9 ? f1(t / 1e9) + "B" : t >= 1e6 ? f0(t / 1e6) + "M" : f0(t));
    const xt = [1e8, 1e9, 1e10, 1e11].filter((t) => t >= tx[0] && t <= tx[tx.length - 1]);
    const xfmt = (t) => (t === tx[0] ? "0" : fmtT(t));
    const uni = Math.log(D.vocab);
    const nearest = (x) => tx.reduce((b, t, i) => (Math.abs(Math.log(t) - Math.log(x)) < Math.abs(Math.log(tx[b]) - Math.log(x)) ? i : b), 0);
    let svg;
    const drawLoss = () => {
      svg = xy($("w05-losschart"), [{ name: null, x: tx, y: CK.map((c) => c.loss), color: C.a, dots: true }], {
        height: 210, xlog: true, ymin: 3, ymax: 12, yticks: [4, 6, 8, 10, 12], xticks: [tx[0], ...xt], xfmt, xtitle: "tokens seen (log scale)", ytitle: "loss (nats/token)",
        hlines: [{ y: uni, label: "uniform guess ln V", color: C.mute }],
        tip: (se, i) => `step ${CK[i].step.toLocaleString()} · ${fmtT(CK[i].tokens)} tokens\nloss ${f2(CK[i].loss)}`,
        onX: (x) => { const i = nearest(x); setStep(i); return tx[i]; },
      });
    };
    const inp = $("w05-step");
    inp.max = CK.length - 1;
    const facts = D.facts;
    function setStep(i) {
      inp.value = i;
      const c = CK[i];
      $("w05-step-out").textContent = c.step.toLocaleString();
      $("w05-when").textContent = `step ${c.step.toLocaleString()} · ${fmtT(c.tokens)} tokens · ${(c.step / CK[CK.length - 1].step * 100).toFixed(c.step && c.step < 1500 ? 2 : 0)}% of training`;
      $("w05-comps").replaceChildren(...D.prompts.map((p) => h("div", { class: "out-box", style: "font-size:.84rem;padding:.6rem .8rem" },
        h("span", { class: "muted", text: p }), h("span", { class: "gen", text: c.completions[p] }))));
      bars($("w05-facts"), facts.map(([q, a], j) => ({ labelNode: [document.createTextNode(`${q.split(" ").slice(-2).join(" ")} →${a}`)], value: c.facts[j], cls: c.facts[j] > 0.3 ? "hot" : "" })), 1);
      $("w05-stats").replaceChildren(...[
        [f2(c.loss), "loss on WikiText"],
        [`${f2(c.copy_nat[0])} → ${f2(c.copy_nat[1])}`, "real passage, 1st → 2nd time"],
        [c.distinct_top1.toLocaleString(), "different top-1 tokens"],
        [`${JSON.stringify(c.most_predicted[0][0])} ${pct(c.most_predicted[0][1], 0)}`, "most predicted token"],
      ].map(([b, t]) => h("div", { class: "stat" }, h("b", { text: b }), h("span", { text: t }))));
      if (svg && svg.setCursor) svg.setCursor(tx[i]);
    }
    drawLoss();
    inp.addEventListener("input", () => setStep(+inp.value));
    setStep(0);

    const cpy = [
      { name: "real text, 1st", x: tx, y: CK.map((c) => c.copy_nat[0]), color: C.a, dots: true },
      { name: "real text, 2nd (copyable)", x: tx, y: CK.map((c) => c.copy_nat[1]), color: C.hot, dots: true },
      { name: "random, 1st", x: tx, y: CK.map((c) => c.copy_rnd[0]), color: C.a, dash: "6 4" },
      { name: "random, 2nd", x: tx, y: CK.map((c) => c.copy_rnd[1]), color: C.hot, dash: "6 4" },
    ];
    const cmax = Math.max(...cpy.flatMap((s_) => s_.y));
    xy($("w05-copy"), cpy, { width: 460, height: 300, xlog: true, ymin: 0, ymax: Math.ceil(cmax / 5) * 5, yticks: [0, 5, 10, 15, 20, 25].filter((t) => t <= cmax + 5),
      xticks: [tx[0], ...xt], xfmt, xtitle: "tokens seen", ytitle: "nats/token", legend: true, hlines: [{ y: uni, color: C.mute }],
      tip: (se, i) => `${se.name}\nstep ${CK[i].step.toLocaleString()}: ${f2(se.y[i])}` });
    const i512 = CK.findIndex((c) => c.step === 512), i1k = CK.findIndex((c) => c.step === 1000);
    if (i512 >= 0 && i1k >= 0) {
      $("w05-copycap").textContent = `Between step 512 and 1,000 the second copy of a real passage drops from ${f2(CK[i512].copy_nat[1])} to ${f2(CK[i1k].copy_nat[1])} nats. Dashed: random tokens, which can only be predicted by copying. Grey: the uniform guess.`;
      N["05.c512"] = f1(CK[i512].copy_nat[1]); N["05.c1k"] = f1(CK[i1k].copy_nat[1]);
      const L0 = CK[0].loss, Lend = CK[CK.length - 1].loss;
      N["05.frac"] = ((L0 - CK[i1k].loss) / (L0 - Lend) * 100).toFixed(0) + "%";
    }
    xy($("w05-variety"), [{ name: null, x: tx, y: CK.map((c) => c.distinct_top1), color: C.a, dots: true }], {
      width: 460, height: 300, xlog: true, ymin: 0, ymax: 1000, yticks: [0, 250, 500, 750, 1000], xticks: [tx[0], ...xt], xfmt, xtitle: "tokens seen",
      tip: (se, i) => `step ${CK[i].step.toLocaleString()}: ${CK[i].distinct_top1} different tokens\ntop: ${JSON.stringify(CK[i].most_predicted[0][0])} ${pct(CK[i].most_predicted[0][1], 0)}` });
    N["05.v0"] = f0(CK[0].distinct_top1);
    const i128 = CK.findIndex((c) => c.step === 128); if (i128 >= 0) N["05.v128"] = f0(CK[i128].distinct_top1);
  }

  document.querySelectorAll("[data-n]").forEach((el) => { if (N[el.dataset.n] != null) el.textContent = N[el.dataset.n]; });

  UI.initTheme();
  UI.initSpy();
})();
