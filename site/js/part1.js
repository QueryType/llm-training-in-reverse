// Part 1 widgets (lessons 01-03). All data comes from window.RESULTS (site/build_data.py).
(function () {
  const { $, h, chip, bars, seg, bindTip, heat, xy, pct, range } = UI;
  const R = window.RESULTS;

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
  }

  // ---------------------------------------------------------------- 02
  if (needs("02", ["w02-heat", "w02-dir", "w02-ans", "w02-tokens"], "02_what_changed.py")) {
    const D = R["02"];
    $("w02-global").textContent = `whole model: ‖Δ‖/‖W‖ = ${pct(D.rel_global)}`;
    const NICE = { input_layernorm: "norm before attn", post_attention_layernorm: "norm before MLP" };
    const g = D.delta_grid.grid;
    const mx = Math.max(...g.flat());
    $("w02-heat").replaceChildren(heat(g.map((r) => r.map((v) => v / mx)), {
      cell: 22, cellH: 18, left: 150, top: 22, label: "relative change per matrix and layer",
      rowLabel: (r) => NICE[D.delta_grid.kinds[r]] || D.delta_grid.kinds[r], colLabel: (c) => (c % 5 === 0 ? c : null),
      tip: (r, c) => `layer ${c} · ${D.delta_grid.kinds[r]}\nchanged by ${pct(g[r][c])}`,
    }));

    const dir = [
      { label: "base", value: D.ppl_base, text: D.ppl_base.toFixed(2) },
      { label: "instruct (Δ)", value: D.ppl_inst, text: D.ppl_inst.toFixed(2), cls: "hot" },
      ...D.random.map((r) => ({ label: `random Δ ×${r.scale}`, value: r.ppl, text: r.ppl.toFixed(2) })),
    ];
    bars($("w02-dir"), dir);

    const A = D.alpha;
    const upd = range("w02-alpha", (i) => A[i].alpha.toFixed(2), (i) => {
      outBox($("w02-ans"), A[i].answer);
      $("w02-ppl").textContent = `perplexity ${A[i].ppl.toFixed(2)}` + (A[i].alpha === 0 ? " · base" : A[i].alpha === 1 ? " · instruct" : "");
    });
    $("w02-alpha").max = A.length - 1;
    upd();

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
      { label: "assistant · before", value: D.assistant_loss[0], text: D.assistant_loss[0].toFixed(2) },
      { label: "assistant · after", value: D.assistant_loss[1], text: D.assistant_loss[1].toFixed(2), cls: "hot" },
      { label: "user · before", value: D.user_loss[0], text: D.user_loss[0].toFixed(2) },
      { label: "user · after", value: D.user_loss[1], text: D.user_loss[1].toFixed(2) },
    ], mx);
  }

  UI.initTheme();
  UI.initSpy();
})();
