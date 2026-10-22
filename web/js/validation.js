// Validation tab: cross-validation results, error analysis, robustness to
// damaged audio, the external dataset test, the safety check and limits.

const Validation = (() => {
  let external = null;
  const fmt2 = (x) => (x === null || x === undefined ? "–" : x.toFixed(2));
  const pctOrDash = (x) => (x === null || x === undefined ? "–" : pct(x));

  function cards() {
    const m = METRICS, c = m.cnn, ds = m.dataset;
    const pm = (k) => `${pct(c[k].mean)} <small>± ${Math.round(c[k].std * 100)}</small>`;
    $("#val-intro").textContent = t("val.intro", { folds: m.folds, children: ds.cv_children, recordings: ds.recordings.toLocaleString(I18N.locale) });
    const list = [
      [c.weighted_accuracy.mean.toFixed(2) + ` <small>± ${c.weighted_accuracy.std.toFixed(2)}</small>`, t("val.card1"), t("val.card1Text")],
      [pm("sensitivity"), t("val.card2"), t("val.card2Text")],
      [pm("specificity"), t("val.card3"), t("val.card3Text")],
      [c.auc.mean.toFixed(2) + ` <small>± ${c.auc.std.toFixed(2)}</small>`, "AUC", t("val.card4Text")],
    ];
    $("#val-cards").innerHTML = list.map(([big, title, text]) =>
      `<div class="card metric"><div class="metric-big">${big}</div><h3>${title}</h3><p class="muted small">${text}</p></div>`).join("");
  }

  function compare() {
    const m = METRICS;
    const bars = [[t("val.barBaseline"), m.baseline.weighted_accuracy.mean, "base"]];
    if (m.comparison) bars.push([t("val.barClean"), m.comparison.cnn.weighted_accuracy.mean, "base2"]);
    bars.push([t("val.barOurs"), m.cnn.weighted_accuracy.mean, "ours"]);
    bars.push([t("val.barTop"), m.challenge_reference.top_hidden_test, "ref"]);
    $("#compare").innerHTML = bars.map(([name, v, cls]) => `
      <div class="cmp-row"><span class="cmp-name">${name}</span>
        <span class="cmp-track"><span class="cmp-bar ${cls}" style="width:${v * 100}%"></span></span>
        <span class="cmp-val">${v.toFixed(2)}</span></div>`).join("");
    $("#compare-note").textContent = t("val.compareNote");
  }

  function confusion() {
    const names = [t("label.0"), t("label.1"), t("label.2")];
    let html = `<div></div>${names.map(n => `<div class="cm-head">${n}</div>`).join("")}`;
    METRICS.confusion.forEach((row, ti) => {
      html += `<div class="cm-side">${names[ti]}</div>`;
      row.forEach((v, p) => { html += `<div class="cm-cell ${ti === p ? "good" : "bad"}">${v}</div>`; });
    });
    $("#confusion").innerHTML = html;
  }

  function grades() {
    const a = METRICS.analysis;
    if (!a) return;
    $("#by-grade").innerHTML = a.by_grade.map(g => {
      const w = (n) => (100 * n / g.n).toFixed(1) + "%";
      return `<div class="stack-row"><span class="stack-name">${t("truth.grade", { g: g.group })} <small>(n=${g.n})</small></span>
        <span class="stack"><span class="seg bad" style="width:${w(g.referred)}"></span><span class="seg warn" style="width:${w(g.rerecord)}"></span><span class="seg miss" style="width:${w(g.missed)}"></span></span>
        <span class="stack-val">${pct(g.referred / g.n)}</span></div>`;
    }).join("");
    const missed = a.by_grade.reduce((s, g) => s + g.missed, 0);
    const faint = (a.by_grade.find(g => g.group === "I/VI") || { missed: 0 }).missed;
    const loud = a.by_grade.find(g => g.group === "III/VI");
    $("#grade-note").textContent = t("val.gradeNote", {
      missed, faint, share: missed ? pct(faint / missed) : "0%",
      loud: loud ? pct(loud.referred / loud.n) : "–",
    });

    $("#by-age").innerHTML = `<thead><tr><th>${t("val.ageCol")}</th><th>${t("val.ageMurmur")}</th><th>${t("val.ageCaught")}</th>
      <th>${t("val.ageNone")}</th><th>${t("val.ageCleared")}</th></tr></thead><tbody>` +
      a.by_age.map(g => `<tr><td>${t("age." + g.group)}</td><td>${g.murmur}</td><td>${pctOrDash(g.sensitivity)}</td>
        <td>${g.no_murmur}</td><td>${pctOrDash(g.specificity)}</td></tr>`).join("") + "</tbody>";
  }

  function robustness() {
    const a = METRICS.analysis;
    if (!a) return;
    const old = METRICS.comparison && Object.fromEntries(METRICS.comparison.robustness.map(r => [r.condition, r]));
    $("#robust-lead").textContent = t(old ? "val.robustLeadCompare" : "val.robustLead");
    const cell = (r, k, fmt) => old ? `<span class="muted">${fmt(old[r.condition][k])} →</span> <b>${fmt(r[k])}</b>` : fmt(r[k]);
    $("#robustness").innerHTML = `<thead><tr><th>${t("val.robCond")}</th><th>${t("val.robScore")}</th><th>${t("val.card2")}</th>
      <th>${t("val.card3")}</th><th>${t("val.robBeat")}</th></tr></thead><tbody>` +
      a.robustness.map(r => `<tr><td>${t("cond." + r.condition)}</td><td>${cell(r, "weighted_accuracy", fmt2)}</td>
        <td>${cell(r, "sensitivity", pct)}</td><td>${cell(r, "specificity", pct)}</td><td>${cell(r, "beat_pass", pct)}</td></tr>`).join("") +
      "</tbody>";
    const phone = a.robustness.find(r => r.condition === "phone");
    const rumble = a.robustness.find(r => r.condition === "rumble");
    $("#robust-note").textContent = old && phone && rumble
      ? t("val.robustNote", { before: pct(old.phone.sensitivity), after: pct(phone.sensitivity),
        rb: fmt2(old.rumble.weighted_accuracy), ra: fmt2(rumble.weighted_accuracy) }) : "";
  }

  async function externalTest() {
    if (!external) {
      try { external = await (await fetch("model/external.json")).json(); } catch { return; }
    }
    const old = METRICS.comparison && METRICS.comparison.external;
    const row = (name, r, o) => `<tr><td>${name}</td><td>${r.recordings}</td><td>${pct(r.abnormal_share)}</td>
      <td>${o ? `<span class="muted">${fmt2(o.auc)} →</span> ` : ""}<b>${fmt2(r.auc)}</b></td>
      <td>${pctOrDash(r.sensitivity)}</td><td>${pctOrDash(r.specificity)}</td><td>${pct(r.beat_pass)}</td></tr>`;
    $("#external").innerHTML = `<thead><tr><th>${t("val.extSet")}</th><th>${t("val.extN")}</th><th>${t("val.extAbn")}</th>
      <th>AUC</th><th>${t("val.extSens")}</th><th>${t("val.extSpec")}</th><th>${t("val.robBeat")}</th></tr></thead><tbody>` +
      Object.entries(external.subsets).map(([k, r]) => row(t("val.extSubset", { k: k.toUpperCase() }), r, old && old.subsets[k])).join("") +
      row(`<b>${t("val.extAll")}</b>`, external.overall, old && old.overall) + "</tbody>";
    $("#external-note").textContent = t("val.externalNote", { auc: fmt2(external.overall.auc), beat: pct(external.overall.beat_pass) });
    limits();
  }

  function safety() {
    const b = LHModel.beatCheck;
    if (!b) return;
    const rows = [[t("val.safetyReal", { n: b.real_recordings.toLocaleString(I18N.locale) }), t("val.accepted", { p: (b.real_pass_rate * 100).toFixed(1) + "%" }), true]];
    Object.entries(b.negatives).forEach(([name, v]) =>
      rows.push([t("neg." + name), t("val.rejected", { k: v.rejected, n: v.trials }), v.rejected === v.trials]));
    $("#safety").innerHTML = rows.map(([name, result, good]) =>
      `<div class="safety-row"><span>${name}</span><b class="${good ? "ok-text" : "warn-text"}">${result}</b></div>`).join("");
  }

  function limits() {
    const c = METRICS.cnn;
    const items = [
      t("val.limit1", { n: Math.max(2, Math.round(1 / Math.max(0.01, 1 - c.sensitivity.mean))) }),
      t("val.limit2"),
      external ? t("val.limit3", { auc: fmt2(external.overall.auc) }) : null,
      t("val.limit4"),
      t("val.limit5"),
      t("val.limit6"),
    ].filter(Boolean);
    $("#limits").innerHTML = items.map(x => `<li>${x}</li>`).join("");
  }

  function render() {
    cards(); compare(); confusion(); grades(); robustness(); safety(); limits();
    externalTest();
  }

  return { render, relabel: render, safety };
})();
