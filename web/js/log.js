// Screening log for a school or village screening day. Stored only in this
// browser (localStorage); nothing is sent anywhere.

const Log = (() => {
  const KEY = "littleheart.log.v1";
  let filter = "all";

  function load() {
    try { return JSON.parse(localStorage.getItem(KEY)) || []; } catch { return []; }
  }

  function store(entries) {
    try { localStorage.setItem(KEY, JSON.stringify(entries)); return true; } catch { return false; }
  }

  function saveCurrent() {
    const d = App.decision();
    if (!d) return;
    const input = $("#child-id");
    const entries = load();
    entries.unshift({
      id: Date.now().toString(36),
      child: input.value.trim() || t("log.unnamed", { n: entries.length + 1 }),
      date: new Date().toISOString(),
      label: d.label,
      present: Math.round(d.present * 100) / 100,
      sites: d.sites.map(s => ({ site: s.site, label: s.label })),
      demo: App.isDemo,
    });
    const btn = $("#save-btn");
    btn.textContent = store(entries) ? t("log.saved") : t("log.saveFailed");
    setTimeout(() => { btn.textContent = t("log.save"); }, 1800);
    input.value = "";
    render();
  }

  const labelText = (label) => t(["log.status0", "log.status1", "log.status2"][label]);

  function render() {
    const entries = load();
    const count = (l) => entries.filter(e => e.label === l).length;
    const stats = [
      [entries.length, t("log.statScreened"), ""],
      [count(1), t("log.statRefer"), "bad"],
      [count(2), t("log.statRerecord"), "warn"],
      [count(0), t("log.statClear"), "ok"],
    ];
    $("#log-stats").innerHTML = stats.map(([n, label, cls]) =>
      `<div class="card metric"><div class="metric-big ${cls}-num">${n}</div><h3>${label}</h3></div>`).join("");

    const shown = entries.filter(e => filter === "all" || String(e.label) === filter);
    const table = $("#log-table");
    if (!shown.length) {
      table.innerHTML = `<tr><td class="muted">${entries.length ? t("log.noneFilter") : t("log.empty")}</td></tr>`;
      return;
    }
    table.innerHTML = `
      <thead><tr><th>${t("log.colChild")}</th><th>${t("log.colDate")}</th><th>${t("log.colResult")}</th>
        <th>${t("log.colSites")}</th><th></th></tr></thead>
      <tbody>${shown.map(e => `
        <tr>
          <td>${escapeHtml(e.child)}${e.demo ? ` <span class="badge">${t("log.demo")}</span>` : ""}</td>
          <td>${new Date(e.date).toLocaleString(I18N.locale, { dateStyle: "short", timeStyle: "short" })}</td>
          <td><span class="badge ${LABEL_CLASS[e.label]}">${labelText(e.label)}</span></td>
          <td class="small">${e.sites.map(s => `${s.site}: ${t("label." + s.label)}`).join(", ")}</td>
          <td><button class="link-btn" data-delete="${e.id}" aria-label="${t("log.delete")}">✕</button></td>
        </tr>`).join("")}</tbody>`;
    table.querySelectorAll("[data-delete]").forEach(b => b.addEventListener("click", () => {
      store(load().filter(e => e.id !== b.dataset.delete));
      render();
    }));
  }

  function escapeHtml(s) {
    return s.replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  function exportCsv() {
    const rows = [["child", "date", "result", "murmur_likelihood", "sites", "demo_case"]];
    load().forEach(e => rows.push([e.child, e.date, ["no murmur", "refer", "record again"][e.label], e.present,
      e.sites.map(s => `${s.site}:${["no murmur", "murmur", "unclear"][s.label]}`).join(" "), e.demo ? "yes" : "no"]));
    const csv = rows.map(r => r.map(v => `"${String(v).replace(/"/g, '""')}"`).join(",")).join("\r\n");
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob(["﻿" + csv], { type: "text/csv" }));
    a.download = `littleheart-screening-${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }

  function init() {
    $("#save-btn").addEventListener("click", saveCurrent);
    $("#log-export").addEventListener("click", exportCsv);
    $("#log-wipe").addEventListener("click", () => {
      if (confirm(t("log.confirmWipe"))) { store([]); render(); }
    });
    document.querySelectorAll("[data-filter]").forEach(b => b.addEventListener("click", () => {
      filter = b.dataset.filter;
      document.querySelectorAll("[data-filter]").forEach(x => x.classList.toggle("active", x === b));
      render();
    }));
    render();
  }

  return { init, relabel: render };
})();
