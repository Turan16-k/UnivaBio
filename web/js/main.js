// Starts the app: tabs, language, modules, model loading and offline support.

(function main() {
  function showTab(name) {
    document.querySelectorAll(".tab").forEach(tb => tb.classList.toggle("active", tb.dataset.tab === name));
    document.querySelectorAll(".panel").forEach(p => p.classList.toggle("active", p.id === "tab-" + name));
  }

  document.querySelectorAll(".tab").forEach(tab => tab.addEventListener("click", () => {
    showTab(tab.dataset.tab);
    history.replaceState(null, "", location.search + "#" + tab.dataset.tab);
  }));
  const fromUrl = location.hash.slice(1);
  if (document.getElementById("tab-" + fromUrl)) showTab(fromUrl);

  document.querySelectorAll(".source").forEach(btn => btn.addEventListener("click", () => {
    document.querySelectorAll(".source").forEach(b => b.classList.toggle("active", b === btn));
    document.querySelectorAll(".source-panel").forEach(p =>
      p.classList.toggle("active", p.id === "source-" + btn.dataset.source));
  }));

  const langSelect = $("#lang");
  langSelect.value = I18N.lang;
  langSelect.addEventListener("change", () => I18N.setLang(langSelect.value));
  I18N.apply();

  App.init();
  Guided.init();
  Quiz.init();
  Log.init();
  Validation.render();

  document.addEventListener("langchange", () => {
    App.relabel(); Guided.relabel(); Quiz.relabel(); Log.relabel(); Validation.relabel();
  });

  LHModel.load().then(() => Validation.safety()).catch(err => {
    console.error(err);
    $("#verdict").innerHTML = `<div class="verdict-icon">!</div><div><div class="verdict-title">${t("load.failTitle")}</div>
      <div class="verdict-text">${t("load.failText")}</div></div>`;
  });

  // Offline support: cache the app, model and recordings after the first visit.
  if ("serviceWorker" in navigator && (location.protocol === "https:" || location.hostname === "localhost")) {
    navigator.serviceWorker.register("sw.js").then(() => navigator.serviceWorker.ready).then(() => {
      const el = $("#offline-status");
      el.hidden = false;
      el.textContent = t("offline.ready");
      el.dataset.i18n = "offline.ready";
    }).catch(err => console.warn("Offline mode unavailable", err));
  }
})();
