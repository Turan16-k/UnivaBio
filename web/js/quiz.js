// "Train your ear": the health worker listens to real recordings, decides
// murmur / no murmur, then sees the cardiologist's answer, the AI's answer
// (from a cross-validation model that never heard that child) and the spectrogram.

const Quiz = (() => {
  let index = 0;
  let answers = [];   // { you: bool, ai: bool } correctness per item
  let revealed = false;
  let audioUrl = null;

  const items = () => METRICS.quiz || [];

  function describe(item) {
    if (!item.murmur) return t("quiz.truthNone");
    const parts = [t("quiz.truthMurmur")];
    if (item.grade !== "nan") parts.push(t("truth.grade", { g: item.grade }));
    if (item.timing !== "nan") parts.push(t("timing." + item.timing));
    if (item.pitch !== "nan") parts.push(t("quiz.pitch." + item.pitch));
    if (item.quality !== "nan") parts.push(t("quiz.quality." + item.quality));
    return parts.join(" · ");
  }

  async function loadAudio(item) {
    const buf = await (await fetch("quiz/" + item.file)).arrayBuffer();
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    audioUrl = URL.createObjectURL(new Blob([buf]));
    return buf;
  }

  async function render() {
    const card = $("#quiz-card");
    const list = items();
    renderScore();
    if (!list.length) { card.innerHTML = `<p>${t("quiz.none")}</p>`; return; }
    if (index >= list.length) return renderEnd(card);

    const item = list[index];
    revealed = false;
    card.innerHTML = `
      <div class="quiz-head">
        <span class="muted small">${t("quiz.progress", { n: index + 1, total: list.length })}</span>
        <span class="badge">${t("site." + item.location)}</span>
      </div>
      <div class="quiz-listen">
        <button class="play big" aria-label="${t("card.play")}">▶</button>
        <div>
          <div class="quiz-question">${t("quiz.question")}</div>
          <div class="muted small">${t("quiz.hint")}</div>
        </div>
      </div>
      <div class="quiz-answers">
        <button class="btn answer" data-answer="1">${t("quiz.yes")}</button>
        <button class="btn answer secondary" data-answer="0">${t("quiz.no")}</button>
      </div>
      <div class="quiz-reveal" hidden></div>
      <audio preload="auto"></audio>`;
    const buf = await loadAudio(item);
    if (items()[index] !== item) return;
    const audio = card.querySelector("audio");
    audio.src = audioUrl;
    const play = card.querySelector(".play");
    play.addEventListener("click", () => (audio.paused ? audio.play() : audio.pause()));
    audio.addEventListener("play", () => { play.textContent = "❚❚"; });
    audio.addEventListener("pause", () => { play.textContent = "▶"; });
    card.querySelectorAll(".answer").forEach(b => b.addEventListener("click", () => answer(item, b.dataset.answer === "1", buf)));
  }

  function answer(item, saysMurmur, buf) {
    if (revealed) return;
    revealed = true;
    const you = saysMurmur === item.murmur;
    const ai = item.ai_flags === item.murmur;
    answers[index] = { you, ai };
    const card = $("#quiz-card");
    card.querySelectorAll(".answer").forEach(b => {
      b.disabled = true;
      if ((b.dataset.answer === "1") === saysMurmur) b.classList.add(you ? "right" : "wrong");
    });
    const box = card.querySelector(".quiz-reveal");
    box.hidden = false;
    box.innerHTML = `
      <div class="reveal-verdict ${you ? "ok" : "bad"}">${you ? t("quiz.right") : t("quiz.wrong")}</div>
      <p><b>${t("quiz.cardiologist")}</b> ${describe(item)}</p>
      <p><b>${t("quiz.ai")}</b> ${t(item.ai_flags ? "quiz.aiYes" : "quiz.aiNo", { p: pct(item.ai_present_prob) })}
        <span class="${ai ? "ok-text" : "bad-text"}">${ai ? "✓" : "✗"}</span></p>
      <div class="spec-wrap"><canvas class="spec"></canvas>
        <span class="axis-label top">1000 Hz</span><span class="axis-label bottom">25 Hz</span></div>
      <p class="muted small">${t(item.murmur ? "quiz.lookMurmur" : "quiz.lookClear")}</p>
      <button class="btn" id="quiz-next">${index + 1 < items().length ? t("quiz.next") : t("quiz.finish")}</button>`;
    const wav = LHFeatures.parseWav(buf);
    if (wav && LHModel.config) Draw.spectrogram(box.querySelector(".spec"), LHFeatures.logMel(wav.samples, LHModel.config.features));
    box.querySelector("#quiz-next").addEventListener("click", () => { index += 1; render(); });
    renderScore();
  }

  function renderScore() {
    const done = answers.filter(Boolean);
    const you = done.filter(a => a.you).length, ai = done.filter(a => a.ai).length;
    $("#scoreboard").innerHTML = `
      <div class="score-row"><span>${t("quiz.you")}</span><b>${you} / ${done.length}</b></div>
      <div class="score-row"><span>${t("quiz.aiName")}</span><b>${ai} / ${done.length}</b></div>
      <p class="muted small">${t("quiz.scoreNote")}</p>`;
  }

  function renderEnd(card) {
    const you = answers.filter(a => a && a.you).length, ai = answers.filter(a => a && a.ai).length;
    card.innerHTML = `
      <h2>${t("quiz.endTitle")}</h2>
      <p>${t("quiz.endText", { you, ai, total: items().length })}</p>
      <p class="muted small">${t("quiz.endNote")}</p>
      <button class="btn" id="quiz-again">${t("quiz.again")}</button>`;
    card.querySelector("#quiz-again").addEventListener("click", () => { index = 0; answers = []; render(); });
  }

  return { init: render, relabel: () => { if (!revealed) render(); else renderScore(); } };
})();
