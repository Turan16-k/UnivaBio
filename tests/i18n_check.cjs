// Checks that every text key used by the page and the scripts exists in every
// language, with the same {placeholders}. Prints problems; exit code 1 if any.
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const web = path.join(__dirname, "..", "web");
const ctx = vm.createContext({ localStorage: undefined, navigator: { language: "en" } });
vm.runInContext(fs.readFileSync(path.join(web, "js", "i18n.js"), "utf8") + "\nthis.I18N = I18N;", ctx);
const { EN, tr, pt, es } = ctx.I18N.keys;
const langs = { tr, pt, es };
const problems = [];

const html = fs.readFileSync(path.join(web, "index.html"), "utf8");
const htmlKeys = [...html.matchAll(/data-i18n(?:-html|-placeholder)?="([^"]+)"/g)].map(m => m[1]);

const jsKeys = new Set();
for (const f of fs.readdirSync(path.join(web, "js"))) {
  if (f === "i18n.js") continue;
  const src = fs.readFileSync(path.join(web, "js", f), "utf8");
  for (const m of src.matchAll(/\bt\("([^"]+)"/g)) if (!m[1].endsWith(".")) jsKeys.add(m[1]);
}
const families = {
  "site.": ["AV", "PV", "TV", "MV", "Phc"], "label.": ["0", "1", "2"],
  "age.": ["Neonate", "Infant", "Child", "Adolescent", "nan"], "sex.": ["Female", "Male"],
  "timing.": ["Early-systolic", "Mid-systolic", "Late-systolic", "Holosystolic"],
  "quiz.pitch.": ["Low", "Medium", "High"], "quiz.quality.": ["Blowing", "Harsh", "Musical"],
  "cond.": ["clean", "noise20", "noise10", "noise5", "phone", "rumble"],
  "neg.": ["white noise", "pink-ish noise", "steady tone 440 Hz", "mains hum 50 Hz", "speech-like babble", "silence with hiss"],
  "guide.place.": ["AV", "PV", "TV", "MV"], "log.status": ["0", "1", "2"],
};
for (const [prefix, ends] of Object.entries(families)) ends.forEach(e => jsKeys.add(prefix + e));

for (const key of jsKeys) {
  if (!(key in EN)) problems.push(`EN missing script key: ${key}`);
}
for (const [name, dict] of Object.entries(langs)) {
  for (const key of htmlKeys) if (!(key in dict)) problems.push(`${name} missing page key: ${key}`);
  for (const key of jsKeys) if (!(key in dict)) problems.push(`${name} missing script key: ${key}`);
  for (const [key, text] of Object.entries(EN)) {
    if (!(key in dict)) continue;
    const vars = (s) => [...s.matchAll(/\{(\w+)\}/g)].map(m => m[1]).sort().join(",");
    if (vars(text) !== vars(dict[key])) problems.push(`${name} placeholders differ for ${key}`);
  }
}

console.log(problems.length ? problems.join("\n") : `ok: ${htmlKeys.length} page keys, ${jsKeys.size} script keys, 4 languages`);
process.exit(problems.length ? 1 : 0);
