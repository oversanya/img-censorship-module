"""Простой веб-демо для Censor Module.

Запуск:
    uvicorn ui_demo:app --reload
Открыть http://127.0.0.1:8000 — кинуть картинку + промпт, получить вердикт.

Это намеренно один самодостаточный файл (HTML/JS встроены), чтобы демо было
легко запустить и читать. Боевые точки входа — censor_guard.app / ui.app.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from censor_guard.pipeline import GuardrailPipeline
from censor_guard.schemas import ModerationRequest, ModerationResponse

app = FastAPI(title="Censor Module — UI Demo")
pipeline = GuardrailPipeline()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/moderate", response_model=ModerationResponse)
def moderate(request: ModerationRequest) -> ModerationResponse:
    return pipeline.moderate(request)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return PAGE


PAGE = r"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Censor Module — Demo</title>
<style>
  :root { --bg:#0f1117; --card:#1a1d27; --line:#2a2f3d; --txt:#e6e8ee; --muted:#9aa3b2;
          --allow:#34d399; --review:#fbbf24; --block:#f87171; --accent:#6366f1; }
  * { box-sizing:border-box; }
  body { margin:0; font:15px/1.5 system-ui,Segoe UI,Roboto,sans-serif; background:var(--bg); color:var(--txt); }
  .wrap { max-width:1080px; margin:0 auto; padding:24px; }
  h1 { font-size:20px; margin:0 0 4px; }
  .sub { color:var(--muted); margin:0 0 20px; font-size:13px; }
  .grid { display:grid; grid-template-columns:1fr 1fr; gap:20px; }
  @media (max-width:860px){ .grid{ grid-template-columns:1fr; } }
  .card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:18px; }
  label { display:block; font-size:12px; color:var(--muted); margin:12px 0 5px; text-transform:uppercase; letter-spacing:.04em; }
  textarea, select { width:100%; background:#11141c; color:var(--txt); border:1px solid var(--line);
                     border-radius:8px; padding:9px 11px; font:inherit; }
  textarea { min-height:80px; resize:vertical; }
  .row { display:flex; gap:12px; }
  .row > div { flex:1; }
  #drop { margin-top:6px; border:1.5px dashed var(--line); border-radius:10px; padding:20px; text-align:center;
          color:var(--muted); cursor:pointer; transition:.15s; }
  #drop.over { border-color:var(--accent); background:#171a24; color:var(--txt); }
  #preview { max-width:100%; max-height:230px; border-radius:8px; margin-top:10px; display:none; }
  button { margin-top:16px; width:100%; padding:11px; border:0; border-radius:8px; cursor:pointer;
           font:600 15px/1 inherit; background:var(--accent); color:#fff; }
  button:disabled { opacity:.5; cursor:not-allowed; }
  .verdict { display:inline-flex; align-items:center; gap:8px; font-weight:700; font-size:18px;
             padding:7px 14px; border-radius:999px; }
  .v-allow { background:rgba(52,211,153,.15); color:var(--allow); }
  .v-review{ background:rgba(251,191,36,.15); color:var(--review); }
  .v-block { background:rgba(248,113,113,.15); color:var(--block); }
  .meta { color:var(--muted); font-size:13px; margin:10px 0; }
  .bar { height:8px; background:#11141c; border-radius:999px; overflow:hidden; margin:6px 0 2px; }
  .bar > i { display:block; height:100%; background:var(--accent); }
  .chip { display:inline-block; background:#11141c; border:1px solid var(--line); border-radius:999px;
          padding:3px 10px; margin:3px 4px 0 0; font-size:12px; }
  table { width:100%; border-collapse:collapse; font-size:13px; margin-top:8px; }
  th,td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--line); vertical-align:top; }
  th { color:var(--muted); font-weight:600; font-size:11px; text-transform:uppercase; }
  .st-ok{ color:var(--allow);} .st-skipped{ color:var(--muted);} .st-error{ color:var(--block);}
  .empty { color:var(--muted); text-align:center; padding:40px 0; }
  details { margin-top:14px; } summary { cursor:pointer; color:var(--muted); font-size:13px; }
  code { background:#11141c; padding:1px 5px; border-radius:5px; font-size:12px; }
  .scores { font-size:12px; color:var(--muted); }
</style>
</head>
<body>
<div class="wrap">
  <h1>Censor Module — Demo</h1>
  <p class="sub">Загрузите изображение и/или введите промпт → получите вердикт пайплайна модерации.</p>
  <div class="grid">
    <div class="card">
      <form id="form">
        <label>Промпт (необязательно)</label>
        <textarea id="prompt" placeholder="Текст запроса к генератору..."></textarea>
        <div class="row">
          <div>
            <label>Scenario</label>
            <select id="scenario">
              <option value="text2image">text2image</option>
              <option value="img2img_stylization">img2img_stylization</option>
              <option value="img2img_editing">img2img_editing</option>
              <option value="output" selected>output</option>
            </select>
          </div>
          <div>
            <label>Stage</label>
            <select id="stage">
              <option value="input">input</option>
              <option value="output" selected>output</option>
            </select>
          </div>
        </div>
        <label>Изображение (необязательно)</label>
        <div id="drop">Перетащите картинку сюда или кликните для выбора</div>
        <input id="file" type="file" accept="image/*" hidden/>
        <img id="preview" alt="preview"/>
        <button id="submit" type="submit">Проверить</button>
      </form>
    </div>

    <div class="card" id="result">
      <div class="empty" id="empty">Результат появится здесь</div>
      <div id="out" style="display:none">
        <span id="verdict" class="verdict"></span>
        <div class="meta" id="reason"></div>
        <label>Confidence</label>
        <div class="bar"><i id="confBar"></i></div>
        <div class="scores" id="confVal"></div>
        <div id="catsWrap" style="display:none">
          <label>Сработавшие категории</label>
          <div id="cats"></div>
        </div>
        <details open>
          <summary>Сигналы сенсоров</summary>
          <table id="signals"><thead><tr><th>Сенсор</th><th>Статус</th><th>Категории</th></tr></thead><tbody></tbody></table>
        </details>
        <details id="fusionBox">
          <summary>Сведение (policy_fusion) — вклад сенсоров</summary>
          <div id="fusion"></div>
        </details>
        <details>
          <summary>Сырой JSON</summary>
          <pre><code id="raw"></code></pre>
        </details>
      </div>
    </div>
  </div>
</div>

<script>
const $ = (s) => document.querySelector(s);
let imageBase64 = "";

const drop = $("#drop"), file = $("#file"), preview = $("#preview");
drop.onclick = () => file.click();
["dragover","dragenter"].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.add("over"); }));
["dragleave","drop"].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", ev => { if (ev.dataTransfer.files[0]) loadFile(ev.dataTransfer.files[0]); });
file.addEventListener("change", () => { if (file.files[0]) loadFile(file.files[0]); });

function loadFile(f) {
  const reader = new FileReader();
  reader.onload = () => {
    imageBase64 = reader.result.split(",")[1] || "";
    preview.src = reader.result;
    preview.style.display = "block";
    drop.textContent = f.name;
  };
  reader.readAsDataURL(f);
}

const VCOLOR = { allow:"--allow", review:"--review", block:"--block" };

const SIGNAL_LABELS = {
  text_guard_heuristic: "Текстовый фильтр промпта",
  ocr_text_guard_heuristic: "Текстовый фильтр OCR-текста",
  ocr_adapter: "OCR (текст на картинке)",
  visual_classifier: "Визуальный классификатор (zero-shot)",
  explicit_content_detector: "Детектор NSFW",
  policy_fusion: "Сведение сигналов (fusion)",
  policy_judge_shieldgemma: "Судья ShieldGemma (эскалация)",
};
const sigLabel = (name) => SIGNAL_LABELS[name] || name;

$("#form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const btn = $("#submit"); btn.disabled = true; btn.textContent = "Проверяю...";
  const payload = {
    scenario: $("#scenario").value,
    stage: $("#stage").value,
    prompt: $("#prompt").value || null,
  };
  if (imageBase64) payload.image_base64 = imageBase64;
  try {
    const res = await fetch("/api/moderate", {
      method: "POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (!res.ok) throw new Error(JSON.stringify(data));
    render(data);
  } catch (err) {
    $("#empty").style.display = "block"; $("#out").style.display = "none";
    $("#empty").textContent = "Ошибка: " + err.message;
  } finally {
    btn.disabled = false; btn.textContent = "Проверить";
  }
});

function render(d) {
  $("#empty").style.display = "none";
  $("#out").style.display = "block";

  const v = $("#verdict");
  v.textContent = d.verdict.toUpperCase();
  v.className = "verdict v-" + d.verdict;
  $("#reason").textContent = d.reason;

  const pct = Math.round((d.confidence || 0) * 100);
  $("#confBar").style.width = pct + "%";
  $("#confBar").style.background = "var(" + (VCOLOR[d.verdict]||"--accent") + ")";
  $("#confVal").textContent = (d.confidence ?? 0).toFixed(4);

  const catsWrap = $("#catsWrap"), cats = $("#cats");
  if (d.categories && d.categories.length) {
    catsWrap.style.display = "block";
    cats.innerHTML = d.categories.map(c => {
      const ev = (d.evidence && d.evidence[c]) ? " ← " + d.evidence[c].map(sigLabel).join(", ") : "";
      return `<span class="chip">${c}${ev}</span>`;
    }).join("");
  } else { catsWrap.style.display = "none"; }

  const tb = $("#signals tbody");
  tb.innerHTML = (d.signals||[]).map(s => {
    const catStr = Object.keys(s.categories||{}).length
      ? Object.entries(s.categories).map(([k,val]) => `${k}: ${val.toFixed(3)}`).join("<br>")
      : (s.reason || "—");
    return `<tr><td title="${s.name}">${sigLabel(s.name)}</td><td class="st-${s.status}">${s.status}</td><td class="scores">${catStr}</td></tr>`;
  }).join("");

  // policy_fusion contributions
  const fusion = (d.signals||[]).find(s => s.name === "policy_fusion");
  const fbox = $("#fusion"), fusionBox = $("#fusionBox");
  const contrib = fusion && fusion.raw && fusion.raw.contributions;
  if (contrib && Object.keys(contrib).length) {
    fusionBox.style.display = "block";
    fbox.innerHTML = Object.entries(contrib).map(([code, list]) => {
      const score = fusion.categories[code] ?? 0;
      const rows = list.map(c => `<tr><td title="${c.sensor}">${sigLabel(c.sensor)}</td><td>${c.score}</td><td>${c.weight}</td></tr>`).join("");
      const agree = (fusion.raw.agreement && fusion.raw.agreement[code]) || 0;
      return `<div style="margin:8px 0"><b>${code}</b> = ${score.toFixed(3)} <span class="scores">(согласие: ${agree})</span>
        <table><thead><tr><th>сенсор</th><th>p</th><th>вес</th></tr></thead><tbody>${rows}</tbody></table></div>`;
    }).join("");
    const esc = fusion.raw.escalation;
    if (esc) fbox.innerHTML += `<div class="scores" style="margin-top:8px">эскалация: needed=${esc.needed}, attempted=${esc.attempted||false}, shieldgemma=${esc.shieldgemma_status||"—"}</div>`;
  } else { fusionBox.style.display = "none"; }

  $("#raw").textContent = JSON.stringify(d, null, 2);
}
</script>
</body>
</html>
"""
