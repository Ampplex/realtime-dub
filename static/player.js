/*
 * Sync model
 * ----------
 * The video clock is the single source of truth; dubbed audio bends to it, never
 * the other way round. When playback starts we capture an anchor pairing
 * AudioContext time with video time. Every dub segment then plays at
 *     anchor.ctx + (segment.start - anchor.video)
 * so a segment is pinned to its position on the SOURCE timeline. Drift cannot
 * accumulate across segments: each one is scheduled against the anchor
 * independently, not relative to the segment before it.
 *
 * Pause / seek / language change invalidate the anchor, so we stop every pending
 * source and re-derive it.
 */
const $ = (id) => document.getElementById(id);

const BUFFER = window.BUFFER_SECONDS || 8;
const LOOKAHEAD = 12;          // schedule segments this far ahead of the playhead

let session = null;            // {id, video_url, tts, translator}
let lang = "ko";               // active audio track
let ctx = null;                // AudioContext
let mediaSrc = null;           // MediaElementSource for the <video> (create once only)
let bedGain = null;            // original-audio bed: music/effects survive under the dub
let dubGain = null;            // dubbed speech bus (current epoch only)
let anchor = null;             // {ctx, video}

// Background bed levels. We cannot separate dialogue from music without a stem
// model (demucs needs ~2GB and runs slower than realtime here), so instead of
// muting the source we duck it under each dubbed line. Original dialogue stays
// faintly audible - voice-over style - but music and effects are preserved.
let BED_LEVEL = 0.55;          // between dubbed lines
const DUCK_RATIO = 0.28;       // fraction of BED_LEVEL while a line is speaking
const RAMP = 0.12;             // seconds, smooths the dip so it does not click
const scheduled = new Map();   // idx -> {src, gain, seg}
const bedScheduled = new Map();// bed chunk idx -> source

// Bumped on every switch/seek/teardown. Any scheduling work that was in flight
// across a bump is stale and must not start: fetch+decode is async, so a chunk
// requested before a language change can otherwise resolve afterwards and play
// the OLD language over the new one.
let epoch = 0;
const decoded = new Map();     // "lang:idx" -> AudioBuffer
let manifest = { segments: [], ready_to: 0, total: 0, stage: "", complete: false };
let poller = null;
let switching = false;

const video = $("video");

// ------------------------------------------------------------------ loading

async function createSession(body, isForm) {
  $("load-err").hidden = true;
  $("load-btn").disabled = true;
  $("load-btn").textContent = "Loading…";
  try {
    const res = await fetch("/api/session", isForm ? { method: "POST", body }
      : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Could not start a session");
    session = data;
    video.src = data.video_url;
    $("stage").hidden = false;
    $("load-card").hidden = true;
    startPolling();
  } catch (e) {
    $("load-err").textContent = String(e.message || e);
    $("load-err").hidden = false;
  } finally {
    $("load-btn").disabled = false;
    $("load-btn").textContent = "Load";
  }
}

if ($("load-btn")) {
  $("load-btn").onclick = () => {
    const p = $("path").value.trim();
    if (p) createSession({ path: p }, false);
  };
}

// click-to-browse upload
// Chrome restores a file input across a reload and fires `change`, which would
// re-upload and re-dub the previous video unasked. Start every load from empty.
$("file").value = "";
$("pick-btn").onclick = () => $("file").click();
$("file").onchange = () => {
  const f = $("file").files[0];
  if (!f) return;
  $("upname").textContent = `uploading ${f.name} (${(f.size / 1048576).toFixed(1)} MB)…`;
  $("upname").hidden = false;
  const fd = new FormData();
  fd.append("file", f);
  fd.append("source_lang", "ko");
  createSession(fd, true);
};

// drag + drop
const drop = $("drop");
let dragDepth = 0;
addEventListener("dragenter", (e) => { e.preventDefault(); if (++dragDepth) drop.classList.add("on"); });
addEventListener("dragleave", (e) => { e.preventDefault(); if (--dragDepth <= 0) drop.classList.remove("on"); });
addEventListener("dragover", (e) => e.preventDefault());
addEventListener("drop", (e) => {
  e.preventDefault(); dragDepth = 0; drop.classList.remove("on");
  const f = e.dataTransfer.files[0];
  if (!f) return;
  const fd = new FormData(); fd.append("file", f); fd.append("source_lang", "ko");
  createSession(fd, true);
});

// ------------------------------------------------------------------ polling

function startPolling() {
  if (poller) clearInterval(poller);
  poller = setInterval(refresh, 700);
  refresh();
}

async function refresh() {
  if (!session) return;
  const q = lang === "ko" ? "hi" : lang;      // still track progress while on original
  try {
    manifest = await (await fetch(`/api/session/${session.id}/manifest?lang=${q}`)).json();
  } catch { return; }
  render();
  if (lang !== "ko" && !video.paused) { scheduleAhead(); scheduleBed(); }
}

function render() {
  const total = manifest.total || 0;
  const ready = manifest.ready_to || 0;
  const pct = total ? Math.min(100, (ready / total) * 100) : 0;
  $("buf").style.width = pct + "%";

  for (const b of document.querySelectorAll(".lang")) {
    const l = b.dataset.lang;
    b.classList.toggle("active", l === lang);
    // Never disable: a slow TTS backend must not become a lockout. Switching is
    // always permitted; if that language has not been voiced up to the current
    // position yet, the bed keeps playing and dubbed lines join as they arrive.
    b.disabled = false;
    const em = b.querySelector("em");
    if (l !== "ko") {
      const secs = (manifest.ready_all || {})[l] ?? 0;
      em.textContent = manifest.complete ? "dubbed"
        : (secs >= video.currentTime + 1 ? "dubbed" : `${secs.toFixed(0)}s ready`);
    }
  }

  $("bedbar").hidden = lang === "ko";

  if (manifest.error) {
    $("status").textContent = "Error: " + manifest.error;
  } else if (manifest.stage === "complete") {
    $("status").textContent = `Dub complete · ${manifest.segments.length} segments · `
      + `${session.translator} + ${session.tts}`;
  } else {
    $("status").textContent = `${manifest.stage} · dubbed ${ready.toFixed(1)}s`
      + (total ? ` / ${total.toFixed(1)}s` : "")
      + ` · need ${BUFFER}s to start · ${session.translator} + ${session.tts}`;
  }

  const q = lang === "ko" ? "hi" : lang;
  $("diag").textContent = manifest.segments.slice(-14).map((s) =>
    `[${s.start.toFixed(2)}-${s.end.toFixed(2)}] slot ${(s.end - s.start).toFixed(2)}s `
    + `dub ${s.dur.toFixed(2)}s x${s.tempo.toFixed(2)}\n   ko: ${s.source_text}\n   ${q}: ${s.text}`
  ).join("\n");
}

// Only governs whether we auto-start playback; it never gates the UI.
function readyToPlay(l) {
  if (manifest.complete) return true;
  const target = (l && l !== "ko") ? l : (lang === "ko" ? "hi" : lang);
  return ((manifest.ready_all || {})[target] ?? 0) >= BUFFER;
}

// ------------------------------------------------------- audio scheduling

function ensureCtx() {
  if (!ctx) {
    ctx = new (window.AudioContext || window.webkitAudioContext)();
    bedGain = ctx.createGain();
    bedGain.connect(ctx.destination);
    dubGain = ctx.createGain();
    dubGain.gain.value = 1;
    dubGain.connect(ctx.destination);
    // Routing the element through Web Audio lets us duck it; muting would kill
    // the music and effects along with the dialogue.
    mediaSrc = ctx.createMediaElementSource(video);
    mediaSrc.connect(bedGain);
    bedGain.gain.value = 1.0;
  }
  if (ctx.state === "suspended") ctx.resume();
  return ctx;
}

function bedTarget() {
  if (lang === "ko") return 1.0;
  return separated() ? BED_LEVEL : BED_LEVEL;   // separated bed needs no duck
}

function resetBed(now) {
  if (!bedGain) return;
  bedGain.gain.cancelScheduledValues(now);
  bedGain.gain.setTargetAtTime(bedTarget(), now, RAMP);
}

function newDubBus() {
  // A fresh bus per epoch. Disconnecting the old one guarantees silence from ANY
  // source still attached to it - including buffers whose fetch resolved after the
  // switch and were therefore never in `scheduled` to be stopped.
  if (!ctx) return null;
  if (dubGain) { try { dubGain.disconnect(); } catch {} }
  dubGain = ctx.createGain();
  dubGain.gain.value = 1;
  dubGain.connect(ctx.destination);
  return dubGain;
}

function clearScheduled() {
  epoch++;
  for (const { src } of scheduled.values()) { try { src.stop(); } catch {} }
  scheduled.clear();
  for (const src of bedScheduled.values()) { try { src.stop(); } catch {} }
  bedScheduled.clear();
  newDubBus();          // orphan the old bus; stale audio can no longer reach output
  if (ctx && bedGain) resetBed(ctx.currentTime);
}

// With separation on, the bed is a real voice-free track: play it at full level
// and keep the video element muted entirely. No ducking, no bleed-through.
function separated() { return manifest.separated && (manifest.beds || []).length > 0; }

async function scheduleBed() {
  if (!separated() || lang === "ko" || !anchor || video.paused) return;
  const myEpoch = epoch;
  const now = video.currentTime;
  for (const b of manifest.beds) {
    if (bedScheduled.has(b.idx)) continue;
    if (b.end < now - 0.25) continue;
    if (b.start > now + LOOKAHEAD) break;
    let buf;
    try {
      const res = await fetch(`/api/session/${session.id}/bed/${b.idx}`);
      if (!res.ok) continue;
      buf = await ensureCtx().decodeAudioData(await res.arrayBuffer());
    } catch { continue; }
    if (myEpoch !== epoch || lang === "ko" || !anchor || video.paused) return;

    const when = anchor.ctx + (b.start - anchor.video);
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(bedGain);
    if (when >= ctx.currentTime) src.start(when);
    else {
      const off = ctx.currentTime - when;
      if (off < buf.duration - 0.05) src.start(ctx.currentTime, off);
      else continue;
    }
    bedScheduled.set(b.idx, src);
  }
}

function setAnchor() {
  anchor = { ctx: ensureCtx().currentTime, video: video.currentTime };
}

async function bufferFor(seg) {
  const key = `${lang}:${seg.idx}`;
  if (decoded.has(key)) return decoded.get(key);
  const res = await fetch(`/api/session/${session.id}/chunk/${lang}/${seg.idx}`);
  if (!res.ok) throw new Error("chunk fetch failed");
  const buf = await ensureCtx().decodeAudioData(await res.arrayBuffer());
  decoded.set(key, buf);
  return buf;
}

async function scheduleAhead() {
  if (lang === "ko" || !anchor || video.paused) return;
  const myEpoch = epoch;
  const myLang = lang;
  const myBus = dubGain || newDubBus();
  const now = video.currentTime;
  for (const seg of manifest.segments) {
    if (scheduled.has(seg.idx)) continue;
    if (seg.end < now - 0.25) continue;              // already gone by
    if (seg.start > now + LOOKAHEAD) break;          // too far out; next poll will get it

    let buf;
    try { buf = await bufferFor(seg); } catch { continue; }
    // Anything that changed the audio timeline while we were fetching invalidates
    // this buffer. Starting it now would layer the previous language on top.
    if (myEpoch !== epoch || myLang !== lang || !anchor || video.paused) return;

    const when = anchor.ctx + (seg.start - anchor.video);
    const src = ensureCtx().createBufferSource();
    const gain = ctx.createGain();
    src.buffer = buf;
    src.connect(gain).connect(myBus);

    // Ducking is only needed when the source voice is still present. With a
    // separated bed there is nothing to duck.
    if (!separated()) {
      const duckAt = Math.max(ctx.currentTime, when - 0.06);
      const liftAt = Math.max(duckAt + 0.05, when + buf.duration - 0.04);
      bedGain.gain.setTargetAtTime(BED_LEVEL * DUCK_RATIO, duckAt, RAMP * 0.6);
      bedGain.gain.setTargetAtTime(BED_LEVEL, liftAt, RAMP);
    }

    if (when >= ctx.currentTime) {
      src.start(when);
    } else {
      // We arrived late (fetch/decode overran). Start mid-buffer so the line stays
      // pinned to its timeline position instead of sliding everything later.
      const offset = ctx.currentTime - when;
      if (offset < buf.duration - 0.05) src.start(ctx.currentTime, offset);
      else continue;
    }
    scheduled.set(seg.idx, { src, gain, seg });
  }
}

// ----------------------------------------------------------- language switch

async function switchTo(next) {
  if (next === lang || switching) return;

  switching = true;
  // Watchdog: `switching` must never be able to stick. If anything below stalls,
  // this releases the guard so the user is not locked out of further switches.
  const guard = setTimeout(() => { switching = false; render(); }, 4000);

  try {
    const from = lang;
    const wasPlaying = !video.paused;
    const resumeAt = video.currentTime;

    video.pause();
    clearScheduled();
    decoded.clear();
    anchor = null;
    lang = next;

    // Korean -> a dub restarts so you hear the dubbed version from the top.
    // Dub -> dub keeps your place, so you can compare languages mid-scene.
    video.currentTime = (from === "ko") ? 0 : resumeAt;
    ensureCtx();
    // Separated: mute the element and play the voice-free bed instead.
    // Not separated: keep the element audible and duck it under each line.
    video.muted = next !== "ko" && separated();
    resetBed(ctx.currentTime);

    // Not awaited: a hung fetch must not hold the switch open. The 700ms poller
    // will deliver the new manifest a moment later anyway.
    refresh().catch(() => {});

    if (wasPlaying || from === "ko") {
      // play() can return a promise that NEVER settles while the element is
      // re-buffering after a seek. Awaiting it is what previously wedged this
      // function and made language switching work exactly once.
      const p = video.play();
      if (p && typeof p.catch === "function") p.catch(() => {});
    }
  } catch (e) {
    console.warn("switchTo failed:", e);
  } finally {
    clearTimeout(guard);
    switching = false;
    render();
  }
}

for (const b of document.querySelectorAll(".lang")) {
  b.onclick = () => switchTo(b.dataset.lang);
}

// background bed level
$("bed").oninput = (e) => {
  BED_LEVEL = Number(e.target.value) / 100;
  $("bedval").textContent = e.target.value + "%";
  if (ctx && bedGain && lang !== "ko") resetBed(ctx.currentTime);
};

// -------------------------------------------------------------- video events

video.addEventListener("play", () => { if (lang !== "ko") { setAnchor(); scheduleAhead(); scheduleBed(); } });
video.addEventListener("pause", clearScheduled);
video.addEventListener("seeking", () => { clearScheduled(); anchor = null; });
video.addEventListener("seeked", () => {
  if (lang !== "ko" && !video.paused) { setAnchor(); scheduleAhead(); scheduleBed(); }
});
video.addEventListener("ratechange", () => {
  if (lang !== "ko" && !video.paused) { clearScheduled(); setAnchor(); scheduleAhead(); }
});
video.addEventListener("timeupdate", () => {
  const cur = manifest.segments.find((s) => video.currentTime >= s.start && video.currentTime < s.end);
  $("caption").textContent = cur ? (lang === "ko" ? cur.source_text : cur.text) : "";
  if (lang !== "ko" && !video.paused) { scheduleAhead(); scheduleBed(); }
});
