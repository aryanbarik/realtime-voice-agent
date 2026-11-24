const $ = (id) => document.getElementById(id);
const startBtn = $("start"), hangBtn = $("hangup"), pttBtn = $("ptt");

let ws = null, micCtx = null, playCtx = null, micNode = null, playNode = null, stream = null;
let turnMode = "auto", holding = false, currentTurn = 0, ending = false;
const v2vHistory = [];
const userLines = new Map(); // turn_id -> element

function setStatus(s) { $("status").textContent = s; }

function addMsg(role, text, turnId, final) {
  let el = role === "user" ? userLines.get(turnId) : null;
  if (!el) {
    el = document.createElement("div");
    el.className = "msg " + role;
    $("transcript").appendChild(el);
    if (role === "user") userLines.set(turnId, el);
  }
  el.textContent = (role === "user" ? "You: " : "Agent: ") + text;
  el.classList.toggle("interim", !final);
  if (role === "user" && final) userLines.delete(turnId);
  $("transcript").scrollTop = $("transcript").scrollHeight;
}

function logTool(m) {
  const d = document.createElement("div");
  d.textContent = `${m.name}(${JSON.stringify(m.args)}) → ${JSON.stringify(m.result)}`;
  $("tools").appendChild(d);
  $("tools").scrollTop = $("tools").scrollHeight;
}

function fmt(v) { return v == null ? "—" : Math.round(v); }

function showMetrics(m) {
  const d = m.derived || {};
  if (d.v2v_ms != null) {
    v2vHistory.push(d.v2v_ms);
    const s = [...v2vHistory].sort((a, b) => a - b);
    const mid = Math.floor(s.length / 2);
    const p50 = s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
    $("p50").textContent = `p50 v2v: ${Math.round(p50)} ms (n=${s.length})`;
    $("v2v").innerHTML = `${Math.round(d.v2v_ms)}<small> ms</small>`;
  }
  $("breakdown").textContent = [
    `endpoint wait  ${fmt(d.endpoint_wait_ms)}`,
    `asr finalize   ${fmt(d.asr_finalize_ms)}`,
    `llm ttft       ${fmt(d.llm_ttft_ms)}`,
    `tools          ${fmt(d.tool_ms)}`,
    `tts ttfa       ${fmt(d.tts_ttfa_ms)}`,
    `send           ${fmt(d.send_ms)}`,
    `barge-in       ${fmt(d.barge_in_ms)}`,
  ].join("\n");
}

function sendJson(o) { if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(o)); }

function onServerJson(m) {
  switch (m.type) {
    case "session":
      turnMode = m.turn_mode;
      pttBtn.hidden = turnMode !== "push_to_talk";
      setStatus(`connected (${m.profile})`);
      break;
    case "transcript":
      addMsg(m.role, m.text, m.turn_id, m.final);
      break;
    case "agent_audio_start":
      currentTurn = m.turn_id;
      break;
    case "agent_audio_end":
      playNode.port.postMessage({ cmd: "end", turn_id: m.turn_id });
      break;
    case "pause": playNode.port.postMessage({ cmd: "pause" }); break;
    case "resume": playNode.port.postMessage({ cmd: "resume" }); break;
    case "clear": playNode.port.postMessage({ cmd: "clear", turn_id: m.turn_id }); break;
    case "tool": logTool(m); break;
    case "turn_metrics": showMetrics(m); break;
    case "error":
      setStatus("error: " + m.message);
      break;
    case "session_end":
      setStatus("call ended: " + m.reason);
      break;
  }
}

async function start() {
  startBtn.disabled = true;
  ending = false;
  $("transcript").textContent = "";
  try {
    // Create both contexts synchronously inside the click handler (autoplay rules).
    micCtx = new AudioContext({ sampleRate: 16000 });
    playCtx = new AudioContext({ sampleRate: 24000 });
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    await micCtx.audioWorklet.addModule("/static/mic-worklet.js");
    await playCtx.audioWorklet.addModule("/static/player-worklet.js");
    await Promise.all([micCtx.resume(), playCtx.resume()]);

    playNode = new AudioWorkletNode(playCtx, "player-worklet");
    playNode.connect(playCtx.destination);
    playNode.port.onmessage = (e) => {
      const d = e.data;
      if (d.event === "cleared") sendJson({ type: "playback_cleared", turn_id: d.turn_id, played_samples: d.played });
      else if (d.event === "drained") sendJson({ type: "playback_done", turn_id: d.turn_id });
    };

    const tenant = new URLSearchParams(location.search).get("tenant") || "demo";
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws?tenant=${encodeURIComponent(tenant)}`);
    ws.binaryType = "arraybuffer";
    ws.onmessage = (e) => {
      if (typeof e.data === "string") return onServerJson(JSON.parse(e.data));
      const samples = new Int16Array(e.data);
      playNode.port.postMessage({ cmd: "push", turn_id: currentTurn, samples }, [samples.buffer]);
    };
    ws.onclose = () => teardown();
    ws.onerror = () => setStatus("connection error");
    await new Promise((res, rej) => { ws.onopen = res; setTimeout(() => rej(new Error("connect timeout")), 10000); });

    micNode = new AudioWorkletNode(micCtx, "mic-worklet");
    micCtx.createMediaStreamSource(stream).connect(micNode);
    micNode.port.onmessage = (e) => {
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      if (turnMode === "auto" || holding) ws.send(e.data);
    };
    hangBtn.disabled = false;
    setStatus("connected");
  } catch (err) {
    setStatus("failed: " + err.message);
    teardown();
  }
}

function teardown() {
  if (stream) stream.getTracks().forEach((t) => t.stop());
  if (micCtx) micCtx.close().catch(() => {});
  if (playCtx) playCtx.close().catch(() => {});
  if (ws && ws.readyState <= WebSocket.OPEN) ws.close();
  stream = micCtx = playCtx = micNode = playNode = ws = null;
  holding = false;
  startBtn.disabled = false;
  hangBtn.disabled = true;
  pttBtn.hidden = true;
}

function hangup() {
  sendJson({ type: "hangup" });
  teardown();
}

function pttDown(e) {
  e.preventDefault();
  if (holding || !ws) return;
  holding = true;
  pttBtn.textContent = "Talking…";
  sendJson({ type: "ptt_start" });
}
function pttUp(e) {
  e.preventDefault();
  if (!holding) return;
  holding = false;
  pttBtn.textContent = "Hold to talk";
  sendJson({ type: "ptt_end" });
}

startBtn.onclick = start;
hangBtn.onclick = hangup;
pttBtn.addEventListener("pointerdown", pttDown);
pttBtn.addEventListener("pointerup", pttUp);
pttBtn.addEventListener("pointerleave", pttUp);
pttBtn.addEventListener("pointercancel", pttUp);
