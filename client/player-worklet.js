// Int16 playback queue with pause/resume/clear and per-turn played-sample counts.
const START_THRESHOLD = 1440; // 60 ms @ 24 kHz

class PlayerWorklet extends AudioWorkletProcessor {
  constructor() {
    super();
    this.chunks = []; // {turn, data: Int16Array, pos}
    this.queued = 0;
    this.started = false;
    this.paused = false;
    this.ended = new Set(); // turn ids whose input is complete
    this.played = new Map(); // turn id -> samples played
    this.port.onmessage = (e) => this.onCmd(e.data);
  }
  onCmd(m) {
    switch (m.cmd) {
      case "push":
        this.chunks.push({ turn: m.turn_id, data: m.samples, pos: 0 });
        this.queued += m.samples.length;
        break;
      case "pause":
        this.paused = true;
        break;
      case "resume":
        this.paused = false;
        break;
      case "clear": {
        this.chunks = [];
        this.queued = 0;
        this.started = false;
        this.paused = false;
        this.ended.delete(m.turn_id);
        const played = this.played.get(m.turn_id) || 0;
        this.port.postMessage({ event: "cleared", turn_id: m.turn_id, played });
        break;
      }
      case "end":
        this.ended.add(m.turn_id);
        this.checkDrained();
        break;
    }
  }
  checkDrained() {
    for (const t of [...this.ended]) {
      if (!this.chunks.some((c) => c.turn === t)) {
        this.ended.delete(t);
        this.port.postMessage({ event: "drained", turn_id: t });
      }
    }
  }
  process(inputs, outputs) {
    const out = outputs[0][0];
    out.fill(0);
    if (this.paused) return true;
    if (!this.started) {
      if (this.queued >= START_THRESHOLD || (this.queued > 0 && this.ended.size > 0)) {
        this.started = true;
      } else {
        this.checkDrained();
        return true;
      }
    }
    let i = 0;
    while (i < out.length && this.chunks.length) {
      const c = this.chunks[0];
      const n = Math.min(out.length - i, c.data.length - c.pos);
      for (let k = 0; k < n; k++) out[i + k] = c.data[c.pos + k] / 32768;
      c.pos += n;
      i += n;
      this.queued -= n;
      this.played.set(c.turn, (this.played.get(c.turn) || 0) + n);
      if (c.pos >= c.data.length) {
        this.chunks.shift();
        this.checkDrained();
      }
    }
    if (!this.chunks.length) this.started = false; // underrun: rebuffer
    return true;
  }
}
registerProcessor("player-worklet", PlayerWorklet);
