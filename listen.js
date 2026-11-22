'use strict';
/* Listening: turns microphone sound into a stream of "chroma" frames
   (how much of each of the 12 note names C, C#, ... B is sounding),
   and matches live frames against a taught passage. */

const HOP_MS = 50;                 // one frame every 50 ms
const NOTE_NAMES = ['C', 'C♯', 'D', 'E♭', 'E', 'F', 'F♯', 'G', 'A♭', 'A', 'B♭', 'B'];

const Listener = {
  ctx: null, stream: null, source: null, analyser: null, timer: null,
  freq: null, time: null, binPc: null, binW: null, kMin: 0, kMax: 0,
  floor: -60,                       // running estimate of the room's noise level (dBFS)
  subscribers: new Set(),

  get running() { return !!this.timer; },

  // Call this synchronously inside a tap handler: iPad only lets audio start from a tap.
  unlock() {
    try {
      if (!this.ctx) {
        const AC = window.AudioContext || window.webkitAudioContext;
        this.ctx = new AC();
      }
      this.ctx.resume();
    } catch (err) { console.warn(err); }
  },

  // Must be called from a tap (browsers only allow audio after a user gesture).
  async start() {
    if (this.running) return;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error('This browser cannot use the microphone. Open the app from its https:// link.');
    }
    if (!this.ctx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      this.ctx = new AC();
    }
    await this.ctx.resume();
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    });
    this.source = this.ctx.createMediaStreamSource(this.stream);
    this.analyser = this.ctx.createAnalyser();
    this.analyser.fftSize = 8192;
    this.analyser.smoothingTimeConstant = 0;
    this.source.connect(this.analyser);
    this.prepareBins();
    this.floor = -60;
    this.timer = setInterval(() => this.tick(), HOP_MS);
  },

  stop() {
    clearInterval(this.timer);
    this.timer = null;
    if (this.source) this.source.disconnect();
    if (this.stream) this.stream.getTracks().forEach(t => t.stop());
    this.source = this.stream = this.analyser = null;
  },

  // Precompute which note name each FFT bin belongs to.
  prepareBins() {
    const sr = this.ctx.sampleRate, n = this.analyser.frequencyBinCount, fft = this.analyser.fftSize;
    this.freq = new Float32Array(n);
    this.time = new Float32Array(fft);
    this.binPc = new Int8Array(n).fill(-1);
    this.binW = new Float32Array(n);
    this.kMin = Math.ceil(100 * fft / sr);   // below ~100 Hz bins are wider than a semitone
    this.kMax = Math.min(n - 1, Math.floor(5000 * fft / sr));
    for (let k = this.kMin; k <= this.kMax; k++) {
      const midi = 69 + 12 * Math.log2((k * sr / fft) / 440);
      const nearest = Math.round(midi);
      this.binPc[k] = ((nearest % 12) + 12) % 12;
      // Bins halfway between two notes count less.
      this.binW[k] = Math.max(0, 1 - 2 * Math.abs(midi - nearest));
    }
  },

  tick() {
    const a = this.analyser;
    if (!a) return;
    a.getFloatTimeDomainData(this.time);
    let sum = 0;
    for (let i = 0; i < this.time.length; i++) sum += this.time[i] * this.time[i];
    const level = 10 * Math.log10(sum / this.time.length + 1e-12);
    // Noise floor follows quiet moments quickly and loud moments slowly.
    this.floor = level < this.floor ? level : this.floor + 0.02;

    a.getFloatFrequencyData(this.freq);
    const chroma = new Float32Array(12);
    for (let k = this.kMin; k <= this.kMax; k++) {
      const w = this.binW[k];
      if (!w) continue;
      chroma[this.binPc[k]] += w * Math.pow(10, this.freq[k] / 20);
    }
    let norm = 0;
    for (let i = 0; i < 12; i++) { chroma[i] = Math.sqrt(chroma[i]); norm += chroma[i] * chroma[i]; }
    norm = Math.sqrt(norm) || 1;
    for (let i = 0; i < 12; i++) chroma[i] /= norm;

    const silent = level < this.floor + 10 || level < -75;
    const frame = { chroma, level, silent, t: performance.now() };
    this.subscribers.forEach(fn => fn(frame));
  },

  subscribe(fn) { this.subscribers.add(fn); return () => this.subscribers.delete(fn); },
};

// Turn a recording (list of frames) into a stored cue: drop quiet frames at both ends.
function makeCue(frames) {
  let a = 0, b = frames.length - 1;
  while (a <= b && frames[a].silent) a++;
  while (b >= a && frames[b].silent) b--;
  const kept = frames.slice(a, b + 1);
  const sounding = kept.filter(f => !f.silent).length;
  if (sounding < 6) return null; // under ~0.3 s of sound: nothing useful was heard
  return {
    hop: HOP_MS / 1000,
    frames: kept.map(f => (f.silent ? null : Array.from(f.chroma, v => Math.round(v * 1000) / 1000))),
    recordedAt: Date.now(),
  };
}

// Center and normalize a chroma vector so that "which notes stand out" is compared, not loudness.
function centered(v) {
  if (!v) return null;
  let mean = 0;
  for (let i = 0; i < 12; i++) mean += v[i];
  mean /= 12;
  const out = new Float32Array(12);
  let n = 0;
  for (let i = 0; i < 12; i++) { out[i] = v[i] - mean; n += out[i] * out[i]; }
  n = Math.sqrt(n);
  if (n < 1e-6) return null;
  for (let i = 0; i < 12; i++) out[i] /= n;
  return out;
}

/* Follows live frames and reports how well the most recent stretch of playing matches the cue.
   It uses streaming "subsequence dynamic time warping": the cue can start at any moment, and
   it may be played anywhere from about twice as fast to much slower than when it was taught. */
class CueMatcher {
  // earlySec: report a match this many seconds before the end of the cue (to turn early).
  constructor(cue, earlySec = 0) {
    this.T = cue.frames.map(centered);
    this.m = this.T.length;
    const hop = cue.hop || HOP_MS / 1000;
    // Never fire before 40% of the cue has been heard, or it would trigger too easily.
    this.fireAt = Math.max(Math.round(0.4 * (this.m - 1)), this.m - 1 - Math.round(earlySec / hop));
    this.D = new Float64Array(this.m).fill(Infinity); // accumulated cost ending at template frame j
    this.L = new Float64Array(this.m).fill(0);        // path length for the same
  }

  reset() { this.D.fill(Infinity); this.L.fill(0); }

  // Returns { similarity 0..1, progress 0..1 } for the best match ending at the end of the cue.
  push(frame) {
    const x = frame.silent ? null : centered(frame.chroma);
    const { T, m, D, L } = this;
    const nD = new Float64Array(m), nL = new Float64Array(m);
    let bestProgress = 0, bestAvg = Infinity;
    for (let j = 0; j < m; j++) {
      // Cost 0 = same notes, 1 = opposite. Silence against sound costs 0.5; silence against silence is free.
      let c;
      if (!x && !T[j]) c = 0.1;
      else if (!x || !T[j]) c = 0.5;
      else {
        let dot = 0;
        for (let i = 0; i < 12; i++) dot += x[i] * T[j][i];
        c = (1 - dot) / 2;
      }
      if (j === 0) { nD[0] = c; nL[0] = 1; }
      else {
        // Steps: advance one cue frame, hold the same cue frame (slower), or skip one (faster).
        // Pick whichever previous point gives the lowest average cost per frame.
        let bd = Infinity, bl = 0, bestAvgHere = Infinity;
        const consider = (d, l) => {
          if (!isFinite(d)) return;
          const avg = (d + c) / (l + 1);
          if (avg < bestAvgHere) { bestAvgHere = avg; bd = d; bl = l; }
        };
        consider(D[j - 1], L[j - 1]);
        consider(D[j], L[j]);
        if (j >= 2) consider(D[j - 2], L[j - 2]);
        nD[j] = isFinite(bd) ? bd + c : Infinity;
        nL[j] = bl + 1;
      }
      const avg = nD[j] / nL[j];
      if (avg < 0.3 && j / (m - 1) > bestProgress) { bestProgress = j / (m - 1); bestAvg = avg; }
    }
    this.D = nD; this.L = nL;
    const f = this.fireAt;
    const endAvg = nD[f] / nL[f];
    // Demand the path really covered the cue (not a single frame stretched out).
    const covered = nL[f] >= (f + 1) * 0.5;
    return {
      similarity: covered && isFinite(endAvg) ? Math.max(0, 1 - endAvg * 2) : 0,
      progress: bestProgress,
    };
  }
}

// Draw a cue (or live frames) as a strip of 12 note rows.
function drawChroma(canvas, frames, opts = {}) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w || !h) return;
  canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const labelW = 30, rowH = h / 12;
  const css = getComputedStyle(document.documentElement);
  const accent = css.getPropertyValue('--accent').trim() || '#3b5bdb';
  const muted = css.getPropertyValue('--muted').trim() || '#888';
  ctx.font = `${Math.min(12, rowH * 0.8)}px -apple-system, sans-serif`;
  ctx.textBaseline = 'middle';
  ctx.fillStyle = muted;
  for (let pc = 0; pc < 12; pc++) ctx.fillText(NOTE_NAMES[pc], 2, h - (pc + 0.5) * rowH);
  const n = opts.columns || frames.length;
  if (!frames.length) return;
  const colW = (w - labelW) / Math.max(n, 1);
  const start = Math.max(0, frames.length - n);
  ctx.fillStyle = accent;
  for (let f = start; f < frames.length; f++) {
    const v = frames[f];
    if (!v) continue;
    let max = 0;
    for (let i = 0; i < 12; i++) if (v[i] > max) max = v[i];
    for (let pc = 0; pc < 12; pc++) {
      const a = max ? Math.pow(v[pc] / max, 2) : 0;
      if (a < 0.08) continue;
      ctx.globalAlpha = a;
      ctx.fillRect(labelW + (f - start) * colW, h - (pc + 1) * rowH + 1, Math.max(1, colW + 0.5), rowH - 2);
    }
  }
  ctx.globalAlpha = 1;
}
