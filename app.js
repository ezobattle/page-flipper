'use strict';
/* Page Flipper, step 1: photograph pages, crop them, keep them as pieces,
   and show them full screen with tap, swipe or pedal page turns. */

// ---------------------------------------------------------------- storage
const DB_NAME = 'pageflipper';
const DB_VERSION = 1;
let dbPromise = null;

function openDB() {
  if (!dbPromise) {
    dbPromise = new Promise((resolve, reject) => {
      const req = indexedDB.open(DB_NAME, DB_VERSION);
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains('pieces')) db.createObjectStore('pieces', { keyPath: 'id' });
        if (!db.objectStoreNames.contains('pages')) db.createObjectStore('pages', { keyPath: 'id' });
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  }
  return dbPromise;
}

async function tx(store, mode, fn) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const t = db.transaction(store, mode);
    const s = t.objectStore(store);
    let result;
    const r = fn(s);
    if (r) r.onsuccess = () => { result = r.result; };
    t.oncomplete = () => resolve(result);
    t.onerror = () => reject(t.error);
    t.onabort = () => reject(t.error);
  });
}
const dbGet = (store, id) => tx(store, 'readonly', s => s.get(id));
const dbAll = (store) => tx(store, 'readonly', s => s.getAll());
const dbPut = (store, value) => tx(store, 'readwrite', s => s.put(value));
const dbDel = (store, id) => tx(store, 'readwrite', s => s.delete(id));

const uid = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 8);

// ---------------------------------------------------------------- helpers
const $ = (id) => document.getElementById(id);
const views = ['library', 'editor', 'crop', 'viewer', 'teach'];
function show(view) {
  for (const v of views) $(v).hidden = v !== view;
  window.scrollTo(0, 0);
}

let toastTimer;
function toast(msg, ms = 2500) {
  const t = $('toast');
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, ms);
}

function canvasToBlob(canvas, type = 'image/jpeg', quality = 0.88) {
  return new Promise((resolve, reject) =>
    canvas.toBlob(b => (b ? resolve(b) : reject(new Error('Could not encode image'))), type, quality));
}

async function loadImage(blob) {
  const url = URL.createObjectURL(blob);
  try {
    const img = new Image();
    img.decoding = 'async';
    img.src = url;
    await img.decode();
    return img;
  } finally {
    URL.revokeObjectURL(url);
  }
}

// Draw an image (optionally rotated by 90° steps) into a canvas no larger than maxSide.
function drawScaled(img, maxSide, rotation = 0) {
  const w0 = img.naturalWidth || img.width, h0 = img.naturalHeight || img.height;
  const scale = Math.min(1, maxSide / Math.max(w0, h0));
  const w = Math.round(w0 * scale), h = Math.round(h0 * scale);
  const swap = rotation % 180 !== 0;
  const c = document.createElement('canvas');
  c.width = swap ? h : w;
  c.height = swap ? w : h;
  const ctx = c.getContext('2d');
  ctx.imageSmoothingQuality = 'high';
  ctx.translate(c.width / 2, c.height / 2);
  ctx.rotate(rotation * Math.PI / 180);
  ctx.drawImage(img, -w / 2, -h / 2, w, h);
  return c;
}

// ---------------------------------------------------------------- image processing
// Solve the 8x8 system for a homography mapping (x,y) -> (u,v) for 4 point pairs.
function homography(from, to) {
  const A = [], b = [];
  for (let i = 0; i < 4; i++) {
    const [x, y] = from[i], [u, v] = to[i];
    A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.push(u);
    A.push([0, 0, 0, x, y, 1, -v * x, -v * y]); b.push(v);
  }
  // Gaussian elimination with partial pivoting
  for (let col = 0; col < 8; col++) {
    let piv = col;
    for (let r = col + 1; r < 8; r++) if (Math.abs(A[r][col]) > Math.abs(A[piv][col])) piv = r;
    [A[col], A[piv]] = [A[piv], A[col]];
    [b[col], b[piv]] = [b[piv], b[col]];
    for (let r = col + 1; r < 8; r++) {
      const f = A[r][col] / A[col][col];
      for (let c = col; c < 8; c++) A[r][c] -= f * A[col][c];
      b[r] -= f * b[col];
    }
  }
  const h = new Array(8);
  for (let r = 7; r >= 0; r--) {
    let s = b[r];
    for (let c = r + 1; c < 8; c++) s -= A[r][c] * h[c];
    h[r] = s / A[r][r];
  }
  return [...h, 1];
}

const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

// Straighten the quad (tl, tr, br, bl) of the source canvas into a flat rectangle.
function warpQuad(src, quad, maxSide = 2000) {
  const [tl, tr, br, bl] = quad;
  let W = Math.max(dist(tl, tr), dist(bl, br));
  let H = Math.max(dist(tl, bl), dist(tr, br));
  const s = Math.min(1, maxSide / Math.max(W, H));
  W = Math.max(1, Math.round(W * s));
  H = Math.max(1, Math.round(H * s));

  // Map output pixel -> source pixel.
  const m = homography([[0, 0], [W, 0], [W, H], [0, H]], quad);
  const sw = src.width, sh = src.height;
  const sd = src.getContext('2d').getImageData(0, 0, sw, sh).data;
  const out = document.createElement('canvas');
  out.width = W; out.height = H;
  const octx = out.getContext('2d');
  const od = octx.createImageData(W, H);
  const o = od.data;
  let k = 0;
  for (let y = 0; y < H; y++) {
    const yy = y + 0.5;
    for (let x = 0; x < W; x++, k += 4) {
      const xx = x + 0.5;
      const d = m[6] * xx + m[7] * yy + 1;
      let sx = (m[0] * xx + m[1] * yy + m[2]) / d - 0.5;
      let sy = (m[3] * xx + m[4] * yy + m[5]) / d - 0.5;
      if (sx < 0) sx = 0; else if (sx > sw - 1.001) sx = sw - 1.001;
      if (sy < 0) sy = 0; else if (sy > sh - 1.001) sy = sh - 1.001;
      const x0 = sx | 0, y0 = sy | 0, fx = sx - x0, fy = sy - y0;
      const i00 = (y0 * sw + x0) * 4, i10 = i00 + 4, i01 = i00 + sw * 4, i11 = i01 + 4;
      for (let c = 0; c < 3; c++) {
        const top = sd[i00 + c] + (sd[i10 + c] - sd[i00 + c]) * fx;
        const bot = sd[i01 + c] + (sd[i11 + c] - sd[i01 + c]) * fx;
        o[k + c] = top + (bot - top) * fy;
      }
      o[k + 3] = 255;
    }
  }
  octx.putImageData(od, 0, 0);
  return out;
}

// Make the paper white and the ink dark, removing uneven lighting and shadows.
// Keeps grey tones so thin lines stay smooth, sharpens slightly, and (optionally)
// erases marks in any color other than black, such as a teacher's red or blue pen.
function enhance(canvas, removeColor = true) {
  const W = canvas.width, H = canvas.height, N = W * H;
  // Estimate the paper color everywhere by shrinking the image a lot (ink is thin, so it mostly averages away).
  let small = canvas, sw = W, sh = H;
  while (sw > 48 || sh > 48) {
    sw = Math.max(1, Math.round(sw / 2)); sh = Math.max(1, Math.round(sh / 2));
    const c = document.createElement('canvas');
    c.width = sw; c.height = sh;
    const cx = c.getContext('2d');
    cx.imageSmoothingQuality = 'high';
    cx.drawImage(small, 0, 0, sw, sh);
    small = c;
  }
  const bg = document.createElement('canvas');
  bg.width = W; bg.height = H;
  const bctx = bg.getContext('2d');
  bctx.imageSmoothingEnabled = true;
  bctx.imageSmoothingQuality = 'high';
  bctx.drawImage(small, 0, 0, W, H);
  const bd = bctx.getImageData(0, 0, W, H).data;

  const ctx = canvas.getContext('2d');
  const img = ctx.getImageData(0, 0, W, H);
  const d = img.data;

  // Brightness relative to the paper (1 = paper, 0 = black), and how colorful each pixel is.
  const L = new Float32Array(N);
  const C = new Float32Array(N);
  for (let p = 0, i = 0; p < N; p++, i += 4) {
    // Ink lowers the background estimate a little; lift it so clean paper lands at or above 1.
    const r = d[i] / Math.max(20, bd[i] * 1.06);
    const g = d[i + 1] / Math.max(20, bd[i + 1] * 1.06);
    const b = d[i + 2] / Math.max(20, bd[i + 2] * 1.06);
    L[p] = removeColor ? Math.max(r, g, b) : 0.299 * r + 0.587 * g + 0.114 * b;
    C[p] = Math.max(r, g, b) - Math.min(r, g, b);
  }

  // Light unsharp mask (3x3) so camera softness doesn't read as blur.
  const amount = 0.7;
  const S = new Float32Array(N);
  for (let y = 0; y < H; y++) {
    const y0 = y > 0 ? y - 1 : 0, y1 = y < H - 1 ? y + 1 : H - 1;
    for (let x = 0; x < W; x++) {
      const x0 = x > 0 ? x - 1 : 0, x1 = x < W - 1 ? x + 1 : W - 1;
      const blur = (L[y0 * W + x0] + L[y0 * W + x] + L[y0 * W + x1] +
                    L[y * W + x0] + L[y * W + x] + L[y * W + x1] +
                    L[y1 * W + x0] + L[y1 * W + x] + L[y1 * W + x1]) / 9;
      const p = y * W + x;
      S[p] = L[p] + amount * (L[p] - blur);
    }
  }

  // Colored ink is bright in at least one color channel (red pen is bright in red, blue pen in blue),
  // while black ink is dark in all of them. Using the brightest channel makes colored marks
  // fade into the paper but keeps black notation, even where a colored mark crosses it.
  let M = null;
  if (removeColor) {
    M = new Float32Array(N);
    for (let p = 0; p < N; p++) {
      // Dark but clearly colored ink (e.g. a dark blue ballpoint) still has some color: push it out too.
      // Skip pixels that are dark in every channel: that's black print, maybe under a pen stroke.
      const w = (C[p] - 0.18) / 0.15, dark = (L[p] - 0.4) / 0.15;
      M[p] = (w < 0 ? 0 : w > 1 ? 1 : w) * (dark < 0 ? 0 : dark > 1 ? 1 : dark);
    }
  }

  // Gentle tone curve: real ink goes black, paper goes white, edges keep their grey.
  const lo = 0.28, hi = 0.9;
  for (let p = 0, i = 0; p < N; p++, i += 4) {
    let v = (S[p] - lo) / (hi - lo);
    v = v < 0 ? 0 : v > 1 ? 1 : v;
    if (M) v = v + (1 - v) * M[p]; // colored marks fade to paper
    d[i] = d[i + 1] = d[i + 2] = Math.round(v * 255);
  }
  ctx.putImageData(img, 0, 0);
  return canvas;
}

// ---------------------------------------------------------------- state
let currentPiece = null;     // piece record being edited / played
let cropQueue = [];          // files waiting to be cropped
let cropJob = null;          // { pageId?, original: Blob, img, rotation, corners, srcCanvas }

// ---------------------------------------------------------------- library
async function renderLibrary() {
  const pieces = (await dbAll('pieces')).sort((a, b) => b.updated - a.updated);
  const grid = $('pieceGrid');
  grid.textContent = '';
  $('emptyLibrary').hidden = pieces.length > 0;
  for (const p of pieces) {
    const btn = document.createElement('button');
    btn.className = 'piece';
    const thumb = document.createElement('div');
    thumb.className = 'thumb';
    if (p.pageIds.length) {
      const page = await dbGet('pages', p.pageIds[0]);
      if (page) thumb.style.backgroundImage = `url(${trackedURL(page.thumb)})`;
    }
    const meta = document.createElement('div');
    meta.className = 'meta';
    meta.innerHTML = '<div class="name"></div><div class="count"></div>';
    meta.querySelector('.name').textContent = p.title || 'Untitled piece';
    meta.querySelector('.count').textContent = `${p.pageIds.length} page${p.pageIds.length === 1 ? '' : 's'}`;
    btn.append(thumb, meta);
    btn.onclick = () => openEditor(p.id);
    grid.append(btn);
  }
}

// Object URLs for thumbnails, released whenever a list is redrawn.
let liveURLs = [];
function trackedURL(blob) {
  const u = URL.createObjectURL(blob);
  liveURLs.push(u);
  return u;
}
function releaseURLs() {
  liveURLs.forEach(u => URL.revokeObjectURL(u));
  liveURLs = [];
}

async function goLibrary() {
  releaseURLs();
  currentPiece = null;
  show('library');
  await renderLibrary();
}

$('newPieceBtn').onclick = async () => {
  const piece = { id: uid(), title: '', pageIds: [], created: Date.now(), updated: Date.now() };
  await dbPut('pieces', piece);
  await openEditor(piece.id);
  $('pieceTitle').focus();
};

// ---------------------------------------------------------------- editor
async function openEditor(id) {
  releaseURLs();
  currentPiece = await dbGet('pieces', id);
  if (!currentPiece) return goLibrary();
  $('pieceTitle').value = currentPiece.title;
  show('editor');
  await renderPages();
}

async function savePiece() {
  currentPiece.updated = Date.now();
  await dbPut('pieces', currentPiece);
}

async function renderPages() {
  releaseURLs();
  const list = $('pageList');
  list.textContent = '';
  const ids = currentPiece.pageIds;
  $('emptyPieces').hidden = ids.length > 0;
  $('playBtn').disabled = ids.length === 0;
  $('teachBtn').disabled = ids.length === 0;
  for (let i = 0; i < ids.length; i++) {
    const page = await dbGet('pages', ids[i]);
    if (!page) continue;
    const li = document.createElement('li');
    const img = document.createElement('img');
    img.src = trackedURL(page.thumb);
    img.alt = `Page ${i + 1}`;
    img.onclick = () => { Listener.unlock(); openViewer(i); };
    const row = document.createElement('div');
    row.className = 'row';
    const label = document.createElement('span');
    label.textContent = i + 1;
    if (page.cue) {
      const t = document.createElement('span');
      t.className = 'taught'; t.textContent = ' 🎹'; t.title = 'Page turn taught';
      label.append(t);
    }
    const mk = (txt, title, fn, disabled) => {
      const b = document.createElement('button');
      b.textContent = txt; b.title = title; b.setAttribute('aria-label', title);
      b.disabled = !!disabled; b.onclick = fn;
      return b;
    };
    row.append(
      label,
      mk('◀', 'Move earlier', () => movePage(i, -1), i === 0),
      mk('▶', 'Move later', () => movePage(i, 1), i === ids.length - 1),
      mk('✂︎', 'Re-crop', () => recropPage(page)),
      mk('🗑', 'Delete page', () => deletePage(i)),
    );
    li.append(img, row);
    list.append(li);
  }
}

async function movePage(i, dir) {
  const ids = currentPiece.pageIds;
  const j = i + dir;
  if (j < 0 || j >= ids.length) return;
  [ids[i], ids[j]] = [ids[j], ids[i]];
  await savePiece();
  await renderPages();
}

let pendingDelete = null;
async function deletePage(i) {
  // Two taps to delete, so a stray tap can't lose a page.
  const id = currentPiece.pageIds[i];
  if (pendingDelete !== id) {
    pendingDelete = id;
    toast('Tap 🗑 again to delete this page');
    setTimeout(() => { if (pendingDelete === id) pendingDelete = null; }, 3000);
    return;
  }
  pendingDelete = null;
  currentPiece.pageIds.splice(i, 1);
  await savePiece();
  await dbDel('pages', id);
  await renderPages();
}

$('pieceTitle').addEventListener('input', () => {
  if (!currentPiece) return;
  currentPiece.title = $('pieceTitle').value.trim();
  clearTimeout(savePiece.t);
  savePiece.t = setTimeout(savePiece, 300);
});
$('editorBack').onclick = async () => { await savePiece(); goLibrary(); };
$('playBtn').onclick = () => { Listener.unlock(); openViewer(0); };

let deleteArmed = false;
$('deletePieceBtn').onclick = async () => {
  if (!deleteArmed) {
    deleteArmed = true;
    $('deletePieceBtn').textContent = 'Tap again to delete the whole piece';
    setTimeout(() => { deleteArmed = false; $('deletePieceBtn').textContent = 'Delete this piece'; }, 3000);
    return;
  }
  deleteArmed = false;
  $('deletePieceBtn').textContent = 'Delete this piece';
  for (const id of currentPiece.pageIds) await dbDel('pages', id);
  await dbDel('pieces', currentPiece.id);
  goLibrary();
};

function onFiles(e) {
  const files = [...e.target.files];
  e.target.value = '';
  if (!files.length) return;
  cropQueue.push(...files);
  if (!cropJob) nextCrop();
}
$('cameraInput').addEventListener('change', onFiles);
$('libraryInput').addEventListener('change', onFiles);

// ---------------------------------------------------------------- crop
const SOURCE_MAX = 4032;   // longest side kept for the original photo (full iPad camera resolution)
const PAGE_MAX = 3000;     // longest side of the straightened page (sharper than the iPad screen)

async function nextCrop() {
  const file = cropQueue.shift();
  if (!file) { cropJob = null; return; }
  try {
    const img = await loadImage(file);
    // Keep a downsized original so the page can be re-cropped later.
    const original = await canvasToBlob(drawScaled(img, SOURCE_MAX), 'image/jpeg', 0.92);
    await startCrop({ original, rotation: 0, corners: null, enhance: false, removeColor: true });
  } catch (err) {
    console.error(err);
    toast('Could not open that photo');
    nextCrop();
  }
}

async function recropPage(page) {
  cropQueue = [];
  await startCrop({
    pageId: page.id, original: page.original, rotation: page.rotation || 0,
    corners: page.corners, enhance: !!page.enhance, removeColor: page.removeColor !== false,
  });
}

async function startCrop(job) {
  job.img = await loadImage(job.original);
  cropJob = job;
  job.srcCanvas = drawScaled(job.img, SOURCE_MAX, job.rotation);
  if (!job.corners) job.corners = defaultCorners(job.srcCanvas);
  $('enhanceToggle').checked = job.enhance;
  $('colorToggle').checked = job.removeColor;
  $('colorToggle').disabled = !job.enhance;
  const left = cropQueue.length;
  $('cropHint').textContent = left
    ? `Drag the corners onto the page edges (${left} more after this)`
    : 'Drag the corners onto the page edges';
  show('crop');
  requestAnimationFrame(drawCrop);
}

function defaultCorners(c) {
  return detectPage(c) || insetCorners(c);
}
function insetCorners(c) {
  const mx = c.width * 0.04, my = c.height * 0.04;
  return [[mx, my], [c.width - mx, my], [c.width - mx, c.height - my], [mx, c.height - my]];
}

// Guess where the sheet of paper is: the largest bright region in the photo.
// Returns [tl, tr, br, bl] in source pixels, or null if nothing convincing is found.
function detectPage(src) {
  try {
    const S = 320 / Math.max(src.width, src.height);
    const w = Math.max(8, Math.round(src.width * S)), h = Math.max(8, Math.round(src.height * S));
    const c = drawScaled(src, 320);
    const d = c.getContext('2d').getImageData(0, 0, w, h).data;
    const n = w * h, g = new Uint8Array(n), hist = new Array(256).fill(0);
    for (let i = 0; i < n; i++) {
      const v = (0.299 * d[i * 4] + 0.587 * d[i * 4 + 1] + 0.114 * d[i * 4 + 2]) | 0;
      g[i] = v; hist[v]++;
    }
    // Otsu threshold splits "paper" from "background".
    let sum = 0; for (let i = 0; i < 256; i++) sum += i * hist[i];
    let sumB = 0, wB = 0, best = 0, thr = 128;
    for (let t = 0; t < 256; t++) {
      wB += hist[t]; if (!wB) continue;
      const wF = n - wB; if (!wF) break;
      sumB += t * hist[t];
      const mB = sumB / wB, mF = (sum - sumB) / wF;
      const between = wB * wF * (mB - mF) * (mB - mF);
      if (between > best) { best = between; thr = t; }
    }
    // Close small gaps (staff lines, notes) so the page is one solid blob.
    const bright = new Uint8Array(n);
    for (let i = 0; i < n; i++) bright[i] = g[i] > thr ? 1 : 0;
    const closed = new Uint8Array(n);
    const R = 3;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      let on = 0;
      for (let dy = -R; dy <= R && !on; dy++) {
        const yy = y + dy; if (yy < 0 || yy >= h) continue;
        for (let dx = -R; dx <= R; dx++) {
          const xx = x + dx; if (xx >= 0 && xx < w && bright[yy * w + xx]) { on = 1; break; }
        }
      }
      closed[y * w + x] = on;
    }
    // Largest connected bright region.
    const label = new Int32Array(n), stack = [];
    let bestLabel = 0, bestSize = 0, next = 1;
    for (let i = 0; i < n; i++) {
      if (!closed[i] || label[i]) continue;
      let size = 0; label[i] = next; stack.push(i);
      while (stack.length) {
        const p = stack.pop(); size++;
        const x = p % w, y = (p / w) | 0;
        if (x > 0 && closed[p - 1] && !label[p - 1]) { label[p - 1] = next; stack.push(p - 1); }
        if (x < w - 1 && closed[p + 1] && !label[p + 1]) { label[p + 1] = next; stack.push(p + 1); }
        if (y > 0 && closed[p - w] && !label[p - w]) { label[p - w] = next; stack.push(p - w); }
        if (y < h - 1 && closed[p + w] && !label[p + w]) { label[p + w] = next; stack.push(p + w); }
      }
      if (size > bestSize) { bestSize = size; bestLabel = next; }
      next++;
    }
    if (bestSize < n * 0.15 || bestSize > n * 0.97) return null;
    // Corners are the extreme points along the two diagonals.
    let tl, tr, br, bl, mn = Infinity, mx = -Infinity, md = Infinity, Md = -Infinity;
    for (let i = 0; i < n; i++) {
      if (label[i] !== bestLabel) continue;
      const x = i % w, y = (i / w) | 0, s = x + y, df = x - y;
      if (s < mn) { mn = s; tl = [x, y]; }
      if (s > mx) { mx = s; br = [x + 1, y + 1]; }
      if (df > Md) { Md = df; tr = [x + 1, y]; }
      if (df < md) { md = df; bl = [x, y + 1]; }
    }
    // The closing step grew the region by R pixels; pull the corners back in.
    const quad = [[tl[0] + R, tl[1] + R], [tr[0] - R, tr[1] + R], [br[0] - R, br[1] - R], [bl[0] + R, bl[1] - R]];
    const k = src.width / w;
    return quad.map(([x, y]) => [Math.max(0, Math.min(src.width, x * k)), Math.max(0, Math.min(src.height, y * k))]);
  } catch (err) {
    console.warn('Page detection failed', err);
    return null;
  }
}

// Layout of the photo inside the crop canvas (CSS pixels).
let cropLayout = { scale: 1, ox: 0, oy: 0 };

function drawCrop() {
  if (!cropJob) return;
  const canvas = $('cropCanvas');
  const stage = $('cropStage');
  const dpr = window.devicePixelRatio || 1;
  const cw = stage.clientWidth, ch = stage.clientHeight;
  canvas.width = Math.round(cw * dpr);
  canvas.height = Math.round(ch * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cw, ch);

  const src = cropJob.srcCanvas;
  const pad = 28;
  const scale = Math.min((cw - pad * 2) / src.width, (ch - pad * 2) / src.height);
  const ox = (cw - src.width * scale) / 2, oy = (ch - src.height * scale) / 2;
  cropLayout = { scale, ox, oy };
  ctx.drawImage(src, ox, oy, src.width * scale, src.height * scale);

  const pts = cropJob.corners.map(([x, y]) => [ox + x * scale, oy + y * scale]);
  // Dim everything outside the page.
  ctx.save();
  ctx.fillStyle = 'rgba(0,0,0,0.55)';
  ctx.beginPath();
  ctx.rect(0, 0, cw, ch);
  ctx.moveTo(...pts[0]); ctx.lineTo(...pts[3]); ctx.lineTo(...pts[2]); ctx.lineTo(...pts[1]); ctx.closePath();
  ctx.fill('evenodd');
  ctx.restore();
  // Outline and handles.
  ctx.strokeStyle = '#4dabf7';
  ctx.lineWidth = 2;
  ctx.beginPath();
  pts.forEach((p, i) => (i ? ctx.lineTo(...p) : ctx.moveTo(...p)));
  ctx.closePath();
  ctx.stroke();
  for (let i = 0; i < 4; i++) {
    ctx.beginPath();
    ctx.arc(pts[i][0], pts[i][1], dragIndex === i ? 18 : 14, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(77,171,247,0.35)';
    ctx.fill();
    ctx.lineWidth = 3;
    ctx.stroke();
  }
  // Magnifier while dragging, so the finger doesn't hide the corner.
  if (dragIndex !== null) {
    const [px, py] = pts[dragIndex];
    const [sx, sy] = cropJob.corners[dragIndex];
    const R = 60, zoom = 3;
    const cx = px < cw / 2 ? cw - R - 16 : R + 16, cy = R + 16;
    ctx.save();
    ctx.beginPath(); ctx.arc(cx, cy, R, 0, Math.PI * 2); ctx.clip();
    ctx.fillStyle = '#000'; ctx.fillRect(cx - R, cy - R, R * 2, R * 2);
    const srcR = R / (scale * zoom);
    ctx.drawImage(src, sx - srcR, sy - srcR, srcR * 2, srcR * 2, cx - R, cy - R, R * 2, R * 2);
    ctx.strokeStyle = '#4dabf7'; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(cx - 12, cy); ctx.lineTo(cx + 12, cy); ctx.moveTo(cx, cy - 12); ctx.lineTo(cx, cy + 12); ctx.stroke();
    ctx.restore();
    ctx.beginPath(); ctx.arc(cx, cy, R, 0, Math.PI * 2); ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke();
  }
}

let dragIndex = null, dragOffset = [0, 0];
const cropCanvas = $('cropCanvas');
cropCanvas.addEventListener('pointerdown', (e) => {
  if (!cropJob) return;
  const r = cropCanvas.getBoundingClientRect();
  const x = e.clientX - r.left, y = e.clientY - r.top;
  const { scale, ox, oy } = cropLayout;
  let best = null, bestD = 60; // generous touch radius
  cropJob.corners.forEach(([cx, cy], i) => {
    const d = Math.hypot(ox + cx * scale - x, oy + cy * scale - y);
    if (d < bestD) { bestD = d; best = i; }
  });
  if (best === null) return;
  dragIndex = best;
  const [cx, cy] = cropJob.corners[best];
  dragOffset = [ox + cx * scale - x, oy + cy * scale - y];
  cropCanvas.setPointerCapture(e.pointerId);
  drawCrop();
});
cropCanvas.addEventListener('pointermove', (e) => {
  if (dragIndex === null) return;
  const r = cropCanvas.getBoundingClientRect();
  const { scale, ox, oy } = cropLayout;
  const src = cropJob.srcCanvas;
  let x = (e.clientX - r.left + dragOffset[0] - ox) / scale;
  let y = (e.clientY - r.top + dragOffset[1] - oy) / scale;
  x = Math.max(0, Math.min(src.width, x));
  y = Math.max(0, Math.min(src.height, y));
  cropJob.corners[dragIndex] = [x, y];
  drawCrop();
});
const endDrag = () => { if (dragIndex !== null) { dragIndex = null; drawCrop(); } };
cropCanvas.addEventListener('pointerup', endDrag);
cropCanvas.addEventListener('pointercancel', endDrag);
window.addEventListener('resize', () => { if (!$('crop').hidden) drawCrop(); });

$('rotateBtn').onclick = () => {
  cropJob.rotation = (cropJob.rotation + 90) % 360;
  cropJob.srcCanvas = drawScaled(cropJob.img, SOURCE_MAX, cropJob.rotation);
  cropJob.corners = defaultCorners(cropJob.srcCanvas);
  drawCrop();
};
$('enhanceToggle').onchange = () => { $('colorToggle').disabled = !$('enhanceToggle').checked; };
$('resetCornersBtn').onclick = () => {
  cropJob.corners = defaultCorners(cropJob.srcCanvas);
  drawCrop();
};
$('cropCancel').onclick = async () => {
  cropJob = null;
  cropQueue = [];
  show('editor');
  await renderPages();
};

$('cropSave').onclick = async () => {
  const job = cropJob;
  if (!job) return;
  $('busy').hidden = false;
  // Let the spinner paint before the heavy work.
  await new Promise(r => requestAnimationFrame(() => setTimeout(r, 30)));
  try {
    job.enhance = $('enhanceToggle').checked;
    job.removeColor = $('colorToggle').checked;
    const page = warpQuad(job.srcCanvas, job.corners, PAGE_MAX);
    if (job.enhance) enhance(page, job.removeColor);
    const blob = await canvasToBlob(page, 'image/jpeg', 0.9);
    const thumb = await canvasToBlob(drawScaled(page, 400), 'image/jpeg', 0.8);
    const record = {
      id: job.pageId || uid(),
      blob, thumb, original: job.original,
      rotation: job.rotation, corners: job.corners, enhance: job.enhance, removeColor: job.removeColor,
      width: page.width, height: page.height,
      cue: job.pageId ? ((await dbGet('pages', job.pageId)) || {}).cue : undefined,
    };
    await dbPut('pages', record);
    if (!job.pageId) currentPiece.pageIds.push(record.id);
    await savePiece();
    requestPersistence();
  } catch (err) {
    console.error(err);
    toast('Saving failed: ' + err.message, 4000);
  } finally {
    $('busy').hidden = true;
  }
  cropJob = null;
  if (cropQueue.length) return nextCrop();
  show('editor');
  await renderPages();
};

// Ask the browser not to evict our photos when storage runs low.
let persistAsked = false;
function requestPersistence() {
  if (persistAsked || !navigator.storage || !navigator.storage.persist) return;
  persistAsked = true;
  navigator.storage.persist().catch(() => {});
}

// ---------------------------------------------------------------- teach
// For each page except the last, Enzo plays the last bar once. We save what the
// microphone heard as a "cue", which the auto-turn will listen for while playing.
const teach = {
  pages: [], index: 0, url: null,
  mode: 'idle',          // idle | recording | testing
  frames: [], unsub: null, matcher: null, liveFrames: [],
};
const MATCH_THRESHOLD = 0.75;   // similarity needed to count as "heard the cue"

async function openTeach() {
  await savePiece();
  teach.pages = [];
  for (const id of currentPiece.pageIds) {
    const p = await dbGet('pages', id);
    if (p) teach.pages.push(p);
  }
  if (!teach.pages.length) return;
  // Start at the first page that hasn't been taught yet.
  const last = Math.max(0, teach.pages.length - 2);
  const firstUntaught = teach.pages.findIndex((p, i) => i <= last && !p.cue);
  teach.index = firstUntaught >= 0 ? firstUntaught : 0;
  $('teachTitle').textContent = `Teach page turns: ${currentPiece.title || 'Untitled piece'}`;
  $('matchLine').style.left = `${MATCH_THRESHOLD * 100}%`;
  show('teach');
  renderTeach();
}

function teachTurnCount() { return Math.max(1, teach.pages.length - 1); }

function renderTeach() {
  const i = teach.index, page = teach.pages[i];
  const single = teach.pages.length === 1;
  if (teach.url) URL.revokeObjectURL(teach.url);
  teach.url = URL.createObjectURL(page.blob);
  $('teachImg').src = teach.url;
  $('teachImg').alt = `Page ${i + 1}`;
  $('teachWhich').textContent = single ? 'Page 1' : `End of page ${i + 1} → ${i + 2}`;
  const state = $('teachState');
  state.textContent = page.cue ? 'Taught ✓' : 'Not taught yet';
  state.classList.toggle('ok', !!page.cue);
  $('teachHelp').innerHTML = single
    ? 'This piece has only one page, so there is nothing to turn. Add more pages first.'
    : `Tap <b>Record</b>, play the <b>last bar or two</b> of page ${i + 1} the way you normally would, then tap <b>Stop</b>. About 3 to 6 seconds is ideal.`;
  $('teachPrev').disabled = i === 0;
  $('teachNext').disabled = i >= teachTurnCount() - 1;
  $('recordBtn').disabled = single;
  $('testBtn').disabled = !page.cue;
  $('forgetBtn').disabled = !page.cue;
  $('testBox').hidden = true;
  drawChroma($('chromaCanvas'), page.cue ? page.cue.frames : []);
}

function setRecordButton(label, stop) {
  const b = $('recordBtn');
  b.textContent = label;
  b.classList.toggle('stop', !!stop);
}

async function ensureMic() {
  try {
    await Listener.start();
    return true;
  } catch (err) {
    console.error(err);
    const msg = err && err.name === 'NotAllowedError'
      ? 'Microphone access was blocked. Allow it for this site in Chrome (or iPad Settings › Chrome › Microphone), then try again.'
      : (err.message || 'Could not start the microphone');
    toast(msg, 6000);
    return false;
  }
}

function stopListening() {
  if (teach.unsub) teach.unsub();
  teach.unsub = null;
  teach.mode = 'idle';
  setRecordButton('● Record');
  $('testBtn').textContent = '👂 Test';
  $('levelBar').style.width = '0';
}

function showLevel(f) {
  // Map roughly -70..-10 dBFS onto the bar.
  const pct = Math.max(0, Math.min(100, (f.level + 70) / 60 * 100));
  $('levelBar').style.width = pct + '%';
}

$('recordBtn').onclick = async () => {
  if (teach.mode === 'recording') return finishRecording();
  if (teach.mode === 'testing') stopListening();
  if (!(await ensureMic())) return;
  teach.mode = 'recording';
  teach.frames = [];
  setRecordButton('■ Stop', true);
  $('testBox').hidden = true;
  $('teachState').textContent = 'Listening… play the last bar now';
  $('teachState').classList.remove('ok');
  teach.unsub = Listener.subscribe((f) => {
    teach.frames.push(f);
    showLevel(f);
    drawChroma($('chromaCanvas'), teach.frames.map(x => (x.silent ? null : x.chroma)), { columns: 120 });
    if (teach.frames.length * HOP_MS > 20000) finishRecording(); // stop by itself after 20 s
  });
};

async function finishRecording() {
  const frames = teach.frames;
  stopListening();
  const cue = makeCue(frames);
  if (!cue) {
    toast("I didn't hear enough. Check that the mic is allowed and play a little louder or longer.", 5000);
    renderTeach();
    return;
  }
  const page = teach.pages[teach.index];
  page.cue = cue;
  await dbPut('pages', page);
  renderTeach();
  toast(`Saved (${(cue.frames.length * cue.hop).toFixed(1)} s). Tap Test and play it again to check.`, 4000);
}

$('testBtn').onclick = async () => {
  if (teach.mode === 'testing') return stopListening();
  if (teach.mode === 'recording') return;
  const page = teach.pages[teach.index];
  if (!page.cue) return;
  if (!(await ensureMic())) return;
  teach.mode = 'testing';
  teach.matcher = new CueMatcher(page.cue);
  teach.liveFrames = [];
  $('testBtn').textContent = '■ Stop test';
  $('testBox').hidden = false;
  let hitUntil = 0, peak = 0, peakAt = 0;
  teach.unsub = Listener.subscribe((f) => {
    showLevel(f);
    const r = teach.matcher.push(f);
    // Show the best score of the last second so the bar is readable.
    if (r.similarity >= peak || f.t - peakAt > 1000) { peak = r.similarity; peakAt = f.t; }
    $('matchBar').style.width = `${Math.round(peak * 100)}%`;
    $('matchBar').style.background = peak >= MATCH_THRESHOLD ? '#2b8a3e' : '#868e96';
    $('matchPct').textContent = `${Math.round(peak * 100)}%`;
    if (r.similarity >= MATCH_THRESHOLD) hitUntil = f.t + 2000;
    $('matchHit').classList.toggle('on', f.t < hitUntil);
    teach.liveFrames.push(f.silent ? null : f.chroma);
    if (teach.liveFrames.length > 120) teach.liveFrames.shift();
  });
};

$('forgetBtn').onclick = async () => {
  stopListening();
  const page = teach.pages[teach.index];
  delete page.cue;
  await dbPut('pages', page);
  renderTeach();
};

$('teachPrev').onclick = () => { stopListening(); teach.index = Math.max(0, teach.index - 1); renderTeach(); };
$('teachNext').onclick = () => { stopListening(); teach.index = Math.min(teachTurnCount() - 1, teach.index + 1); renderTeach(); };
$('teachBack').onclick = async () => {
  stopListening();
  Listener.stop();
  if (teach.url) URL.revokeObjectURL(teach.url);
  teach.url = null;
  show('editor');
  await renderPages();
};
$('teachBtn').onclick = () => openTeach();
window.addEventListener('resize', () => {
  if ($('teach').hidden) return;
  const page = teach.pages[teach.index];
  if (teach.mode === 'idle' && page) drawChroma($('chromaCanvas'), page.cue ? page.cue.frames : []);
});

// ---------------------------------------------------------------- viewer
const viewer = {
  pages: [],        // page records
  urls: new Map(),  // page index -> object URL
  index: 0,         // first page on screen
  wakeLock: null,
  chromeTimer: null,
  badgeTimer: null,
};

// Two pages side by side when the screen is wider than tall, unless the user chose otherwise.
function layoutMode() {
  try { return localStorage.getItem('layout') || 'auto'; } catch { return 'auto'; }
}
function perScreen() {
  const mode = layoutMode();
  if (mode === 'one') return 1;
  if (mode === 'two') return 2;
  return window.innerWidth > window.innerHeight * 1.15 ? 2 : 1;
}

function urlFor(i) {
  if (i < 0 || i >= viewer.pages.length) return null;
  if (!viewer.urls.has(i)) viewer.urls.set(i, URL.createObjectURL(viewer.pages[i].blob));
  return viewer.urls.get(i);
}

async function openViewer(startIndex = 0) {
  await savePiece();
  viewer.pages = [];
  for (const id of currentPiece.pageIds) {
    const p = await dbGet('pages', id);
    if (p) viewer.pages.push(p);
  }
  if (!viewer.pages.length) return;
  viewer.index = Math.min(startIndex, viewer.pages.length - 1);
  $('viewerTitle').textContent = currentPiece.title || 'Untitled piece';
  show('viewer');
  enterFullscreen();
  keepAwake();
  renderSpread();
  flashChrome();
  startAutoTurn();
}

function renderSpread() {
  const n = perScreen();
  // In two-page mode keep spreads aligned to even pages: 1-2, 3-4, ...
  if (n === 2) viewer.index -= viewer.index % 2;
  const spread = $('spread');
  spread.textContent = '';
  for (let k = 0; k < n; k++) {
    const i = viewer.index + k;
    const url = urlFor(i);
    if (url) {
      const img = document.createElement('img');
      img.src = url;
      img.alt = `Page ${i + 1}`;
      spread.append(img);
    } else if (n === 2) {
      const end = document.createElement('div');
      end.className = 'end-card';
      end.textContent = 'End';
      spread.append(end);
    }
  }
  // Warm up the next and previous spreads so turns are instant.
  for (const i of [viewer.index + n, viewer.index + n + 1, viewer.index - n]) {
    const u = urlFor(i);
    if (u) { const im = new Image(); im.src = u; }
  }
  const last = Math.min(viewer.index + n, viewer.pages.length);
  $('pageBadge').textContent = n === 2 && last > viewer.index + 1
    ? `${viewer.index + 1}–${last} / ${viewer.pages.length}`
    : `${viewer.index + 1} / ${viewer.pages.length}`;
  flashBadge();
  armCue();
}

function turn(dir, auto = false) {
  if (!auto) auto_.cooldownUntil = performance.now() + (dir < 0 ? 4000 : 1500);
  const n = perScreen();
  const next = viewer.index + dir * n;
  if (next < 0 || next >= viewer.pages.length) {
    flashBadge();
    return;
  }
  const spread = $('spread');
  spread.classList.add(dir > 0 ? 'turn-next' : 'turn-prev');
  setTimeout(() => {
    viewer.index = next;
    renderSpread();
    spread.classList.remove('turn-next', 'turn-prev');
  }, 110);
}
// Exposed so the auto-flip step can call it later.
window.pageFlipper = { next: () => turn(1), prev: () => turn(-1) };

// ---------------------------------------------------------------- auto-turn
// While playing, listen for the taught last bar of the page on screen (the right-hand page
// in two-page view) and turn when it's heard.
const EARLY_OPTIONS = [
  { sec: 0, label: 'Turn: on last note' },
  { sec: 0.5, label: 'Turn: ½ s early' },
  { sec: 1, label: 'Turn: 1 s early' },
  { sec: 2, label: 'Turn: 2 s early' },
];
const auto_ = { unsub: null, matcher: null, cuePage: -1, cooldownUntil: 0, lastBlink: 0 };

function pref(key, fallback) { try { const v = localStorage.getItem(key); return v === null ? fallback : v; } catch { return fallback; } }
function setPref(key, v) { try { localStorage.setItem(key, v); } catch {} }
const autoEnabled = () => pref('autoTurn', 'on') === 'on';
const earlySec = () => Number(pref('turnEarly', '0.5'));

function updateAutoButtons() {
  $('autoBtn').textContent = `🎹 Auto-turn: ${autoEnabled() ? 'on' : 'off'}`;
  const opt = EARLY_OPTIONS.find(o => o.sec === earlySec()) || EARLY_OPTIONS[1];
  $('earlyBtn').textContent = opt.label;
  $('earlyBtn').hidden = !autoEnabled();
}

async function startAutoTurn() {
  updateAutoButtons();
  const anyCue = viewer.pages.some(p => p.cue);
  if (!autoEnabled() || !anyCue) { $('autoDot').hidden = true; return; }
  try {
    await Listener.start();
  } catch (err) {
    console.error(err);
    toast(err && err.name === 'NotAllowedError'
      ? 'Auto-turn needs the microphone. Allow it for this site in Chrome, then reopen the piece.'
      : 'Auto-turn could not start the microphone.', 6000);
    $('autoDot').hidden = true;
    return;
  }
  if ($('viewer').hidden) { Listener.stop(); return; } // closed while we were starting
  if (auto_.unsub) auto_.unsub();
  auto_.unsub = Listener.subscribe(onAutoFrame);
  $('autoDot').hidden = false;
  armCue();
}

function stopAutoTurn() {
  if (auto_.unsub) auto_.unsub();
  auto_.unsub = null;
  auto_.matcher = null;
  auto_.cuePage = -1;
  Listener.stop();
  $('autoDot').hidden = true;
}

// Pick the cue for the page now on screen.
function armCue() {
  if (!auto_.unsub) return;
  const n = perScreen();
  const pageIdx = Math.min(viewer.index + n - 1, viewer.pages.length - 1);
  const hasNext = viewer.index + n < viewer.pages.length;
  const page = viewer.pages[pageIdx];
  const dot = $('autoDot');
  if (hasNext && page && page.cue) {
    if (auto_.cuePage !== pageIdx || !auto_.matcher) {
      auto_.matcher = new CueMatcher(page.cue, earlySec());
      auto_.cuePage = pageIdx;
    }
    dot.classList.remove('quiet');
    $('autoText').textContent = `Listening for end of page ${pageIdx + 1}`;
  } else {
    auto_.matcher = null;
    auto_.cuePage = -1;
    dot.classList.add('quiet');
    $('autoText').textContent = hasNext ? `Page ${pageIdx + 1} not taught` : 'Last page';
  }
}

function onAutoFrame(f) {
  const dot = $('autoDot');
  // The dot pulses with your playing so you can see the mic is hearing you.
  const s = 1 + Math.max(0, Math.min(1, (f.level + 60) / 40)) * 0.6;
  dot.querySelector('.dot').style.transform = `scale(${s.toFixed(2)})`;
  dot.classList.toggle('on', !!auto_.matcher);
  if (!auto_.matcher) return;
  const r = auto_.matcher.push(f);
  dot.classList.toggle('near', r.similarity > MATCH_THRESHOLD * 0.8);
  if (r.similarity >= MATCH_THRESHOLD && f.t > auto_.cooldownUntil) {
    auto_.cooldownUntil = f.t + 2500;
    auto_.matcher = null;
    auto_.cuePage = -1;
    turn(1, true);
  }
}

$('autoBtn').onclick = () => {
  Listener.unlock();
  setPref('autoTurn', autoEnabled() ? 'off' : 'on');
  if (autoEnabled()) {
    if (!viewer.pages.some(p => p.cue)) toast('No page turns taught yet. Use 🎹 Teach turns on the piece page first.', 4000);
    startAutoTurn();
  } else {
    stopAutoTurn();
    updateAutoButtons();
  }
  flashChrome();
};
$('earlyBtn').onclick = () => {
  const i = EARLY_OPTIONS.findIndex(o => o.sec === earlySec());
  const next = EARLY_OPTIONS[(i + 1) % EARLY_OPTIONS.length];
  setPref('turnEarly', String(next.sec));
  updateAutoButtons();
  auto_.cuePage = -1; // rebuild the matcher with the new timing
  armCue();
  flashChrome();
};

function flashBadge() {
  const b = $('pageBadge');
  b.classList.remove('fade');
  clearTimeout(viewer.badgeTimer);
  viewer.badgeTimer = setTimeout(() => b.classList.add('fade'), 1500);
}
function flashChrome() {
  const c = $('viewerChrome');
  c.classList.remove('hide');
  clearTimeout(viewer.chromeTimer);
  viewer.chromeTimer = setTimeout(() => c.classList.add('hide'), 3000);
}

async function closeViewer() {
  stopAutoTurn();
  releaseWake();
  exitFullscreen();
  viewer.urls.forEach(u => URL.revokeObjectURL(u));
  viewer.urls.clear();
  viewer.pages = [];
  show('editor');
  await renderPages();
}

// Taps: left third goes back, the rest goes forward, a tap near the top shows the controls.
// Swipes: left = next, right = previous.
let touchStart = null;
const viewerEl = $('viewer');
viewerEl.addEventListener('pointerdown', (e) => {
  if (e.target.closest('.viewer-chrome')) return;
  touchStart = { x: e.clientX, y: e.clientY, t: Date.now() };
});
viewerEl.addEventListener('pointerup', (e) => {
  if (!touchStart || e.target.closest('.viewer-chrome')) { touchStart = null; return; }
  const dx = e.clientX - touchStart.x, dy = e.clientY - touchStart.y;
  touchStart = null;
  if (Math.abs(dx) > 40 && Math.abs(dx) > Math.abs(dy)) {
    turn(dx < 0 ? 1 : -1);
    return;
  }
  if (Math.abs(dx) > 15 || Math.abs(dy) > 15) return; // a drag that wasn't a swipe
  const w = window.innerWidth, h = window.innerHeight;
  if (e.clientY < h * 0.12) { flashChrome(); return; }
  turn(e.clientX < w / 3 ? -1 : 1);
});
viewerEl.addEventListener('pointercancel', () => { touchStart = null; });

// Bluetooth page-turner pedals and keyboards send these keys.
document.addEventListener('keydown', (e) => {
  if ($('viewer').hidden) return;
  const next = ['ArrowRight', 'ArrowDown', 'PageDown', ' ', 'Enter'];
  const prev = ['ArrowLeft', 'ArrowUp', 'PageUp', 'Backspace'];
  if (next.includes(e.key)) { e.preventDefault(); turn(1); }
  else if (prev.includes(e.key)) { e.preventDefault(); turn(-1); }
  else if (e.key === 'Escape') closeViewer();
});

$('viewerClose').onclick = closeViewer;
$('layoutBtn').onclick = () => {
  const order = ['auto', 'one', 'two'];
  const label = { auto: 'Layout: automatic (2 pages in landscape)', one: 'Layout: always 1 page', two: 'Layout: always 2 pages' };
  const next = order[(order.indexOf(layoutMode()) + 1) % order.length];
  try { localStorage.setItem('layout', next); } catch {}
  toast(label[next]);
  renderSpread();
  flashChrome();
};
window.addEventListener('resize', () => { if (!$('viewer').hidden) renderSpread(); });

// Keep the screen on while playing.
async function keepAwake() {
  try {
    if ('wakeLock' in navigator) viewer.wakeLock = await navigator.wakeLock.request('screen');
  } catch { /* not allowed right now; fine */ }
}
function releaseWake() {
  if (viewer.wakeLock) viewer.wakeLock.release().catch(() => {});
  viewer.wakeLock = null;
}
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible' && !$('viewer').hidden) keepAwake();
});

function enterFullscreen() {
  const el = document.documentElement;
  try {
    if (el.requestFullscreen) el.requestFullscreen().catch(() => {});
    else if (el.webkitRequestFullscreen) el.webkitRequestFullscreen();
  } catch { /* not supported in this browser */ }
}
function exitFullscreen() {
  try {
    if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
    else if (document.webkitFullscreenElement) document.webkitExitFullscreen();
  } catch {}
}

// ---------------------------------------------------------------- start
if ('serviceWorker' in navigator && location.protocol === 'https:') {
  navigator.serviceWorker.register('sw.js').catch(() => {});
}
goLibrary().catch(err => {
  console.error(err);
  toast('Storage is not available in this browser window');
});
