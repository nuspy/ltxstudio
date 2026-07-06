// Rilevatore firma di provenienza ATF — port JS ON-DEVICE di core/stego.py.
// I carrier ±1 derivano dallo stesso stream SHA256 del desktop → rilevamento identico, senza rete.
// Input: ImageData (da <canvas>.getImageData). Output: { found, z, text }.

const KEY_STR = 'AIToolsForge::provenance::v1::do-not-share';
const GRID = 8;
const CANON = 80;
const NONCE_BANK = 24;
const Z_THRESH = 5.0;
const PAYLOAD_TEXT_BYTES = 5;
const PAYLOAD_BITS = (PAYLOAD_TEXT_BYTES + 1) * 8;

const enc = new TextEncoder();
const KEY = enc.encode(KEY_STR);

// ---- maschera banda medio-frequenza (r=u+v in [4,26], esclusi DC) ----
const MASK: number[] = (() => {
  const idx: number[] = [];
  for (let u = 0; u < CANON; u++)
    for (let v = 0; v < CANON; v++) {
      const r = u + v;
      if (r >= 4 && r <= 26 && !(u < 2 && v < 2)) idx.push(u * CANON + v);
    }
  return idx;
})();
const N_COEFF = MASK.length;

// ---- stream SHA256 → n valori ±1 (MSB-first per byte), identico a Python _sha_bits ----
async function shaBits(keyMaterial: Uint8Array, n: number): Promise<Float32Array> {
  const need = Math.ceil(n / 8);
  const buf = new Uint8Array(Math.ceil(need / 32) * 32);
  let off = 0;
  let counter = 0;
  while (off < need) {
    const cb = new Uint8Array(4);
    new DataView(cb.buffer).setUint32(0, counter, false);
    const m = new Uint8Array(keyMaterial.length + 4);
    m.set(keyMaterial, 0);
    m.set(cb, keyMaterial.length);
    const d = new Uint8Array(await crypto.subtle.digest('SHA-256', m));
    buf.set(d, off);
    off += 32;
    counter++;
  }
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const bit = (buf[i >> 3] >> (7 - (i & 7))) & 1;
    out[i] = bit * 2 - 1;
  }
  return out;
}

function km(label: string, nonce: number, tile: number): Uint8Array {
  const l = enc.encode(label);
  const b = new Uint8Array(KEY.length + l.length + 6);
  b.set(KEY, 0);
  b.set(l, KEY.length);
  const dv = new DataView(b.buffer);
  dv.setUint32(KEY.length + l.length, nonce, false);
  dv.setUint16(KEY.length + l.length + 4, tile, false);
  return b;
}

// carrier presence per tile: matrice [NONCE_BANK × N_COEFF], precalcolata
let CMAT: Float32Array[][] | null = null; // CMAT[tile][nonce] = Float32Array(N_COEFF)

export async function init(): Promise<void> {
  if (CMAT) return;
  CMAT = [];
  for (let tile = 0; tile < GRID * GRID; tile++) {
    const rows: Float32Array[] = [];
    for (let n = 0; n < NONCE_BANK; n++) rows.push(await shaBits(km('C', n, tile), N_COEFF));
    CMAT.push(rows);
  }
}

async function payloadMatrix(nonce: number, tile: number): Promise<Float32Array[]> {
  const flat = await shaBits(km('P', nonce, tile), PAYLOAD_BITS * N_COEFF);
  const rows: Float32Array[] = [];
  for (let b = 0; b < PAYLOAD_BITS; b++) rows.push(flat.subarray(b * N_COEFF, (b + 1) * N_COEFF));
  return rows;
}

// ---- CRC8 + bit→testo (identico a Python) ----
function crc8(data: Uint8Array): number {
  let c = 0;
  for (const b of data) {
    c ^= b;
    for (let i = 0; i < 8; i++) c = c & 0x80 ? ((c << 1) ^ 0x07) & 0xff : (c << 1) & 0xff;
  }
  return c;
}
function bitsToText(signs: Float32Array): string | null {
  const bytes = new Uint8Array(PAYLOAD_TEXT_BYTES + 1);
  for (let i = 0; i < bytes.length; i++) {
    let v = 0;
    for (let b = 0; b < 8; b++) v = (v << 1) | (signs[i * 8 + b] > 0 ? 1 : 0);
    bytes[i] = v;
  }
  if (crc8(bytes.subarray(0, PAYLOAD_TEXT_BYTES)) !== bytes[PAYLOAD_TEXT_BYTES]) return null;
  let end = PAYLOAD_TEXT_BYTES;
  while (end > 0 && bytes[end - 1] === 0) end--;
  try {
    return new TextDecoder().decode(bytes.subarray(0, end));
  } catch {
    return null;
  }
}

// ---- primitive immagine ----
function luma(img: ImageData): { y: Float32Array; w: number; h: number } {
  const { data, width: w, height: h } = img;
  const y = new Float32Array(w * h);
  for (let i = 0, p = 0; i < y.length; i++, p += 4) y[i] = data[p] * 0.299 + data[p + 1] * 0.587 + data[p + 2] * 0.114;
  return { y, w, h };
}
function resizeTile(y: Float32Array, w: number, x0: number, y0: number, tw: number, th: number, out: Float32Array): void {
  // bilinear tile (tw×th) → CANON×CANON in `out`
  for (let j = 0; j < CANON; j++) {
    const sy = (j / CANON) * th;
    const jy = Math.min(th - 1, sy | 0);
    for (let i = 0; i < CANON; i++) {
      const sx = (i / CANON) * tw;
      const ix = Math.min(tw - 1, sx | 0);
      out[j * CANON + i] = y[(y0 + jy) * w + (x0 + ix)];
    }
  }
}

// DCT-II ortonormale via matrice (== cv2.dct)
const DCTC = (() => {
  const C = new Float32Array(CANON * CANON);
  for (let k = 0; k < CANON; k++)
    for (let x = 0; x < CANON; x++) {
      C[k * CANON + x] = k === 0 ? Math.sqrt(1 / CANON) : Math.sqrt(2 / CANON) * Math.cos((Math.PI * (2 * x + 1) * k) / (2 * CANON));
    }
  return C;
})();
function dctMask(tile: Float32Array): Float32Array {
  // tmp = C @ tile ; D = tmp @ C^T ; poi prende i coeff della maschera
  const tmp = new Float32Array(CANON * CANON);
  for (let r = 0; r < CANON; r++)
    for (let c = 0; c < CANON; c++) {
      let s = 0;
      for (let k = 0; k < CANON; k++) s += DCTC[r * CANON + k] * tile[k * CANON + c];
      tmp[r * CANON + c] = s;
    }
  const D = new Float32Array(N_COEFF);
  let mean = 0;
  for (let mi = 0; mi < N_COEFF; mi++) {
    const idx = MASK[mi];
    const r = (idx / CANON) | 0;
    const c = idx % CANON;
    let s = 0;
    for (let k = 0; k < CANON; k++) s += tmp[r * CANON + k] * DCTC[c * CANON + k];
    D[mi] = s;
    mean += s;
  }
  mean /= N_COEFF;
  for (let mi = 0; mi < N_COEFF; mi++) D[mi] -= mean;
  return D;
}

function* tiles(w: number, h: number) {
  const ys: number[] = [];
  const xs: number[] = [];
  for (let i = 0; i <= GRID; i++) {
    ys.push(Math.round((i * h) / GRID));
    xs.push(Math.round((i * w) / GRID));
  }
  let idx = 0;
  for (let a = 0; a < GRID; a++)
    for (let b = 0; b < GRID; b++) yield { idx: idx++, y0: ys[a], y1: ys[a + 1], x0: xs[b], x1: xs[b + 1] };
}

// ---- rilevamento ----
export type Detection = { found: boolean; z: number; text: string | null };

export async function detect(img: ImageData): Promise<Detection> {
  await init();
  const { y, w, h } = luma(img);
  const corr = new Float64Array(NONCE_BANK);
  const Ds: Float32Array[] = [];
  const tileIdx: number[] = [];
  let energy = 0;
  let ncoef = 0;
  const buf = new Float32Array(CANON * CANON);
  for (const t of tiles(w, h)) {
    const tw = t.x1 - t.x0;
    const th = t.y1 - t.y0;
    if (tw < 8 || th < 8) continue;
    resizeTile(y, w, t.x0, t.y0, tw, th, buf);
    const D = dctMask(buf);
    Ds.push(D);
    tileIdx.push(t.idx);
    for (let mi = 0; mi < N_COEFF; mi++) energy += D[mi] * D[mi];
    ncoef += N_COEFF;
    const cm = CMAT![t.idx];
    for (let n = 0; n < NONCE_BANK; n++) {
      const car = cm[n];
      let s = 0;
      for (let mi = 0; mi < N_COEFF; mi++) s += car[mi] * D[mi];
      corr[n] += s;
    }
  }
  if (!ncoef || energy <= 0) return { found: false, z: 0, text: null };
  const sigma = Math.sqrt(energy / ncoef);
  let best = 0;
  let bestZ = -1e9;
  for (let n = 0; n < NONCE_BANK; n++) {
    const z = corr[n] / (sigma * Math.sqrt(ncoef));
    if (z > bestZ) {
      bestZ = z;
      best = n;
    }
  }
  const found = bestZ >= Z_THRESH;
  let text: string | null = null;
  if (found) {
    const pcorr = new Float64Array(PAYLOAD_BITS);
    for (let ti = 0; ti < Ds.length; ti++) {
      const pm = await payloadMatrix(best, tileIdx[ti]);
      const D = Ds[ti];
      for (let b = 0; b < PAYLOAD_BITS; b++) {
        const row = pm[b];
        let s = 0;
        for (let mi = 0; mi < N_COEFF; mi++) s += row[mi] * D[mi];
        pcorr[b] += s;
      }
    }
    const signs = new Float32Array(PAYLOAD_BITS);
    for (let b = 0; b < PAYLOAD_BITS; b++) signs[b] = pcorr[b] > 0 ? 1 : -1;
    text = bitsToText(signs);
  }
  return { found, z: bestZ, text };
}
