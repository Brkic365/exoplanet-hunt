// Small canvas plotting helpers shared by every chart on the page (no dependencies).

export const token = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

export const reducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/** Fit a canvas to its CSS box at device pixel ratio; returns {ctx, w, h} in CSS px. */
export function fit(canvas) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const { width, height } = canvas.getBoundingClientRect();
  const w = Math.max(1, Math.round(width)), h = Math.max(1, Math.round(height));
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

export function scale(d0, d1, r0, r1, log = false) {
  if (log) {
    const a = Math.log10(d0), b = Math.log10(d1);
    const f = (v) => r0 + ((Math.log10(v) - a) / (b - a)) * (r1 - r0);
    f.invert = (p) => 10 ** (a + ((p - r0) / (r1 - r0)) * (b - a));
    f.domain = [d0, d1]; f.log = true;
    return f;
  }
  const f = (v) => r0 + ((v - d0) / (d1 - d0 || 1)) * (r1 - r0);
  f.invert = (p) => d0 + ((p - r0) / (r1 - r0)) * (d1 - d0);
  f.domain = [d0, d1];
  return f;
}

export function niceTicks(lo, hi, n = 5) {
  const span = hi - lo;
  if (!(span > 0)) return [lo];
  const raw = span / n, mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

export function logTicks(lo, hi) {
  const out = [];
  for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) {
    for (const m of [1, 2, 5]) {
      const v = m * 10 ** e;
      if (v >= lo * 0.999 && v <= hi * 1.001) out.push(v);
    }
  }
  return out;
}

export const fmt = (v, d = 0) => (v == null || !isFinite(v) ? "–" : Number(v).toLocaleString("en-US", { maximumFractionDigits: d, minimumFractionDigits: d }));

/** Axes with a faint grid. frame = {l, r, t, b} plot rectangle in CSS px. */
export function axes(ctx, frame, x, y, { xticks, yticks, xlabel, ylabel, xfmt = (v) => fmt(v, 1), yfmt = (v) => fmt(v) } = {}) {
  const line = token("--line"), muted = token("--muted");
  ctx.save();
  const base = ctx.globalAlpha;
  ctx.font = `11px ${token("--mono")}`;
  ctx.fillStyle = muted;
  ctx.strokeStyle = line;
  ctx.lineWidth = 1;
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (const v of xticks || []) {
    const px = Math.round(x(v)) + 0.5;
    if (px < frame.l - 1 || px > frame.r + 1) continue;
    ctx.globalAlpha = base * 0.6;
    ctx.beginPath(); ctx.moveTo(px, frame.t); ctx.lineTo(px, frame.b); ctx.stroke();
    ctx.globalAlpha = base;
    ctx.fillText(xfmt(v), px, frame.b + 6);
  }
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (const v of yticks || []) {
    const py = Math.round(y(v)) + 0.5;
    if (py < frame.t - 1 || py > frame.b + 1) continue;
    ctx.globalAlpha = base * 0.6;
    ctx.beginPath(); ctx.moveTo(frame.l, py); ctx.lineTo(frame.r, py); ctx.stroke();
    ctx.globalAlpha = base;
    ctx.fillText(yfmt(v), frame.l - 6, py);
  }
  if (xlabel) {
    ctx.textAlign = "right"; ctx.textBaseline = "bottom";
    ctx.fillText(xlabel, frame.r, frame.b - 4);
  }
  if (ylabel) {
    ctx.textAlign = "left"; ctx.textBaseline = "top";
    ctx.fillText(ylabel, frame.l + 6, frame.t + 4);
  }
  ctx.restore();
}

/** Robust y-range: percentiles of finite values, padded. */
export function yrange(arrs, lo = 0.5, hi = 99.5, pad = 0.08) {
  const v = [];
  for (const a of arrs) for (const x of a) if (x != null && isFinite(x)) v.push(x);
  if (!v.length) return [0, 1];
  v.sort((a, b) => a - b);
  const q = (p) => v[Math.min(v.length - 1, Math.max(0, Math.round((p / 100) * (v.length - 1))))];
  let a = q(lo), b = q(hi);
  if (a === b) { a -= 1; b += 1; }
  const s = (b - a) * pad;
  return [a - s, b + s];
}

export const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);
export const lerp = (a, b, t) => a + (b - a) * t;

/** Score colour ramp (low = cool slate, high = amber); returns [r, g, b] in 0..1. */
export function scoreColor(s) {
  const stops = [[0, [0.23, 0.29, 0.48]], [0.5, [0.5, 0.64, 0.91]], [1, [0.95, 0.71, 0.29]]];
  s = Math.max(0, Math.min(1, s));
  for (let i = 1; i < stops.length; i++) {
    if (s <= stops[i][0]) {
      const [a, ca] = stops[i - 1], [b, cb] = stops[i];
      const t = (s - a) / (b - a);
      return ca.map((c, k) => c + (cb[k] - c) * t);
    }
  }
  return stops[stops.length - 1][1];
}

export const rgb = (c, a = 1) => `rgba(${Math.round(c[0] * 255)},${Math.round(c[1] * 255)},${Math.round(c[2] * 255)},${a})`;

/** Redraw on resize. */
export function onResize(el, fn) {
  const ro = new ResizeObserver(() => fn());
  ro.observe(el);
  return ro;
}
