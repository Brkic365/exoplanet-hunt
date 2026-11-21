// Injection-recovery map: 5,000 fake planets planted in real sector-75 light curves.
// Hover or tap a cell to see one injected planet of that size and period that the
// pipeline found, and one it missed (folded at the injected period).
import { axes, fit, fmt, niceTicks, onResize, scale, scoreColor, rgb, token, yrange } from "./plot.js";

export function drawFold(canvas, ex, color) {
  const { ctx, w, h } = fit(canvas);
  ctx.clearRect(0, 0, w, h);
  if (!ex) {
    ctx.fillStyle = token("--muted");
    ctx.font = `12px ${token("--body")}`;
    ctx.textAlign = "center";
    ctx.fillText("No injection in this cell ended up here", w / 2, h / 2);
    return;
  }
  const fr = { l: 50, r: w - 10, t: 10, b: h - 24 };
  const hw = Math.max(...ex.x.map(Math.abs));
  const yr = yrange([ex.y], 1, 99, 0.1);
  yr[0] = Math.min(yr[0], -ex.depth_ppm * 1.3);
  const x = scale(-hw, hw, fr.l, fr.r), y = scale(yr[0], yr[1], fr.b, fr.t);
  // injected transit window
  ctx.fillStyle = token("--line");
  ctx.fillRect(x(-ex.dur_h / 2), fr.t, x(ex.dur_h / 2) - x(-ex.dur_h / 2), fr.b - fr.t);
  axes(ctx, fr, x, y, { xticks: niceTicks(-hw, hw, 4), yticks: niceTicks(yr[0], yr[1], 3), xlabel: "hours", xfmt: (v) => fmt(v) });
  ctx.save();
  ctx.beginPath(); ctx.rect(fr.l, fr.t, fr.r - fr.l, fr.b - fr.t); ctx.clip();
  ctx.fillStyle = token("--data"); ctx.globalAlpha = 0.45;
  ex.x.forEach((xv, i) => ex.y[i] != null && ctx.fillRect(x(xv) - 0.8, y(ex.y[i]) - 0.8, 1.6, 1.6));
  ctx.globalAlpha = 1;
  // injected depth
  ctx.setLineDash([3, 3]); ctx.strokeStyle = token("--muted");
  ctx.beginPath(); ctx.moveTo(fr.l, y(-ex.depth_ppm)); ctx.lineTo(fr.r, y(-ex.depth_ppm)); ctx.stroke();
  ctx.setLineDash([]);
  ctx.strokeStyle = color; ctx.lineWidth = 2;
  ctx.beginPath();
  ex.bx.forEach((bx, i) => (i ? ctx.lineTo(x(bx), y(ex.by[i])) : ctx.moveTo(x(bx), y(ex.by[i]))));
  ctx.stroke();
  ctx.restore();
}

export async function initHeatmap(root) {
  const data = await fetch("data/heatmap.json").then((r) => r.json());
  const canvas = root.querySelector(".heat canvas");
  const pEdges = [...new Set(data.cells.flatMap((c) => c.p))].sort((a, b) => a - b);
  const rEdges = [...new Set(data.cells.flatMap((c) => c.r))].sort((a, b) => a - b);
  const nx = pEdges.length - 1, ny = rEdges.length - 1;
  const cellAt = (i, j) => data.cells.find((c) => c.p[0] === pEdges[i] && c.r[0] === rEdges[j]);
  let sel = { i: 1, j: 6 }, hov = null;
  const foundC = root.querySelector(".ex-found canvas"), missC = root.querySelector(".ex-missed canvas");
  const foundH = root.querySelector(".ex-found .mini-head"), missH = root.querySelector(".ex-missed .mini-head");
  const cellH = root.querySelector(".cell-head");

  let geo = null;
  function draw() {
    const { ctx, w, h } = fit(canvas);
    ctx.clearRect(0, 0, w, h);
    const fr = { l: 44, r: w - 18, t: 8, b: h - 40 };
    const cw = (fr.r - fr.l) / nx, ch = (fr.b - fr.t) / ny;
    geo = { fr, cw, ch };
    ctx.font = `11px ${token("--mono")}`;
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) {
      const c = cellAt(i, j);
      if (!c) continue;
      const x0 = fr.l + i * cw, y0 = fr.b - (j + 1) * ch;
      ctx.fillStyle = rgb(scoreColor(c.frac), 0.12 + 0.88 * c.frac);
      ctx.fillRect(x0 + 1, y0 + 1, cw - 2, ch - 2);
      ctx.fillStyle = c.frac > 0.55 ? "#0b0e14" : token("--fg");
      if (cw > 34) ctx.fillText(`${Math.round(c.frac * 100)}`, x0 + cw / 2, y0 + ch / 2);
    }
    const ring = (s, color, lw) => {
      if (!s) return;
      ctx.strokeStyle = color; ctx.lineWidth = lw;
      ctx.strokeRect(fr.l + s.i * cw + 1, fr.b - (s.j + 1) * ch + 1, cw - 2, ch - 2);
    };
    ring(hov, token("--muted"), 1);
    ring(sel, token("--fg"), 2);
    // axis labels at bin edges
    ctx.fillStyle = token("--muted");
    ctx.textBaseline = "top";
    pEdges.forEach((e, i) => ctx.fillText(fmt(e, e < 1 ? 1 : e % 1 ? 1 : 0), fr.l + i * cw, fr.b + 6));
    ctx.fillText("orbital period (days)", (fr.l + fr.r) / 2, fr.b + 22);
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    rEdges.forEach((e, j) => ctx.fillText(fmt(e, e % 1 ? 2 : 0).replace(/0$/, ""), fr.l - 6, fr.b - j * ch));
    ctx.save();
    ctx.translate(10, (fr.t + fr.b) / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = "center";
    ctx.fillText("planet radius (R⊕)", 0, 0);
    ctx.restore();
  }

  function pick(e) {
    if (!geo) return null;
    const r = canvas.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top, { fr, cw, ch } = geo;
    const i = Math.floor((mx - fr.l) / cw), j = Math.floor((fr.b - my) / ch);
    return i >= 0 && i < nx && j >= 0 && j < ny ? { i, j } : null;
  }

  function select(s) {
    if (!s) return;
    sel = s;
    const c = cellAt(s.i, s.j);
    draw();
    if (!c) return;
    cellH.innerHTML = `<b>${fmt(c.p[0], 1)}–${fmt(c.p[1], 1)} d · ${c.r[0]}–${c.r[1]} R⊕</b> · ${c.found} of ${c.n} recovered (${Math.round(c.frac * 100)}%)`;
    const head = (ex, kind) => ex
      ? `<span><b class="${kind}">${kind}</b> · ${fmt(ex.rp_re, 1)} R⊕, P ${fmt(ex.period, 2)} d, ${fmt(ex.depth_ppm)} ppm</span><span>expected SNR ${fmt(ex.snr, 1)} · TIC ${ex.tic}, T ${fmt(ex.tmag, 1)}</span>`
      : `<span><b class="${kind}">${kind}</b></span><span>none in this cell</span>`;
    foundH.innerHTML = head(c.examples.found, "found");
    missH.innerHTML = head(c.examples.missed, "missed");
    drawFold(foundC, c.examples.found, token("--known"));
    drawFold(missC, c.examples.missed, token("--fp"));
  }

  canvas.addEventListener("pointermove", (e) => { const s = pick(e); if (JSON.stringify(s) !== JSON.stringify(hov)) { hov = s; if (e.pointerType === "mouse" && s) select(s); else draw(); } });
  canvas.addEventListener("pointerleave", () => { hov = null; draw(); });
  canvas.addEventListener("click", (e) => select(pick(e)));
  canvas.addEventListener("keydown", (e) => {
    const d = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, 1], ArrowDown: [0, -1] }[e.key];
    if (!d) return;
    e.preventDefault();
    select({ i: Math.max(0, Math.min(nx - 1, sel.i + d[0])), j: Math.max(0, Math.min(ny - 1, sel.j + d[1])) });
  });
  onResize(canvas, () => select(sel));
  onResize(foundC, () => select(sel));
  select(sel);
  return data;
}
