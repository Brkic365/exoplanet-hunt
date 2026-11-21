// "One star, five stages": a sticky chart that animates the star's real data from
// raw flux to classifier verdict as the reader scrolls through the steps.
import { axes, ease, fit, fmt, lerp, logTicks, niceTicks, onResize, reducedMotion, scale, token, yrange } from "./plot.js";

const STAGES = ["Ingest", "Detrend", "Search", "Fold", "Verdict"];
const cache = new Map();

async function loadStar(id) {
  if (!cache.has(id)) cache.set(id, fetch(`data/stars/${id}.json`).then((r) => r.json()).then(prepare));
  return cache.get(id);
}

function robustStd(a) {
  const v = a.filter((x) => x != null).sort((p, q) => p - q);
  if (!v.length) return NaN;
  const med = v[v.length >> 1];
  const dev = v.map((x) => Math.abs(x - med)).sort((p, q) => p - q);
  return 1.4826 * dev[dev.length >> 1];
}

function prepare(s) {
  const lc = s.lc, f = s.fold;
  const T = lc.t[lc.t.length - 1];
  const tt0 = f.t0 - lc.t0; // mid-transit in plot time
  // phase (hours from nearest mid-transit) of each binned flat point
  const ph = lc.t.map((t) => {
    if (f.period) return ((((t - tt0) / f.period + 0.5) % 1 + 1) % 1 - 0.5) * f.period * 24;
    return (t - tt0) * 24;
  });
  const hw = Math.max(...f.x.map(Math.abs), ...f.bx.map(Math.abs));
  s._ = {
    T, tt0, ph, hw,
    yRaw: yrange([lc.raw]), yFlat: yrange([lc.flat]), yFold: yrange([f.y, f.by], 1, 99.5, 0.12),
    sRaw: robustStd(lc.raw.map((v, i) => (lc.trend[i] == null || v == null ? null : v - lc.trend[i]))),
    sRawTotal: robustStd(lc.raw), sFlat: robustStd(lc.flat),
  };
  // keep the dip in view even when outliers dominate
  const minBin = Math.min(...f.by.filter((v) => v != null));
  s._.yFold[0] = Math.min(s._.yFold[0], minBin * 1.25);
  return s;
}

export function initWalkthrough(root, index) {
  const canvas = root.querySelector(".plot canvas");
  const verdictEl = root.querySelector(".verdict");
  const title = root.querySelector(".walk-head h3");
  const sub = root.querySelector(".walk-head .sub");
  const bar = root.querySelector(".stages");
  const steps = [...root.querySelectorAll(".step")];
  const picker = root.querySelector(".picker");
  const byId = new Map(index.map((d) => [d.id, d]));

  let star = null, stage = 0, from = 0, p = 1, anim = 0, fade = 1;

  STAGES.forEach((name, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.setAttribute("aria-label", `Stage ${i + 1}: ${name}`);
    b.addEventListener("click", () => steps[i].scrollIntoView({ behavior: reducedMotion() ? "auto" : "smooth", block: "center" }));
    bar.append(b);
  });

  // chips: the walkthrough set + the two CTOI stars + the blend
  const chipIds = index.filter((d) => d.group === "known" || d.group === "ctoi").map((d) => d.id).concat(["tic139526205"]);
  for (const id of chipIds) {
    const d = byId.get(id);
    if (!d) continue;
    const b = document.createElement("button");
    b.type = "button";
    b.className = "chip";
    b.dataset.id = id;
    const color = d.group === "known" ? "var(--known)" : d.group === "ctoi" ? "var(--accent)" : "var(--fp)";
    b.innerHTML = `<span class="dot" style="background:${color}"></span>${d.label}${d.group === "fp" ? " · blend" : d.group === "ctoi" ? " · lone dip" : ""}`;
    b.addEventListener("click", () => show(id));
    picker.append(b);
  }

  function goto(i) {
    if (i === stage) return;
    from = stage; stage = i; p = 0;
    bar.querySelectorAll("button").forEach((b, k) => (k === i ? b.setAttribute("aria-current", "step") : b.removeAttribute("aria-current")));
    steps.forEach((s, k) => s.classList.toggle("on", k === i));
    animate(Math.abs(stage - from) === 1 ? 1100 : 450);
  }

  function animate(ms) {
    cancelAnimationFrame(anim);
    if (reducedMotion()) { p = 1; fade = 1; draw(); return; }
    const t0 = performance.now(), p0 = p, f0 = fade;
    const tick = (now) => {
      const k = Math.min(1, (now - t0) / ms);
      p = lerp(p0, 1, k); fade = lerp(f0, 1, Math.min(1, k * 2));
      draw();
      if (k < 1) anim = requestAnimationFrame(tick);
    };
    anim = requestAnimationFrame(tick);
  }

  async function show(id, { scroll = false } = {}) {
    const s = await loadStar(id);
    star = s;
    picker.querySelectorAll(".chip").forEach((c) => c.setAttribute("aria-pressed", String(c.dataset.id === id)));
    const meta = byId.get(id) || {};
    title.textContent = s.label;
    sub.textContent = `TIC ${s.tic} · sector ${s.sector} · T = ${fmt(s.star.tmag, 1)} · ${fmt(s.star.teff)} K`;
    fillFacts(s, meta);
    fillVerdict(s);
    from = stage; p = 1; fade = 0;
    animate(500);
    if (scroll) root.scrollIntoView({ behavior: reducedMotion() ? "auto" : "smooth", block: "start" });
  }

  function fillFacts(s, meta) {
    const f = s.fold, _ = s._, isSingle = s.method === "single";
    const set = (k, html) => { const el = root.querySelector(`[data-fact="${k}"]`); if (el) el.innerHTML = html; };
    set(0, `${fmt(s.lc.t.length)} ten-minute bins over ${fmt(_.T, 1)} days` +
      `<br>T = ${fmt(s.star.tmag, 2)} · T<sub>eff</sub> ${fmt(s.star.teff)} K · R★ ${fmt(s.star.rstar, 2)} R☉` +
      (s.blurb ? `<br>${s.blurb}` : ""));
    set(1, `scatter ${fmt(_.sRawTotal)} ppm raw → ${fmt(_.sFlat)} ppm detrended`);
    if (isSingle) {
      set(2, `box scan at ${fmt(s.search.dur_h, 1)} h: best event at day ${fmt(s.search.peak.t, 2)}, SES ${fmt(s.search.peak.value, 1)} (threshold 9)`);
    } else {
      const pk = s.search.peak;
      const same = Math.abs(pk.period - s.search.signal_period) / s.search.signal_period < 0.01;
      set(2, `${fmt(s.search.n_periods)} trial periods · peak P = ${fmt(pk.period, 4)} d, SDE ${fmt(pk.sde, 1)}` +
        (same ? "" : `<br>signal kept: P = ${fmt(s.search.signal_period, 4)} d (second search pass)`) +
        (s.known ? `<br>ExoFOP period: ${fmt(s.known.toi_period, 4)} d` : ""));
    }
    set(3, `${isSingle ? "single event" : `P = ${fmt(f.period, 4)} d`} · duration ${fmt(f.dur_h, 1)} h · depth ${fmt(s.verdict.depth_ppm)} ppm` +
      ` · implied radius ${fmt(s.verdict.rp_re, 1)} R⊕` + (s.known?.toi_radius_re ? ` (ExoFOP ${fmt(s.known.toi_radius_re, 1)} R⊕)` : ""));
  }

  function fillVerdict(s) {
    const v = s.verdict, flags = (v.flags || "").split(/[ ,;|]+/).filter(Boolean);
    const check = (label, val, bad, unit = "") =>
      `<div class="check${bad ? " bad" : ""}"><span class="muted">${label}</span><span class="num">${val == null ? "–" : fmt(val, 1) + unit}</span></div>`;
    const single = s.method === "single";
    const split = v.split === "test" ? "held-out test star" : v.split ? `in the ${v.split} set` : "";
    verdictEl.innerHTML = `
      <div class="gauge">
        <div class="gauge-row"><span class="eyebrow">Classifier score · planet-like</span><span class="num">${fmt(v.score, 2)}</span></div>
        <div class="gauge-bar"><i style="width:${Math.round((v.score ?? 0) * 100)}%"></i></div>
        <div class="muted" style="font-size:.82rem">CNN ${fmt(v.p_cnn, 2)} · gradient boosting ${fmt(v.p_gbm, 2)}` +
      (v.baseline != null ? ` · BLS-statistics baseline ${fmt(v.baseline, 2)}` : "") + (split ? ` · ${split}` : "") + `</div>
      </div>
      <div class="checks">
        ${check("SNR", v.snr, false)}
        ${check("odd / even", v.oddeven_sigma, flags.includes("oddeven"), " σ")}
        ${check("secondary", v.sec_snr, flags.includes("secondary"), " SNR")}
        ${check("centroid shift", v.centroid_sigma, flags.includes("centroid"), " σ")}
        ${check("implied radius", v.rp_re, flags.includes("toobig"), " R⊕")}
      </div>
      <p class="hand">${
        s.hand ? `<b>Hand-vetting: ${s.hand.verdict}.</b> ${s.hand.reason}`
        : s.known ? `<b>ExoFOP: ${s.known.disposition === "CP" ? "confirmed planet" : "known planet"}.</b> Recovered blind: the pipeline did not use the TOI catalogue to find it.`
        : ""}${single ? ` <span class="muted">The classifier is trained on periodic TOIs and scores lone dips low; single events are ranked by SNR and judged by hand.</span>` : ""}</p>`;
  }

  // ------------------------------------------------------------- drawing
  function frameRect(w, h) {
    const narrow = w < 420;
    return { l: narrow ? 46 : 58, r: w - 12, t: 14, b: h - (narrow ? 26 : 30) };
  }

  function draw() {
    if (!star) return;
    const { ctx, w, h } = fit(canvas);
    ctx.clearRect(0, 0, w, h);
    const fr = frameRect(w, h);
    const e = ease(Math.max(0, Math.min(1, p)));
    ctx.save();
    ctx.globalAlpha = fade;
    const forward = stage > from;
    if (Math.abs(stage - from) === 1) {
      // adjacent: play scene(max) forward or backward
      const sc = Math.max(stage, from), q = forward ? e : 1 - e;
      scene(ctx, fr, sc, q);
    } else {
      scene(ctx, fr, stage, 1);
    }
    ctx.restore();
    verdictEl.style.opacity = String(stage === 4 ? (Math.abs(stage - from) === 1 ? e : 1) * fade : stage === 3 && from === 4 ? 1 - e : 0);
    verdictEl.style.pointerEvents = stage === 4 ? "auto" : "none";
  }

  const C = () => ({ data: token("--data"), accent: token("--accent"), fg: token("--fg"), muted: token("--muted"), known: token("--known"), fp: token("--fp") });

  function timeAxes(ctx, fr, yr, alpha = 1, ylabel = "flux (ppm)") {
    const x = scale(0, star._.T, fr.l, fr.r), y = scale(yr[0], yr[1], fr.b, fr.t);
    ctx.save(); ctx.globalAlpha *= alpha;
    axes(ctx, fr, x, y, { xticks: niceTicks(0, star._.T, 6), yticks: niceTicks(yr[0], yr[1], 5), xlabel: "days", ylabel, xfmt: (v) => fmt(v), yfmt: (v) => fmt(v) });
    ctx.restore();
    return { x, y };
  }

  function points(ctx, xs, ys, color, r = 1.6, alpha = 0.8) {
    ctx.save();
    ctx.fillStyle = color;
    ctx.globalAlpha *= alpha;
    for (let i = 0; i < xs.length; i++) {
      if (ys[i] == null || xs[i] == null) continue;
      ctx.fillRect(xs[i] - r / 2, ys[i] - r / 2, r, r);
    }
    ctx.restore();
  }

  function clip(ctx, fr) { ctx.beginPath(); ctx.rect(fr.l, fr.t - 2, fr.r - fr.l, fr.b - fr.t + 4); ctx.clip(); }

  function scene(ctx, fr, sc, q) {
    const s = star, lc = s.lc, _ = s._, c = C();
    if (sc === 0) {
      const { x, y } = timeAxes(ctx, fr, _.yRaw);
      ctx.save(); clip(ctx, fr);
      points(ctx, lc.t.map(x), lc.raw.map((v) => (v == null ? null : y(v))), c.data, 1.8, q);
      ctx.restore();
      return;
    }
    if (sc === 1) {
      const d = Math.min(1, q * 2), m = ease(Math.max(0, (q - 0.5) * 2));
      const yr = [lerp(_.yRaw[0], _.yFlat[0], m), lerp(_.yRaw[1], _.yFlat[1], m)];
      const { x, y } = timeAxes(ctx, fr, yr);
      ctx.save(); clip(ctx, fr);
      const ys = lc.raw.map((v, i) => {
        if (v == null) return null;
        const target = lc.flat[i] == null ? v - (lc.trend[i] ?? 0) : lc.flat[i];
        return y(lerp(v, target, m));
      });
      points(ctx, lc.t.map(x), ys, c.data, 1.8, 0.8);
      // trend line, drawn left to right, then flattened with the data
      ctx.strokeStyle = c.accent; ctx.lineWidth = 1.6; ctx.globalAlpha *= 0.95 * (1 - m * 0.6);
      ctx.beginPath();
      const nDraw = Math.floor(lc.t.length * d);
      let pen = false;
      for (let i = 0; i < nDraw; i++) {
        const tv = lc.trend[i];
        if (tv == null || (i > 0 && lc.t[i] - lc.t[i - 1] > 0.1)) { pen = false; if (tv == null) continue; }
        const py = y(lerp(tv, 0, m));
        pen ? ctx.lineTo(x(lc.t[i]), py) : ctx.moveTo(x(lc.t[i]), py);
        pen = true;
      }
      ctx.stroke();
      ctx.restore();
      return;
    }
    if (sc === 2) {
      // flat points fade out; search statistic draws in
      const fadeOut = 1 - Math.min(1, q * 1.6), drawIn = Math.max(0, (q - 0.15) / 0.85);
      if (fadeOut > 0) {
        const { x, y } = timeAxes(ctx, fr, _.yFlat, fadeOut);
        ctx.save(); clip(ctx, fr);
        points(ctx, lc.t.map(x), lc.flat.map((v) => (v == null ? null : y(v))), c.data, 1.8, 0.8 * fadeOut);
        ctx.restore();
      }
      if (drawIn > 0) searchScene(ctx, fr, drawIn);
      return;
    }
    if (sc === 3) {
      // periodogram fades out, flat points come back and fold onto the phase axis
      const back = Math.min(1, q / 0.2), m = ease(Math.max(0, (q - 0.2) / 0.8));
      if (back < 1) searchScene(ctx, fr, 1, 1 - back);
      foldScene(ctx, fr, m, back);
      return;
    }
    if (sc === 4) {
      foldScene(ctx, fr, 1, 1 - 0.88 * q);
    }
  }

  function searchScene(ctx, fr, progress, alpha = 1) {
    const s = star, c = C(), sr = s.search;
    ctx.save(); ctx.globalAlpha *= alpha;
    let x, y;
    if (sr.kind === "periodogram") {
      const lo = sr.x[0], hi = sr.x[sr.x.length - 1];
      const ymax = Math.max(...sr.y) * 1.12, ymin = Math.min(0, Math.min(...sr.y));
      x = scale(lo, hi, fr.l, fr.r, true); y = scale(ymin, ymax, fr.b, fr.t);
      axes(ctx, fr, x, y, { xticks: logTicks(lo, hi), yticks: niceTicks(ymin, ymax, 4), xlabel: "trial period (days)", ylabel: "BLS power (σ)", xfmt: (v) => fmt(v, v < 1 ? 1 : 0), yfmt: (v) => fmt(v) });
    } else {
      const ymax = Math.max(...sr.y, 10) * 1.12, ymin = Math.min(...sr.y, -2);
      x = scale(0, s._.T, fr.l, fr.r); y = scale(ymin, ymax, fr.b, fr.t);
      axes(ctx, fr, x, y, { xticks: niceTicks(0, s._.T, 6), yticks: niceTicks(ymin, ymax, 4), xlabel: "days", ylabel: `single-event SNR, ${fmt(sr.dur_h, 1)} h box` });
      // threshold
      ctx.save(); ctx.setLineDash([4, 4]); ctx.strokeStyle = c.muted; ctx.globalAlpha *= 0.8;
      ctx.beginPath(); ctx.moveTo(fr.l, y(9)); ctx.lineTo(fr.r, y(9)); ctx.stroke(); ctx.restore();
    }
    ctx.save(); clip(ctx, fr);
    ctx.strokeStyle = c.data; ctx.lineWidth = 1.2;
    ctx.beginPath();
    const n = Math.floor(sr.x.length * progress);
    for (let i = 0; i < n; i++) {
      const px = x(sr.x[i]), py = y(sr.y[i] ?? 0);
      if (i && sr.kind === "ses" && sr.x[i] - sr.x[i - 1] > 0.5) { ctx.moveTo(px, py); continue; }
      i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
    }
    ctx.stroke();
    ctx.restore();
    if (progress > 0.98) {
      const mark = sr.kind === "periodogram" ? sr.signal_period : sr.peak.t;
      const px = x(mark);
      ctx.strokeStyle = c.accent; ctx.fillStyle = c.accent; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(px, fr.t + 18); ctx.lineTo(px, fr.b); ctx.stroke();
      ctx.font = `600 12px ${token("--mono")}`;
      const label = sr.kind === "periodogram" ? `P = ${fmt(sr.signal_period, 3)} d` : `day ${fmt(sr.peak.t, 2)} · SES ${fmt(sr.peak.value, 1)}`;
      ctx.textAlign = px > (fr.l + fr.r) / 2 ? "right" : "left";
      ctx.fillText(label, px + (ctx.textAlign === "left" ? 6 : -6), fr.t + 28);
      if (s.known && sr.kind === "periodogram") {
        ctx.fillStyle = c.known;
        ctx.font = `11px ${token("--mono")}`;
        ctx.fillText(`ExoFOP ${fmt(s.known.toi_period, 3)} d`, px + (ctx.textAlign === "left" ? 6 : -6), fr.t + 44);
      }
    }
    ctx.restore();
  }

  function foldScene(ctx, fr, m, alpha) {
    const s = star, lc = s.lc, _ = s._, f = s.fold, c = C();
    const yr = [lerp(_.yFlat[0], _.yFold[0], m), lerp(_.yFlat[1], _.yFold[1], m)];
    const xT = scale(0, _.T, fr.l, fr.r), xP = scale(-_.hw, _.hw, fr.l, fr.r), y = scale(yr[0], yr[1], fr.b, fr.t);
    ctx.save(); ctx.globalAlpha *= alpha;
    // axes crossfade from time to phase
    ctx.save(); ctx.globalAlpha *= 1 - m;
    axes(ctx, fr, xT, y, { xticks: niceTicks(0, _.T, 6), yticks: niceTicks(yr[0], yr[1], 5), xlabel: "days", ylabel: "flux (ppm)", xfmt: (v) => fmt(v) });
    ctx.restore();
    ctx.save(); ctx.globalAlpha *= m;
    axes(ctx, fr, xP, y, { xticks: niceTicks(-_.hw, _.hw, 6), yticks: niceTicks(yr[0], yr[1], 5), xlabel: s.method === "single" ? "hours from the event" : "hours from mid-transit", xfmt: (v) => fmt(v, Math.abs(v) < 2 && v % 1 ? 1 : 0) });
    ctx.restore();
    ctx.save(); clip(ctx, fr);
    // binned flat points moving from time to phase
    const xs = lc.t.map((t, i) => lerp(xT(t), xP(_.ph[i]), m));
    points(ctx, xs, lc.flat.map((v) => (v == null ? null : y(v))), c.data, 1.8, 0.8 * (1 - 0.75 * m));
    if (m > 0.6) {
      const k = (m - 0.6) / 0.4;
      points(ctx, f.x.map(xP), f.y.map((v) => (v == null ? null : y(v))), c.data, 1.6, 0.55 * k);
      ctx.strokeStyle = c.accent; ctx.lineWidth = 2.2; ctx.globalAlpha *= k;
      ctx.beginPath();
      f.bx.forEach((bx, i) => (i ? ctx.lineTo(xP(bx), y(f.by[i])) : ctx.moveTo(xP(bx), y(f.by[i]))));
      ctx.stroke();
    }
    ctx.restore();
    ctx.restore();
  }

  onResize(canvas, draw);

  // steps drive the stage
  const io = new IntersectionObserver((entries) => {
    for (const en of entries) if (en.isIntersecting) goto(steps.indexOf(en.target));
  }, { rootMargin: "-45% 0px -45% 0px" });
  steps.forEach((s) => io.observe(s));
  steps[0].classList.add("on");
  bar.firstChild.setAttribute("aria-current", "step");

  return { show };
}
