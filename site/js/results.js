// Validation charts, scoreboard and the hand-vetted signal cards.
import { axes, fit, fmt, niceTicks, onResize, scale, token } from "./plot.js";

function lineChart(canvas, { bins, series, ylabel, xlabel }) {
  const draw = () => {
    const { ctx, w, h } = fit(canvas);
    ctx.clearRect(0, 0, w, h);
    const fr = { l: 44, r: w - 10, t: 12, b: h - 40 };
    const x = scale(-0.5, bins.length - 0.5, fr.l, fr.r), y = scale(0, 1, fr.b, fr.t);
    axes(ctx, fr, x, y, { yticks: [0, 0.25, 0.5, 0.75, 1], yfmt: (v) => `${Math.round(v * 100)}%`, ylabel });
    ctx.fillStyle = token("--muted");
    ctx.font = `10px ${token("--mono")}`;
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    const step = w < 420 ? 2 : 1;
    bins.forEach((b, i) => i % step === 0 && ctx.fillText(b, x(i), fr.b + 6));
    ctx.fillText(xlabel, (fr.l + fr.r) / 2, fr.b + 22);
    for (const s of series) {
      ctx.strokeStyle = s.color; ctx.fillStyle = s.color; ctx.lineWidth = s.width || 2;
      ctx.setLineDash(s.dash || []);
      ctx.beginPath();
      s.values.forEach((v, i) => (i ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v))));
      ctx.stroke();
      ctx.setLineDash([]);
      s.values.forEach((v, i) => { ctx.beginPath(); ctx.arc(x(i), y(v), 2.6, 0, Math.PI * 2); ctx.fill(); });
    }
  };
  onResize(canvas, draw);
}

const binLabel = (b) => {
  const m = b.match(/\(([\d.]+),\s*([\d.]+)\]/);
  if (!m) return b;
  const lo = +m[1], hi = +m[2];
  return hi >= 1000 ? `>${fmt(lo)}` : `${fmt(lo)}–${fmt(hi)}`;
};

export function initResults(metrics, stars, { onWalk }) {
  const rv = metrics.recall_vs_snr;
  // skip the first bin (SNR 3–5): nothing is found there and it only stretches the axis
  const sl = (a) => a.slice(1);
  lineChart(document.querySelector("#chart-snr canvas"), {
    bins: sl(rv.bins).map(binLabel), xlabel: "expected transit SNR", ylabel: "recovered",
    series: [
      { values: sl(rv.inj_tls), color: token("--data") },
      { values: sl(rv.inj_bls), color: token("--accent"), dash: [5, 4], width: 1.6 },
      { values: sl(rv.toi_tls), color: token("--known") },
    ],
  });
  const se = metrics.single_vs_snr;
  lineChart(document.querySelector("#chart-single canvas"), {
    bins: se.bins.map(binLabel), xlabel: "expected single-event SNR", ylabel: "recovered",
    series: [{ values: se.recall, color: token("--data") }],
  });

  // classifier bars (ROC AUC, planet vs false positive, held-out stars)
  const cl = metrics.classifier;
  const models = [
    ["logreg_bls", "Logistic regression on BLS statistics"],
    ["gbm_bls", "Gradient boosting on BLS statistics (baseline)"],
    ["cnn_views", "CNN on folded views only"],
    ["cnn_views_vet", "CNN on views + vetting numbers"],
    ["gbm_vet", "Gradient boosting on BLS + vetting numbers"],
    ["ensemble", "Mean of the two above (used)"],
  ];
  const lo = 0.5;
  document.querySelector("#bars-cls").innerHTML = models.map(([k, label]) => {
    const v = cl[k]?.auc;
    return `<div class="bar${k === "ensemble" ? " hi" : ""}"><span class="lbl">${label}</span><span class="track"><i style="width:${((v - lo) / (1 - lo)) * 100}%"></i></span><span class="num">${fmt(v, 2)}</span></div>`;
  }).join("");

  const bench = metrics.bench;
  const name = { tls_cpu: "Transit Least Squares (reference)", bls_cpu_numba: "Own BLS, CPU (numba)", bls_cuda: "Own BLS, CUDA kernel" };
  const max = Math.max(...bench.map((b) => b.stars_per_s));
  document.querySelector("#bars-speed").innerHTML = bench.filter((b) => b.hw.startsWith("laptop")).concat(bench.filter((b) => !b.hw.startsWith("laptop"))).map((b) =>
    `<div class="bar${b.backend === "bls_cuda" ? " hi" : ""}"><span class="lbl">${name[b.backend]} · ${b.hw.startsWith("laptop") ? (b.backend === "bls_cuda" ? "laptop GPU (RTX 4050)" : "laptop CPU, 16 threads") : b.hw}</span><span class="track"><i style="width:${(b.stars_per_s / max) * 100}%"></i></span><span class="num">${fmt(b.stars_per_s, b.stars_per_s < 10 ? 1 : 0)}</span></div>`).join("");

  const fu = metrics.funnel_s105;
  const rows = [
    ["M and K dwarfs searched", fu.stars_searched],
    ["signals detected (periodic + lone dips)", fu.signals],
    ["after removing sector-wide systematics and failed vetting", fu.pass_all],
    ["not already a TOI, CTOI or SPOC TCE, SNR ≥ 10", 21],
    ["new candidates after hand-vetting", 0],
  ];
  document.querySelector("#funnel").innerHTML = rows.map(([l, v], i) =>
    `<div class="row${i === rows.length - 1 ? " end" : ""}"><span>${l}</span><span class="num">${fmt(v)}</span></div>`).join("");

  // candidate cards
  const groups = [
    ["ctoi", "New transits on known single-transit community TOIs", "Real dips, on stars that already have a single-transit community TOI from an earlier sector, so most likely the same objects transiting again. No CTOI submission; worth passing to the owners of those CTOIs."],
    ["fp", "False positives: blends and eclipsing binaries", "Mostly caught by the centroid check: the light moves toward a neighbouring star during the dip."],
    ["variability", "Stellar variability", "Rotation or pulsation troughs that a box search scores as dips."],
    ["noise", "Noise and systematics", "Folds driven by a few outliers, ramps, or bad stretches of data."],
    ["inconclusive", "Inconclusive", "Too weak to submit."],
  ];
  const pillText = { ctoi: "new transit", fp: "false positive", variability: "variability", noise: "noise", inconclusive: "inconclusive" };
  const el = document.querySelector("#cands");
  el.innerHTML = groups.map(([g, title, blurb]) => {
    const items = stars.filter((s) => s.group === g);
    if (!items.length) return "";
    return `<div class="cand-group"><div class="section-head" style="margin:0;gap:.4rem"><h3>${title} <span class="muted num">${items.length}</span></h3><p>${blurb}</p></div>
      <div class="cards">${items.map((s) => `
        <article class="card${g === "ctoi" ? " feature" : ""}">
          <div class="card-top"><h3>${s.label}</h3><span class="pill ${g}">${pillText[g]}</span></div>
          <div class="kv">
            <span>${s.method === "single" ? "lone dip at" : "period"}<b>${s.method === "single" ? `BTJD ${fmt(s.t0, 2)}` : `${fmt(s.period, 3)} d`}</b></span>
            <span>depth<b>${fmt(s.depth_ppm)} ppm</b></span>
            <span>SNR<b>${fmt(s.snr, 1)}</b></span>
          </div>
          <p>${s.reason}</p>
          <div class="card-actions">
            <button class="btn" type="button" data-walk="${s.id}">Walk through</button>
            <button class="btn" type="button" data-sheet="${s.id}" aria-expanded="false">Vetting sheet</button>
          </div>
          <img class="sheet" alt="Vetting sheet for ${s.label}" data-src="data/sheets/${s.id}.png" hidden>
        </article>`).join("")}</div></div>`;
  }).join("");
  el.addEventListener("click", (e) => {
    const w = e.target.closest("[data-walk]");
    if (w) onWalk(w.dataset.walk);
    const b = e.target.closest("[data-sheet]");
    if (b) {
      const img = b.closest(".card").querySelector("img.sheet");
      if (!img.src) img.src = img.dataset.src;
      img.hidden = !img.hidden;
      b.setAttribute("aria-expanded", String(!img.hidden));
    }
  });
}
