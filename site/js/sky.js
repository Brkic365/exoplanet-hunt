// Hero: every star the pipeline searched, on a Hammer-Aitoff sky map (WebGL points).
// Colour = best classifier score of a signal that survived the systematics filter
// and vetting; glowing points are the stars with a full walkthrough below.
import { fit, fmt, reducedMotion, scoreColor, token } from "./plot.js";

const CENTER_RA = 300; // degrees; puts the sector-105 southern field near the middle
const SQ2 = Math.SQRT2;

function hammer(raDeg, decDeg) {
  let l = ((raDeg - CENTER_RA + 540) % 360) - 180; // [-180, 180)
  const lam = (l * Math.PI) / 180, phi = (decDeg * Math.PI) / 180;
  const d = Math.sqrt(1 + Math.cos(phi) * Math.cos(lam / 2));
  // RA increases to the left on a sky map
  return [-(2 * SQ2 * Math.cos(phi) * Math.sin(lam / 2)) / d, (SQ2 * Math.sin(phi)) / d];
}

const VS = `
attribute vec2 a_pos; attribute float a_size; attribute vec4 a_color; attribute float a_glow; attribute float a_seed;
uniform vec2 u_scale; uniform float u_time; uniform float u_dpr; uniform float u_ps;
varying vec4 v_color; varying float v_glow; varying float v_twinkle;
void main() {
  gl_Position = vec4(a_pos * u_scale, 0.0, 1.0);
  float appear = clamp((u_time - a_seed * 1.6 - a_glow * 1.4) / 0.6, 0.0, 1.0);
  float pulse = a_glow > 0.0 ? 0.75 + 0.25 * sin(u_time * 1.7 + a_seed * 40.0) : 1.0;
  v_twinkle = appear * (0.82 + 0.18 * sin(u_time * (0.6 + a_seed) + a_seed * 90.0));
  v_color = a_color; v_glow = a_glow;
  gl_PointSize = a_size * u_ps * u_dpr * (a_glow > 0.0 ? 5.0 * pulse : 1.0);
}`;
const FS = `
precision mediump float;
varying vec4 v_color; varying float v_glow; varying float v_twinkle;
void main() {
  float r = length(gl_PointCoord - 0.5) * 2.0;
  if (r > 1.0) discard;
  float a;
  if (v_glow > 0.0) {
    float core = smoothstep(0.24, 0.12, r);
    float halo = exp(-r * r * 6.0) * 0.55;
    a = core + halo;
  } else {
    a = smoothstep(1.0, 0.35, r);
  }
  gl_FragColor = vec4(v_color.rgb, v_color.a * a * v_twinkle);
}`;

function compile(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}

const CLASS_STYLE = {
  0: { c: [0.5, 0.57, 0.72], a: 0.5 },
  1: null, // by score
  2: { c: [0.95, 0.71, 0.29], a: 0.85, glow: 1 },
  3: { c: [1.0, 0.8, 0.42], a: 1.0, glow: 1.4 },
  4: { c: [0.36, 0.81, 0.7], a: 0.9 },
  5: { c: [0.36, 0.81, 0.7], a: 1.0, glow: 1.2 },
};

export async function initSky(root, { onPick, walkIds }) {
  const canvas = root.querySelector("canvas.gl");
  const overlay = root.querySelector("canvas.ov");
  const tip = root.querySelector(".tip");
  const [sky, back] = await Promise.all([
    fetch("data/sky.json").then((r) => r.json()),
    fetch("data/backdrop.bin").then((r) => r.arrayBuffer()),
  ]);

  // ---- vertex data: backdrop first (drawn under), then searched stars
  const nb = back.byteLength / 4;
  const bra = new Uint16Array(back, 0, nb), bdec = new Int16Array(back, nb * 2, nb);
  const n = sky.n, N = nb + n;
  const pos = new Float32Array(N * 2), size = new Float32Array(N), col = new Float32Array(N * 4),
    glow = new Float32Array(N), seed = new Float32Array(N);
  let rnd = 7;
  const rand = () => ((rnd = (rnd * 16807) % 2147483647) / 2147483647);
  for (let i = 0; i < nb; i++) {
    const [x, y] = hammer((bra[i] / 65535) * 360, bdec[i] / 300);
    pos.set([x, y], i * 2);
    size[i] = 1.1;
    col.set([0.42, 0.47, 0.6, 0.16], i * 4);
    seed[i] = rand() * 0.3;
  }
  const proj = new Float32Array(n * 2);
  for (let j = 0; j < n; j++) {
    const i = nb + j;
    const [x, y] = hammer(sky.ra[j], sky.dec[j]);
    pos.set([x, y], i * 2);
    proj.set([x, y], j * 2);
    const t = sky.tmag[j] ?? 12;
    size[i] = Math.max(1.3, Math.min(4.2, 1.4 + (13.5 - t) * 0.5));
    const cls = sky.cls[j], st = CLASS_STYLE[cls];
    if (st) {
      col.set([...st.c, st.a], i * 4);
      glow[i] = st.glow || 0;
    } else {
      col.set([...scoreColor(sky.score[j] ?? 0), 0.95], i * 4);
      size[i] += 0.6;
    }
    seed[i] = 0.25 + rand() * 0.6;
  }

  const gl = canvas.getContext("webgl", { premultipliedAlpha: false, antialias: false, alpha: true });
  if (!gl) {
    root.classList.add("no-webgl");
    return;
  }
  const prog = gl.createProgram();
  gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, VS));
  gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, FS));
  gl.linkProgram(prog);
  gl.useProgram(prog);
  const attr = (name, data, k) => {
    const b = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(prog, name);
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, k, gl.FLOAT, false, 0, 0);
  };
  attr("a_pos", pos, 2); attr("a_size", size, 1); attr("a_color", col, 4); attr("a_glow", glow, 1); attr("a_seed", seed, 1);
  const uScale = gl.getUniformLocation(prog, "u_scale"), uTime = gl.getUniformLocation(prog, "u_time"),
    uDpr = gl.getUniformLocation(prog, "u_dpr"), uPs = gl.getUniformLocation(prog, "u_ps");
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE);

  // ---- layout: fit the 2:1 map inside the canvas (frame() redraws; it loops unless motion is reduced)
  let W = 1, H = 1, sx = 1, sy = 1, dpr = 1, ps = 1;
  const toPx = (x, y) => [W / 2 + x * sx * (W / 2), H / 2 - y * sy * (H / 2)];
  function layout() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    const r = canvas.getBoundingClientRect();
    W = Math.max(1, r.width); H = Math.max(1, r.height);
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
    const k = Math.min(W / (4 * SQ2), H / (2 * SQ2)) * 0.96; // px per map unit
    sx = (2 * k) / W; sy = (2 * k) / H;
    ps = Math.max(0.45, Math.min(1, k / 230)); // smaller points on small maps, so dense fields do not saturate
    gl.viewport(0, 0, canvas.width, canvas.height);
    drawOverlay();
  }

  // ---- overlay: graticule, labels, hover ring
  let hover = -1;
  function drawOverlay() {
    const { ctx, w, h } = fit(overlay);
    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = token("--line");
    ctx.lineWidth = 1;
    const path = (pts) => {
      ctx.beginPath();
      pts.forEach(([x, y], i) => { const [px, py] = toPx(x, y); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); });
      ctx.stroke();
    };
    for (let dec = -60; dec <= 60; dec += 30) {
      const pts = [];
      for (let l = -179.9; l <= 179.9; l += 2) pts.push(hammer(CENTER_RA + l, dec));
      path(pts);
    }
    for (let l = -180; l <= 180; l += 60) {
      const pts = [];
      for (let dec = -90; dec <= 90; dec += 2) pts.push(hammer(CENTER_RA + (l === 180 ? 179.99 : l === -180 ? -179.99 : l), dec));
      path(pts);
    }
    ctx.fillStyle = token("--faint");
    ctx.font = `10px ${token("--mono")}`;
    ctx.textAlign = "center";
    for (const ra of [0, 60, 120, 180, 240]) {
      const [px, py] = toPx(...hammer(ra, 0));
      ctx.fillText(`${ra}°`, px, py - 4);
    }
    // field labels
    ctx.textAlign = "left";
    ctx.fillStyle = token("--muted");
    const lab = (ra, dec, text) => { const [px, py] = toPx(...hammer(ra, dec)); ctx.fillText(text, px, py); };
    const narrow = w < 600;
    lab(250, 84, narrow ? "Sector 75 · 2024" : "Sector 75 · 2024 · TOI hosts + field stars");
    lab(330, -70, narrow ? "Sector 105 · 2026" : "Sector 105 · 2026 · 15,511 M & K dwarfs");
    if (hover >= 0) {
      const [px, py] = toPx(proj[hover * 2], proj[hover * 2 + 1]);
      ctx.strokeStyle = token("--fg");
      ctx.beginPath(); ctx.arc(px, py, 7, 0, Math.PI * 2); ctx.stroke();
    }
  }

  const CLS_LABEL = sky.classes;
  function nearest(mx, my) {
    let best = -1, bd = 12 * 12;
    for (let j = 0; j < n; j++) {
      const [px, py] = toPx(proj[j * 2], proj[j * 2 + 1]);
      const d = (px - mx) ** 2 + (py - my) ** 2;
      // prefer interesting stars when close
      const bias = sky.cls[j] >= 2 ? 0.35 : 1;
      if (d * bias < bd) { bd = d * bias; best = j; }
    }
    return best;
  }
  function showTip(j, mx, my) {
    if (j < 0) { tip.hidden = true; return; }
    const id = `tic${sky.tic[j]}`;
    const walk = walkIds.has(id);
    tip.innerHTML = `<b>TIC ${sky.tic[j]}</b> · sector ${sky.sector[j]}<br>T = ${fmt(sky.tmag[j], 1)}` +
      (sky.score[j] != null ? ` · score ${fmt(sky.score[j], 2)}` : "") +
      `<br><span style="color:var(--muted)">${CLS_LABEL[sky.cls[j]]}</span>` +
      (walk ? `<br><span style="color:var(--accent)">Click to walk through this star</span>` : "");
    tip.hidden = false;
    const r = root.getBoundingClientRect();
    const left = Math.min(mx, r.width - tip.offsetWidth - 24);
    tip.style.left = `${Math.max(0, left)}px`;
    tip.style.top = `${Math.min(my, r.height - tip.offsetHeight - 24)}px`;
  }
  canvas.addEventListener("pointermove", (e) => {
    const r = canvas.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    const j = nearest(mx, my);
    if (j !== hover) { hover = j; drawOverlay(); }
    showTip(j, mx, my);
    canvas.style.cursor = j >= 0 && walkIds.has(`tic${sky.tic[j]}`) ? "pointer" : "crosshair";
  });
  canvas.addEventListener("pointerleave", () => { hover = -1; tip.hidden = true; drawOverlay(); });
  canvas.addEventListener("click", (e) => {
    const r = canvas.getBoundingClientRect();
    const j = nearest(e.clientX - r.left, e.clientY - r.top);
    if (j >= 0 && walkIds.has(`tic${sky.tic[j]}`)) onPick(`tic${sky.tic[j]}`);
  });
  overlay.style.pointerEvents = "none";

  const t0 = performance.now();
  const still = reducedMotion();
  let visible = true, raf = 0;
  const kick = () => { if (!raf) raf = requestAnimationFrame(frame); };
  new ResizeObserver(() => { layout(); kick(); }).observe(canvas);
  layout();
  new IntersectionObserver(([e]) => { visible = e.isIntersecting; if (visible) kick(); }).observe(canvas);
  function frame(now) {
    raf = 0;
    const t = still ? 10 : (now - t0) / 1000;
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.uniform2f(uScale, sx, sy);
    gl.uniform1f(uTime, t);
    gl.uniform1f(uDpr, dpr);
    gl.uniform1f(uPs, ps);
    gl.drawArrays(gl.POINTS, 0, N);
    if (visible && !still) kick();
  }
  kick();

  const counts = sky.cls.reduce((a, c) => ((a[c] = (a[c] || 0) + 1), a), {});
  return { n, counts };
}
