// Wires the page together. Every section loads its own data from data/ (see
// scripts/week4_export.py), so each one can be lifted into another site alone.
import { initSky } from "./sky.js";
import { initWalkthrough } from "./walkthrough.js";
import { initHeatmap } from "./heatmap.js";
import { initResults } from "./results.js";

const [stars, metrics] = await Promise.all([
  fetch("data/stars.json").then((r) => r.json()),
  fetch("data/metrics.json").then((r) => r.json()),
]);

const walk = initWalkthrough(document.querySelector("#walk"), stars);
walk.show("tic147950620");

const pick = (id) => walk.show(id, { scroll: true });
initSky(document.querySelector("#sky"), { onPick: pick, walkIds: new Set(stars.map((s) => s.id)) }).catch((e) => console.error(e));
initHeatmap(document.querySelector("#heatmap")).catch((e) => console.error(e));
initResults(metrics, stars, { onWalk: pick });
