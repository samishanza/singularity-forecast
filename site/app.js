// Singularity Forecast front end: plain SVG charts, no dependencies.
const NS = "http://www.w3.org/2000/svg";
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const SERIES = [
  { key: "ac", label: "Automated coder", color: "var(--series-3)" },
  { key: "sar", label: "Superhuman AI researcher", color: "var(--series-4)" },
  { key: "siar", label: "Superintelligent AI researcher", color: "var(--series-2)" },
  { key: "asi", label: "Superintelligence (ASI)", color: "var(--series-1)" },
];

const $ = (id) => document.getElementById(id);
function el(tag, attrs = {}, parent) {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
}
function h(tag, text, cls) {
  const n = document.createElement(tag);
  if (text != null) n.textContent = text;
  if (cls) n.className = cls;
  return n;
}
function yearToDate(y) {
  const yr = Math.floor(y);
  const start = Date.UTC(yr, 0, 1), end = Date.UTC(yr + 1, 0, 1);
  return new Date(start + (end - start) * (y - yr));
}
function fmtMonth(y) {
  if (y == null || !isFinite(y)) return "after 2200";
  if (y > 2200) return "after 2200";
  const d = yearToDate(y);
  return `${MONTHS[d.getUTCMonth()]} ${d.getUTCFullYear()}`;
}
function fmtPct(p) {
  if (p == null) return "—";
  if (p > 0.995) return ">99%";
  if (p < 0.005) return "<1%";
  return `${Math.round(p * 100)}%`;
}
function fmtHours(min) {
  const hrs = min / 60;
  if (hrs < 1) return `${Math.round(min)} min`;
  if (hrs < 48) return `${hrs.toFixed(hrs < 10 ? 1 : 0)} h`;
  const workWeeks = hrs / 40;
  if (workWeeks < 8) return `${workWeeks.toFixed(1)} work-weeks`;
  const workMonths = hrs / 167;
  if (workMonths < 24) return `${workMonths.toFixed(1)} work-months`;
  return `${(hrs / 2000).toFixed(1)} work-years`;
}

// ---------- tooltip
const tip = $("tip");
function showTip(evt, head, rows) {
  tip.replaceChildren();
  tip.appendChild(h("div", head, "tt-head"));
  for (const r of rows) {
    const row = h("div", null, "tt-row");
    if (r.color) { const k = h("span", null, "tt-key"); k.style.background = r.color; row.appendChild(k); }
    row.appendChild(h("b", r.value));
    row.appendChild(h("span", r.label));
    tip.appendChild(row);
  }
  tip.hidden = false;
  const pad = 14, w = tip.offsetWidth, hh = tip.offsetHeight;
  let x = evt.clientX + pad, y = evt.clientY + pad;
  if (x + w > window.innerWidth - 8) x = evt.clientX - w - pad;
  if (y + hh > window.innerHeight - 8) y = evt.clientY - hh - pad;
  tip.style.left = `${x}px`; tip.style.top = `${y}px`;
}
function hideTip() { tip.hidden = true; }

// ---------- generic frame
function frame(container, { width = 960, height = 360, m = { t: 12, r: 24, b: 34, l: 52 } } = {}) {
  container.replaceChildren();
  const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, role: "img" }, container);
  const iw = width - m.l - m.r, ih = height - m.t - m.b;
  const g = el("g", { transform: `translate(${m.l},${m.t})` }, svg);
  return { svg, g, iw, ih, m, width, height };
}
const lin = (d0, d1, r0, r1) => (v) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);

function xAxisYears(g, x, ih, y0, y1, step = 1) {
  for (let y = Math.ceil(y0); y <= y1; y += step) {
    const px = x(y);
    el("line", { x1: px, x2: px, y1: 0, y2: ih, stroke: "var(--grid)", "stroke-width": 1 }, g);
    const t = el("text", { x: px, y: ih + 20, "text-anchor": "middle" }, g);
    t.textContent = String(y);
  }
  el("line", { x1: 0, x2: x(y1), y1: ih, y2: ih, stroke: "var(--axis)", "stroke-width": 1 }, g);
}
function yAxis(g, y, ticks, iw, fmt) {
  for (const v of ticks) {
    const py = y(v);
    el("line", { x1: 0, x2: iw, y1: py, y2: py, stroke: "var(--grid)", "stroke-width": 1 }, g);
    const t = el("text", { x: -8, y: py + 4, "text-anchor": "end" }, g);
    t.textContent = fmt(v);
  }
}
function pathD(xs, ys) {
  let d = "";
  for (let i = 0; i < xs.length; i++) {
    if (!isFinite(ys[i])) continue;
    d += (d ? "L" : "M") + xs[i].toFixed(1) + "," + ys[i].toFixed(1);
  }
  return d;
}
function bandD(xs, lo, hi) {
  const top = xs.map((x, i) => `${x.toFixed(1)},${hi[i].toFixed(1)}`);
  const bot = xs.map((x, i) => `${x.toFixed(1)},${lo[i].toFixed(1)}`).reverse();
  return `M${top.join("L")}L${bot.join("L")}Z`;
}
function crosshair(f, xsData, xScale, onMove) {
  const line = el("line", { y1: 0, y2: f.ih, stroke: "var(--axis)", "stroke-width": 1, visibility: "hidden" }, f.g);
  const hit = el("rect", { x: 0, y: 0, width: f.iw, height: f.ih, fill: "transparent", tabindex: 0 }, f.g);
  const pick = (evt) => {
    const pt = f.svg.createSVGPoint(); pt.x = evt.clientX; pt.y = evt.clientY;
    const loc = pt.matrixTransform(f.g.getScreenCTM().inverse());
    let best = 0, bd = Infinity;
    xsData.forEach((v, i) => { const d = Math.abs(xScale(v) - loc.x); if (d < bd) { bd = d; best = i; } });
    const px = xScale(xsData[best]);
    line.setAttribute("x1", px); line.setAttribute("x2", px); line.setAttribute("visibility", "visible");
    onMove(evt, best);
  };
  hit.addEventListener("pointermove", pick);
  hit.addEventListener("pointerleave", () => { line.setAttribute("visibility", "hidden"); hideTip(); });
  return hit;
}
function table(container, headers, rows, numCols = []) {
  const t = document.createElement("table");
  const tr = document.createElement("tr");
  headers.forEach((hd, i) => { const th = h("th", hd); if (numCols.includes(i)) th.className = "num"; tr.appendChild(th); });
  const thead = document.createElement("thead"); thead.appendChild(tr); t.appendChild(thead);
  const tb = document.createElement("tbody");
  for (const r of rows) {
    const row = document.createElement("tr");
    r.forEach((c, i) => { const td = h("td", c); if (numCols.includes(i)) td.className = "num"; row.appendChild(td); });
    tb.appendChild(row);
  }
  t.appendChild(tb);
  container.replaceChildren(t);
}

// ---------- charts
function cdfChart(F) {
  const legend = $("cdf-legend");
  legend.replaceChildren();
  for (const s of SERIES) {
    const sp = h("span"); const i = h("i"); i.style.background = s.color; sp.appendChild(i);
    sp.appendChild(document.createTextNode(s.label)); legend.appendChild(sp);
  }
  const y0 = Math.floor(F.now), y1 = 2045;
  const years = [];
  for (let y = y0; y <= y1 + 1e-9; y += 0.25) years.push(+y.toFixed(2));
  // the CDF at quarter steps, from the quarterly density
  const cdf = {};
  for (const s of SERIES) {
    const d = F.density[s.key];
    let acc = d.before; const vals = [];
    for (let i = 0; i < years.length; i++) {
      vals.push(acc);
      acc += d.prob[i] || 0;
    }
    cdf[s.key] = vals;
  }
  const f = frame($("cdf"), { height: 340 });
  const x = lin(y0, y1, 0, f.iw), y = lin(0, 1, f.ih, 0);
  yAxis(f.g, y, [0, 0.25, 0.5, 0.75, 1], f.iw, (v) => `${v * 100}%`);
  xAxisYears(f.g, x, f.ih, y0, y1, 2);
  const xs = years.map(x);
  for (const s of SERIES) {
    el("path", { d: pathD(xs, cdf[s.key].map(y)), fill: "none", stroke: s.color, "stroke-width": 2,
      "stroke-linejoin": "round", "stroke-linecap": "round" }, f.g);
  }
  // direct label: the ASI median, where its curve crosses 50%
  const med = F.milestones.asi.quantiles["0.5"];
  if (med && med < y1) {
    el("circle", { cx: x(med), cy: y(0.5), r: 4.5, fill: "var(--series-1)", stroke: "var(--surface-1)", "stroke-width": 2 }, f.g);
    const t = el("text", { x: x(med) + 10, y: y(0.5) + 18, class: "dl" }, f.g);
    t.textContent = `ASI median: ${fmtMonth(med)}`;
  }
  crosshair(f, years, x, (evt, i) => showTip(evt, `By ${fmtMonth(years[i])}`,
    SERIES.map((s) => ({ color: s.color, value: fmtPct(cdf[s.key][i]), label: s.label })).reverse()));
  const rows = Object.keys(F.milestones.asi.p_by_year_end).filter((yy) => +yy <= 2050).map((yy) =>
    [`End of ${yy}`, ...SERIES.map((s) => fmtPct(F.milestones[s.key].p_by_year_end[yy]))]);
  table($("cdf-table"), ["By", ...SERIES.map((s) => s.label)], rows, [1, 2, 3, 4]);
}

function densityChart(F) {
  const d = F.density.asi;
  const y0 = d.start, y1 = 2045;
  const n = Math.round((y1 - y0) / d.step);
  const probs = d.prob.slice(0, n);
  const f = frame($("density"), { height: 300 });
  const x = lin(y0, y1, 0, f.iw);
  const maxP = Math.max(...probs) * 1.15;
  const step = maxP > 0.08 ? 0.02 : maxP > 0.04 ? 0.01 : 0.005;
  const top = Math.ceil(maxP / step) * step;
  const y = lin(0, top, f.ih, 0);
  const ticks = []; for (let v = 0; v <= top + 1e-9; v += step) ticks.push(+v.toFixed(4));
  yAxis(f.g, y, ticks, f.iw, (v) => `${(v * 100).toFixed(v < 0.01 && v > 0 ? 1 : 0)}%`);
  xAxisYears(f.g, x, f.ih, y0, y1, 2);
  const slot = f.iw / n, bw = Math.min(24, slot - 2);
  probs.forEach((p, i) => {
    const qx = y0 + i * d.step;
    const px = x(qx) + (slot - bw) / 2, py = y(p), hgt = f.ih - py;
    const r = Math.min(4, bw / 2, hgt);
    // 4px rounded data end, square at the baseline
    const dpath = hgt <= 0 ? "" :
      `M${px},${f.ih}V${py + r}Q${px},${py} ${px + r},${py}H${px + bw - r}Q${px + bw},${py} ${px + bw},${py + r}V${f.ih}Z`;
    const bar = el("path", { d: dpath, fill: "var(--series-1)" }, f.g);
    const hit = el("rect", { x: x(qx), y: 0, width: slot, height: f.ih, fill: "transparent", tabindex: 0 }, f.g);
    const q = Math.round((qx - Math.floor(qx)) * 4) + 1;
    const label = `Q${q} ${Math.floor(qx + 1e-6)}`;
    const on = (evt) => { bar.setAttribute("opacity", 0.75); showTip(evt, label, [{ color: "var(--series-1)", value: fmtPct(p), label: "chance ASI arrives this quarter" }]); };
    hit.addEventListener("pointermove", on);
    hit.addEventListener("focus", (evt) => on({ clientX: hit.getBoundingClientRect().x, clientY: hit.getBoundingClientRect().y }));
    hit.addEventListener("pointerleave", () => { bar.setAttribute("opacity", 1); hideTip(); });
  });
  const later = 1 - d.before - probs.reduce((a, b) => a + b, 0);
  $("density-note").textContent = `Not shown: ${fmtPct(later)} chance ASI arrives after 2045 (${fmtPct(d.after)} after 2060).`;
}

function historyChart(F, H) {
  const rows = H.filter((r) => r.asi && r.asi["0.5"] != null);
  const ts = rows.map((r) => {
    const [Y, M, D] = r.date.split("-").map(Number);
    const s = Date.UTC(Y, 0, 1), e = Date.UTC(Y + 1, 0, 1);
    return Y + (Date.UTC(Y, M - 1, D) - s) / (e - s);
  });
  const med = rows.map((r) => r.asi["0.5"]), lo = rows.map((r) => r.asi["0.25"]), hi = rows.map((r) => r.asi["0.75"]);
  const acMed = rows.map((r) => r.ac["0.5"]);
  const f = frame($("history"), { height: 320 });
  const x0 = Math.floor(ts[0] * 4) / 4, x1 = Math.max(ts[ts.length - 1], F.now) + 0.05;
  const yMin = Math.floor(Math.min(...acMed, ...lo)) , yMax = Math.min(2050, Math.ceil(Math.max(...hi)) + 1);
  const x = lin(x0, x1, 0, f.iw), y = lin(yMin, yMax, f.ih, 0);
  const yt = []; for (let v = yMin; v <= yMax; v += yMax - yMin > 12 ? 4 : 2) yt.push(v);
  yAxis(f.g, y, yt, f.iw, (v) => String(v));
  for (let q = Math.ceil(x0 * 2) / 2; q <= x1; q += 0.5) {
    const px = x(q);
    el("line", { x1: px, x2: px, y1: 0, y2: f.ih, stroke: "var(--grid)", "stroke-width": 1 }, f.g);
    const t = el("text", { x: px, y: f.ih + 20, "text-anchor": "middle" }, f.g);
    t.textContent = q % 1 === 0 ? String(q) : `Jul ${Math.floor(q)}`;
  }
  el("line", { x1: 0, x2: f.iw, y1: f.ih, y2: f.ih, stroke: "var(--axis)" }, f.g);
  const xs = ts.map(x);
  el("path", { d: bandD(xs, lo.map((v) => y(Math.min(v, yMax))), hi.map((v) => y(Math.min(v, yMax)))), fill: "var(--series-1)", opacity: 0.12 }, f.g);
  el("path", { d: pathD(xs, acMed.map(y)), fill: "none", stroke: "var(--series-3)", "stroke-width": 2, "stroke-linejoin": "round" }, f.g);
  el("path", { d: pathD(xs, med.map(y)), fill: "none", stroke: "var(--series-1)", "stroke-width": 2, "stroke-linejoin": "round" }, f.g);
  const last = xs.length - 1;
  const lab = (val, color, text, dy) => {
    el("circle", { cx: xs[last], cy: y(val), r: 4.5, fill: color, stroke: "var(--surface-1)", "stroke-width": 2 }, f.g);
    const t = el("text", { x: xs[last] - 8, y: y(val) + dy, class: "dl", "text-anchor": "end" }, f.g);
    t.textContent = text;
  };
  lab(med[last], "var(--series-1)", `ASI median ${fmtMonth(med[last])}`, -10);
  lab(acMed[last], "var(--series-3)", `Automated coder median ${fmtMonth(acMed[last])}`, 20);
  crosshair(f, ts, x, (evt, i) => showTip(evt, `Forecast as of ${rows[i].date}${rows[i].kind === "backfill" ? " (back-filled)" : ""}`, [
    { color: "var(--series-1)", value: fmtMonth(med[i]), label: "ASI median" },
    { color: "var(--series-1)", value: `${fmtMonth(lo[i])} – ${fmtMonth(hi[i])}`, label: "middle 50%" },
    { color: "var(--series-3)", value: fmtMonth(acMed[i]), label: "automated coder median" },
  ]));
  table($("history-table"), ["Forecast date", "Kind", "ASI median", "ASI middle 50%", "Automated coder median"],
    rows.map((r) => [r.date, r.kind, fmtMonth(r.asi["0.5"]), `${fmtMonth(r.asi["0.25"])} – ${fmtMonth(r.asi["0.75"])}`, fmtMonth(r.ac["0.5"])]));
}

function horizonChart(F) {
  const H = F.horizon;
  const t1 = 2030;
  const keep = H.t.map((t, i) => i).filter((i) => H.t[i] <= t1);
  const ts = keep.map((i) => H.t[i]);
  const yMin = 0, yMax = 22; // log2 minutes: 1 min .. ~70 work-years
  const clamp = (v) => Math.max(yMin, Math.min(yMax, v));
  const f = frame($("horizon"), { height: 380, m: { t: 12, r: 24, b: 34, l: 96 } });
  const x = lin(H.t[0], t1, 0, f.iw), y = lin(yMin, yMax, f.ih, 0);
  const ticks = [0, Math.log2(15), Math.log2(60), Math.log2(480), Math.log2(2400), Math.log2(10020), Math.log2(120000), Math.log2(1200000)];
  const names = ["1 min", "15 min", "1 hour", "1 workday", "1 work-week", "1 work-month", "1 work-year", "10 work-years"];
  ticks.forEach((v, i) => {
    el("line", { x1: 0, x2: f.iw, y1: y(v), y2: y(v), stroke: "var(--grid)" }, f.g);
    const t = el("text", { x: -8, y: y(v) + 4, "text-anchor": "end" }, f.g); t.textContent = names[i];
  });
  xAxisYears(f.g, x, f.ih, H.t[0], t1, 1);
  const xs = ts.map(x);
  const pick = (k) => keep.map((i) => y(clamp(H[k][i])));
  el("path", { d: bandD(xs, pick("p10"), pick("p90")), fill: "var(--series-1)", opacity: 0.10 }, f.g);
  el("path", { d: bandD(xs, pick("p25"), pick("p75")), fill: "var(--series-1)", opacity: 0.16 }, f.g);
  el("path", { d: pathD(xs, pick("p50")), fill: "none", stroke: "var(--series-1)", "stroke-width": 2 }, f.g);
  // threshold band: the automated-coder threshold's middle 50%
  for (const o of H.obs) {
    const cx = x(o.t), cy = y(clamp(o.y));
    const isNow = o.kind === "nowcast";
    const c = el("circle", { cx, cy, r: 4.5, fill: isNow ? "var(--surface-1)" : "var(--series-1)",
      stroke: isNow ? "var(--series-1)" : "var(--surface-1)", "stroke-width": 2 }, f.g);
    const hit = el("circle", { cx, cy, r: 12, fill: "transparent", tabindex: 0 }, f.g);
    hit.addEventListener("pointermove", (evt) => showTip(evt, `${o.name} · ${fmtMonth(o.t)}`, [
      { color: "var(--series-1)", value: fmtHours(2 ** o.y), label: isNow ? "estimated from ECI" : "METR measurement" },
      { value: `${fmtHours(2 ** (o.y - 1.2816 * o.se))} – ${fmtHours(2 ** (o.y + 1.2816 * o.se))}`, label: "80% range" },
    ]));
    hit.addEventListener("pointerleave", hideTip);
  }
  const legend = document.createElement("div"); legend.className = "legend";
  const add = (cls, text) => { const sp = h("span"); const i = h("i", null, cls); if (cls !== "hollow") i.style.background = "var(--series-1)"; sp.appendChild(i); sp.appendChild(document.createTextNode(text)); legend.appendChild(sp); };
  add("dot", "METR measurement"); add("hollow", "Estimated from Epoch Capabilities Index"); add("", "Model median"); add("band", "Model 50% and 80% ranges");
  $("horizon").prepend(legend);
  table($("horizon-table"), ["Model", "Date", "Source", "50% horizon"], H.obs.map((o) => [o.name, fmtMonth(o.t), o.kind === "metr" ? "METR" : "ECI estimate", fmtHours(2 ** o.y)]), [3]);
}

function tiles(F) {
  const M = F.milestones, T = F.trend;
  const asi = M.asi;
  $("hero-asi").textContent = fmtMonth(asi.quantiles["0.5"]);
  $("hero-range").textContent = `80% range: ${fmtMonth(asi.quantiles["0.1"])} to ${fmtMonth(asi.quantiles["0.9"])}`;
  const yr = (y) => asi.p_by_year_end[String(y)];
  const items = [
    ["Automated coder, median", fmtMonth(M.ac.quantiles["0.5"]), `80% range ${fmtMonth(M.ac.quantiles["0.1"])} – ${fmtMonth(M.ac.quantiles["0.9"])}`],
    ["Chance of ASI by end of 2030", fmtPct(yr(2030)), `by end of 2035: ${fmtPct(yr(2035))}`],
    ["Frontier time horizon now", fmtHours(T.horizon_hours_now[1] * 60), `80% range ${fmtHours(T.horizon_hours_now[0] * 60)} – ${fmtHours(T.horizon_hours_now[2] * 60)}`],
    ["Doubling time now", `${Math.round(T.doubling_days_now[1])} days`, `chance the trend is accelerating: ${fmtPct(T.regime_post.super)}`],
  ];
  const box = $("tiles"); box.replaceChildren();
  for (const [l, v, n] of items) {
    const t = h("div", null, "tile");
    t.append(h("p", l, "t-label"), h("p", v, "t-value"), h("p", n, "t-note"));
    box.appendChild(t);
  }
}

function sensitivity(S) {
  table($("sensitivity"), ["Assumption", "ASI median", "80% range"],
    S.rows.map((r) => [r.case, fmtMonth(r.asi[1]), `${fmtMonth(r.asi[0])} – ${fmtMonth(r.asi[2])}`]), [1]);
}
function backtest(B) {
  const rows = Object.entries(B).map(([k, v]) => [k, v.mean_lpd.toFixed(2), fmtPct(v.cover50), fmtPct(v.cover80), String(v.n_scored)]);
  table($("backtest"), ["Trend prior", "Log score", "In 50% range", "In 80% range", "Points"], rows, [1, 2, 3, 4]);
  const n = h("p", "Higher log score is better. A well-calibrated model puts about 50% and 80% of later measurements inside its 50% and 80% ranges; lower coverage means past fits were overconfident, mostly by underpredicting how fast horizons grew.", "note");
  $("backtest").appendChild(n);
}
function events(F) {
  const box = $("events"); box.replaceChildren();
  if (!F.events.length) { box.appendChild(h("p", "No events accepted yet.", "note")); return; }
  for (const e of F.events) {
    const d = h("div", null, "event");
    const ratio = Math.exp(e.log_lr[0] - e.log_lr[1]);
    d.append(h("p", e.title, "e-title"), h("p", `${e.date} · makes the accelerating regime ${ratio.toFixed(2)}× more likely relative to the steady one`, "e-meta"));
    box.appendChild(d);
  }
}
function sources(F) {
  const rows = Object.entries(F.data.sources).map(([k, v]) => [k, v.fetched ? v.fetched.slice(0, 10) : `failed (${v.error || "?"})`, v.bytes ? `${Math.round(v.bytes / 1024)} KB` : "—"]);
  table($("sources"), ["Dataset", "Last fetched", "Size"], rows, [2]);
  const map = F.data.eci_map;
  if (map) $("sources").appendChild(h("p", `ECI → horizon map: log₂(minutes) = ${map.a.toFixed(2)} + ${map.b.toFixed(3)} × ECI, fitted on ${map.n} models; ${F.data.n_nowcast} newer frontier models estimated from it. Latest METR measurement: ${fmtMonth(F.data.metr_last)}.`, "note"));
}

async function main() {
  const get = (p) => fetch(p, { cache: "no-cache" }).then((r) => r.json());
  const [F, H, S, B] = await Promise.all([get("data/forecast.json"), get("data/history.json"), get("data/sensitivity.json"), get("data/backtest.json")]);
  $("meta").textContent = `Updated ${F.generated_at.replace("T", " ").slice(0, 16)} UTC · posterior effective sample size ${Math.round(F.trend.ess).toLocaleString()}`;
  tiles(F); cdfChart(F); densityChart(F); historyChart(F, H); horizonChart(F);
  sensitivity(S); backtest(B); events(F); sources(F);
  let raf;
  window.addEventListener("resize", () => { cancelAnimationFrame(raf); raf = requestAnimationFrame(() => { cdfChart(F); densityChart(F); historyChart(F, H); horizonChart(F); }); });
}
main().catch((e) => { $("meta").textContent = `Could not load the forecast data: ${e.message}`; });
