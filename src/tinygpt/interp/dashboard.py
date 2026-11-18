"""Static, single-file HTML feature dashboard.

Data for the selected latents is embedded as JSON, and a small vanilla-JS viewer renders
a searchable list plus, per feature: stats, an activation histogram, logit effects and
top-activating contexts with per-token highlighting. No external resources are loaded.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import torch

from tinygpt.interp.features import HIST_EDGES, FeatureStats, context_of
from tinygpt.sae.sae import SAE
from tinygpt.tokenizer.bpe import BPETokenizer


@torch.no_grad()
def build_feature_records(
    latents: list[int],
    stats: FeatureStats,
    sae: SAE,
    acts: torch.Tensor,
    tokens: torch.Tensor,
    block_size: int,
    tok: BPETokenizer,
    effects: dict[int, dict[str, list[tuple[int, float]]]],
    n_examples: int = 12,
    labels: dict[int, dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Everything the dashboard shows, for each latent in `latents`."""
    recs = []
    for i in latents:
        examples = []
        for j in range(min(n_examples, stats.top_vals.shape[0])):
            v = float(stats.top_vals[j, i])
            if v <= 0:
                break
            row = int(stats.top_rows[j, i])
            ids, start, tgt = context_of(row, tokens, block_size)
            z = sae.encode(acts[start : start + len(ids)].float())[:, i].tolist()
            examples.append(
                {
                    "toks": [tok.decode([t]) for t in ids],
                    "acts": [round(a, 3) for a in z],
                    "target": tgt,
                    "max": round(v, 3),
                }
            )
        rec = {
            "id": i,
            "freq": float(stats.freq[i]),
            "mean": float(stats.mean_active[i]),
            "hist": stats.hist[i].astype(int).tolist(),
            "boost": [(tok.decode([t]), round(s, 2)) for t, s in effects[i]["boost"]],
            "suppress": [(tok.decode([t]), round(s, 2)) for t, s in effects[i]["suppress"]],
            "examples": examples,
        }
        if labels and i in labels:
            rec["label"] = labels[i]
        recs.append(rec)
    return recs


def write_dashboard(
    path: str | Path, records: list[dict[str, Any]], title: str, summary: dict[str, Any]
) -> None:
    data = json.dumps({"features": records, "edges": HIST_EDGES.tolist(), "summary": summary})
    data = data.replace("</", "<\\/")  # keep the JSON from closing the <script> element
    Path(path).write_text(
        _TEMPLATE.replace("__TITLE__", html.escape(title)).replace("__DATA__", data)
    )


_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root { --bg:#fbfbf9; --fg:#1d1d1b; --muted:#6b6b66; --line:#e3e2dc; --card:#fff;
        --hi:255,120,40; --accent:#2f5fb3; }
@media (prefers-color-scheme: dark) { :root { --bg:#161615; --fg:#ecebe6; --muted:#9a9993;
        --line:#30302d; --card:#1f1f1d; --accent:#8fb0ec; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg);
       font:14px/1.45 -apple-system, system-ui, sans-serif; }
header { padding:14px 16px; border-bottom:1px solid var(--line); }
header h1 { font-size:17px; margin:0 0 4px; } header p { margin:0; color:var(--muted); }
.wrap { display:grid; grid-template-columns:260px 1fr; min-height:calc(100vh - 70px); }
nav { border-right:1px solid var(--line); overflow:auto; max-height:calc(100vh - 70px); }
nav input { width:calc(100% - 16px); margin:8px; padding:6px 8px; border:1px solid var(--line);
            background:var(--card); color:var(--fg); border-radius:6px; }
nav a { display:block; padding:5px 12px; color:var(--fg); text-decoration:none;
        border-bottom:1px solid var(--line); font-size:13px; }
nav a.on { background:var(--line); } nav a small { color:var(--muted); }
main { padding:16px; overflow:auto; max-height:calc(100vh - 70px); }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px;
        margin-bottom:12px; }
.row { display:flex; gap:12px; flex-wrap:wrap; }
.row .card { flex:1 1 260px; }
h2 { margin:0 0 8px; font-size:16px; } h3 { margin:0 0 6px; font-size:13px;
     color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }
.ex { font-family:ui-monospace, Menlo, monospace; font-size:12.5px; padding:4px 0;
      border-bottom:1px dashed var(--line); white-space:pre-wrap; word-break:break-word; }
.ex span { border-radius:2px; } .ex .tgt { outline:1.5px solid var(--accent); }
.ex b { color:var(--muted); font-weight:500; margin-right:6px; }
table { border-collapse:collapse; width:100%; font-size:13px; }
td { padding:2px 6px; border-bottom:1px solid var(--line);
     font-family:ui-monospace, Menlo, monospace; }
.lab { padding:6px 8px; border-radius:6px; background:var(--line); margin-top:6px; }
@media (max-width:700px) { .wrap { grid-template-columns:1fr; }
  nav { max-height:200px; border-right:none; border-bottom:1px solid var(--line); } }
</style></head><body>
<header><h1>__TITLE__</h1><p id="sum"></p></header>
<div class="wrap"><nav><input id="q" placeholder="filter: id, label or token"><div id="list">
</div></nav><main id="main"></main></div>
<script>
const D = __DATA__;
const esc = s => s.replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const vis = s => esc(s).replace(/\n/g, '⏎');
document.getElementById('sum').textContent = Object.entries(D.summary)
  .map(([k, v]) => k + ': ' + v).join(' · ');
function hist(h) {
  const W = 300, H = 70, m = Math.max(1, ...h), bw = W / h.length;
  let s = `<svg viewBox="0 0 ${W} ${H + 14}" width="100%" role="img" aria-label="histogram">`;
  h.forEach((c, i) => { const bh = c / m * H;
    s += `<rect x="${i * bw}" y="${H - bh}" width="${bw - 1}" height="${bh}" fill="currentColor"
           opacity=".55"><title>10^${D.edges[i].toFixed(1)}: ${c}</title></rect>`; });
  s += `<text x="0" y="${H + 12}" font-size="10" fill="currentColor">10^${D.edges[0]}</text>`;
  s += `<text x="${W}" y="${H + 12}" font-size="10" text-anchor="end"
         fill="currentColor">10^${D.edges[D.edges.length - 1]}</text></svg>`;
  return s; }
function show(id) {
  const f = D.features.find(x => x.id === id); if (!f) return;
  document.querySelectorAll('nav a').forEach(a => a.classList.toggle('on', +a.dataset.id === id));
  const lab = f.label ? `<div class="lab"><b>${esc(f.label.label)}</b> — ${esc(f.label.confidence)}
     ${f.label.description ? '<br>' + esc(f.label.description) : ''}</div>` : '';
  const tbl = rows => '<table>' + rows.map(([t, v]) =>
     `<tr><td>${vis(t)}</td><td style="text-align:right">${v}</td></tr>`).join('') + '</table>';
  const exs = f.examples.map(e => { const m = Math.max(...e.acts, 1e-6);
     return `<div class="ex"><b>${e.max.toFixed(2)}</b>` + e.toks.map((t, j) =>
       `<span class="${j === e.target ? 'tgt' : ''}" title="${e.acts[j]}"
        style="background:rgba(var(--hi),${(Math.max(e.acts[j], 0) / m * .85).toFixed(2)})">${vis(t)}</span>`)
       .join('') + '</div>'; }).join('');
  document.getElementById('main').innerHTML = `
   <div class="card"><h2>Feature ${f.id}</h2>
     fires on ${(f.freq * 100).toFixed(3)}% of tokens · mean activation when active
     ${f.mean.toFixed(3)}${lab}</div>
   <div class="row"><div class="card"><h3>Activation histogram (log10)</h3>${hist(f.hist)}</div>
     <div class="card"><h3>Logits boosted</h3>${tbl(f.boost)}</div>
     <div class="card"><h3>Logits suppressed</h3>${tbl(f.suppress)}</div></div>
   <div class="card"><h3>Top-activating contexts</h3>${exs || '<i>never fired</i>'}</div>`;
  history.replaceState(null, '', '#' + id); }
function list(q) {
  q = (q || '').toLowerCase();
  const items = D.features.filter(f => !q || String(f.id) === q ||
     (f.label && f.label.label.toLowerCase().includes(q)) ||
     f.examples.some(e => e.toks[e.target].toLowerCase().includes(q)));
  document.getElementById('list').innerHTML = items.map(f =>
     `<a href="#${f.id}" data-id="${f.id}">#${f.id} <small>${f.label ? esc(f.label.label) :
       (f.examples[0] ? vis(f.examples[0].toks[f.examples[0].target]) : '—')}</small></a>`).join('');
  document.querySelectorAll('nav a').forEach(a => a.onclick = e => { e.preventDefault();
     show(+a.dataset.id); }); }
document.getElementById('q').oninput = e => list(e.target.value);
list(); show(+location.hash.slice(1) || D.features[0].id);
</script></body></html>
"""
