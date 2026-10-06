(function () {
  const $ = (s) => document.querySelector(s);
  const state = { tickers: new Set() };
  const MODE_LABEL = { vector: 'Vector only', keyword: 'Keyword only', hybrid: 'Hybrid', hybrid_rerank: 'Hybrid + rerank' };

  // Theme toggle
  $('#theme').addEventListener('click', () => {
    const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    try { localStorage.setItem('fr-theme', next); } catch (e) {}
  });

  // Tabs
  const tabs = [['#tab-ask', '#view-ask'], ['#tab-evals', '#view-evals']];
  tabs.forEach(([t, v]) => $(t).addEventListener('click', () => {
    tabs.forEach(([t2, v2]) => { $(t2).setAttribute('aria-selected', String(t2 === t)); $(v2).hidden = v2 !== v; });
    if (v === '#view-evals') loadEvals();
  }));

  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  // Company chips from the indexed filings
  fetch('/api/filings').then((r) => r.ok ? r.json() : []).then((rows) => {
    const seen = new Map();
    rows.forEach((r) => { if (!seen.has(r.ticker)) seen.set(r.ticker, r.company); });
    const box = $('#tickers');
    if (!seen.size) { box.innerHTML = '<span class="help">No filings indexed yet. Run: filings-rag ingest</span>'; return; }
    seen.forEach((company, ticker) => {
      const b = document.createElement('button');
      b.type = 'button'; b.className = 'chip'; b.textContent = ticker; b.title = company;
      b.setAttribute('aria-pressed', 'false');
      b.addEventListener('click', () => {
        const on = !state.tickers.has(ticker);
        on ? state.tickers.add(ticker) : state.tickers.delete(ticker);
        b.setAttribute('aria-pressed', String(on));
      });
      box.appendChild(b);
    });
  });

  document.querySelectorAll('.ex').forEach((b) => b.addEventListener('click', () => {
    $('#q').value = b.textContent; $('#ask-form').requestSubmit();
  }));

  function renderAnswer(data) {
    const cited = new Map(data.citations.map((c) => [c.n, c]));
    // Link [n] markers to their source entries
    const html = esc(data.answer).replace(/\[(\d+)\]/g, (m, n) =>
      cited.has(Number(n)) ? `<a class="cite" href="#src-${n}" aria-label="Source ${n}">[${n}]</a>` : '');
    $('#answer-text').innerHTML = html;
    $('#sources').innerHTML = data.citations.length
      ? data.citations.map((c) => `<li id="src-${c.n}" value="${c.n}"><span class="where">${esc(c.company)} 10-K FY${c.fiscal_year}, ${esc(c.section)}</span>
           · <a href="${esc(c.url)}" target="_blank" rel="noopener">Open filing</a><p class="snip">${esc(c.snippet)}…</p></li>`).join('')
      : '<li>No sources cited.</li>';
    const t = data.timings_ms || {};
    const total = Object.values(t).reduce((a, b) => a + b, 0);
    $('#meta').textContent = `${MODE_LABEL[data.mode] || data.mode} · ${data.model} · ${Math.round(total)} ms` +
      (t.rerank_ms ? ` (rerank ${Math.round(t.rerank_ms)} ms, answer ${Math.round(t.generate_ms || 0)} ms)` : '');
    $('#answer').hidden = false;
  }

  $('#ask-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const q = $('#q').value.trim();
    if (q.length < 3) return;
    const btn = $('#ask-btn'); const status = $('#status');
    btn.disabled = true; status.className = 'status'; status.textContent = 'Searching the filings…';
    try {
      const r = await fetch('/api/ask', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q, tickers: [...state.tickers], mode: $('#mode').value }),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || `Request failed (${r.status})`);
      status.textContent = '';
      renderAnswer(data);
    } catch (err) {
      status.className = 'status err'; status.textContent = err.message;
    } finally { btn.disabled = false; }
  });

  // Evaluations view
  const pct = (x) => x == null ? '–' : `${Math.round(x * 100)}%`;
  const meter = (x) => x == null ? '' : `<div class="meter" aria-hidden="true"><span style="width:${Math.max(2, x * 100)}%"></span></div>`;
  let evalsLoaded = false;
  async function loadEvals() {
    if (evalsLoaded) return;
    const box = $('#evals');
    const r = await fetch('/api/evals/latest');
    if (!r.ok) { box.innerHTML = '<p class="lead">No evaluation results yet. Run <code>filings-rag eval</code>.</p>'; return; }
    const d = await r.json(); evalsLoaded = true;
    const ret = d.retrieval || {};
    const modes = Object.keys(ret);
    const best = modes.reduce((b, m) => (ret[m].mrr > (ret[b]?.mrr ?? -1) ? m : b), modes[0]);
    let html = `<h2 class="h-small">Retrieval: does the right section come back?</h2>
      <p class="lead">${d.n_questions} questions, top ${d.k} chunks. Hit rate = the expected 10-K section is among them;
      MRR rewards ranking it first.</p>
      <table><thead><tr><th>Method</th><th>Hit rate</th><th></th><th>MRR</th><th>Median search</th></tr></thead><tbody>` +
      modes.map((m) => `<tr class="${m === best ? 'best' : ''}"><td>${MODE_LABEL[m] || m}</td><td class="num">${pct(ret[m].hit_rate)}</td>
        <td>${meter(ret[m].hit_rate)}</td><td class="num">${ret[m].mrr.toFixed(2)}</td><td class="num">${Math.round(ret[m].median_ms)} ms</td></tr>`).join('') +
      '</tbody></table>';
    const g = d.generation;
    if (g) {
      const rows = [['Faithfulness', g.faithfulness, 'Claims in the answer that the sources support'],
        ['Answer relevancy', g.answer_relevancy, 'How directly the answer addresses the question'],
        ['Context precision', g.context_precision, 'Useful sources ranked above useless ones'],
        ['Cited answers', g.citation_rate, 'Answers with at least one valid citation'],
        ['Correct refusals', g.abstention_rate, 'Unanswerable questions answered with "not in the filings"']];
      html += `<h2 class="h-small">Answers: graded by ${esc(g.judge)} (RAGAS)</h2>
        <p class="lead">Answer model: ${esc(g.model)}, retrieval: ${MODE_LABEL[g.mode] || g.mode}.</p>
        <table><thead><tr><th>Metric</th><th>Score</th><th></th><th>What it measures</th></tr></thead><tbody>` +
        rows.map(([n, v, w]) => `<tr><td>${n}</td><td class="num">${pct(v)}</td><td>${meter(v)}</td><td>${w}</td></tr>`).join('') +
        '</tbody></table>';
    }
    html += `<p class="meta">Run on ${esc(d.run_at)} · embeddings: ${esc(d.embed_model)}</p>`;
    box.innerHTML = html;
  }
})();
