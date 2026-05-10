"""HTML / CSS / JS templates served by :mod:`active_inference.web.server`.

Kept as plain string constants — no Jinja, no f-string templates buried in
heredocs. Everything renders to a single-page app with tabs; data is
fetched via ``fetch()`` from the JSON endpoints exposed by the server.

Design constraints (matching the rest of the package):

* No third-party CDNs, no external CSS frameworks. The page must render
  with the network unplugged.
* No external JS. Vanilla JavaScript defined inline.
* Markdown is rendered server-side by a small converter in
  :mod:`active_inference.web.server` so the browser doesn't need a
  library.

Visual identity
---------------

A monochrome dark theme (pure black background, white text, gray
chromework) with a single red accent reserved for active state and
emphasis: chapter tabs that are selected, "Render" buttons, errors,
the favicon dot. Avoid using red for anything ambient — it has to keep
its signal.
"""

from __future__ import annotations


# A tiny SVG favicon: white "ai" mark on black with a red dot. Serving SVG
# directly works in every modern browser and avoids a binary asset.
FAVICON_SVG = """\
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <rect width="64" height="64" rx="10" fill="#000"/>
  <rect x="3" y="3" width="58" height="58" rx="8" fill="none"
        stroke="#262626" stroke-width="1"/>
  <text x="32" y="40" text-anchor="middle"
        font-family="-apple-system, Segoe UI, Helvetica, Arial, sans-serif"
        font-size="28" font-weight="700" fill="#fafafa"
        letter-spacing="-1">ai</text>
  <circle cx="50" cy="14" r="4" fill="#ef4444"/>
</svg>
"""


CSS = """\
:root {
    --bg: #000000;
    --bg-1: #0a0a0a;
    --bg-2: #121212;
    --panel: #161616;
    --panel-hi: #1c1c1c;
    --border: #262626;
    --border-hi: #3a3a3a;
    --text: #fafafa;
    --text-2: #d4d4d4;
    --muted: #8a8a8a;
    --muted-2: #6b6b6b;
    --accent: #ef4444;
    --accent-strong: #f87171;
    --accent-soft: rgba(239, 68, 68, 0.10);
    --accent-line: rgba(239, 68, 68, 0.45);
    --good: #d4d4d4;          /* success uses tone, not chroma */
    --code-bg: #0a0a0a;
    --shadow: 0 6px 22px rgba(0, 0, 0, 0.55);
    --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas,
            "Liberation Mono", monospace;
    --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial,
            sans-serif;
}
* { box-sizing: border-box; }
html, body {
    margin: 0;
    padding: 0;
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    font-size: 14px;
    line-height: 1.5;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}
a { color: var(--text); }
a:hover { color: var(--accent); }
kbd {
    font-family: var(--mono);
    background: var(--panel);
    border: 1px solid var(--border);
    border-bottom-width: 2px;
    padding: 1px 6px;
    border-radius: 4px;
    font-size: 11px;
    color: var(--text-2);
}
header {
    display: grid;
    grid-template-columns: 1fr auto;
    align-items: center;
    padding: 16px 24px;
    border-bottom: 1px solid var(--border);
    background: var(--bg-1);
    position: sticky;
    top: 0;
    z-index: 30;
}
header .brand {
    display: flex;
    align-items: center;
    gap: 12px;
}
header .logo {
    width: 28px;
    height: 28px;
    border-radius: 6px;
    background: #000;
    border: 1px solid var(--border-hi);
    display: grid;
    place-items: center;
    font-weight: 700;
    color: var(--text);
    position: relative;
}
header .logo::after {
    content: "";
    position: absolute;
    top: 3px;
    right: 3px;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--accent);
}
header h1 {
    margin: 0;
    font-size: 15px;
    font-weight: 600;
    letter-spacing: 0.1px;
}
header .subtitle {
    color: var(--muted);
    font-size: 12px;
    margin-top: 2px;
}
header .pill {
    display: inline-block;
    margin-left: 6px;
    padding: 2px 8px;
    border: 1px solid var(--border-hi);
    border-radius: 999px;
    font-size: 10.5px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
header .meta {
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: var(--mono);
    font-size: 11.5px;
    color: var(--muted);
}
header .meta .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 8px var(--accent);
}
main {
    display: grid;
    grid-template-columns: 252px 1fr;
    min-height: calc(100vh - 60px);
}
nav.tabs {
    border-right: 1px solid var(--border);
    padding: 10px 0 32px;
    background: var(--bg-1);
    position: sticky;
    top: 60px;
    align-self: start;
    max-height: calc(100vh - 60px);
    overflow-y: auto;
}
nav.tabs::-webkit-scrollbar { width: 8px; }
nav.tabs::-webkit-scrollbar-thumb { background: var(--border-hi); border-radius: 4px; }
nav.tabs button {
    display: block;
    width: 100%;
    padding: 9px 18px;
    background: transparent;
    border: 0;
    border-left: 2px solid transparent;
    color: var(--text-2);
    text-align: left;
    font-size: 13px;
    cursor: pointer;
    font-family: inherit;
    transition: background 90ms ease, color 90ms ease, border-color 90ms ease;
}
nav.tabs button:hover { color: var(--text); background: var(--bg-2); }
nav.tabs button.active {
    color: var(--text);
    border-left-color: var(--accent);
    background: var(--bg-2);
}
nav.tabs button.active .nav-sub { color: var(--accent-strong); }
nav.tabs .nav-label {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 8px;
}
nav.tabs .nav-sub {
    font-size: 11px;
    color: var(--muted-2);
    font-family: var(--mono);
}
nav.tabs .nav-section-title {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    color: var(--muted-2);
    padding: 16px 18px 6px;
    border-top: 1px solid var(--border);
    margin-top: 6px;
}
nav.tabs .nav-section-title:first-child {
    border-top: 0;
    margin-top: 0;
    padding-top: 6px;
}
section.content { padding: 22px 28px 60px; max-width: 1300px; }
section.content h2 {
    margin: 0 0 4px;
    font-size: 22px;
    font-weight: 600;
    letter-spacing: -0.2px;
}
section.content h2 .accent-bar {
    display: inline-block;
    width: 3px;
    height: 18px;
    background: var(--accent);
    margin-right: 10px;
    border-radius: 2px;
    vertical-align: -3px;
}
section.content .lead {
    color: var(--muted);
    margin: 0 0 22px;
    font-size: 13px;
}
section.content h3 {
    margin: 26px 0 10px;
    font-size: 13px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.15em;
    font-weight: 600;
}
section.content h3 .count {
    color: var(--muted-2);
    font-family: var(--mono);
    font-size: 11px;
    margin-left: 6px;
    letter-spacing: 0.04em;
    text-transform: none;
}
.search-row {
    display: flex;
    gap: 8px;
    align-items: center;
    margin: 0 0 18px;
}
.search-row input {
    flex: 1;
    background: var(--bg-1);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 8px 12px;
    border-radius: 6px;
    font: inherit;
    outline: none;
}
.search-row input:focus { border-color: var(--accent-line); }
.search-row .hint {
    color: var(--muted-2);
    font-size: 11.5px;
    font-family: var(--mono);
}
.metrics {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-bottom: 20px;
}
.metric {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 14px;
}
.metric .label {
    color: var(--muted-2);
    font-size: 10.5px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
}
.metric .value {
    margin-top: 4px;
    font-size: 22px;
    font-weight: 600;
    font-family: var(--mono);
    letter-spacing: -0.5px;
}
.metric.accent .value { color: var(--accent); }
.panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px 18px;
    margin-bottom: 18px;
}
.gallery {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 12px;
}
.gallery figure {
    margin: 0;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    transition: border-color 120ms ease, transform 120ms ease;
}
.gallery figure:hover { border-color: var(--border-hi); }
.gallery a.thumb {
    display: block;
    background: #000;
    border-bottom: 1px solid var(--border);
    cursor: zoom-in;
}
.gallery img {
    width: 100%;
    height: auto;
    display: block;
}
.gallery figcaption {
    padding: 9px 12px;
    font-size: 12px;
    color: var(--text-2);
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 4px 10px;
    align-items: baseline;
}
.gallery figcaption .fname {
    font-family: var(--mono);
    font-size: 11.5px;
    color: var(--text);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.gallery figcaption .badge {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--muted);
    border: 1px solid var(--border-hi);
    padding: 1px 6px;
    border-radius: 999px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.gallery figcaption .badge.gif { color: var(--accent); border-color: var(--accent-line); }
.gallery figcaption .meta-line {
    grid-column: 1 / -1;
    color: var(--muted-2);
    font-family: var(--mono);
    font-size: 10.5px;
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
}
.gallery figcaption .meta-line .from {
    color: var(--muted);
    cursor: pointer;
}
.gallery figcaption .meta-line .from:hover { color: var(--accent); }

.script-list {
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
}
.script-row {
    display: grid;
    grid-template-columns: 86px 1fr auto;
    align-items: center;
    gap: 14px;
    padding: 11px 14px;
    border-bottom: 1px solid var(--border);
    background: var(--panel);
}
.script-row:last-child { border-bottom: 0; }
.script-row:hover { background: var(--panel-hi); }
.script-row .kind {
    font-family: var(--mono);
    text-transform: uppercase;
    font-size: 10px;
    letter-spacing: 0.1em;
    color: var(--muted);
    border: 1px solid var(--border-hi);
    padding: 2px 6px;
    border-radius: 4px;
    text-align: center;
    background: var(--bg-1);
}
.script-row .kind.example { color: var(--text); }
.script-row .kind.animation { color: var(--accent); border-color: var(--accent-line); }
.script-row .kind.visualize { color: var(--text-2); }
.script-row .kind.interactive { color: var(--accent-strong); border-color: var(--accent-line); }
.script-row .kind.concept { color: var(--muted); }
.script-row .info { min-width: 0; }
.script-row .name {
    font-family: var(--mono);
    font-size: 13px;
    color: var(--text);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.script-row .name .example-tag {
    font-size: 10px;
    color: var(--accent);
    font-weight: 600;
    margin-left: 6px;
    letter-spacing: 0.04em;
    background: var(--accent-soft);
    padding: 1px 5px;
    border-radius: 3px;
}
.script-row .docstring {
    color: var(--muted);
    font-size: 12px;
    margin-top: 2px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.script-row .path {
    color: var(--muted-2);
    font-family: var(--mono);
    font-size: 10.5px;
    margin-top: 3px;
}
.script-row .meta-chips {
    display: flex;
    gap: 6px;
    margin-top: 4px;
    color: var(--muted-2);
    font-family: var(--mono);
    font-size: 10px;
}
.script-row .meta-chips span { color: var(--muted-2); }
.script-row .actions { display: flex; gap: 6px; }

button.btn {
    padding: 6px 12px;
    border-radius: 6px;
    background: transparent;
    color: var(--text-2);
    border: 1px solid var(--border-hi);
    cursor: pointer;
    font-size: 12px;
    font-family: inherit;
    transition: border-color 120ms ease, color 120ms ease, background 120ms ease;
}
button.btn:hover { border-color: var(--text-2); color: var(--text); }
button.btn[disabled] { opacity: 0.45; cursor: not-allowed; }
button.btn.primary {
    border-color: var(--accent);
    color: var(--accent);
    background: var(--accent-soft);
}
button.btn.primary:hover { background: rgba(239, 68, 68, 0.18); color: var(--accent-strong); }
button.btn.ghost { border-color: transparent; color: var(--muted); }
button.btn.ghost:hover { color: var(--text); }
button.btn .spinner {
    display: inline-block;
    width: 10px;
    height: 10px;
    border: 2px solid currentColor;
    border-right-color: transparent;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    margin-right: 6px;
    vertical-align: -1px;
}
@keyframes spin { to { transform: rotate(360deg); } }

.docs-body, .markdown {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px 24px;
    overflow-x: auto;
    color: var(--text-2);
}
.markdown h1 { font-size: 22px; margin: 0 0 14px; color: var(--text); }
.markdown h2 { font-size: 17px; margin: 18px 0 10px; color: var(--text); }
.markdown h2::before {
    content: "";
    display: inline-block;
    width: 3px;
    height: 14px;
    background: var(--accent);
    margin-right: 8px;
    border-radius: 1px;
    vertical-align: -2px;
}
.markdown h3 { font-size: 14px; margin: 16px 0 8px; color: var(--text); text-transform: uppercase; letter-spacing: 0.12em; }
.markdown p { margin: 0 0 12px; color: var(--text-2); }
.markdown a { color: var(--accent); text-decoration: none; border-bottom: 1px solid transparent; }
.markdown a:hover { border-bottom-color: var(--accent); }
.markdown code {
    font-family: var(--mono);
    background: var(--code-bg);
    padding: 1px 5px;
    border-radius: 4px;
    font-size: 12px;
    border: 1px solid var(--border);
    color: var(--text);
}
.markdown pre {
    background: var(--code-bg);
    padding: 12px 14px;
    border-radius: 6px;
    border: 1px solid var(--border);
    overflow-x: auto;
}
.markdown pre code {
    background: transparent;
    padding: 0;
    border: 0;
    font-size: 12.5px;
    color: var(--text);
}
.markdown table {
    width: 100%;
    border-collapse: collapse;
    margin: 8px 0 14px;
    font-size: 13px;
}
.markdown th, .markdown td {
    border: 1px solid var(--border);
    padding: 6px 10px;
    text-align: left;
    color: var(--text-2);
}
.markdown th {
    background: var(--bg-1);
    color: var(--text);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-size: 11.5px;
    font-weight: 600;
}
.markdown blockquote {
    border-left: 2px solid var(--accent);
    margin: 12px 0;
    padding: 4px 14px;
    color: var(--muted);
    background: var(--accent-soft);
    border-radius: 0 6px 6px 0;
}
.markdown ul, .markdown ol { padding-left: 22px; margin: 0 0 12px; color: var(--text-2); }
.markdown li { margin-bottom: 3px; }
.markdown hr { border: 0; border-top: 1px solid var(--border); margin: 16px 0; }

.empty {
    color: var(--muted-2);
    font-style: italic;
    padding: 14px;
    border: 1px dashed var(--border);
    border-radius: 6px;
    background: var(--bg-1);
}
.notice {
    background: var(--bg-1);
    border: 1px solid var(--border-hi);
    color: var(--muted);
    padding: 10px 14px;
    border-radius: 6px;
    font-size: 12px;
    margin-bottom: 12px;
}
.notice.warn {
    border-left: 3px solid var(--accent);
    color: var(--text-2);
}

/* Dialog (command popups, render output) */
.dialog {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.78);
    display: none;
    align-items: center;
    justify-content: center;
    z-index: 50;
    padding: 20px;
}
.dialog.open { display: flex; }
.dialog .card {
    background: var(--bg-1);
    border: 1px solid var(--border-hi);
    border-radius: 10px;
    width: min(760px, 95vw);
    max-height: 85vh;
    overflow-y: auto;
    padding: 20px 22px;
    box-shadow: var(--shadow);
}
.dialog .card header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: transparent;
    border: 0;
    padding: 0 0 12px;
    position: static;
}
.dialog .card h3 { margin: 0; font-size: 16px; color: var(--text); text-transform: none; letter-spacing: 0; }
.dialog .card .header-actions { display: flex; gap: 8px; }
.dialog pre {
    background: var(--code-bg);
    padding: 12px 14px;
    border-radius: 6px;
    border: 1px solid var(--border);
    overflow-x: auto;
    font-size: 12.5px;
    color: var(--text);
}
.dialog .tabs-row {
    display: flex;
    gap: 6px;
    margin-bottom: 10px;
}
.dialog .tabs-row button {
    background: transparent;
    border: 0;
    color: var(--muted);
    font-family: var(--mono);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 4px 8px;
    cursor: pointer;
    border-bottom: 2px solid transparent;
}
.dialog .tabs-row button.active { color: var(--text); border-bottom-color: var(--accent); }

/* Lightbox (figure preview) */
.lightbox {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.92);
    display: none;
    align-items: center;
    justify-content: center;
    z-index: 60;
    padding: 30px;
}
.lightbox.open { display: flex; }
.lightbox img {
    max-width: 95vw;
    max-height: 90vh;
    border: 1px solid var(--border-hi);
    border-radius: 4px;
    background: #000;
}
.lightbox .caption {
    position: absolute;
    bottom: 12px;
    left: 50%;
    transform: translateX(-50%);
    color: var(--muted);
    font-family: var(--mono);
    font-size: 11.5px;
    background: var(--bg-1);
    padding: 4px 10px;
    border: 1px solid var(--border);
    border-radius: 4px;
}
.lightbox .close {
    position: absolute;
    top: 16px;
    right: 18px;
    background: transparent;
    border: 1px solid var(--border-hi);
    color: var(--text);
    width: 30px;
    height: 30px;
    border-radius: 50%;
    font-size: 16px;
    cursor: pointer;
}

/* Toast */
.toast {
    position: fixed;
    bottom: 24px;
    right: 24px;
    background: var(--bg-1);
    border: 1px solid var(--border-hi);
    border-left: 3px solid var(--text-2);
    color: var(--text);
    padding: 10px 14px;
    border-radius: 6px;
    font-size: 13px;
    box-shadow: var(--shadow);
    display: none;
    max-width: 380px;
}
.toast.bad { border-left-color: var(--accent); color: var(--accent-strong); }
.toast.good { border-left-color: var(--text-2); }
.toast.show { display: block; }

footer {
    padding: 14px 24px;
    border-top: 1px solid var(--border);
    color: var(--muted-2);
    font-size: 11.5px;
    font-family: var(--mono);
    display: flex;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
}
footer .accent { color: var(--accent); }

/* Responsive */
@media (max-width: 880px) {
    main { grid-template-columns: 1fr; }
    nav.tabs {
        position: static;
        max-height: none;
        border-right: 0;
        border-bottom: 1px solid var(--border);
    }
    .metrics { grid-template-columns: repeat(2, 1fr); }
}
"""


JS = r"""
const state = {
    chapters: [],
    docPages: [],
    activeTab: null,
    chapterCache: new Map(),  // number → payload
    busy: new Set(),
    scriptFilter: '',
};

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

function toast(message, kind = 'good', ms = 3500) {
    const el = $('#toast');
    el.textContent = message;
    el.classList.remove('good', 'bad');
    el.classList.add(kind, 'show');
    clearTimeout(toast._t);
    toast._t = setTimeout(() => el.classList.remove('show'), ms);
}

async function fetchJSON(url, opts) {
    const resp = await fetch(url, opts);
    if (!resp.ok) {
        const text = await resp.text();
        let detail;
        try { detail = JSON.parse(text).error || text; } catch { detail = text; }
        throw new Error(`${resp.status}: ${detail}`);
    }
    return resp.json();
}

async function loadIndex() {
    const data = await fetchJSON('/api/index');
    state.chapters = data.chapters;
    state.docPages = data.docs;
    renderNav();
    if (state.chapters.length) {
        activate(`chapter-${state.chapters[0].number}`);
    } else {
        activate('about');
    }
}

function renderNav() {
    const nav = $('#tabs');
    nav.innerHTML = '';
    nav.appendChild(navSection('Chapters'));
    for (const ch of state.chapters) {
        nav.appendChild(navButton(
            `chapter-${ch.number}`,
            `Chapter ${String(ch.number).padStart(2, '0')}`,
            `${ch.figure_count}/${ch.scripts.length}`,
            ch.subtitle,
        ));
    }
    nav.appendChild(navSection('Documentation'));
    for (const doc of state.docPages) {
        nav.appendChild(navButton(`doc-${doc.id}`, doc.title, '', doc.subtitle));
    }
    nav.appendChild(navSection('Meta'));
    nav.appendChild(navButton('about', 'About'));
}

function navSection(title) {
    const div = document.createElement('div');
    div.className = 'nav-section-title';
    div.textContent = title;
    return div;
}

function navButton(id, label, suffix = '', subtitle = '') {
    const b = document.createElement('button');
    b.dataset.tab = id;
    const top = document.createElement('div');
    top.className = 'nav-label';
    const labelEl = document.createElement('span');
    labelEl.textContent = label;
    top.appendChild(labelEl);
    if (suffix) {
        const sub = document.createElement('span');
        sub.className = 'nav-sub';
        sub.textContent = suffix;
        top.appendChild(sub);
    }
    b.appendChild(top);
    if (subtitle) {
        const note = document.createElement('div');
        note.style.color = 'var(--muted-2)';
        note.style.fontSize = '11px';
        note.style.marginTop = '2px';
        note.textContent = subtitle;
        b.appendChild(note);
    }
    b.addEventListener('click', () => activate(id));
    return b;
}

function activate(id) {
    state.activeTab = id;
    state.scriptFilter = '';
    $$('#tabs button').forEach(b => b.classList.toggle('active', b.dataset.tab === id));
    const content = $('#content');
    content.innerHTML = '<div class="empty">Loading...</div>';
    if (id === 'about') {
        renderAbout(content);
    } else if (id.startsWith('chapter-')) {
        renderChapter(content, parseInt(id.slice('chapter-'.length), 10));
    } else if (id.startsWith('doc-')) {
        renderDoc(content, id.slice('doc-'.length));
    }
}

function renderAbout(root) {
    const scriptTotal = state.chapters.reduce((acc, c) => acc + c.scripts.length, 0);
    const figureTotal = state.chapters.reduce((acc, c) => acc + c.figure_count, 0);
    root.innerHTML = `
        <h2><span class="accent-bar"></span>About this UI</h2>
        <p class="lead">A local browser front end for the chapter orchestrators.</p>
        <div class="metrics">
            <div class="metric"><div class="label">Chapters</div><div class="value">${state.chapters.length}</div></div>
            <div class="metric"><div class="label">Scripts</div><div class="value">${scriptTotal}</div></div>
            <div class="metric"><div class="label">Cached figures</div><div class="value">${figureTotal}</div></div>
            <div class="metric accent"><div class="label">Docs linked</div><div class="value">${state.docPages.length}</div></div>
        </div>
        <div class="markdown">
            <p>This page is served by <code>active_inference.web</code>, a stdlib-only
            HTTP server that shares discovery with the text menu (<code>./run.sh</code>).
            Pick a chapter on the left to browse its figures, animations, and the
            scripts that produced them.</p>
            <p>The <strong>Render</strong> button on each script re-runs the
            orchestrator with <code>--save</code> under <code>MPLBACKEND=Agg</code>;
            the gallery refreshes once new files are on disk. Interactive scripts
            open a matplotlib slider window on the host running the server.</p>
            <p>Theme: mono with a single red accent — reserved for active state,
            primary actions, and warnings. Press <kbd>/</kbd> to focus the script
            filter inside any chapter tab.</p>
            <p style="border-top:1px solid var(--border);padding-top:14px;margin-top:18px;color:var(--muted);font-size:12.5px">
                This is an open-source companion maintained by the
                <a href="https://activeinference.institute/" target="_blank" rel="noopener noreferrer">Active Inference Institute</a>.
                Live, cohort-based reading groups for the book run continuously —
                register at
                <a href="https://textbook-group.activeinference.institute/" target="_blank" rel="noopener noreferrer">textbook-group.activeinference.institute</a>.
            </p>
        </div>
    `;
}

async function renderDoc(root, docId) {
    try {
        const data = await fetchJSON(`/api/doc/${encodeURIComponent(docId)}`);
        root.innerHTML = `
            <h2><span class="accent-bar"></span>${escapeHtml(data.title)}</h2>
            <p class="lead">Source: <code>${escapeHtml(data.source)}</code></p>
            <div class="markdown">${data.html}</div>
        `;
    } catch (err) {
        root.innerHTML = `<div class="empty">Failed to load document: ${escapeHtml(err.message)}</div>`;
    }
}

async function renderChapter(root, number) {
    try {
        const data = await fetchJSON(`/api/chapter/${number}`);
        state.chapterCache.set(number, data);
        const examples = data.scripts.filter(s => s.kind === 'example' || s.kind === 'concept');
        const animations = data.scripts.filter(s => s.kind === 'animation');
        const visualizations = data.scripts.filter(s => s.kind === 'visualize');
        const interactives = data.scripts.filter(s => s.kind === 'interactive');
        const animationFigs = data.figures.filter(f => f.kind === 'animation');
        const staticFigs = data.figures.filter(f => f.kind !== 'animation');

        root.innerHTML = `
            <h2><span class="accent-bar"></span>${data.title}
                ${data.subtitle ? `<span style="color:var(--muted);font-weight:400;font-size:15px;margin-left:8px">${escapeHtml(data.subtitle)}</span>` : ''}</h2>
            <p class="lead"><code>${escapeHtml(data.relative)}</code> ·
                ${data.scripts.length} scripts · ${data.figures.length} rendered files</p>
            <div class="metrics">
                <div class="metric"><div class="label">Examples</div><div class="value">${examples.length}</div></div>
                <div class="metric"><div class="label">Animations</div><div class="value">${animations.length}</div></div>
                <div class="metric"><div class="label">Visualizations</div><div class="value">${visualizations.length}</div></div>
                <div class="metric accent"><div class="label">Cached figures</div><div class="value">${data.figures.length}</div></div>
            </div>

            <div class="search-row">
                <input id="script-filter" type="text"
                       placeholder="Filter scripts and figures by name or docstring..." autocomplete="off" />
                <span class="hint">press <kbd>/</kbd> to focus · <kbd>Esc</kbd> to clear</span>
            </div>

            <h3>Figures <span class="count">· ${staticFigs.length} static</span></h3>
            <div id="figures-static"></div>

            ${animationFigs.length ? `<h3>Animations <span class="count">· ${animationFigs.length} GIF</span></h3>
            <div id="figures-anim"></div>` : ''}

            <h3>Examples &amp; concept demos <span class="count">· ${examples.length}</span></h3>
            <div id="examples-section"></div>

            ${animations.length ? `<h3>Animation scripts <span class="count">· ${animations.length}</span></h3>
            <div id="animations-section"></div>` : ''}

            ${visualizations.length ? `<h3>Diagnostic visualizations <span class="count">· ${visualizations.length}</span></h3>
            <div id="visualizations-section"></div>` : ''}

            ${interactives.length ? `<h3>Interactive scripts <span class="count">· slider windows</span></h3>
            <div id="interactives-section"></div>` : ''}

            ${data.readme_html ? `<h3>Chapter README</h3>
            <div class="markdown">${data.readme_html}</div>` : ''}

            <h3>Documentation</h3>
            <div id="docs-section"></div>
        `;

        renderGallery($('#figures-static'), staticFigs, number);
        if (animationFigs.length) renderGallery($('#figures-anim'), animationFigs, number);
        renderScriptList($('#examples-section'), examples, number);
        if (animations.length) renderScriptList($('#animations-section'), animations, number);
        if (visualizations.length) renderScriptList($('#visualizations-section'), visualizations, number);
        if (interactives.length) renderScriptList($('#interactives-section'), interactives, number);
        renderDocLinks($('#docs-section'), data.docs);
        wireFilter(number);
    } catch (err) {
        root.innerHTML = `<div class="empty">Failed to load chapter: ${escapeHtml(err.message)}</div>`;
    }
}

function renderGallery(host, figures, chapter) {
    if (!host) return;
    host.innerHTML = '';
    if (!figures.length) {
        host.innerHTML = `<div class="notice warn">No figures yet — click <strong>Render</strong> on
            any script below to populate this gallery.</div>`;
        return;
    }
    const grid = document.createElement('div');
    grid.className = 'gallery';
    for (const f of figures) {
        const fig = document.createElement('figure');
        fig.dataset.name = f.name.toLowerCase();
        fig.dataset.from = (f.generated_by || '').toLowerCase();
        const dim = (f.width && f.height) ? `${f.width}×${f.height}` : '';
        const generator = f.generated_by ? `<span class="from" data-script="${escapeAttr(f.generated_by)}"
            title="Jump to the script that produced this figure">${escapeHtml(f.generated_by)}</span>` : '';
        fig.innerHTML = `
            <a class="thumb" data-url="${escapeAttr(f.url)}" data-name="${escapeAttr(f.name)}"
               data-caption="${escapeAttr(`${f.name} · ${dim} · ${f.size_human}`)}">
                <img loading="lazy" src="${escapeAttr(f.url)}" alt="${escapeAttr(f.name)}">
            </a>
            <figcaption>
                <span class="fname" title="${escapeAttr(f.name)}">${escapeHtml(f.name)}</span>
                <span class="badge ${f.kind === 'animation' ? 'gif' : ''}">${(f.extension || 'png').toUpperCase()}</span>
                <span class="meta-line">
                    ${dim ? `<span>${dim}</span>` : ''}
                    <span>${f.size_human}</span>
                    <span>${f.mtime_human || ''}</span>
                    ${generator}
                </span>
            </figcaption>
        `;
        fig.querySelector('.thumb').addEventListener('click', (ev) => {
            ev.preventDefault();
            openLightbox(f.url, ev.currentTarget.dataset.caption);
        });
        const from = fig.querySelector('.from');
        if (from) {
            from.addEventListener('click', (ev) => {
                ev.preventDefault();
                ev.stopPropagation();
                scrollToScript(from.dataset.script);
            });
        }
        grid.appendChild(fig);
    }
    host.appendChild(grid);
}

function renderScriptList(host, scripts, chapter) {
    if (!host) return;
    host.innerHTML = '';
    if (!scripts.length) {
        host.innerHTML = '<div class="empty">(none in this section)</div>';
        return;
    }
    const wrap = document.createElement('div');
    wrap.className = 'script-list';
    for (const s of scripts) {
        const row = document.createElement('div');
        row.className = 'script-row';
        row.id = `script-${s.name.replace(/[^a-z0-9_]/gi, '_')}`;
        row.dataset.name = s.name.toLowerCase();
        row.dataset.docstring = (s.docstring || '').toLowerCase();
        const exampleTag = s.example_number
            ? `<span class="example-tag">§ ${s.example_number}</span>` : '';
        const meta = [];
        if (s.size_human) meta.push(s.size_human);
        if (s.mtime_human) meta.push(s.mtime_human);
        const interactive = s.kind === 'interactive';
        const primaryLabel = interactive ? 'Launch on host' : 'Render';
        const primaryAction = interactive ? 'launch' : 'run';
        row.innerHTML = `
            <div class="kind ${s.kind}">${s.kind}</div>
            <div class="info">
                <div class="name">${escapeHtml(s.name)}${exampleTag}</div>
                ${s.docstring ? `<div class="docstring" title="${escapeAttr(s.docstring)}">${escapeHtml(s.docstring)}</div>` : ''}
                <div class="path">${escapeHtml(s.relative)}</div>
                <div class="meta-chips">
                    ${meta.map(m => `<span>${escapeHtml(m)}</span>`).join('')}
                </div>
            </div>
            <div class="actions">
                <button class="btn ghost" data-action="cmd">Command</button>
                <button class="btn primary" data-action="${primaryAction}">${primaryLabel}</button>
            </div>
        `;
        row.querySelector('[data-action="cmd"]').addEventListener('click',
            () => showCommandDialog(s, chapter));
        row.querySelector(`[data-action="${primaryAction}"]`).addEventListener('click',
            (ev) => interactive ? launchInteractive(s, ev.currentTarget)
                                : runScript(s, chapter, ev.currentTarget));
        wrap.appendChild(row);
    }
    host.appendChild(wrap);
}

function renderDocLinks(host, docs) {
    if (!host) return;
    host.innerHTML = '';
    if (!docs.length) {
        host.innerHTML = '<div class="empty">(no docs linked for this chapter)</div>';
        return;
    }
    const ul = document.createElement('ul');
    ul.style.lineHeight = '1.9';
    ul.style.fontSize = '13px';
    for (const d of docs) {
        const li = document.createElement('li');
        const a = document.createElement('a');
        a.href = '#';
        a.style.color = 'var(--accent)';
        a.textContent = d.title;
        a.addEventListener('click', (ev) => { ev.preventDefault(); activate(`doc-${d.id}`); });
        li.appendChild(a);
        const span = document.createElement('span');
        span.style.color = 'var(--muted-2)';
        span.style.marginLeft = '8px';
        span.style.fontFamily = 'var(--mono)';
        span.style.fontSize = '11px';
        span.textContent = '· ' + d.path;
        li.appendChild(span);
        ul.appendChild(li);
    }
    host.appendChild(ul);
}

function wireFilter(chapter) {
    const input = $('#script-filter');
    if (!input) return;
    input.addEventListener('input', (ev) => {
        applyFilter(ev.currentTarget.value.toLowerCase().trim());
    });
    input.addEventListener('keydown', (ev) => {
        if (ev.key === 'Escape') {
            ev.currentTarget.value = '';
            applyFilter('');
        }
    });
}

function applyFilter(q) {
    state.scriptFilter = q;
    $$('.script-row').forEach(row => {
        const hit = !q || row.dataset.name.includes(q) ||
                    (row.dataset.docstring && row.dataset.docstring.includes(q));
        row.style.display = hit ? '' : 'none';
    });
    $$('.gallery figure').forEach(fig => {
        const hit = !q || fig.dataset.name.includes(q) ||
                    (fig.dataset.from && fig.dataset.from.includes(q));
        fig.style.display = hit ? '' : 'none';
    });
}

async function runScript(script, chapter, btn) {
    if (state.busy.has(script.name)) return;
    state.busy.add(script.name);
    const original = btn.textContent;
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span>Rendering';
    let result = null;
    try {
        result = await fetchJSON('/api/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ chapter, script: script.name }),
        });
        if (result.returncode === 0) {
            toast(`Rendered ${script.name}`, 'good');
        } else {
            toast(`${script.name} exited ${result.returncode}`, 'bad', 6000);
        }
        if (state.activeTab === `chapter-${chapter}`) {
            // Re-render the chapter to refresh figures + metadata.
            activate(state.activeTab);
        }
    } catch (err) {
        toast(`Failed: ${err.message}`, 'bad', 6000);
    } finally {
        state.busy.delete(script.name);
        btn.disabled = false;
        btn.textContent = original;
        if (result && (result.stdout_tail || result.stderr_tail)) {
            renderRunOutput(script, result);
        }
    }
}

function renderRunOutput(script, result) {
    if (result.returncode === 0 &&
        (!result.stderr_tail || result.stderr_tail.length === 0)) return;
    showOutputDialog(script, result);
}

async function launchInteractive(script, btn) {
    if (state.busy.has(script.name)) return;
    state.busy.add(script.name);
    const original = btn.textContent;
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span>Launching';
    try {
        const data = await fetchJSON('/api/launch-interactive', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ script: script.name }),
        });
        toast(`Opened ${script.name} on host (pid=${data.pid})`, 'good');
    } catch (err) {
        toast(`Could not launch: ${err.message}`, 'bad', 6000);
    } finally {
        state.busy.delete(script.name);
        btn.disabled = false;
        btn.textContent = original;
    }
}

function showCommandDialog(script, chapter) {
    const lines = script.kind === 'interactive'
        ? [
            `# Open ${script.name} in an interactive matplotlib window.`,
            `# Set MPLBACKEND only if you need to override the system default.`,
            `uv run python ${script.relative}`,
        ]
        : [
            `# Re-render ${script.name} into`,
            `# output/figures/chapter_${String(chapter).padStart(2, '0')}/`,
            `uv run python ${script.relative} --save`,
            ``,
            `# Or via the text menu:`,
            `./run.sh --script ${script.name.replace('.py', '')}`,
        ];
    const cmd = lines.join('\n');
    openDialog({
        title: script.name,
        bodyHtml: `<pre id="cmd-body">${escapeHtml(cmd)}</pre>`,
        actions: [
            { label: 'Copy', onclick: async () => {
                try {
                    await navigator.clipboard.writeText(cmd);
                    toast('Copied to clipboard', 'good');
                } catch (err) {
                    toast(`Clipboard unavailable: ${err.message}`, 'bad');
                }
            }},
        ],
    });
}

function showOutputDialog(script, result) {
    const tab = (label, body, isErr) => {
        const safe = (body || []).map(escapeHtml).join('\n');
        return `<pre data-pane="${label}" style="display:${isErr ? 'block' : 'block'}">${safe || '(empty)'}</pre>`;
    };
    const stderr = result.stderr_tail || [];
    const stdout = result.stdout_tail || [];
    openDialog({
        title: `${script.name} · exit ${result.returncode}`,
        bodyHtml: `
            <div class="tabs-row">
                <button class="active" data-pane="stderr">stderr (${stderr.length})</button>
                <button data-pane="stdout">stdout (${stdout.length})</button>
            </div>
            <pre data-pane="stderr">${stderr.map(escapeHtml).join('\n') || '(empty)'}</pre>
            <pre data-pane="stdout" style="display:none">${stdout.map(escapeHtml).join('\n') || '(empty)'}</pre>
        `,
    });
    // Hook up tab toggling
    const card = $('#dialog .card');
    card.querySelectorAll('[data-pane]').forEach(btn => {
        if (btn.tagName !== 'BUTTON') return;
        btn.addEventListener('click', () => {
            card.querySelectorAll('.tabs-row button').forEach(b => b.classList.toggle('active', b === btn));
            const target = btn.dataset.pane;
            card.querySelectorAll('pre[data-pane]').forEach(p => {
                p.style.display = p.dataset.pane === target ? 'block' : 'none';
            });
        });
    });
}

function openDialog({ title, bodyHtml, actions = [] }) {
    const dialog = $('#dialog');
    $('#dialog-title').textContent = title;
    $('#dialog-body').innerHTML = bodyHtml;
    const actionsEl = $('#dialog-actions');
    actionsEl.innerHTML = '';
    for (const a of actions) {
        const b = document.createElement('button');
        b.className = 'btn';
        b.textContent = a.label;
        b.addEventListener('click', a.onclick);
        actionsEl.appendChild(b);
    }
    dialog.classList.add('open');
}

function openLightbox(url, caption) {
    const lb = $('#lightbox');
    $('#lightbox-img').src = url;
    $('#lightbox-caption').textContent = caption;
    lb.classList.add('open');
}

function scrollToScript(scriptName) {
    if (!scriptName) return;
    const id = 'script-' + scriptName.replace(/[^a-z0-9_]/gi, '_');
    const el = document.getElementById(id);
    if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'center' });
        const oldBg = el.style.background;
        el.style.background = 'var(--accent-soft)';
        setTimeout(() => el.style.background = oldBg, 1200);
    }
}

function escapeHtml(s) {
    return String(s)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}
function escapeAttr(s) { return escapeHtml(s); }

document.addEventListener('keydown', (ev) => {
    if (ev.key === '/' && !['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) {
        const input = $('#script-filter');
        if (input) { ev.preventDefault(); input.focus(); }
    } else if (ev.key === 'Escape') {
        $('#dialog').classList.remove('open');
        $('#lightbox').classList.remove('open');
    }
});

document.addEventListener('DOMContentLoaded', () => {
    $('#dialog-close').addEventListener('click', () => $('#dialog').classList.remove('open'));
    $('#dialog').addEventListener('click', (ev) => {
        if (ev.target.id === 'dialog') $('#dialog').classList.remove('open');
    });
    $('#lightbox-close').addEventListener('click', () => $('#lightbox').classList.remove('open'));
    $('#lightbox').addEventListener('click', (ev) => {
        if (ev.target.id === 'lightbox') $('#lightbox').classList.remove('open');
    });
    loadIndex().catch(err => {
        $('#content').innerHTML = `<div class="empty">Failed to load: ${escapeHtml(err.message)}</div>`;
    });
});
"""


INDEX_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Active Inference Fundamentals — Local UI</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#000000">
<meta name="description" content="Local browser interface for the Active Inference Fundamentals Python companion.">
<meta name="robots" content="noindex">
<link rel="icon" type="image/svg+xml" href="/favicon.svg">
<link rel="stylesheet" href="/static/app.css">
</head>
<body>
<header>
    <div class="brand">
        <div class="logo">ai</div>
        <div>
            <h1>Active Inference Fundamentals
                <span class="pill">v{version}</span>
            </h1>
            <div class="subtitle">Python companion to Namjoshi, MIT Press 2026 — runs entirely on your machine</div>
        </div>
    </div>
    <div class="meta"><span class="dot"></span>{host}:{port}</div>
</header>
<main>
<nav class="tabs" id="tabs"></nav>
<section class="content" id="content">
    <div class="empty">Loading…</div>
</section>
</main>
<footer>
    <span>
        Open-source project hosted by the
        <a href="https://activeinference.institute/" target="_blank" rel="noopener noreferrer">Active Inference Institute</a>
        · join an ongoing textbook group:
        <a href="https://textbook-group.activeinference.institute/" target="_blank" rel="noopener noreferrer">textbook-group.activeinference.institute</a>
    </span>
    <span><span class="accent">Ctrl+C</span> in the terminal to stop · press <kbd>/</kbd> to search · <kbd>Esc</kbd> to dismiss</span>
</footer>
<div class="dialog" id="dialog">
    <div class="card">
        <header>
            <h3 id="dialog-title"></h3>
            <div class="header-actions">
                <div id="dialog-actions"></div>
                <button class="btn" id="dialog-close">Close</button>
            </div>
        </header>
        <div id="dialog-body"></div>
    </div>
</div>
<div class="lightbox" id="lightbox">
    <button class="close" id="lightbox-close" aria-label="Close">×</button>
    <img id="lightbox-img" alt="figure">
    <div class="caption" id="lightbox-caption"></div>
</div>
<div class="toast" id="toast"></div>
<script src="/static/app.js"></script>
</body>
</html>
"""


def render_index_html(*, host: str, port: int, version: str) -> str:
    """Substitute the per-instance values into the SPA shell."""
    return INDEX_HTML.format(host=host, port=port, version=version)
