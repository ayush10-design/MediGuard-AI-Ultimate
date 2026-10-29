"""Shared UI helpers: theme injection, sidebar, headers, cards, cursor, scan animation."""
from __future__ import annotations

import base64
import html
from urllib.parse import quote

import streamlit as st
import streamlit.components.v1 as components

import config

# Small inline line icons (stroke-based, 24px grid)
ICONS = {
    "dashboard": '<rect x="3" y="3" width="7" height="9" rx="2"/><rect x="14" y="3" width="7" height="5" rx="2"/><rect x="14" y="12" width="7" height="9" rx="2"/><rect x="3" y="16" width="7" height="5" rx="2"/>',
    "lab": '<path d="M9 3h6M10 3v6L4.5 18.5A2 2 0 0 0 6.2 21.5h11.6a2 2 0 0 0 1.7-3L14 9V3"/><path d="M7 15h10"/>',
    "camera": '<path d="M4 7h3l2-3h6l2 3h3a1 1 0 0 1 1 1v11a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V8a1 1 0 0 1 1-1z"/><circle cx="12" cy="13" r="4"/>',
    "form": '<rect x="4" y="3" width="16" height="18" rx="2"/><path d="M8 8h8M8 12h8M8 16h5"/>',
    "chart": '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>',
    "data": '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>',
    "history": '<path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5M12 7v5l3 2"/>',
    "report": '<path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><path d="M14 3v6h6M8 13h8M8 17h6"/>',
    "settings": '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1A1.7 1.7 0 0 0 4.6 9a1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5v.5"/>',
}


def icon(name: str, size: int = 24) -> str:
    return (f'<svg viewBox="0 0 24 24" width="{size}" height="{size}" fill="none" stroke="currentColor" '
            f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{ICONS.get(name, ICONS["info"])}</svg>')


def logo_svg() -> str:
    try:
        return (config.ASSETS_DIR / "logo.svg").read_text(encoding="utf-8")
    except OSError:
        return ""


def get_settings() -> dict:
    if "settings" not in st.session_state:
        st.session_state["settings"] = config.load_settings()
    return st.session_state["settings"]


def apply_theme() -> None:
    css = ""
    for name in ("styles.css", "animations.css"):
        try:
            css += (config.ASSETS_DIR / name).read_text(encoding="utf-8")
        except OSError:
            pass
    s = get_settings()
    if not s.get("animations", True):
        css += "*,*::before,*::after{animation:none!important;transition:none!important}"
    if s.get("custom_cursor", True):
        css += cursor_css()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    cursor_highlight(s.get("custom_cursor", True) and s.get("animations", True))


# ----------------------------------------------------------------- cursor
_SYRINGE = """<svg xmlns='http://www.w3.org/2000/svg' width='32' height='32' viewBox='0 0 32 32'>
<g transform='rotate(45 16 16)'>
<line x1='1.5' y1='16' x2='10' y2='16' stroke='#5A6E8C' stroke-width='1.4' stroke-linecap='round'/>
<rect x='9.5' y='14' width='2.5' height='4' rx='.6' fill='#142B4F'/>
<rect x='12' y='12' width='12' height='8' rx='1.6' fill='#EAF2FE' stroke='#1769E0' stroke-width='1.3'/>
<rect x='12.8' y='13.2' width='6.5' height='5.6' rx='.8' fill='#1769E0' opacity='.75'/>
<path d='M15 12v2M17.5 12v2M20 12v2' stroke='#1769E0' stroke-width='.8'/>
<rect x='24' y='15' width='4.5' height='2' fill='#142B4F'/>
<rect x='28.3' y='11.5' width='2' height='9' rx='.8' fill='#142B4F'/>
</g></svg>"""


def cursor_css() -> str:
    uri = "data:image/svg+xml;utf8," + quote(" ".join(_SYRINGE.split()))
    # Only on precise pointers (mouse/trackpad); text fields keep the I-beam; links/buttons keep pointer semantics.
    return f"""
@media (pointer: fine) {{
  .stApp, .stApp div, .stApp section, .stApp p, .stApp span, .stApp img, .stApp canvas {{ cursor: url("{uri}") 5 5, auto; }}
  .stApp button, .stApp a, .stApp [role="button"], .stApp [role="tab"], .stApp [role="option"], .stApp label, .stApp summary,
  .stApp [data-baseweb="select"] * {{ cursor: url("{uri}") 5 5, pointer; }}
  .stApp input, .stApp textarea, .stApp [contenteditable="true"] {{ cursor: text !important; }}
}}"""


def cursor_highlight(enabled: bool) -> None:
    """Soft blue glow following the mouse. Injected into the parent page; never intercepts clicks."""
    script = """
<script>
(function(){
  try {
    const d = window.parent.document, w = window.parent;
    const existing = d.getElementById('mg-cursor-glow');
    const enabled = %s;
    const reduce = w.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const fine = w.matchMedia('(pointer: fine)').matches;
    if (!enabled || reduce || !fine) { if (existing) existing.remove(); return; }
    if (existing) return;
    const g = d.createElement('div');
    g.id = 'mg-cursor-glow';
    g.setAttribute('aria-hidden', 'true');
    g.style.cssText = 'position:fixed;left:0;top:0;width:26px;height:26px;margin:-13px 0 0 -13px;border-radius:50%%;' +
      'background:radial-gradient(circle,rgba(23,105,224,.32),rgba(23,105,224,0) 70%%);pointer-events:none;z-index:999999;' +
      'opacity:0;transition:opacity .2s;will-change:transform';
    d.body.appendChild(g);
    let x=0,y=0,tx=0,ty=0;
    d.addEventListener('mousemove', e => { tx=e.clientX; ty=e.clientY; g.style.opacity='1'; }, {passive:true});
    d.addEventListener('mouseleave', () => { g.style.opacity='0'; });
    (function loop(){ x+=(tx-x)*.22; y+=(ty-y)*.22; g.style.transform='translate('+x+'px,'+y+'px)'; w.requestAnimationFrame(loop); })();
  } catch (e) { /* parent not reachable: silently skip decorative effect */ }
})();
</script>""" % ("true" if enabled else "false")
    components.html(script, height=0, width=0)


# ----------------------------------------------------------------- layout pieces
def sidebar_brand() -> None:
    from utils.model_utils import model_status
    from utils import gemini_service
    from utils.ocr_service import tesseract_available
    ms = model_status()
    with st.sidebar:
        st.markdown(f'<div class="mg-brand">{logo_svg()}<div><b>MediGuard AI</b><small>Medicine screening workstation</small></div></div>',
                    unsafe_allow_html=True)

    def chip(ok, yes, no, warn=False):
        cls = "ok" if ok else ("warn" if warn else "bad")
        return f'<span class="mg-chip {cls}"><span class="dot"></span>{yes if ok else no}</span>'

    st.sidebar.markdown(
        '<div class="mg-side-status">'
        f'<div><span>Model</span>{chip(ms["trained"], "Trained", "Not trained", warn=True)}</div>'
        f'<div><span>Gemini</span>{chip(gemini_service.is_configured(), "Configured", "Off", warn=True)}</div>'
        f'<div><span>Tesseract</span>{chip(tesseract_available(), "Found", "Missing", warn=True)}</div>'
        '</div>', unsafe_allow_html=True)


def page_header(title: str, subtitle: str, icon_name: str = "info", right_html: str = "") -> None:
    st.markdown(
        f'<div class="mg-page-head"><div class="mg-head-left"><div class="mg-page-icon">{icon(icon_name)}</div>'
        f'<div><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></div></div><div>{right_html}</div></div>',
        unsafe_allow_html=True)


def chip(text: str, kind: str = "info") -> str:
    return f'<span class="mg-chip {kind}"><span class="dot"></span>{html.escape(text)}</span>'


def note(text: str, kind: str = "") -> None:
    st.markdown(f'<div class="mg-note {kind}">{text}</div>', unsafe_allow_html=True)


def stat(value, label: str, tone: str = "") -> str:
    return (f'<div class="mg-card mg-stat {tone}"><div class="bar"></div><div class="v">{html.escape(str(value))}</div>'
            f'<div class="l">{html.escape(label)}</div></div>')


def card(title: str, body: str, soft: bool = False) -> str:
    return f'<div class="mg-card {"soft" if soft else ""}"><h3>{html.escape(title)}</h3><p>{body}</p></div>'


def image_data_uri(image_bytes: bytes, mime: str = "image/jpeg") -> str:
    return f"data:{mime};base64,{base64.b64encode(image_bytes).decode()}"


STAGES = ["Extracting packaging information", "Preparing image features", "Running neural network", "Generating detailed report"]


def scan_html(image_uri: str, states: list[str], detail: str = "") -> str:
    """states: one of pending|active|done|skipped|failed for each stage."""
    marks = {"done": "✓", "failed": "!", "skipped": "–"}
    items = "".join(f'<li class="{s}"><span class="ic">{marks.get(s, "")}</span>{html.escape(label)}'
                    f'{" — skipped" if s == "skipped" else ""}</li>' for label, s in zip(STAGES, states))
    scanning = any(s == "active" for s in states)
    line = '<div class="line"></div>' if scanning else ""
    return (f'<div class="mg-scan"><img src="{image_uri}" alt="Medicine image being analysed"><div class="grid"></div>{line}'
            '<div class="corner c1"></div><div class="corner c2"></div><div class="corner c3"></div><div class="corner c4"></div></div>'
            f'<ul class="mg-stages" aria-live="polite">{items}</ul>'
            + (f'<div class="mg-note">{html.escape(detail)}</div>' if detail else ""))


def prob_bars_html(scores: dict[str, float]) -> str:
    rows = ""
    for name, v in sorted(scores.items(), key=lambda kv: -kv[1]):
        tone = "fake" if name.lower() == "fake" else "real" if name.lower() == "real" else ""
        rows += (f'<div class="mg-prob"><span>{html.escape(name)}</span><div class="track">'
                 f'<div class="fill {tone}" style="width:{v*100:.1f}%"></div></div><b>{v*100:.1f}%</b></div>')
    return rows


SOURCE_CLASS = {"Visible text (Gemini)": "g", "Manually edited": "m", "Manually entered": "m", "Not detected": "x"}


def fields_table_html(fields: dict) -> str:
    rows = ""
    for f in fields.values():
        src = f.get("status", "")
        cls = SOURCE_CLASS.get(src, "g" if "Tesseract" in src else "")
        rows += (f'<tr><td>{html.escape(f["label"])}</td><td>{html.escape(str(f["value"]))}</td>'
                 f'<td><span class="mg-src {cls}">{html.escape(src)}</span></td></tr>')
    return f'<table class="mg-table"><tr><th>Field</th><th>Value</th><th>Source</th></tr>{rows}</table>'


def section_title(letter: str, title: str) -> None:
    st.markdown(f'<div class="mg-section-title"><div class="k">{letter}</div><h3>{html.escape(title)}</h3></div>',
                unsafe_allow_html=True)
