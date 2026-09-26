"""Turn a pyte screen into something reviewable: colour-preserving ANSI and HTML.

The plain `screen.display` throws away every attribute, which is most of what
"what does it look like" means. These walk the cell buffer instead.
"""

NAMED = {
    "black": "#3b4252", "red": "#d35f5f", "green": "#6a9955",
    "brown": "#b8860b", "yellow": "#d7ba7d", "blue": "#569cd6",
    "magenta": "#c586c0", "cyan": "#4ec9b0", "white": "#d4d4d4",
    "default": None,
}
ANSI_FG = {"black": 30, "red": 31, "green": 32, "brown": 33, "yellow": 33,
           "blue": 34, "magenta": 35, "cyan": 36, "white": 37}


def _rows(screen):
    for y in range(screen.lines):
        row = screen.buffer[y]
        cells = [row[x] for x in range(screen.columns)]
        while cells and cells[-1].data == " " and cells[-1].bg == "default":
            cells.pop()
        yield cells


def to_ansi(screen):
    out = []
    for cells in _rows(screen):
        line, cur = [], None
        for c in cells:
            key = (c.fg, c.bold, c.reverse)
            if key != cur:
                codes = ["0"]
                if c.bold:
                    codes.append("1")
                if c.reverse:
                    codes.append("7")
                if c.fg in ANSI_FG:
                    codes.append(str(ANSI_FG[c.fg]))
                elif c.fg != "default" and len(c.fg) == 6:
                    codes.append("38;2;%d;%d;%d" % (
                        int(c.fg[0:2], 16), int(c.fg[2:4], 16), int(c.fg[4:6], 16)))
                line.append("\x1b[" + ";".join(codes) + "m")
                cur = key
            line.append(c.data)
        line.append("\x1b[0m")
        out.append("".join(line))
    while out and not out[-1].replace("\x1b[0m", "").strip():
        out.pop()
    return "\n".join(out)


def _css(c):
    bits = []
    col = NAMED.get(c.fg, "#" + c.fg if len(c.fg) == 6 else None) if c.fg != "default" else None
    if col:
        bits.append("color:%s" % col)
    if c.bold:
        bits.append("font-weight:600")
    return ";".join(bits)


def _esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def to_html(screen):
    out = []
    for cells in _rows(screen):
        line, buf, style = [], [], None
        for c in cells:
            st = _css(c)
            if st != style:
                if buf:
                    line.append(_span(style, "".join(buf)))
                buf, style = [], st
            buf.append(c.data)
        if buf:
            line.append(_span(style, "".join(buf)))
        out.append("".join(line) or "&nbsp;")
    while len(out) > 1 and out[-1] == "&nbsp;":
        out.pop()
    return "\n".join(out)


def _span(style, text):
    if not style:
        return _esc(text)
    return '<span style="%s">%s</span>' % (style, _esc(text))


PAGE = """<!doctype html><meta charset="utf-8"><title>%(title)s</title>
<style>
 :root{color-scheme:dark}
 body{background:#14161a;color:#d4d4d4;font-family:ui-monospace,"JetBrains Mono",Menlo,monospace;
      margin:0;padding:32px 24px;line-height:1.35}
 h1{font-size:18px;font-weight:600;margin:0 0 4px}
 .sub{color:#7f8792;font-size:13px;margin-bottom:28px}
 .shot{margin:0 0 30px}
 .cap{color:#4ec9b0;font-size:13px;margin:0 0 8px;font-weight:600}
 .cap .t{color:#7f8792;font-weight:400}
 pre{background:#0d0f12;border:1px solid #262b33;border-radius:8px;
     padding:14px 16px;margin:0;overflow-x:auto;font-size:12.5px;white-space:pre}
</style>
<h1>%(title)s</h1>
<div class="sub">%(sub)s</div>
%(body)s
"""


def page(title, sub, shots):
    body = []
    for s in shots:
        body.append('<div class="shot"><div class="cap">%s <span class="t">'
                    '&middot; t=%.2fs</span></div><pre>%s</pre></div>'
                    % (_esc(s["label"]), s["t"], s["html"]))
    return PAGE % {"title": _esc(title), "sub": _esc(sub), "body": "\n".join(body)}
