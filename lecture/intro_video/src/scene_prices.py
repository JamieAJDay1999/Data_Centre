"""A year of British electricity prices as a carpet plot (days across, time of day down)."""
from pathlib import Path

import numpy as np

from engine import *  # noqa: F401,F403
from engine import W, H

GRID = np.load(Path(__file__).resolve().parents[1] / "assets" / "prices_2025.npy")   # (96, 365)
CW, CH = 3, 7
X0, Y0 = 150, 250
RX = 1300
PW, PH = 365 * CW, 96 * CH

STOPS = [(0, (0.05, 0.14, 0.40)), (45, (0.08, 0.50, 0.72)), (80, (0.36, 0.28, 0.55)),
         (115, (0.92, 0.42, 0.16)), (170, (1.00, 0.80, 0.40)), (260, (1.0, 0.97, 0.85))]
NEG = (0.70, 1.00, 0.85)


def cmap(v):
    v = np.asarray(v, np.float32)
    out = np.zeros(v.shape + (3,), np.float32)
    xs = np.array([s[0] for s in STOPS], np.float32)
    for k in range(3):
        out[..., k] = np.interp(v, xs, [s[1][k] for s in STOPS])
    out[v < 0] = NEG
    return out


def _surface(rgb, alpha=None):
    """numpy HxWx3 float (0..1) -> cairo surface scaled to the plot size."""
    h, w = rgb.shape[:2]
    a = np.ones((h, w), np.float32) if alpha is None else alpha
    big = np.repeat(np.repeat(np.concatenate([rgb, a[..., None]], -1), CH, 0), CW, 1)
    bgra = np.empty(big.shape, np.uint8)
    premul = big[..., :3] * big[..., 3:4]
    bgra[..., 0] = (premul[..., 2] * 255).astype(np.uint8)
    bgra[..., 1] = (premul[..., 1] * 255).astype(np.uint8)
    bgra[..., 2] = (premul[..., 0] * 255).astype(np.uint8)
    bgra[..., 3] = (big[..., 3] * 255).astype(np.uint8)
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, big.shape[1], big.shape[0])
    stride = surf.get_stride()
    buf = np.ndarray((big.shape[0], stride // 4, 4), np.uint8, surf.get_data())
    buf[:, :big.shape[1]] = bgra
    surf.mark_dirty()
    return surf


_CACHE = {}


def surfaces():
    if not _CACHE:
        rgb = cmap(GRID)
        _CACHE["carpet"] = _surface(rgb)
        dark = np.zeros_like(rgb)
        _CACHE["dim_evening"] = _surface(dark, np.where(GRID > 115, 0.0, 0.72).astype(np.float32))
        _CACHE["dim_cheap"] = _surface(dark, np.where(GRID < 45, 0.0, 0.75).astype(np.float32))
        _CACHE["dim_neg"] = _surface(dark, np.where(GRID < 0, 0.0, 0.85).astype(np.float32))
        glow = np.ones_like(rgb) * np.array(NEG, np.float32)
        _CACHE["neg"] = _surface(glow, (GRID < 0).astype(np.float32))
    return _CACHE


def legend(ctx, a):
    if a <= 0:
        return
    x, y, w = RX, Y0 + 30, 460
    for i in range(w):
        v = -20 + 220 * i / w
        c = cmap(np.array([v]))[0]
        rgba(ctx, c, a)
        ctx.rectangle(x + i, y - 26, 1.2, 16)
        ctx.fill()
    for v, lab in ((-20, "< £0"), (0, "£0"), (100, "£100"), (200, "£200+ /MWh")):
        xx = x + (v + 20) / 220 * w
        text(ctx, lab, xx, y + 14, 19, SEMI, MUTED, a, "center" if v < 200 else "center")


def draw(ctx, t, S):
    c = S["cues"]
    background(ctx, glow=(CYAN, 0.04, 900))
    sf = surfaces()
    eyebrow(ctx, "04", "A grid that changes by the minute", t, 0.2)
    rise_text(ctx, "Britain's electricity price, every quarter-hour of 2025", 96, 180, 50, t, 0.3, HEAD, INK, dur=0.8)

    # Reveal sweeping January -> December.
    rv = ease_io(prog(t, c["carpet"] - 0.3, c["carpet"] + 3.6))
    xr = X0 + PW * rv
    ctx.save()
    ctx.rectangle(X0, Y0, PW * rv, PH)
    ctx.clip()
    ctx.set_source_surface(sf["carpet"], X0, Y0)
    ctx.paint()
    for key, cue, end in (("dim_evening", "evening", "cheap"), ("dim_cheap", "cheap", "negative"),
                          ("dim_neg", "negative", "valuable")):
        a = fade(t, c[cue] - 0.15, c[end] - 0.2, 0.5, 0.5)
        if a > 0:
            ctx.set_source_surface(sf[key], X0, Y0)
            ctx.paint_with_alpha(a)
    na = fade(t, c["negative"] - 0.1, c["valuable"] - 0.2, 0.4, 0.5)
    if na > 0:
        pulse = 0.55 + 0.45 * math.sin((t - c["negative"]) * 6)
        ctx.set_source_surface(sf["neg"], X0, Y0)
        ctx.paint_with_alpha(na * pulse)
    ctx.restore()
    if 0 < rv < 1:
        g = cairo.LinearGradient(xr - 60, 0, xr + 2, 0)
        g.add_color_stop_rgba(0, 1, 1, 1, 0)
        g.add_color_stop_rgba(1, 0.8, 0.95, 1, 0.55)
        ctx.set_source(g)
        ctx.rectangle(xr - 60, Y0, 62, PH)
        ctx.fill()
    # Frame and axes.
    fa = ease_out(prog(t, c["carpet"] - 0.6, c["carpet"] + 0.4))
    rgba(ctx, CYAN, 0.25 * fa)
    ctx.set_line_width(1)
    ctx.rectangle(X0 - 0.5, Y0 - 0.5, PW + 1, PH + 1)
    ctx.stroke()
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    starts = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    for m, d0 in zip(months, starts):
        text(ctx, m.upper(), X0 + (d0 + 15) * CW, Y0 + PH + 36, 21, COND, MUTED, fa, "center", 0.12)
    for hh in (0, 6, 12, 18, 24):
        yy = Y0 + hh * 4 * CH
        text(ctx, f"{hh:02d}:00", X0 - 14, yy + 7, 20, SEMI, MUTED, fa, "right")
    legend(ctx, ease_out(prog(t, c["carpet"] + 0.5, c["carpet"] + 1.3)))

    # Evening peak bracket.
    ea = fade(t, c["evening"] - 0.15, c["cheap"] - 0.2, 0.5, 0.5)
    if ea > 0:
        y1, y2 = Y0 + 16 * 4 * CH, Y0 + 20 * 4 * CH
        rgba(ctx, AMBER, 0.9 * ea)
        ctx.set_line_width(2)
        ctx.rectangle(X0 - 4, y1, PW + 8, y2 - y1)
        ctx.stroke()
        rgba(ctx, AMBER, 0.7 * ea)
        ctx.move_to(X0 + PW + 4, (y1 + y2) / 2)
        ctx.line_to(RX - 12, (y1 + y2) / 2)
        ctx.stroke()
        text(ctx, "EVENING PEAKS", RX, (y1 + y2) / 2 - 2, 34, COND, AMBER, ea, tracking=0.12)
        text(ctx, "High demand; gas sets the price", RX, (y1 + y2) / 2 + 36, 27, SEMI, INK, ea)
    ca = fade(t, c["cheap"] - 0.15, c["negative"] - 0.2, 0.5, 0.5)
    if ca > 0:
        text(ctx, "WINDY HOURS", RX, Y0 + 230, 34, COND, (0.35, 0.75, 1.0), ca, tracking=0.12)
        text(ctx, "Cheap, low-carbon power", RX, Y0 + 268, 27, SEMI, INK, ca)
    if na > 0:
        cnt = int(672 * ease_out(prog(t, c["negative"], c["negative"] + 1.6)))
        text(ctx, f"{cnt}", RX, Y0 + 330, 120, HEAD, NEG, na, glow=0.5)
        text(ctx, "QUARTER-HOURS BELOW £0/MWh", RX, Y0 + 378, 30, COND, NEG, na, tracking=0.10)
        text(ctx, "Buyers were paid to take power", RX, Y0 + 416, 27, SEMI, INK, na)

    # Closing message.
    va = ease_out(prog(t, c["valuable"] - 0.2, c["valuable"] + 0.6))
    if va > 0:
        rise_text(ctx, "FLEXIBLE DEMAND", RX, Y0 + 230, 34, t, c["valuable"] - 0.2, COND, GREEN, tracking=0.12, a=va)
        rise_text(ctx, "Use more when power is", RX, Y0 + 290, 40, t, c["valuable"] - 0.1, SEMI, INK, a=va)
        rise_text(ctx, "clean and cheap,", RX, Y0 + 340, 40, t, c["valuable"], SEMI, (0.35, 0.75, 1.0), a=va)
        rise_text(ctx, "less when the system", RX, Y0 + 400, 40, t, c["valuable"] + 0.1, SEMI, INK, a=va)
        rise_text(ctx, "is stretched.", RX, Y0 + 450, 40, t, c["valuable"] + 0.2, SEMI, AMBER, a=va)

    stats_a = ease_out(prog(t, c["carpet"] + 3.0, c["carpet"] + 3.8))
    text(ctx, "Min −£32  ·  Median £81  ·  Max £898 per MWh", RX, Y0 + PH - 6, 23, SEMI, MUTED, stats_a)
    source(ctx, "Source: LCCC Intermittent Market Reference Price (GB day-ahead), 2025, local time", t,
           c["carpet"] + 1.0)
    vignette(ctx, 0.45)
    return {}
