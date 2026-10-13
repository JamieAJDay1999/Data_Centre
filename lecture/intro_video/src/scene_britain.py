"""Britain: where data centres are, the connection queue, growth zones and the wind squeeze."""
import json
import math
import random
from pathlib import Path

from engine import *  # noqa: F401,F403
from engine import W, H

GEO = json.loads((Path(__file__).resolve().parents[1] / "assets" / "geography.json").read_text())
LAT0, LON0, K = 54.3, -2.9, 108.0
CX, CY = 640, 560
COSL = math.cos(math.radians(LAT0))


def P(lon, lat, view=None):
    if view is None:
        return CX + (lon - LON0) * K * COSL, CY - (lat - LAT0) * K
    lon0, lat0, k, cx, cy = view
    return cx + (lon - lon0) * k * COSL, cy - (lat - lat0) * k


# Main island first (largest polygon), for the coastline draw-on.
GB = sorted(GEO["landPolygons"], key=len, reverse=True)
CONTEXT = GEO["contextLandPolygons"]
ROUTES = []
for r in GEO["contextBackbone"]:
    pts = r["route"]
    if pts[0][1] < pts[-1][1]:
        pts = pts[::-1]          # run north -> south
    ROUTES.append(pts)

CLUSTERS = [  # lon, lat, weight (stylised concentrations of existing sites)
    (-0.01, 51.51, 1.0), (-0.59, 51.51, 1.0), (-0.27, 51.53, 0.7), (-0.44, 51.50, 0.8),
    (-0.45, 51.46, 0.5), (-0.10, 51.37, 0.35), (-0.06, 51.65, 0.35), (0.22, 51.45, 0.35),
    (-0.97, 51.45, 0.45), (-1.09, 51.27, 0.2), (-0.76, 52.04, 0.25), (-2.99, 51.58, 0.5),
    (-3.18, 51.48, 0.3), (-2.59, 51.45, 0.2), (-1.89, 52.48, 0.35), (-2.24, 53.48, 0.55),
    (-1.55, 53.80, 0.3), (-1.61, 54.97, 0.2), (-3.19, 55.95, 0.25), (-4.25, 55.86, 0.2),
]
_R = random.Random(4)
DC_DOTS = []
for (lon, lat, w) in CLUSTERS:
    for _ in range(int(4 + 14 * w)):
        DC_DOTS.append((lon + _R.gauss(0, 0.10 * (0.6 + w)), lat + _R.gauss(0, 0.06 * (0.6 + w)),
                        _R.uniform(1.6, 3.2), _R.random() * 6, lat < 51.75 and lon > -1.2))

ZONES = [("Lanarkshire", -3.98, 55.86, "right"), ("North East", -1.51, 55.13, "right"),
         ("North Wales", -4.30, 53.25, "left"), ("Oxfordshire", -1.23, 51.66, "right"),
         ("South Wales", -2.99, 51.58, "left")]
WIND = [(-3.0, 58.15, True), (-2.6, 58.3, True), (-1.85, 56.62, True), (-2.3, 56.28, True),
        (-4.3, 57.3, True), (-4.0, 55.55, True), (-3.6, 57.85, True), (-5.0, 56.3, True),
        (1.9, 54.85, False), (1.75, 53.95, False), (2.1, 52.35, False), (-3.55, 54.05, False),
        (1.2, 54.2, False)]


def land(ctx, t, view, coast_p=1.0, a=1.0):
    for poly in CONTEXT:
        if len(poly) < 3:
            continue
        ctx.new_path()
        for i, (lon, lat) in enumerate(poly):
            x, y = P(lon, lat, view)
            (ctx.move_to if i == 0 else ctx.line_to)(x, y)
        ctx.close_path()
        rgba(ctx, (0.045, 0.065, 0.10), a)
        ctx.fill_preserve()
        rgba(ctx, (0.25, 0.35, 0.5), 0.25 * a)
        ctx.set_line_width(1)
        ctx.stroke()
    if view is None:
        mg = cairo.LinearGradient(960, 0, 1320, 0)
        mg.add_color_stop_rgba(0, 0.012, 0.022, 0.045, 0.0)
        mg.add_color_stop_rgba(1, 0.012, 0.022, 0.045, 1.0)
        ctx.set_source(mg)
        ctx.rectangle(960, 0, W - 960, H)
        ctx.fill()
    for k, poly in enumerate(GB):
        ctx.new_path()
        for i, (lon, lat) in enumerate(poly):
            x, y = P(lon, lat, view)
            (ctx.move_to if i == 0 else ctx.line_to)(x, y)
        ctx.close_path()
        rgba(ctx, (0.075, 0.105, 0.16), a * ease_out(clamp(coast_p * 1.3 - 0.3)))
        ctx.fill()
        pts = [P(lon, lat, view) for lon, lat in poly] + [P(*poly[0], view)]
        draw_polyline(ctx, pts, coast_p if k == 0 else ease_out(clamp(coast_p * 2 - 1)))
        glow_stroke(ctx, CYAN, 1.5, 0.75 * a, glow=2.0, layers=2)


def graticule(ctx, a):
    ctx.set_line_width(1)
    rgba(ctx, CYAN, 0.05 * a)
    for lon in range(-12, 6, 2):
        x1, y1 = P(lon, 48)
        x2, y2 = P(lon, 62)
        ctx.move_to(x1, y1)
        ctx.line_to(x2, y2)
    for lat in range(48, 62):
        x1, y1 = P(-14, lat)
        x2, y2 = P(6, lat)
        ctx.move_to(x1, y1)
        ctx.line_to(x2, y2)
    ctx.stroke()


def dcs(ctx, t, a, view=None, scale=1.0, london_boost=0.0):
    if a <= 0:
        return
    for (lon, lat, r, ph, lnd) in DC_DOTS:
        x, y = P(lon, lat, view)
        pulse = 0.75 + 0.25 * math.sin(t * 2.2 + ph)
        col = mix(CYAN, (0.8, 0.95, 1.0), 0.3)
        dot(ctx, x, y, r * scale, col, a * pulse * (1 + 0.3 * london_boost * lnd), halo=2.2)


def panel_title(ctx, s, t, t0, a, y=180):
    rise_text(ctx, s, 1060, y, 50, t, t0, HEAD, INK, a=a, dur=0.7)


def stat(ctx, big, unit, label, x, y, t, t0, a, col=CYAN):
    p = ease_out(prog(t, t0, t0 + 0.7))
    if p <= 0:
        return
    w = text(ctx, big, x, y + (1 - p) * 20, 96, HEAD, col, a * p, glow=0.3)
    text(ctx, unit, x + w + 14, y + (1 - p) * 20, 40, SEMI, col, a * p)
    text(ctx, label, x, y + 46 + (1 - p) * 20, 27, SEMI, INK, 0.85 * a * p)


def queue_bars(ctx, t, c, a):
    if a <= 0:
        return
    x0, y0, L = 1060, 330, 700
    sc = L / 60.0
    rows = [("CONNECTED DATA CENTRES TODAY", 2.4, "≈ 2.4 GW", CYAN, c["queue"] - 0.6),
            ("GB PEAK ELECTRICITY DEMAND", 45.0, "≈ 45 GW", INK, c["peak"] - 0.3),
            ("DATA-CENTRE CONNECTION REQUESTS", 52.0, "50+ GW", MAGENTA, c["queue"] + 0.1)]
    order = [0, 2, 1]
    for slot, idx in enumerate(order):
        label, v, vs, col, t0 = rows[idx]
        y = y0 + slot * 150
        p = ease_out(prog(t, t0, t0 + 1.4))
        if p <= 0:
            continue
        text(ctx, label, x0, y, 26, COND, col, a * ease_out(p * 2), tracking=0.12)
        rrect(ctx, x0, y + 18, L, 46, 6)
        rgba(ctx, (0.08, 0.11, 0.17), 0.9 * a)
        ctx.fill()
        rrect(ctx, x0, y + 18, max(10, v * sc * p), 46, 6)
        rgba(ctx, col, (0.85 if col != INK else 0.6) * a)
        ctx.fill()
        text(ctx, vs, x0 + max(10, v * sc * p) + 16, y + 54, 34, HEAD, col, a * ease_out(p * 2))
    # Peak line across the bars.
    pa = a * ease_out(prog(t, c["peak"] + 0.6, c["peak"] + 1.2))
    if pa > 0:
        xx = x0 + 45 * sc
        ctx.set_dash([8, 6])
        rgba(ctx, INK, 0.7 * pa)
        ctx.set_line_width(2)
        ctx.move_to(xx, y0 - 10)
        ctx.line_to(xx, y0 + 3 * 150 - 60)
        ctx.stroke()
        ctx.set_dash([])


def pin(ctx, x, y, t, t0, label, align, a):
    p = prog(t, t0, t0 + 0.6)
    if p <= 0:
        return
    drop = (1 - ease_back(p, 2.2)) * 40
    al = a * ease_out(p * 2)
    rg = cairo.RadialGradient(x, y, 0, x, y, 26)
    rg.add_color_stop_rgba(0, GOLD[0], GOLD[1], GOLD[2], 0.35 * al)
    rg.add_color_stop_rgba(1, GOLD[0], GOLD[1], GOLD[2], 0)
    ctx.set_source(rg)
    ctx.arc(x, y, 26, 0, 2 * math.pi)
    ctx.fill()
    yy = y - drop
    ctx.new_path()
    ctx.move_to(x, yy)
    ctx.curve_to(x - 4, yy - 10, x - 13, yy - 16, x - 13, yy - 26)
    ctx.arc(x, yy - 26, 13, math.pi, 2 * math.pi)
    ctx.curve_to(x + 13, yy - 16, x + 4, yy - 10, x, yy)
    ctx.close_path()
    rgba(ctx, GOLD, al)
    ctx.fill()
    dot(ctx, x, yy - 26, 4.5, (0.1, 0.08, 0.02), al, halo=0)
    dx = 22 if align == "right" else -22
    text(ctx, label.upper(), x + dx, yy - 18, 23, COND, GOLD, al, "left" if align == "right" else "right", 0.1)
    # Ripple on landing.
    rp = prog(t, t0 + 0.45, t0 + 1.4)
    if 0 < rp < 1:
        rgba(ctx, GOLD, 0.6 * (1 - rp) * a)
        ctx.set_line_width(2)
        ctx.save()
        ctx.translate(x, y)
        ctx.scale(1, 0.5)
        ctx.arc(0, 0, 10 + 40 * rp, 0, 2 * math.pi)
        ctx.restore()
        ctx.stroke()


def magnifier(ctx, t, c, a):
    if a <= 0:
        return
    lx, ly = P(-0.35, 51.5)
    cx, cy, R = 1660, 760, 150
    # Leader from London to the lens.
    rgba(ctx, RED, 0.6 * a)
    ctx.set_line_width(1.5)
    ctx.move_to(lx + 18, ly)
    ctx.line_to(cx - R, cy)
    ctx.stroke()
    ctx.arc(lx, ly, 18, 0, 2 * math.pi)
    ctx.stroke()
    ctx.save()
    ctx.arc(cx, cy, R * ease_out(a), 0, 2 * math.pi)
    ctx.clip()
    rgba(ctx, (0.02, 0.03, 0.05), 0.95)
    ctx.paint()
    view = (-0.35, 51.5, K * 6.0, cx, cy)
    for poly in GB[:1]:
        ctx.new_path()
        for i, (lon, lat) in enumerate(poly):
            x, y = P(lon, lat, view)
            (ctx.move_to if i == 0 else ctx.line_to)(x, y)
        ctx.close_path()
        rgba(ctx, (0.075, 0.105, 0.16), 1)
        ctx.fill()
    # Congested local network: a mesh of lines pulsing red.
    rnd = random.Random(9)
    nodes = [(cx + rnd.uniform(-150, 150), cy + rnd.uniform(-120, 120)) for _ in range(16)]
    ctx.set_line_width(2)
    for i, (x1, y1) in enumerate(nodes):
        for (x2, y2) in nodes[i + 1:]:
            if math.dist((x1, y1), (x2, y2)) < 110:
                pulse = 0.5 + 0.5 * math.sin(t * 5 + x1 * 0.05)
                rgba(ctx, mix(AMBER, RED, pulse), 0.55 * a)
                ctx.move_to(x1, y1)
                ctx.line_to(x2, y2)
                ctx.stroke()
    dcs(ctx, t, a, view, 1.8)
    ctx.restore()
    rgba(ctx, RED, 0.85 * a)
    ctx.set_line_width(2.5)
    ctx.arc(cx, cy, R * ease_out(a), 0, 2 * math.pi)
    ctx.stroke()
    text(ctx, "WEST LONDON", cx, cy - R - 16, 26, COND, RED, a, "center", 0.14)


def draw(ctx, t, S):
    c = S["cues"]
    background(ctx, top=(0.012, 0.025, 0.05), glow=(CYAN, 0.05, 700), glow_pos=(0.33, 0.5))
    graticule(ctx, ease_out(prog(t, 0, 1.5)))
    coast = ease_io(prog(t, 0.1, 2.6))
    land(ctx, t, None, coast)

    # Backbone network, then north->south power flows during the wind section.
    na = ease_out(prog(t, 1.5, 3.0))
    wind_a = ease_out(prog(t, c["wind"] - 0.3, c["wind"] + 0.8))
    curtail = ease_out(prog(t, c["curtail"] - 0.1, c["curtail"] + 1.0))
    for pts in ROUTES:
        sp = [P(*q) for q in pts]
        draw_polyline(ctx, sp, 1.0)
        rgba(ctx, CYAN, (0.16 + 0.2 * wind_a) * na)
        ctx.set_line_width(1.6)
        ctx.stroke()
        if wind_a > 0:
            north = pts[0][1] > 55.4
            flow_particles(ctx, sp, t, GREEN, n=10 if north else 6, speed=0.12 * (1 - 0.6 * curtail * north),
                           r=2.6, a=wind_a * (0.9 if north else 0.6), offset=hash(str(pts[0])) % 7 / 7)

    # Data-centre clusters.
    da = ease_out(prog(t, c["now"] - 0.2, c["now"] + 1.0))
    dcs(ctx, t, da, london_boost=ease_out(prog(t, c["london"], c["london"] + 0.8)))

    # Boundary constraint between Scotland and England.
    if wind_a > 0:
        ba = wind_a * (0.5 + 0.5 * math.sin(t * 4)) * (0.6 + 0.4 * curtail)
        x1, y1 = P(-3.6, 55.25)
        x2, y2 = P(-1.4, 55.55)
        ctx.set_dash([10, 7])
        ctx.move_to(x1, y1)
        ctx.line_to(x2, y2)
        rgba(ctx, RED, 0.5 + 0.5 * ba)
        ctx.set_line_width(4)
        ctx.stroke()
        ctx.set_dash([])
        text(ctx, "BOTTLENECK", x1 - 14, y1 + 8, 24, COND, RED, wind_a, "right", 0.14)

    # Wind farms.
    for k, (lon, lat, north) in enumerate(WIND):
        wa = wind_a * ease_out(prog(t, c["wind"] + k * 0.08, c["wind"] + 0.6 + k * 0.08))
        if wa <= 0:
            continue
        x, y = P(lon, lat)
        stopped = curtail if north else 0.0
        col = mix(GREEN, (0.45, 0.5, 0.55), stopped)
        speed = 1.6 * (1 - stopped) + 0.02
        turbine(ctx, x, y + 18, 40, t * speed, col, wa, 1.0, phase=k)
    if curtail > 0:
        x, y = P(-8.4, 57.6)
        text(ctx, "£ PAID TO", x - 10, y - 18, 26, COND, AMBER, curtail, "right", 0.12)
        text(ctx, "SWITCH OFF", x - 10, y + 12, 26, COND, AMBER, curtail, "right", 0.12)

    # Growth zones.
    za = 1 - ease_in(prog(t, c["wind"] - 0.8, c["wind"] + 0.2)) * 0.6
    span = max(0.6, c["swales"] - c["zones"])
    for k, (name, lon, lat, align) in enumerate(ZONES):
        x, y = P(lon, lat)
        pin(ctx, x, y, t, c["zones"] + 0.3 + k * span / 4.6, name, align, za)

    # West London magnifier.
    ma = fade(t, c["west"] - 0.1, c["queue"] + 0.2, 0.6, 0.6)
    magnifier(ctx, t, c, ma)

    # Right-hand panel, in phases.
    eyebrow(ctx, "03", "Britain", t, 0.3)
    p1 = fade(t, c["now"] - 0.3, c["queue"] - 0.6, 0.6, 0.5)
    panel_title(ctx, "Where they are today", t, c["now"] - 0.3, p1)
    stat(ctx, "≈ 2.4", "GW", "of data centres connected in Great Britain", 1060, 360, t, c["now"], p1)
    stat(ctx, "≈ 7.6", "TWh", "of electricity a year", 1060, 530, t, c["london"], p1, col=TEAL)
    wa_ = fade(t, c["west"] + 0.2, c["queue"] - 0.6, 0.6, 0.5)
    if wa_ > 0:
        text(ctx, "West London (2022): new housing", 1060, 760, 28, SEMI, RED, wa_)
        text(ctx, "told to wait years for a grid", 1060, 798, 28, SEMI, RED, wa_)
        text(ctx, "connection", 1060, 836, 28, SEMI, RED, wa_)

    p2 = fade(t, c["queue"] - 0.6, c["zones"] - 0.2, 0.6, 0.5)
    panel_title(ctx, "The connection queue", t, c["queue"] - 0.6, p2)
    queue_bars(ctx, t, c, p2)

    p3 = fade(t, c["zones"] - 0.2, c["wind"] - 0.4, 0.6, 0.5)
    panel_title(ctx, "AI Growth Zones", t, c["zones"] - 0.2, p3)
    if p3 > 0:
        text(ctx, "Designated by government, 2025–26, to fast-track", 1060, 250, 27, SEMI, MUTED, p3)
        text(ctx, "AI data-centre build-out", 1060, 286, 27, SEMI, MUTED, p3)
        for k, (name, *_r) in enumerate(ZONES):
            rise_text(ctx, f"{k + 1:02d}", 1060, 380 + k * 80, 34, t, c["zones"] + 0.3 + k * span / 4.6, HEAD,
                      GOLD, a=p3)
            rise_text(ctx, name, 1130, 380 + k * 80, 40, t, c["zones"] + 0.3 + k * span / 4.6, SEMI, INK, a=p3)

    p4 = fade(t, c["wind"] - 0.4, S["dur"] + 1, 0.6, 0.1)
    panel_title(ctx, "Power north, demand south", t, c["wind"] - 0.4, p4)
    if p4 > 0:
        rise_text(ctx, "Wind in Scotland and the North Sea", 1060, 290, 30, t, c["wind"], SEMI, GREEN, a=p4)
        rise_text(ctx, "Most demand, and most data centres, in the south", 1060, 336, 30, t, c["wind"] + 0.4,
                  SEMI, CYAN, a=p4)
        rise_text(ctx, "When the network is full, wind farms are paid to stop", 1060, 382, 30, t,
                  c["curtail"] + 0.2, SEMI, AMBER, a=p4)
        for k, (word, cue) in enumerate((("WHERE", "where"), ("WHEN", "when"))):
            p = ease_back(prog(t, c[cue] - 0.05, c[cue] + 0.5), 1.4)
            if p > 0:
                text(ctx, word, 1060, 600 + k * 150, 140 * (0.8 + 0.2 * p), HEAD, INK if k == 0 else GREEN,
                     p4 * clamp(p), glow=0.5)
        ap = ease_out(prog(t, c["when"] + 0.3, c["when"] + 0.9))
        text(ctx, "Location and timing both matter", 1060, 830, 34, SEMI, MUTED, p4 * ap)

    src = [("NESO FES 2025; NESO evidence to Parliament (2026)", c["now"], c["zones"] - 0.3),
           ("GOV.UK AI Growth Zones (2025–26)", c["zones"], c["wind"] - 0.3),
           ("Wind and network shown schematically; site clusters indicative", c["wind"], S["dur"] + 1)]
    for s_, a0, a1 in src:
        source(ctx, s_, t, a0, fade(t, a0, a1, 0.6, 0.4))
    vignette(ctx, 0.45)
    return {}
