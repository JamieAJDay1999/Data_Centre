"""Flexibility: balancing the grid, the three data-centre levers, and real-world evidence."""
import random

from engine import *  # noqa: F401,F403
from engine import W, H


# ------------------------------------------------------------- part A: balance
def wind_supply(u):
    return 0.52 + 0.16 * math.sin(u * 9.0 + 0.6) + 0.08 * math.sin(u * 23.0 + 1.3) + 0.05 * math.sin(u * 41.0)


def demand(u):
    return 0.62 + 0.14 * math.sin((u - 0.3) * 2 * math.pi) + 0.03 * math.sin(u * 15)


def part_balance(ctx, t, c, a):
    if a <= 0:
        return
    x0, y0, w, h = 160, 250, 1600, 720
    gas = fade(t, c["fossil"] - 0.2, c["demand"] + 0.8, 0.8, 1.2)
    flexd = ease_io(prog(t, c["demand"] - 0.1, c["demand"] + 2.0))
    scroll = t * 0.035
    n = 160
    us = [k / (n - 1) for k in range(n)]
    pts_d, pts_w, pts_s = [], [], []
    for u in us:
        uu = u + scroll
        wv = wind_supply(uu)
        dv = demand(uu)
        dflex = lerp(dv, wv + 0.02, flexd)
        pts_d.append((x0 + u * w, y0 + h - dflex * h))
        pts_w.append((x0 + u * w, y0 + h - wv * h))
        pts_s.append((x0 + u * w, y0 + h - dflex * h))
    # Axes.
    rgba(ctx, MUTED, 0.35 * a)
    ctx.set_line_width(1)
    ctx.move_to(x0, y0 + h)
    ctx.line_to(x0 + w, y0 + h)
    ctx.stroke()
    text(ctx, "TIME", x0 + w, y0 + h + 34, 20, COND, MUTED, a, "right", 0.14)
    text(ctx, "POWER", x0 - 16, y0 + 10, 20, COND, MUTED, a, "right", 0.14)
    # Gas fills the gap between wind and demand.
    if gas > 0:
        ctx.new_path()
        for i, (x, y) in enumerate(pts_d):
            (ctx.move_to if i == 0 else ctx.line_to)(x, y)
        for (x, yw), (_, yd) in zip(reversed(pts_w), reversed(pts_d)):
            ctx.line_to(x, max(yw, yd))
        ctx.close_path()
        rgba(ctx, AMBER, 0.28 * gas * a)
        ctx.fill()
        gx, gy = x0 + w * 0.5, y0 + 60
        text(ctx, "FOSSIL PLANTS FILL THE GAP", gx, gy, 28, COND, AMBER, gas * a, "center", 0.12)
    # Wind supply.
    wa = ease_out(prog(t, c["balance"] - 0.2, c["balance"] + 0.8))
    draw_polyline(ctx, pts_w, 1)
    glow_stroke(ctx, GREEN, 3.5, a * wa, glow=1.5, layers=2)
    text(ctx, "WIND & SOLAR", pts_w[8][0], pts_w[8][1] + 46, 24, COND, GREEN, a * wa, tracking=0.12)
    # Demand.
    draw_polyline(ctx, pts_d, ease_io(prog(t, 0.0, 1.2)))
    glow_stroke(ctx, CYAN, 4, a, glow=1.5, layers=2)
    label = "DEMAND" if flexd < 0.5 else "FLEXIBLE DEMAND FOLLOWS SUPPLY"
    la = a * clamp(abs(flexd - 0.5) * 4)
    text(ctx, label, pts_d[-30][0], pts_d[-30][1] - 26, 26, COND, CYAN if flexd < 0.5 else TEAL, la, "center", 0.12)
    # Frequency readout.
    fa = a * ease_out(prog(t, c["balance"] - 0.2, c["balance"] + 0.6))
    f = 50.0 + 0.03 * math.sin(t * 3.1) + 0.02 * math.sin(t * 7.7)
    panel(ctx, 1500, 190, 300, 110, fa, 16)
    text(ctx, "GRID FREQUENCY", 1524, 226, 20, COND, MUTED, fa, tracking=0.14)
    text(ctx, f"{f:.2f} Hz", 1524, 282, 50, HEAD, GREEN, fa)


# ------------------------------------------------------------- part B: levers
PX = [96, 690, 1284]
PW_, PY, PH_ = 540, 250, 680


def lever_panel(ctx, k, num, ttl, sub, col, t, t0, a):
    p = ease_out(prog(t, t0 - 0.3, t0 + 0.5))
    x = PX[k]
    al = a * (0.35 + 0.65 * p)
    rrect(ctx, x, PY, PW_, PH_, 22)
    rgba(ctx, (0.035, 0.06, 0.10), 0.88 * a)
    ctx.fill_preserve()
    rgba(ctx, col, (0.2 + 0.6 * p) * a)
    ctx.set_line_width(1.5 + 1.5 * p)
    ctx.stroke()
    text(ctx, num, x + 32, PY + 60, 34, HEAD, col, al)
    text(ctx, ttl, x + 90, PY + 60, 38, HEAD, INK, al)
    text(ctx, sub, x + 32, PY + 104, 25, SEMI, MUTED, al)
    return p


def lever_work(ctx, t, c, a):
    p = lever_panel(ctx, 0, "01", "Workload", "Shift computing in time", CYAN, t, c["work"], a)
    if p <= 0:
        return
    x0, y0, w, h = PX[0] + 32, PY + 160, PW_ - 64, 300
    base_y = y0 + h
    # Plentiful-power windows (green bands).
    ga = a * p * ease_out(prog(t, c["wait"] - 0.6, c["wait"] + 0.2))
    for (u0, u1) in ((0.04, 0.27), (0.50, 0.66)):
        rgba(ctx, GREEN, 0.14 * ga)
        ctx.rectangle(x0 + u0 * w, y0, (u1 - u0) * w, h)
        ctx.fill()
    if ga > 0:
        text(ctx, "PLENTIFUL POWER", x0 + 0.155 * w, y0 + 26, 19, COND, GREEN, ga, "center", 0.1)
    # Inflexible base.
    rgba(ctx, (0.30, 0.38, 0.52), 0.9 * a * p)
    ctx.new_path()
    ctx.move_to(x0, base_y)
    for k in range(41):
        u = k / 40
        ctx.line_to(x0 + u * w, base_y - (0.30 + 0.06 * math.sin(u * 6.28 - 1.0)) * h)
    ctx.line_to(x0 + w, base_y)
    ctx.close_path()
    ctx.fill()
    # Flexible blocks that move from the evening peak into green windows.
    sh = ease_io(prog(t, c["shift"] - 0.1, c["shift"] + 2.0))
    blocks = [(0.72, 0.08), (0.76, 0.24), (0.83, 0.12), (0.80, 0.40)]
    targets = [(0.08, 0.06), (0.14, 0.08), (0.53, 0.06), (0.20, 0.10)]
    for k, ((u, lvl), (tu, tl)) in enumerate(zip(blocks, targets)):
        uu = lerp(u, tu, sh)
        hb = 0.13 * h
        yb = base_y - 0.42 * h - lerp(lvl, tl, sh) * h * 0.6
        rrect(ctx, x0 + uu * w, yb, 0.07 * w, hb, 4)
        rgba(ctx, CYAN, 0.9 * a * p)
        ctx.fill()
    text(ctx, "00:00", x0, base_y + 30, 18, SEMI, MUTED, a * p)
    text(ctx, "24:00", x0 + w, base_y + 30, 18, SEMI, MUTED, a * p, "right")
    rgba(ctx, AMBER, 0.6 * a * p)
    ctx.rectangle(x0 + 0.70 * w, y0, 0.20 * w, 3)
    ctx.fill()
    text(ctx, "PEAK", x0 + 0.80 * w, y0 - 8, 19, COND, AMBER, a * p, "center", 0.1)
    # Legend chips.
    la = a * ease_out(prog(t, c["work"] + 0.8, c["work"] + 1.4))
    rrect(ctx, x0, base_y + 60, 22, 22, 4)
    rgba(ctx, (0.30, 0.38, 0.52), la)
    ctx.fill()
    text(ctx, "Must run now: video calls, search", x0 + 34, base_y + 79, 23, SEMI, INK, la)
    lb = a * ease_out(prog(t, c["wait"] - 0.4, c["wait"] + 0.2))
    rrect(ctx, x0, base_y + 104, 22, 22, 4)
    rgba(ctx, CYAN, lb)
    ctx.fill()
    text(ctx, "Can wait: AI training, backups", x0 + 34, base_y + 123, 23, SEMI, INK, lb)


def lever_batt(ctx, t, c, a):
    p = lever_panel(ctx, 1, "02", "UPS batteries", "Energy already stored on site", GREEN, t, c["batt"], a)
    if p <= 0:
        return
    x0, y0 = PX[1] + 32, PY + 160
    lvl = 0.85 - 0.35 * ease_io(prog(t, c["ms"] - 0.2, c["ms"] + 2.5))
    battery(ctx, x0 + 10, y0 + 40, 110, 210, lvl, GREEN, a * p)
    # Response chart: signal at t*, battery instant, gas slow.
    cx0, cy0, cw, chh = x0 + 170, y0 + 40, 300, 210
    rgba(ctx, MUTED, 0.4 * a * p)
    ctx.set_line_width(1)
    ctx.move_to(cx0, cy0 + chh)
    ctx.line_to(cx0 + cw, cy0 + chh)
    ctx.stroke()
    run = prog(t, c["batt"] + 0.6, c["ms"] + 1.2)
    ts = 0.25
    pts_b, pts_g = [], []
    for k in range(101):
        u = k / 100
        if u > run:
            break
        b = 0 if u < ts else 1.0
        g = 0 if u < ts else 1 - math.exp(-(u - ts) / 0.28)
        pts_b.append((cx0 + u * cw, cy0 + chh - b * chh * 0.85))
        pts_g.append((cx0 + u * cw, cy0 + chh - g * chh * 0.85))
    if len(pts_b) > 1:
        draw_polyline(ctx, pts_g, 1)
        ctx.set_dash([7, 6])
        rgba(ctx, AMBER, 0.8 * a * p)
        ctx.set_line_width(3)
        ctx.stroke()
        ctx.set_dash([])
        draw_polyline(ctx, pts_b, 1)
        glow_stroke(ctx, GREEN, 4, a * p, glow=1.5, layers=2)
    sa = a * p * ease_out(prog(t, c["batt"] + 0.4, c["batt"] + 1.0))
    xs = cx0 + ts * cw
    ctx.set_dash([3, 5])
    rgba(ctx, INK, 0.5 * sa)
    ctx.move_to(xs, cy0 - 10)
    ctx.line_to(xs, cy0 + chh)
    ctx.stroke()
    ctx.set_dash([])
    text(ctx, "GRID SIGNAL", xs, cy0 - 18, 18, COND, INK, sa, "center", 0.1)
    text(ctx, "UPS", cx0 + cw - 4, cy0 + 22, 22, COND, GREEN, sa, "right", 0.1)
    text(ctx, "Gas plant", cx0 + cw - 4, cy0 + 110, 21, SEMI, AMBER, sa, "right")
    ma = a * ease_out(prog(t, c["ms"] - 0.1, c["ms"] + 0.5))
    text(ctx, "Almost instant", x0, y0 + 360, 44, HEAD, GREEN, ma, glow=0.3)
    text(ctx, "Sub-second response, ideal for", x0, y0 + 404, 24, SEMI, INK, ma)
    text(ctx, "frequency services", x0, y0 + 436, 24, SEMI, INK, ma)


def lever_cool(ctx, t, c, a):
    p = lever_panel(ctx, 2, "03", "Cooling", "Thermal inertia + chilled water", ICE, t, c["therm"], a)
    if p <= 0:
        return
    x0, y0, w, h = PX[2] + 32, PY + 160, PW_ - 64, 260
    ev = prog(t, c["therm"] + 0.3, c["trial"] - 1.0)
    # Allowed temperature band.
    band_top, band_bot = y0 + 30, y0 + 150
    rgba(ctx, ICE, 0.10 * a * p)
    ctx.rectangle(x0, band_top, w, band_bot - band_top)
    ctx.fill()
    text(ctx, "ALLOWED TEMPERATURE BAND", x0 + 8, band_top + 26, 18, COND, ICE, a * p, tracking=0.1)
    # Chiller power (bottom) and temperature (in band).
    pts_T, pts_P = [], []
    for k in range(101):
        u = k / 100
        if u > ev:
            break
        off = 0.25 < u < 0.75
        P = 0.15 if off else 0.8
        if u < 0.25:
            T = 0.25
        elif u < 0.75:
            T = 0.25 + 0.65 * (u - 0.25) / 0.5
        else:
            T = 0.9 - 0.65 * min(1, (u - 0.75) / 0.2)
        pts_T.append((x0 + u * w, band_bot - T * (band_bot - band_top)))
        pts_P.append((x0 + u * w, y0 + h - P * 80))
    if len(pts_T) > 1:
        draw_polyline(ctx, pts_T, 1)
        glow_stroke(ctx, mix(ICE, AMBER, 0.5), 3.5, a * p, glow=1.5, layers=2)
        draw_polyline(ctx, pts_P, 1)
        glow_stroke(ctx, CYAN, 3.5, a * p, glow=1.5, layers=2)
    text(ctx, "Server temperature", x0 + w, band_top - 10, 21, SEMI, mix(ICE, AMBER, 0.5), a * p, "right")
    text(ctx, "Chiller power", x0 + w, y0 + h + 28, 21, SEMI, CYAN, a * p, "right")
    da = a * ease_out(prog(t, c["drift"] - 0.2, c["drift"] + 0.4))
    text(ctx, "Chillers ease off,", x0, y0 + 360, 40, HEAD, ICE, da)
    text(ctx, "servers stay in range", x0, y0 + 404, 40, HEAD, ICE, da)


# ------------------------------------------------------------- part C: evidence
def part_evidence(ctx, t, c, a):
    if a <= 0:
        return
    # Card A: London trial.
    xa, ya, wa_, ha = 96, 260, 1000, 640
    pa = ease_out(prog(t, c["trial"] - 0.3, c["trial"] + 0.5))
    rrect(ctx, xa, ya + (1 - pa) * 30, wa_, ha, 22)
    rgba(ctx, (0.035, 0.06, 0.10), 0.9 * a * pa)
    ctx.fill_preserve()
    rgba(ctx, TEAL, 0.6 * a * pa)
    ctx.set_line_width(2)
    ctx.stroke()
    text(ctx, "LONDON  ·  DECEMBER 2025", xa + 36, ya + 62, 28, COND, TEAL, a * pa, tracking=0.14)
    text(ctx, "National Grid  ×  Emerald AI  ×  Nebius", xa + 36, ya + 110, 34, HEAD, INK, a * pa)
    cx0, cy0, cw, chh = xa + 36, ya + 170, wa_ - 300, 330
    run = prog(t, c["trial"] + 0.2, c["forty"] + 1.4)
    pts = []
    for k in range(201):
        u = k / 200
        if u > run:
            break
        ev = smooth(prog(u, 0.38, 0.44)) * (1 - smooth(prog(u, 0.70, 0.76)))
        noise = 0.015 * math.sin(u * 140) + 0.01 * math.sin(u * 377)
        v = 0.92 - 0.40 * 0.92 * ev + noise
        pts.append((cx0 + u * cw, cy0 + chh - v * chh))
    rgba(ctx, AMBER, 0.12 * a * pa)
    ctx.rectangle(cx0 + 0.38 * cw, cy0, 0.38 * cw, chh)
    ctx.fill()
    text(ctx, "GRID EVENT", cx0 + 0.57 * cw, cy0 + 26, 20, COND, AMBER, a * pa * clamp(run * 3 - 1), "center", 0.12)
    rgba(ctx, MUTED, 0.4 * a * pa)
    ctx.set_line_width(1)
    ctx.move_to(cx0, cy0 + chh)
    ctx.line_to(cx0 + cw, cy0 + chh)
    ctx.stroke()
    text(ctx, "AI CLUSTER POWER", cx0, cy0 + chh + 34, 20, COND, MUTED, a * pa, tracking=0.12)
    if len(pts) > 1:
        draw_polyline(ctx, pts, 1)
        glow_stroke(ctx, TEAL, 4, a * pa, glow=1.6, layers=2)
    fa = a * ease_back(prog(t, c["forty"] - 0.1, c["forty"] + 0.6), 1.5)
    text(ctx, "−40%", xa + wa_ - 36, ya + 340, 120, HEAD, TEAL, clamp(fa), "right", glow=0.5)
    text(ctx, "up to", xa + wa_ - 36, ya + 230, 30, SEMI, INK, clamp(fa), "right")
    text(ctx, "on grid signals,", xa + wa_ - 36, ya + 390, 26, SEMI, INK, clamp(fa), "right")
    text(ctx, "critical work protected", xa + wa_ - 36, ya + 424, 26, SEMI, INK, clamp(fa), "right")

    # Card B: Google.
    xb, yb, wb, hb = 1124, 260, 700, 640
    pb = ease_out(prog(t, c["google"] - 0.3, c["google"] + 0.5))
    if pb > 0:
        rrect(ctx, xb, yb + (1 - pb) * 30, wb, hb, 22)
        rgba(ctx, (0.035, 0.06, 0.10), 0.9 * a * pb)
        ctx.fill_preserve()
        rgba(ctx, GOLD, 0.6 * a * pb)
        ctx.set_line_width(2)
        ctx.stroke()
        text(ctx, "UNITED STATES  ·  2025–26", xb + 36, yb + 62, 28, COND, GOLD, a * pb, tracking=0.14)
        text(ctx, "Google", xb + 36, yb + 110, 34, HEAD, INK, a * pb)
        g = 1.0 * ease_out(prog(t, c["google"] + 0.2, c["google"] + 1.8))
        text(ctx, f"≈ {g:.1f}", xb + 36, yb + 330, 150, HEAD, GOLD, a * pb, glow=0.5)
        text(ctx, "GW", xb + 470, yb + 330, 70, HEAD, GOLD, a * pb)
        text(ctx, "of data-centre demand that can be", xb + 36, yb + 410, 28, SEMI, INK, a * pb)
        text(ctx, "curtailed when grids are stressed,", xb + 36, yb + 448, 28, SEMI, INK, a * pb)
        text(ctx, "through agreements with utilities", xb + 36, yb + 486, 28, SEMI, MUTED, a * pb)
        for k in range(5):
            bolt(ctx, xb + 520 + k * 34, yb + 200 + 10 * math.sin(t * 3 + k), 30, GOLD, a * pb * (0.4 + 0.6 * ((int(t * 4) + k) % 5 == 0)))


def draw(ctx, t, S):
    c = S["cues"]
    background(ctx, glow=(TEAL, 0.05, 900))
    tA = c["work"] - 0.7
    tC = c["trial"] - 0.7
    aA = 1 - ease_in(prog(t, tA - 0.3, tA + 0.4))
    aB = ease_out(prog(t, tA, tA + 0.7)) * (1 - ease_in(prog(t, tC - 0.3, tC + 0.4)))
    aC = ease_out(prog(t, tC, tC + 0.7))
    eyebrow(ctx, "06", "Flexibility", t, 0.2)
    rise_text(ctx, "Supply and demand must balance, every second", 96, 180, 54, t, 0.3, HEAD, INK, dur=0.8, a=aA)
    part_balance(ctx, t, c, aA)
    # Three-lever teaser icons at the end of part A.
    rise_text(ctx, "Three data-centre levers", 96, 180, 54, t, tA + 0.1, HEAD, INK, dur=0.8, a=aB)
    if aB > 0:
        lever_work(ctx, t, c, aB)
        lever_batt(ctx, t, c, aB)
        lever_cool(ctx, t, c, aB)
    rise_text(ctx, "Already happening", 96, 180, 54, t, tC + 0.1, HEAD, INK, dur=0.8, a=aC)
    part_evidence(ctx, t, c, aC)
    source(ctx, "Sources: National Grid / Emerald AI / Nebius trial (Dec 2025); Google utility agreements, via "
                "DatacenterDynamics (2026)", t, tC + 0.6, aC)
    source(ctx, "Illustrative profiles", t, tA + 0.5, aB)
    vignette(ctx, 0.45)
    return {}
