"""The scale of demand: global growth to 2030, then the AI rack."""
import random

from engine import *  # noqa: F401,F403
from engine import W, H

_R = random.Random(21)
STREAKS = [(_R.uniform(0, W), _R.uniform(0, H), _R.uniform(80, 260), _R.uniform(0.3, 1.0)) for _ in range(70)]


def streaks(ctx, t, a):
    """Faint data streams drifting upward in the background."""
    for (x, y, sp, br) in STREAKS:
        yy = (y - t * sp) % (H + 200) - 100
        g = cairo.LinearGradient(x, yy, x, yy + 90)
        g.add_color_stop_rgba(0, CYAN[0], CYAN[1], CYAN[2], 0.0)
        g.add_color_stop_rgba(1, CYAN[0], CYAN[1], CYAN[2], 0.10 * br * a)
        ctx.set_source(g)
        ctx.rectangle(x, yy, 1.5, 90)
        ctx.fill()


def bars(ctx, t, c, a):
    if a <= 0:
        return
    base_y, top_y = 900, 300           # 0 and 1000 TWh
    sc = (base_y - top_y) / 1000.0
    bx = [700, 1120]
    bw = 230
    # Axis and gridlines.
    ctx.set_line_width(1)
    for v in (0, 250, 500, 750, 1000):
        y = base_y - v * sc
        rgba(ctx, MUTED, (0.25 if v else 0.6) * a)
        ctx.move_to(560, y)
        ctx.line_to(1450, y)
        ctx.stroke()
        text(ctx, f"{v}", 540, y + 7, 22, SEMI, MUTED, 0.8 * a, "right")
    text(ctx, "TWh / year", 540, top_y - 40, 22, COND, MUTED, 0.8 * a, "right", 0.08)

    # 2025 bar.
    p1 = ease_out(prog(t, c["y2025"] - 0.2, c["y2025"] + 1.4))
    v1 = 485 * p1
    if p1 > 0:
        h1 = v1 * sc
        g = cairo.LinearGradient(0, base_y - h1, 0, base_y)
        g.add_color_stop_rgba(0, CYAN[0], CYAN[1], CYAN[2], 0.95 * a)
        g.add_color_stop_rgba(1, CYAN[0], CYAN[1], CYAN[2], 0.35 * a)
        ctx.set_source(g)
        rrect(ctx, bx[0], base_y - h1, bw, h1, 6)
        ctx.fill()
        text(ctx, counter(v1), bx[0] + bw / 2, base_y - h1 - 22, 64, HEAD, INK, a, "center")
    text(ctx, "2025", bx[0] + bw / 2, base_y + 46, 32, COND, INK, a * ease_out(p1 * 3), "center", 0.1)

    # 2030 bar with AI-focused share.
    p2 = ease_out(prog(t, c["y2030"] - 0.1, c["y2030"] + 1.6))
    v2 = 485 + (950 - 485) * p2
    if p2 > 0:
        h2 = v2 * sc
        g = cairo.LinearGradient(0, base_y - h2, 0, base_y)
        g.add_color_stop_rgba(0, CYAN[0], CYAN[1], CYAN[2], 0.95 * a)
        g.add_color_stop_rgba(1, CYAN[0], CYAN[1], CYAN[2], 0.35 * a)
        ctx.set_source(g)
        rrect(ctx, bx[1], base_y - h2, bw, h2, 6)
        ctx.fill()
        pa = ease_out(prog(t, c["ai"] - 0.2, c["ai"] + 1.0))
        if pa > 0:
            ha = 465 * sc * pa
            g = cairo.LinearGradient(0, base_y - h2, 0, base_y - h2 + ha)
            g.add_color_stop_rgba(0, MAGENTA[0], MAGENTA[1], MAGENTA[2], 0.95 * a)
            g.add_color_stop_rgba(1, MAGENTA[0], MAGENTA[1], MAGENTA[2], 0.55 * a)
            ctx.set_source(g)
            rrect(ctx, bx[1], base_y - h2, bw, ha, 6)
            ctx.fill()
            la = a * ease_out(prog(t, c["ai"] + 0.3, c["ai"] + 1.0))
            yy = base_y - h2 + ha / 2
            rgba(ctx, MAGENTA, 0.8 * la)
            ctx.set_line_width(1.5)
            ctx.move_to(bx[1] + bw + 12, yy)
            ctx.line_to(bx[1] + bw + 60, yy)
            ctx.stroke()
            text(ctx, "AI-FOCUSED", bx[1] + bw + 74, yy - 8, 30, COND, MAGENTA, la, tracking=0.12)
            text(ctx, "≈ 465 TWh, tripling", bx[1] + bw + 74, yy + 26, 26, SEMI, INK, la)
        text(ctx, "≈ " + counter(v2), bx[1] + bw / 2, base_y - h2 - 22, 64, HEAD, INK, a, "center")
    text(ctx, "2030", bx[1] + bw / 2, base_y + 46, 32, COND, INK, a * ease_out(p2 * 3), "center", 0.1)

    # Growth arrow.
    ga = a * ease_out(prog(t, c["y2030"] + 0.9, c["y2030"] + 1.6))
    if ga > 0:
        x1, y1 = bx[0] + bw + 20, base_y - 485 * sc - 70
        x2, y2 = bx[1] - 20, base_y - 950 * sc + 110
        rgba(ctx, GREEN, ga)
        ctx.set_line_width(3)
        ctx.move_to(x1, y1)
        ctx.curve_to(x1 + 80, y1 - 10, x2 - 120, y2 + 60, x2, y2)
        ctx.stroke()
        text(ctx, "× 2", (x1 + x2) / 2 - 70, (y1 + y2) / 2 + 10, 48, HEAD, GREEN, ga, "center")

    # Japan reference line.
    ja = a * ease_out(prog(t, c["japan"] - 0.1, c["japan"] + 0.6))
    if ja > 0:
        y = base_y - 950 * sc
        ctx.set_dash([10, 8])
        rgba(ctx, GOLD, 0.85 * ja)
        ctx.set_line_width(2)
        ctx.move_to(560, y)
        ctx.line_to(560 + (1450 - 560) * ease_io(prog(t, c["japan"] - 0.1, c["japan"] + 0.8)), y)
        ctx.stroke()
        ctx.set_dash([])
        text(ctx, "≈ JAPAN'S ENTIRE ELECTRICITY USE", 580, y + 38, 26, COND, GOLD, ja, tracking=0.12)


def rack(ctx, x, y, w, h, t, density, col, a, glow=0.0, coolant=0.0):
    rrect(ctx, x, y, w, h, 8)
    rgba(ctx, (0.07, 0.09, 0.14), a)
    ctx.fill_preserve()
    rgba(ctx, col, 0.5 * a)
    ctx.set_line_width(2)
    ctx.stroke()
    n = 22
    rnd = random.Random(int(w))
    for k in range(n):
        uy = y + 16 + k * (h - 32) / n
        busy = rnd.random() < density
        rrect(ctx, x + 12, uy, w - 24, (h - 32) / n - 4, 2)
        rgba(ctx, (0.13, 0.16, 0.23) if busy else (0.08, 0.10, 0.14), a)
        ctx.fill()
        if busy:
            for j in range(3 if density > 0.6 else 1):
                on = 1.0 if int(t * rnd.uniform(1, 5) + rnd.random() * 5) % 3 else 0.2
                dot(ctx, x + 26 + j * 12, uy + ((h - 32) / n - 4) / 2, 2.4, col, a * on, halo=1.5)
            if density > 0.6:
                rgba(ctx, col, 0.5 * a)
                ctx.rectangle(x + 70, uy + ((h - 32) / n - 4) / 2 - 1, (w - 100) * (0.4 + 0.6 * rnd.random()), 2)
                ctx.fill()
    if glow > 0:
        rg = cairo.RadialGradient(x + w / 2, y + h / 2, 0, x + w / 2, y + h / 2, w * 1.6)
        rg.add_color_stop_rgba(0, col[0], col[1], col[2], 0.25 * glow * a)
        rg.add_color_stop_rgba(1, col[0], col[1], col[2], 0)
        ctx.set_source(rg)
        ctx.paint()
    if coolant > 0:
        pts = [(x - 40, y + h - 30)]
        for k in range(6):
            yy = y + h - 60 - k * (h - 120) / 5
            pts += [(x + 16, yy), (x + w - 16, yy - 20 if k % 2 else yy)]
        pts += [(x + w - 16, y + 40), (x + w + 40, y + 40)]
        pts = []
        for k in range(7):
            yy = y + h - 40 - k * (h - 80) / 6
            pts += [(x + 18, yy), (x + w - 18, yy)] if k % 2 == 0 else [(x + w - 18, yy), (x + 18, yy)]
        pts = [(x - 60, pts[0][1])] + pts + [(x - 60, pts[-1][1])]
        draw_polyline(ctx, pts, ease_io(coolant))
        glow_stroke(ctx, ICE, 3.0, 0.8 * a, glow=1.5, layers=2)
        flow_particles(ctx, pts, t, ICE, n=26, speed=0.12, r=3, a=a * coolant)


def power_meter(ctx, x, y, w, frac, label, col, a):
    rrect(ctx, x, y, w, 16, 8)
    rgba(ctx, (0.1, 0.13, 0.2), a)
    ctx.fill()
    if frac > 0:
        rrect(ctx, x, y, max(16, w * frac), 16, 8)
        rgba(ctx, col, 0.9 * a)
        ctx.fill()
    text(ctx, label, x, y - 16, 30, HEAD, col, a)


def racks_part(ctx, t, c, a):
    if a <= 0:
        return
    # Conventional rack.
    oa = a * ease_out(prog(t, c["old"] - 0.3, c["old"] + 0.5))
    rack(ctx, 260, 280, 210, 560, t, 0.45, CYAN, oa)
    text(ctx, "CONVENTIONAL RACK", 365, 900, 26, COND, MUTED, oa, "center", 0.12)
    power_meter(ctx, 255, 980, 220, 0.07 * ease_out(prog(t, c["old"], c["old"] + 1.0)), "5–10 kW", CYAN, oa)
    # AI rack.
    na = a * ease_out(prog(t, c["new"] - 0.4, c["new"] + 0.4))
    heat = ease_out(prog(t, c["new"], c["new"] + 1.5))
    col = mix(CYAN, MAGENTA, heat)
    cool = ease_out(prog(t, c["liquid"] - 0.1, c["liquid"] + 1.4))
    rack(ctx, 640, 280, 230, 560, t, 0.95, col, na, glow=heat * (1 - 0.6 * cool), coolant=cool)
    text(ctx, "AI RACK", 755, 900, 26, COND, MUTED, na, "center", 0.12)
    power_meter(ctx, 600, 980, 310, 0.92 * heat, "100+ kW", MAGENTA, na)
    if cool > 0:
        text(ctx, "LIQUID COOLING", 755, 250, 26, COND, ICE, na * cool, "center", 0.14)

    # 400 homes.
    ha = a * ease_out(prog(t, c["homes"] - 0.2, c["homes"] + 0.3))
    if ha > 0:
        x0, y0, cols, rows, sp = 1060, 300, 25, 16, 31
        n = cols * rows
        shown = int(n * ease_io(prog(t, c["homes"] - 0.1, c["homes"] + 1.6)))
        for k in range(shown):
            i, j = k % cols, k // cols
            house(ctx, x0 + i * sp, y0 + j * sp + 22, 20, mix(GOLD, AMBER, (i + j) / (cols + rows)), ha * 0.9)
        text(ctx, "≈ 400", x0, 880, 84, HEAD, GOLD, ha, glow=0.3)
        text(ctx, "British homes' average electricity demand", x0 + 250, 846, 28, SEMI, INK, ha)
        text(ctx, "in a single cabinet", x0 + 250, 882, 28, SEMI, MUTED, ha)


def draw(ctx, t, S):
    c = S["cues"]
    background(ctx, glow=(CYAN, 0.05, 900))
    streaks(ctx, t, 1.0)
    t_sw = c["old"] - 0.9
    a1 = 1 - ease_in(prog(t, t_sw - 0.2, t_sw + 0.5))
    a2 = ease_out(prog(t, t_sw + 0.3, t_sw + 1.0))
    eyebrow(ctx, "02", "The scale of demand", t, 0.2)
    rise_text(ctx, "Global data-centre electricity use", 96, 180, 54, t, 0.4, HEAD, INK, a=a1, dur=0.8)
    bars(ctx, t, c, a1)
    source(ctx, "Source: IEA, Key Questions on Energy and AI (2026), central projection", t, 0.8, a1)
    rise_text(ctx, "The AI step-change", 96, 180, 54, t, t_sw + 0.3, HEAD, INK, a=a2, dur=0.8)
    racks_part(ctx, t, c, a2)
    source(ctx, "AI rack: e.g. 72-GPU systems at ~120–140 kW. Homes: ~2,700 kWh/yr typical UK use (Ofgem)", t,
           t_sw + 0.8, a2)
    vignette(ctx, 0.5)
    return {}
