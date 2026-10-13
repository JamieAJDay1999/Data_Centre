"""The low-carbon toolkit: four levers as animated cards; the last one zooms into the next scene."""
from engine import *  # noqa: F401,F403
from engine import W, H

CW_, CH_, GAP = 404, 660, 28
CX0 = (W - (4 * CW_ + 3 * GAP)) // 2
CY0 = 250


def card_frame(ctx, x, y, num, title, col, a, hi):
    rrect(ctx, x, y, CW_, CH_, 22)
    rgba(ctx, (0.035, 0.06, 0.10), 0.88 * a)
    ctx.fill_preserve()
    rgba(ctx, col, (0.25 + 0.55 * hi) * a)
    ctx.set_line_width(1.5 + 1.5 * hi)
    ctx.stroke()
    if hi > 0:
        rg = cairo.RadialGradient(x + CW_ / 2, y + 180, 0, x + CW_ / 2, y + 180, 300)
        rg.add_color_stop_rgba(0, col[0], col[1], col[2], 0.14 * hi * a)
        rg.add_color_stop_rgba(1, col[0], col[1], col[2], 0)
        ctx.set_source(rg)
        rrect(ctx, x, y, CW_, CH_, 22)
        ctx.fill()
    text(ctx, num, x + 30, y + 56, 30, HEAD, col, a)
    text(ctx, title, x + 30, y + 440, 40, HEAD, INK, a)


def illus_eff(ctx, x, y, t, c, a):
    cx, cy, R = x + CW_ / 2, y + 290, 130
    v = 1.6 - 0.5 * ease_io(prog(t, c["eff"] + 0.3, c["eff"] + 2.2))
    ctx.set_line_width(14)
    for k in range(60):
        f0, f1 = k / 60, (k + 0.8) / 60
        val = 1.0 + 0.8 * f0
        col = mix(GREEN, AMBER, clamp((val - 1.1) / 0.5))
        rgba(ctx, col, 0.25 * a if val > v else 0.9 * a)
        ctx.new_path()
        ctx.arc(cx, cy, R, math.pi + f0 * math.pi, math.pi + f1 * math.pi)
        ctx.stroke()
    ang = math.pi + (v - 1.0) / 0.8 * math.pi
    rgba(ctx, INK, a)
    ctx.set_line_width(4)
    ctx.move_to(cx, cy)
    ctx.line_to(cx + math.cos(ang) * (R - 24), cy + math.sin(ang) * (R - 24))
    ctx.stroke()
    dot(ctx, cx, cy, 8, INK, a, halo=0)
    text(ctx, f"PUE {v:.2f}", cx, cy - 40, 34, HEAD, mix(GREEN, AMBER, clamp((v - 1.1) / 0.5)), a, "center")
    # Snowflake-ish airflow lines and a droplet.
    for k in range(5):
        yy = y + 120 + k * 16
        u = (t * 0.6 + k * 0.21) % 1.0
        rgba(ctx, ICE, 0.5 * a * (1 - u))
        ctx.set_line_width(2)
        ctx.move_to(x + 40 + u * 60, yy)
        ctx.line_to(x + 90 + u * 60, yy)
        ctx.stroke()


def illus_clean(ctx, x, y, t, c, a):
    turbine(ctx, x + 150, y + 230, 120, t, GREEN, a, 1.2)
    # Sun.
    sx, sy = x + 300, y + 130
    dot(ctx, sx, sy, 26, GOLD, a, halo=2)
    for k in range(10):
        th = k * math.pi / 5 + t * 0.3
        rgba(ctx, GOLD, 0.7 * a)
        ctx.set_line_width(3)
        ctx.move_to(sx + math.cos(th) * 36, sy + math.sin(th) * 36)
        ctx.line_to(sx + math.cos(th) * 50, sy + math.sin(th) * 50)
        ctx.stroke()
    # Hourly matching bars.
    hp = ease_out(prog(t, c["hourly"] - 0.2, c["hourly"] + 0.6))
    if hp > 0:
        bx, by, bw = x + 40, y + 360, (CW_ - 80) / 24
        sweep = ease_io(prog(t, c["hourly"], c["hourly"] + 1.6))
        for h in range(24):
            supply = 0.5 + 0.45 * math.sin((h - 6) / 24 * 2 * math.pi) * 0.6 + 0.25 * math.sin(h * 0.9)
            matched = supply >= 0.62
            col = GREEN if (matched and h / 24 < sweep) else (0.30, 0.36, 0.46)
            rgba(ctx, col, 0.9 * a * hp)
            ctx.rectangle(bx + h * bw + 1, by - 70, bw - 3, 70)
            ctx.fill()
        text(ctx, "00", bx, by + 24, 18, SEMI, MUTED, a * hp)
        text(ctx, "24 h", bx + 24 * bw, by + 24, 18, SEMI, MUTED, a * hp, "right")


def illus_heat(ctx, x, y, t, c, a):
    # Data centre block.
    rrect(ctx, x + 36, y + 170, 110, 150, 8)
    rgba(ctx, (0.15, 0.2, 0.3), a)
    ctx.fill()
    for k in range(6):
        dot(ctx, x + 60 + (k % 3) * 30, y + 200 + (k // 3) * 40, 3, CYAN, a, halo=1.5)
    # Pipe.
    pts = [(x + 146, y + 280), (x + 210, y + 280), (x + 230, y + 300), (x + 360, y + 300)]
    draw_polyline(ctx, pts, 1)
    glow_stroke(ctx, AMBER, 6, 0.6 * a, glow=1.2, layers=2)
    flow_particles(ctx, pts, t, GOLD, n=8, speed=0.35, r=4, a=a)
    # Houses glowing.
    warm = ease_out(prog(t, c["odense"] - 0.3, c["odense"] + 1.2))
    for k in range(3):
        hx = x + 250 + k * 50
        house(ctx, hx, y + 270, 42, mix((0.3, 0.35, 0.45), AMBER, warm), a)
        rgba(ctx, GOLD, 0.9 * a * warm)
        ctx.rectangle(hx - 6, y + 250, 12, 12)
        ctx.fill()
    # Heat shimmer.
    for k in range(6):
        u = (t * 0.4 + k / 6) % 1
        dot(ctx, x + 70 + k * 12, y + 160 - u * 90, 2.5, AMBER, a * (1 - u) * 0.8, halo=1.6)


def illus_flex(ctx, x, y, t, c, a, hi):
    bx, by, w = x + 30, y + 300, CW_ - 60
    pts_base, pts_flex = [], []
    for k in range(80):
        u = k / 79
        base = 0.55 + 0.25 * math.sin(u * 2 * math.pi - 1.2)
        shift = 0.18 * math.sin(u * 2 * math.pi * 2 + t * 1.5) * (0.4 + 0.6 * hi)
        pts_base.append((bx + u * w, by - base * 160))
        pts_flex.append((bx + u * w, by - (base + shift) * 160))
    draw_polyline(ctx, pts_base, 1)
    rgba(ctx, MUTED, 0.5 * a)
    ctx.set_dash([6, 6])
    ctx.set_line_width(2)
    ctx.stroke()
    ctx.set_dash([])
    draw_polyline(ctx, pts_flex, 1)
    glow_stroke(ctx, TEAL, 3.5, a, glow=1.5, layers=2)
    for k, (xx, yy) in enumerate(pts_flex[::16]):
        bolt(ctx, xx, yy - 30, 22, GOLD, a * hi * 0.8)


CARDS = [("01", "Efficiency", "eff", CYAN, ["Free cooling with outside", "air or water; liquid cooling", "for dense AI chips"]),
         ("02", "Clean supply", "clean", GREEN, ["Wind and solar contracts,", "increasingly matched", "hour by hour (24/7)"]),
         ("03", "Heat reuse", "heat", AMBER, ["District heating: Odense,", "Denmark; Old Oak and", "Park Royal, London"]),
         ("04", "Flexibility", "flex", TEAL, ["Shift and shape demand", "to match a renewable", "power system"])]


def draw(ctx, t, S):
    c = S["cues"]
    background(ctx, glow=(TEAL, 0.05, 900))
    zoom = ease_io(prog(t, c["flex"] + 0.9, S["dur"] + 0.3))
    head_a = 1 - zoom
    eyebrow(ctx, "05", "The low-carbon toolkit", t, 0.2, a=head_a)
    rise_text(ctx, "How do we make data centres low-carbon?", 96, 180, 54, t, 0.3, HEAD, INK, dur=0.8, a=head_a)

    order = [k for k in range(4)]
    for k in order:
        num, ttl, cue, col, lines = CARDS[k]
        x, y = CX0 + k * (CW_ + GAP), CY0
        t0 = c[cue] - 0.3
        p = ease_out(prog(t, t0, t0 + 0.7))
        if p <= 0:
            continue
        nxt = CARDS[k + 1][2] if k < 3 else None
        hi = 1.0 if k == 3 else 1 - 0.7 * ease_out(prog(t, c[nxt] - 0.3, c[nxt] + 0.4)) if nxt else 1.0
        hi *= p
        a = p * (1 - zoom if k < 3 else 1)
        ctx.save()
        if k == 3 and zoom > 0:
            s = 1 + 3.2 * zoom
            ccx, ccy = x + CW_ / 2, y + 290
            ctx.translate(ccx + (W / 2 - ccx) * zoom, ccy + (H / 2 - ccy) * zoom)
            ctx.scale(s, s)
            ctx.translate(-ccx, -ccy)
        ctx.translate(0, (1 - p) * 40)
        card_frame(ctx, x, y, num, ttl, col, a, hi)
        if k == 0:
            illus_eff(ctx, x, y, t, c, a)
        elif k == 1:
            illus_clean(ctx, x, y, t, c, a)
        elif k == 2:
            illus_heat(ctx, x, y, t, c, a)
        else:
            illus_flex(ctx, x, y, t, c, a, hi)
        for j, ln in enumerate(lines):
            text(ctx, ln, x + 30, y + 500 + j * 38, 27, SEMI, MUTED if j else INK, a)
        ctx.restore()
    rgba(ctx, BG, zoom * 0.9)
    ctx.paint()
    vignette(ctx, 0.45)
    return {}
