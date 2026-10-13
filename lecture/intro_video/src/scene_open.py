"""Cold open: dolly down a server aisle, thermal scan on 'heat', then title card."""
import math
import random

from engine import *  # noqa: F401,F403
from engine import W, H

F = 900.0            # focal length (px)
HORIZON = H * 0.50
EYE = 1.30           # camera height (m)
XF = 1.15            # rack face offset from aisle centre (m)
RACK_H = 2.20
RACK_W = 0.62
CEIL = 3.0
FOG = (0.020, 0.045, 0.080)

LED_COLS = [CYAN, CYAN, TEAL, TEAL, GREEN, GREEN, (0.85, 0.95, 1.0), AMBER]

# Per-rack server layouts (seeded so they are stable across frames/workers).
_R = random.Random(11)
RACKS = []
for i in range(400):
    units, u = [], 0.10
    style = _R.random()
    while u < 1.95:
        size = _R.choice((0.0445, 0.089, 0.089, 0.089, 0.178))
        if u + size > 1.97:
            break
        blank = _R.random() < 0.12
        bar = (not blank) and style > 0.45 and _R.random() < 0.55
        leds = []
        if not blank:
            for k in range(_R.randint(1, 3)):
                leds.append((0.06 + 0.035 * k, _R.choice(LED_COLS), _R.uniform(0.5, 4.0), _R.random() * 9))
        units.append((u, size, blank, bar, leds))
        u += size + 0.004
    RACKS.append(units)


def proj(x, y, z, cam):
    cx, cy, cz = cam
    d = max(z - cz, 0.05)
    return W / 2 + F * (x - cx) / d, HORIZON - F * (y - cy) / d, d


def fogf(d, amt=15.0):
    return 1 - math.exp(-d / amt)


def blink(seed, rate, t):
    k = int(t * rate + seed)
    h = (k * 2654435761 + int(seed * 1000)) & 0xFFFF
    return 1.0 if (h % 7) > 1 else 0.2


def quad(ctx, pts):
    ctx.move_to(*pts[0][:2])
    for p in pts[1:]:
        ctx.line_to(*p[:2])
    ctx.close_path()


def corridor(ctx, t, cam, heat=0.0):
    cx, cy, cz = cam
    near, far = cz + 0.30, cz + 46

    # Base: ceiling, far glow and floor.
    g = cairo.LinearGradient(0, 0, 0, H)
    g.add_color_stop_rgb(0, 0.016, 0.026, 0.045)
    g.add_color_stop_rgb(0.5, *FOG)
    g.add_color_stop_rgb(1, 0.012, 0.020, 0.036)
    ctx.set_source(g)
    ctx.paint()
    rg = cairo.RadialGradient(W / 2, HORIZON, 0, W / 2, HORIZON, 380)
    rg.add_color_stop_rgba(0, 0.25, 0.55, 0.75, 0.30)
    rg.add_color_stop_rgba(1, 0.25, 0.55, 0.75, 0.0)
    ctx.set_source(rg)
    ctx.paint()

    # Floor: glossy tiles with the reflection of the ceiling lights.
    a = proj(-XF, 0, near, cam)
    b = proj(XF, 0, near, cam)
    c = proj(XF, 0, far, cam)
    d = proj(-XF, 0, far, cam)
    quad(ctx, [a, b, c, d])
    fg = cairo.LinearGradient(0, H, 0, HORIZON)
    fg.add_color_stop_rgb(0, 0.030, 0.045, 0.065)
    fg.add_color_stop_rgb(1, *FOG)
    ctx.set_source(fg)
    ctx.fill()
    # Light reflection streak down the aisle.
    for wv, al in ((0.35, 0.10), (0.12, 0.18)):
        quad(ctx, [proj(-wv, 0, near, cam), proj(wv, 0, near, cam), proj(wv * 0.2, 0, far, cam),
                   proj(-wv * 0.2, 0, far, cam)])
        sg = cairo.LinearGradient(0, H, 0, HORIZON)
        sg.add_color_stop_rgba(0, 0.6, 0.85, 1.0, al * 0.5)
        sg.add_color_stop_rgba(1, 0.6, 0.85, 1.0, al)
        ctx.set_source(sg)
        ctx.fill()
    ctx.set_line_width(1.0)
    for xl in (-0.575, 0.0, 0.575):
        p1, p2 = proj(xl, 0, near, cam), proj(xl, 0, far, cam)
        lg = cairo.LinearGradient(p1[0], p1[1], p2[0], p2[1])
        lg.add_color_stop_rgba(0, 0.3, 0.5, 0.65, 0.30)
        lg.add_color_stop_rgba(1, 0.3, 0.5, 0.65, 0.0)
        ctx.set_source(lg)
        ctx.move_to(*p1[:2])
        ctx.line_to(*p2[:2])
        ctx.stroke()
    z = math.ceil(near / 0.6) * 0.6
    while z < far:
        pa, pb = proj(-XF, 0, z, cam), proj(XF, 0, z, cam)
        rgba(ctx, (0.3, 0.5, 0.65), 0.22 * (1 - fogf(pa[2])))
        ctx.move_to(*pa[:2])
        ctx.line_to(*pb[:2])
        ctx.stroke()
        z += 0.6

    # Ceiling: dark plane, cable trays, and cross-aisle light panels.
    quad(ctx, [proj(-XF, CEIL, near, cam), proj(XF, CEIL, near, cam), proj(XF, CEIL, far, cam),
               proj(-XF, CEIL, far, cam)])
    cg = cairo.LinearGradient(0, 0, 0, HORIZON)
    cg.add_color_stop_rgb(0, 0.025, 0.035, 0.055)
    cg.add_color_stop_rgb(1, *FOG)
    ctx.set_source(cg)
    ctx.fill()
    for xl in (-0.85, -0.78, 0.78, 0.85):
        p1, p2 = proj(xl, 2.55, near, cam), proj(xl, 2.55, far, cam)
        lg = cairo.LinearGradient(p1[0], p1[1], p2[0], p2[1])
        lg.add_color_stop_rgba(0, 0.35, 0.55, 0.75, 0.45)
        lg.add_color_stop_rgba(1, 0.35, 0.55, 0.75, 0.0)
        ctx.set_source(lg)
        ctx.set_line_width(2.0)
        ctx.move_to(*p1[:2])
        ctx.line_to(*p2[:2])
        ctx.stroke()
    z = math.ceil(near / 1.86) * 1.86
    panels = []
    while z < far:
        panels.append(z)
        z += 1.86
    for z in reversed(panels):
        pts = [proj(-0.55, CEIL - 0.02, z, cam), proj(0.55, CEIL - 0.02, z, cam),
               proj(0.55, CEIL - 0.02, z + 0.28, cam), proj(-0.55, CEIL - 0.02, z + 0.28, cam)]
        f = fogf(pts[0][2], 20)
        quad(ctx, pts)
        rgba(ctx, mix((0.85, 0.95, 1.0), FOG, f * 0.8), 0.95)
        ctx.fill()

    # Racks, far to near, both sides.
    i0, i1 = int(near / RACK_W) - 1, int(far / RACK_W) + 1
    for i in range(i1, i0 - 1, -1):
        if i < 0 or i % 16 == 15:
            continue
        z0 = i * RACK_W + 0.01
        z1 = z0 + RACK_W - 0.02
        if z1 < near:
            continue
        z0c = max(z0, near)
        for side in (-1, 1):
            x = side * XF
            corners = [proj(x, 0, z0c, cam), proj(x, RACK_H, z0c, cam),
                       proj(x, RACK_H, z1, cam), proj(x, 0, z1, cam)]
            depth = (corners[0][2] + corners[2][2]) / 2
            f = fogf(depth)
            quad(ctx, corners)
            # Lit from above: top of rack slightly brighter.
            lg = cairo.LinearGradient(0, corners[1][1], 0, corners[0][1])
            lg.add_color_stop_rgb(0, *mix((0.10, 0.13, 0.19), FOG, f))
            lg.add_color_stop_rgb(1, *mix((0.045, 0.06, 0.09), FOG, f))
            ctx.set_source(lg)
            ctx.fill()
            # Rack frame.
            rgba(ctx, (0.45, 0.65, 0.85), 0.35 * (1 - f))
            ctx.set_line_width(max(0.6, 3.0 / depth))
            ctx.move_to(*corners[0][:2])
            ctx.line_to(*corners[1][:2])
            ctx.stroke()

            units = RACKS[(i * 2 + (side > 0)) % len(RACKS)]
            detail = depth < 24
            for (u, size, blank, bar, leds) in units:
                if detail:
                    ua, ub = z0c + 0.025 if z0 < near else z0 + 0.025, z1 - 0.025
                    if ub <= ua:
                        continue
                    pts = [proj(x, u, ua, cam), proj(x, u + size, ua, cam),
                           proj(x, u + size, ub, cam), proj(x, u, ub, cam)]
                    quad(ctx, pts)
                    col = (0.035, 0.045, 0.065) if blank else (0.085, 0.105, 0.150)
                    rgba(ctx, mix(col, FOG, f))
                    ctx.fill()
                    if bar:
                        hb = u + size * 0.5
                        p1, p2 = proj(x, hb, ua + 0.12, cam), proj(x, hb, ub - 0.03, cam)
                        rgba(ctx, mix(CYAN, FOG, f * 0.8), 0.55 * (1 - f * 0.6))
                        ctx.set_line_width(max(0.5, 0.008 * F / depth))
                        ctx.move_to(*p1[:2])
                        ctx.line_to(*p2[:2])
                        ctx.stroke()
                for (lz, col, rate, seed) in leds:
                    zz = z0 + lz
                    if zz < near:
                        continue
                    px, py, dd = proj(x, u + size * 0.5, zz, cam)
                    on = blink(seed, rate, t)
                    r = clamp(0.0075 * F / dd, 0.5, 2.6)
                    al = on * (1 - fogf(dd, 18) * 0.9)
                    dot(ctx, px, py, r, mix(col, FOG, fogf(dd, 30) * 0.5), al, halo=0)
                    if dd < 14:
                        rx, ry, _ = proj(x, -(u + size * 0.5), zz, cam)
                        rgba(ctx, col, al * 0.10)
                        ctx.arc(rx, ry, r * 1.8, 0, 2 * math.pi)
                        ctx.fill()
            # Heat bloom on the rack face (shows through the thermal pass).
            if heat > 0:
                hx, hy, _ = proj(x, RACK_H * 0.62, (z0c + z1) / 2, cam)
                r = 0.9 * F / depth
                hg = cairo.RadialGradient(hx, hy, 0, hx, hy, r)
                hg.add_color_stop_rgba(0, 0.55, 0.55, 0.55, 0.40 * heat * (1 - f))
                hg.add_color_stop_rgba(1, 0.55, 0.55, 0.55, 0)
                ctx.set_source(hg)
                ctx.arc(hx, hy, r, 0, 2 * math.pi)
                ctx.fill()


def tags(ctx, S, t):
    """Kinetic tags for 'ask an AI', 'stream a film', 'tap to pay'."""
    line = S["lines"][0]
    words = {w[2].lower().strip(",."): w[0] for w in line["words"]}
    items = [("ASK AN AI", words.get("ask", 0.5)), ("STREAM A FILM", words.get("stream", 1.5)),
             ("TAP TO PAY", words.get("tap", 2.5))]
    end = line["end"] + 1.0
    for k, (label, t0) in enumerate(items):
        a = fade(t, t0 - 0.05, end + k * 0.12, 0.45, 0.6)
        if a <= 0:
            continue
        cx = W / 2 + (k - 1) * 340
        w = text_width(ctx, label, COND, 28, 0.18) + 48
        y = H * 0.79 + (1 - ease_out(prog(t, t0 - 0.05, t0 + 0.4))) * 16
        rrect(ctx, cx - w / 2, y - 32, w, 50, 25)
        rgba(ctx, (0.02, 0.05, 0.09), 0.72 * a)
        ctx.fill_preserve()
        rgba(ctx, CYAN, 0.6 * a)
        ctx.set_line_width(1.5)
        ctx.stroke()
        text(ctx, label, cx, y + 2, 28, COND, INK, a, "center", 0.18)


def draw(ctx, t, S):
    cues = S["cues"]
    t_heat, t_dc = cues["heat"], cues["dc"]
    t_title = S["lines"][-1]["end"] + 0.25

    cam = (0.05 * math.sin(t * 0.33), EYE + 0.03 * math.sin(t * 0.45), 2.0 + 0.95 * t + 0.015 * t * t)
    heat = fade(t, t_heat - 0.1, t_dc + 0.5, 0.5, 0.9)
    corridor(ctx, t, cam, heat=heat)

    # Heat shimmer rising from the racks.
    if heat > 0:
        rnd = random.Random(3)
        for k in range(80):
            side = rnd.choice((-1, 1))
            x0 = W / 2 + side * rnd.uniform(160, 940)
            sp = rnd.uniform(50, 150)
            y = H * 0.85 - ((t * sp + rnd.random() * 950) % 950)
            x = x0 + 20 * math.sin(t * 1.7 + k)
            dot(ctx, x, y, rnd.uniform(1.5, 3.5), (0.75, 0.75, 0.75),
                0.6 * heat * clamp((y - 60) / 300), halo=1.8)

    # Fade up from black; a single status LED first.
    intro = ease_io(prog(t, 1.0, 3.2))
    if intro < 1:
        rgba(ctx, (0, 0, 0), 1 - intro)
        ctx.paint()
        led_a = fade(t, 0.15, 3.0, 0.2, 1.0) * (1.0 if int(t * 2.2) % 2 == 0 else 0.25)
        dot(ctx, W / 2, H / 2, 3.0, CYAN, led_a, halo=4)

    tags(ctx, S, t)

    # Big word on 'data centres'.
    a = fade(t, t_dc - 0.05, t_title + 0.2, 0.5, 0.6)
    if a > 0:
        rise_text(ctx, "DATA CENTRES", W / 2, H * 0.80, 40, t, t_dc - 0.05, COND, INK, "center",
                  0.5, dur=0.7, a=a)

    # Title card.
    tp = prog(t, t_title, t_title + 1.0)
    if tp > 0:
        rgba(ctx, (0.0, 0.01, 0.02), 0.66 * ease_io(tp))
        ctx.paint()
        cy = H / 2 - 10
        rise_text(ctx, "LECTURE  INTRODUCTION", W / 2, cy - 140, 26, t, t_title + 0.2, COND, CYAN,
                  "center", 0.32, dur=0.8)
        rise_text(ctx, "Low-Carbon & Flexible", W / 2, cy - 22, 122, t, t_title + 0.35, HEAD, INK,
                  "center", 0.0, dur=1.0, dy=30)
        rise_text(ctx, "Data Centres", W / 2, cy + 112, 122, t, t_title + 0.55, HEAD, CYAN,
                  "center", 0.0, dur=1.0, dy=30, glow=0.6)
        lp = ease_io(prog(t, t_title + 0.7, t_title + 2.0))
        if lp > 0:
            rgba(ctx, CYAN, 0.8)
            ctx.set_line_width(2)
            ctx.move_to(W / 2 - 400 * lp, cy + 168)
            ctx.line_to(W / 2 + 400 * lp, cy + 168)
            ctx.stroke()
        rise_text(ctx, "Powering the digital world on a net-zero grid", W / 2, cy + 226, 34,
                  t, t_title + 1.0, SEMI, MUTED, "center", 0.02, dur=0.9)

    vignette(ctx, 0.6)

    fx = {}
    if heat > 0:
        sweep = ease_io(prog(t, t_heat - 0.15, t_heat + 1.1))
        fx["thermal"] = heat
        fx["scan"] = -60 + sweep * (W + 120) if sweep < 1 else None
    return fx
