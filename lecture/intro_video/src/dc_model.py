"""Isometric data-centre facility model shared by several scenes.

World units are roughly metres. x runs to the lower right of the screen,
y to the lower left, z up. The viewer looks from +x +y.
"""
import math
import random

from engine import *  # noqa: F401,F403
from engine import Iso

HALL = (8.0, 2.0, 22.0, 12.0)     # x0, y0, x1, y1
WALL_H = 3.6
RACK_X0, RACK_X1, RACK_W = 10.0, 19.6, 0.6
RACK_D, RACK_H = 0.9, 2.2
ROWS = [(3.0, "front"), (5.0, "back"), (7.6, "front"), (9.6, "back")]   # y0, face shown (+y)
COLD = [(3.9, 5.0), (8.5, 9.6)]    # cold aisle y-ranges
HOT = [(5.9, 7.6)]                 # hot aisle y-range
CRAH = [(8.2, 3.7, 1.0, 1.5), (8.2, 8.3, 1.0, 1.5)]   # x, y, dx, dy
CHILLERS = [(10.5, -1.6), (14.0, -1.6), (17.5, -1.6)]
SUB = (1.0, 5.5, 3.2, 3.0)         # substation footprint x, y, dx, dy
UPS = (4.8, 2.6, 2.4, 3.2)
GEN = [(4.6, 8.4), (4.6, 10.4)]
PYLON = (-3.0, 9.5)

_R = random.Random(5)
RACK_LEDS = {(r, i): [(_R.uniform(0.08, 0.52), _R.uniform(0.3, 2.0), _R.choice((CYAN, CYAN, TEAL, GREEN)),
                       _R.uniform(0.6, 3.0), _R.random() * 8) for _ in range(7)]
             for r in range(4) for i in range(20)}


def fit_iso(bounds, rect, zmax=4.0):
    """Iso projection that fits a world bounding box into a screen rect."""
    x0, y0, x1, y1 = bounds
    probe = Iso(0, 0, 1.0)
    pts = [probe.p(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (0, zmax)]
    minx, maxx = min(p[0] for p in pts), max(p[0] for p in pts)
    miny, maxy = min(p[1] for p in pts), max(p[1] for p in pts)
    rx, ry, rw, rh = rect
    s = min(rw / (maxx - minx), rh / (maxy - miny))
    return Iso(rx + (rw - (maxx - minx) * s) / 2 - minx * s, ry + (rh - (maxy - miny) * s) / 2 - miny * s, s)


def iso_lerp(a, b, u):
    return Iso(lerp(a.ox, b.ox, u), lerp(a.oy, b.oy, u), lerp(a.s, b.s, u))


def _blink(seed, rate, t):
    k = int(t * rate + seed)
    return 1.0 if ((k * 2654435761) & 0xFF) % 6 > 0 else 0.2


def pylon(ctx, iso, x, y, h, a, color=(0.55, 0.65, 0.8)):
    if a <= 0:
        return
    rgba(ctx, color, 0.8 * a)
    ctx.set_line_width(max(1.0, iso.s * 0.05))
    b = 0.5
    legs = [(x - b, y - b), (x + b, y - b), (x + b, y + b), (x - b, y + b)]
    for lx, ly in legs:
        iso.line(ctx, (lx, ly, 0), (x, y, h))
    ctx.stroke()
    for zz, wdt in ((h * 0.72, 1.6), (h * 0.9, 1.1)):
        iso.line(ctx, (x - wdt, y + wdt * 0.0, zz), (x + wdt, y, zz))
        ctx.stroke()


def draw_facility(ctx, iso, t, st):
    """st: dict of 0..1 visibilities: base, grid, ups, gen, it, cool, cold, hot, power, water."""
    g = st.get
    base = g("base", 1.0)

    # Ground plate with grid.
    if base > 0:
        iso.poly(ctx, [(-4, -3.5, 0), (24, -3.5, 0), (24, 14.5, 0), (-4, 14.5, 0)])
        rgba(ctx, (0.035, 0.055, 0.085), 0.9 * base)
        ctx.fill()
        ctx.set_line_width(1.0)
        for gx in range(-4, 25, 2):
            iso.line(ctx, (gx, -3.5, 0), (gx, 14.5, 0))
        for gy in range(-3, 15, 2):
            iso.line(ctx, (-4, gy, 0), (24, gy, 0))
        rgba(ctx, CYAN, 0.06 * base)
        ctx.stroke()

    # Chillers (behind the hall).
    c = g("cool", 0)
    for k, (cx, cy) in enumerate(CHILLERS):
        a = ease_out(prog(c, k * 0.15, k * 0.15 + 0.6))
        if a <= 0:
            continue
        lift = (1 - a) * 1.5
        iso.box(ctx, cx, cy, lift, 3.0, 2.2, 1.6, (0.58, 0.68, 0.80), a, edge=ICE, edge_a=0.3)
        for f in range(2):
            fx, fy = cx + 0.8 + f * 1.4, cy + 1.1
            sx, sy = iso.p(fx, fy, 1.6 + lift)
            rx, ry = iso.s * 0.55, iso.s * 0.32
            ctx.save()
            ctx.translate(sx, sy)
            ctx.scale(rx, ry)
            ctx.arc(0, 0, 1, 0, 2 * math.pi)
            ctx.restore()
            rgba(ctx, (0.12, 0.16, 0.22), a)
            ctx.fill()
            ang = t * 6 + f
            ctx.set_line_width(1.5)
            for bl in range(4):
                th = ang + bl * math.pi / 2
                ctx.move_to(sx, sy)
                ctx.line_to(sx + math.cos(th) * rx * 0.9, sy + math.sin(th) * ry * 0.9)
            rgba(ctx, ICE, 0.7 * a)
            ctx.stroke()

    # Hall: floor slab and the two back walls (cutaway).
    hx0, hy0, hx1, hy1 = HALL
    if base > 0:
        iso.box(ctx, hx0, hy0, 0, hx1 - hx0, hy1 - hy0, 0.15, (0.10, 0.13, 0.18), base)
        iso.poly(ctx, [(hx0, hy0, 0.15), (hx1, hy0, 0.15), (hx1, hy0, WALL_H), (hx0, hy0, WALL_H)])
        rgba(ctx, (0.08, 0.11, 0.17), 0.85 * base)
        ctx.fill()
        iso.poly(ctx, [(hx0, hy0, 0.15), (hx0, hy1, 0.15), (hx0, hy1, WALL_H), (hx0, hy0, WALL_H)])
        rgba(ctx, (0.06, 0.085, 0.13), 0.85 * base)
        ctx.fill()

    # Aisle tints.
    for (ya, yb) in COLD:
        a = g("cold", 0)
        if a > 0:
            iso.poly(ctx, [(RACK_X0, ya, 0.16), (RACK_X1, ya, 0.16), (RACK_X1, yb, 0.16), (RACK_X0, yb, 0.16)])
            rgba(ctx, ICE, 0.22 * a)
            ctx.fill()
    for (ya, yb) in HOT:
        a = g("hot", 0)
        if a > 0:
            iso.poly(ctx, [(RACK_X0, ya, 0.16), (RACK_X1, ya, 0.16), (RACK_X1, yb, 0.16), (RACK_X0, yb, 0.16)])
            rgba(ctx, RED, 0.20 * a)
            ctx.fill()

    # Air handlers against the back wall.
    for k, (x, y, dx, dy) in enumerate(CRAH):
        a = ease_out(prog(c, 0.2 + k * 0.15, 0.8 + k * 0.15))
        if a > 0:
            iso.box(ctx, x, y, 0.15, dx, dy, 2.6, (0.60, 0.70, 0.82), a, edge=ICE, edge_a=0.25)

    # Rack rows.
    it = g("it", 0)
    nr = int(round((RACK_X1 - RACK_X0) / RACK_W))
    for r, (ry, face) in enumerate(ROWS):
        for i in range(nr):
            u = (i + r * 3) / (nr + 9)
            a = ease_out(prog(it, u * 0.7, u * 0.7 + 0.3))
            if a <= 0:
                continue
            x = RACK_X0 + i * RACK_W
            drop = (1 - a) * 1.2
            iso.box(ctx, x + 0.02, ry, 0.15 + drop, RACK_W - 0.04, RACK_D, RACK_H,
                    (0.20, 0.25, 0.34), a, top=0.9, left=0.55, right=0.38, edge=(0.4, 0.6, 0.85), edge_a=0.18)
            fy = ry + RACK_D
            if face == "front":
                for (lx, lz, col, rate, seed) in RACK_LEDS[(r, i)]:
                    sx, sy = iso.p(x + lx, fy, 0.15 + drop + lz)
                    dot(ctx, sx, sy, max(1.0, iso.s * 0.045), col, a * _blink(seed, rate, t), halo=1.6)
            else:
                hot = g("hot", 0)
                for lz in (0.6, 1.2, 1.8):
                    p1 = iso.p(x + 0.1, fy, 0.15 + drop + lz)
                    p2 = iso.p(x + 0.5, fy, 0.15 + drop + lz)
                    ctx.move_to(*p1)
                    ctx.line_to(*p2)
                rgba(ctx, mix((0.35, 0.4, 0.5), AMBER, hot), 0.6 * a)
                ctx.set_line_width(max(1.0, iso.s * 0.04))
                ctx.stroke()

    # Substation and pylon (left).
    gr = g("grid", 0)
    if gr > 0:
        a = ease_out(prog(gr, 0, 0.6))
        sx, sy, sdx, sdy = SUB
        iso.poly(ctx, [(sx - 0.3, sy - 0.3, 0.02), (sx + sdx + 0.3, sy - 0.3, 0.02),
                       (sx + sdx + 0.3, sy + sdy + 0.3, 0.02), (sx - 0.3, sy + sdy + 0.3, 0.02)])
        rgba(ctx, CYAN, 0.10 * a)
        ctx.fill()
        for k in range(3):
            iso.box(ctx, sx + k * 1.1, sy + 0.6, 0, 0.8, 1.6, 1.3 + 0.2 * (k == 1), (0.42, 0.50, 0.62), a,
                    edge=CYAN, edge_a=0.25)
        pylon(ctx, iso, PYLON[0], PYLON[1], 5.0, a)
    # UPS / battery room.
    up = g("ups", 0)
    if up > 0:
        a = ease_out(prog(up, 0, 0.6))
        x, y, dx, dy = UPS
        iso.box(ctx, x, y, 0, dx, dy, 0.12, (0.10, 0.14, 0.18), a)
        for k in range(4):
            for j in range(2):
                iso.box(ctx, x + 0.2 + j * 1.1, y + 0.25 + k * 0.75, 0.12 + (1 - a), 0.9, 0.55, 1.5,
                        (0.25, 0.42, 0.34), a, edge=GREEN, edge_a=0.35)
                lvl = 0.55 + 0.4 * math.sin(t * 0.8 + k + j)
                p1 = iso.p(x + 0.3 + j * 1.1, y + 0.8 + k * 0.75, 0.3 + (1 - a))
                p2 = iso.p(x + 0.3 + j * 1.1, y + 0.8 + k * 0.75, 0.3 + (1 - a) + 1.1 * lvl)
                ctx.move_to(*p1)
                ctx.line_to(*p2)
                rgba(ctx, GREEN, 0.9 * a)
                ctx.set_line_width(max(1.5, iso.s * 0.07))
                ctx.stroke()
    # Generators.
    ge = g("gen", 0)
    for k, (x, y) in enumerate(GEN):
        a = ease_out(prog(ge, k * 0.2, k * 0.2 + 0.6))
        if a <= 0:
            continue
        iso.box(ctx, x, y, (1 - a) * 1.2, 2.8, 1.4, 1.4, (0.55, 0.50, 0.42), a, edge=AMBER, edge_a=0.3)
        iso.box(ctx, x + 2.2, y + 0.4, 1.4 + (1 - a) * 1.2, 0.3, 0.3, 0.8, (0.45, 0.42, 0.38), a)

    # Front wall outline (cutaway edges).
    if base > 0:
        ctx.set_line_width(1.4)
        rgba(ctx, CYAN, 0.35 * base)
        iso.line(ctx, (hx0, hy1, WALL_H), (hx1, hy1, WALL_H))
        iso.line(ctx, (hx1, hy0, WALL_H), (hx1, hy1, WALL_H))
        iso.line(ctx, (hx0, hy0, WALL_H), (hx1, hy0, WALL_H))
        iso.line(ctx, (hx0, hy0, WALL_H), (hx0, hy1, WALL_H))
        for (px, py) in ((hx1, hy1), (hx1, hy0), (hx0, hy1)):
            iso.line(ctx, (px, py, 0.15), (px, py, WALL_H))
        ctx.stroke()
        iso.line(ctx, (hx0, hy1, 0.15), (hx1, hy1, 0.15))
        iso.line(ctx, (hx1, hy0, 0.15), (hx1, hy1, 0.15))
        ctx.stroke()

    draw_flows(ctx, iso, t, st)


def power_path(iso):
    pts3 = [(PYLON[0], PYLON[1], 4.6), (1.0, 7.0, 1.4), (3.4, 7.0, 1.4), (6.0, 4.2, 1.6),
            (8.0, 4.2, 3.0), (10.0, 4.4, 3.0), (19.6, 4.4, 3.0)]
    return [iso.p(*p) for p in pts3]


def power_path2(iso):
    pts3 = [(6.0, 4.2, 1.6), (8.0, 4.2, 3.0), (10.0, 9.0, 3.0), (19.6, 9.0, 3.0)]
    return [iso.p(*p) for p in pts3]


def gen_path(iso):
    return [iso.p(*p) for p in [(5.5, 8.6, 1.4), (5.5, 6.2, 1.6), (6.0, 5.0, 1.6)]]


def water_paths(iso):
    supply = [(12.0, 0.6, 1.2), (8.7, 0.6, 1.2), (8.7, 0.6, 3.3), (8.7, 4.4, 3.3), (8.7, 4.4, 2.75)]
    supply2 = [(8.7, 4.4, 3.3), (8.7, 9.0, 3.3), (8.7, 9.0, 2.75)]
    ret = [(9.0, 4.6, 2.75), (9.0, 4.6, 3.5), (9.0, 0.9, 3.5), (9.0, 0.9, 1.4), (15.5, 0.9, 1.4)]
    return ([iso.p(*p) for p in supply], [iso.p(*p) for p in supply2], [iso.p(*p) for p in ret])


def draw_flows(ctx, iso, t, st):
    g = st.get
    pw = g("power", 0)
    if pw > 0:
        path = power_path(iso)
        draw_polyline(ctx, path, ease_io(min(1.0, pw * 1.5)))
        glow_stroke(ctx, CYAN, max(1.2, iso.s * 0.05), 0.55 * min(1, pw * 2), glow=2.0, layers=2)
        if pw > 0.6:
            path2 = power_path2(iso)
            draw_polyline(ctx, path2, ease_io((pw - 0.6) / 0.4))
            glow_stroke(ctx, CYAN, max(1.2, iso.s * 0.05), 0.45, glow=2.0, layers=2)
        flow_particles(ctx, path, t, CYAN, n=22, speed=0.10, r=max(1.8, iso.s * 0.07), a=clamp(pw * 2 - 0.4))
    gn = g("gen", 0) * g("genline", 1.0)
    if gn > 0:
        path = gen_path(iso)
        draw_polyline(ctx, path, ease_io(gn))
        ctx.set_dash([6, 6])
        glow_stroke(ctx, AMBER, 1.6, 0.6 * gn, glow=1.0, layers=1)
        ctx.set_dash([])
    wa = g("water", 0)
    if wa > 0:
        sup, sup2, ret = water_paths(iso)
        for pts, col in ((sup, ICE), (sup2, ICE), (ret, AMBER)):
            draw_polyline(ctx, pts, ease_io(wa))
            glow_stroke(ctx, col, max(1.5, iso.s * 0.06), 0.5 * wa, glow=1.5, layers=2)
        flow_particles(ctx, sup, t, ICE, n=10, speed=0.18, r=max(1.5, iso.s * 0.06), a=wa)
        flow_particles(ctx, sup2, t, ICE, n=6, speed=0.25, r=max(1.5, iso.s * 0.06), a=wa)
        flow_particles(ctx, ret, t, AMBER, n=10, speed=0.18, r=max(1.5, iso.s * 0.06), a=wa)
    # Airflow: cold air along cold aisles at floor level, hot air back along the ceiling.
    ca = g("cold", 0)
    if ca > 0:
        for (ya, yb) in COLD:
            ym = (ya + yb) / 2
            for k in range(18):
                u = (k / 18 + t * 0.12) % 1.0
                x = lerp(9.3, RACK_X1, u)
                z = 0.35 + 0.9 * u + 0.2 * math.sin(k * 1.7)
                yy = ym + (0.45 * math.sin(k * 2.3))
                sx, sy = iso.p(x, yy, z)
                dot(ctx, sx, sy, max(1.4, iso.s * 0.06), ICE, ca * min(u / 0.1, (1 - u) / 0.1, 1), halo=1.8)
    ha = g("hot", 0)
    if ha > 0:
        for (ya, yb) in HOT:
            ym = (ya + yb) / 2
            for k in range(20):
                u = (k / 20 + t * 0.12) % 1.0
                x = lerp(RACK_X1, 9.3, u)
                z = 1.0 + 1.9 * min(1.0, u * 3) + 0.15 * math.sin(k * 1.3)
                yy = ym + 0.5 * math.sin(k * 2.1)
                sx, sy = iso.p(x, yy, z)
                dot(ctx, sx, sy, max(1.4, iso.s * 0.06), mix(AMBER, RED, 0.4), ha * min(u / 0.1, (1 - u) / 0.1, 1),
                    halo=1.8)


def callout(ctx, iso, anchor3, lx, ly, title_s, sub, a, color=CYAN, align="left"):
    """Leader line from a 3D anchor to a screen-space label."""
    if a <= 0:
        return
    ax, ay = iso.p(*anchor3)
    ex = lx - 12 if align == "left" else lx + 12
    p = ease_out(a)
    draw_polyline(ctx, [(ax, ay), (ax + (ex - ax) * 0.0, ly), (ex, ly)] if abs(ax - ex) > 1 else [(ax, ay), (ex, ly)], p)
    rgba(ctx, color, 0.75 * a)
    ctx.set_line_width(1.4)
    ctx.stroke()
    dot(ctx, ax, ay, 4, color, a, halo=2.5)
    text(ctx, title_s, lx, ly - 6, 29, COND, color, a, align, 0.14)
    if sub:
        text(ctx, sub, lx, ly + 26, 24, SEMI, INK, 0.85 * a, align)
