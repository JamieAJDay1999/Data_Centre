"""Inside a data centre: isometric cutaway built up with the narration, then PUE."""
from engine import *  # noqa: F401,F403
from engine import W, H
import dc_model as M

ISO_A = M.fit_iso((-3.5, -2.5, 23, 13.0), (110, 150, 1700, 900))
ISO_B = M.fit_iso((-3.5, -2.5, 23, 13.0), (10, 230, 1000, 760))


def pue_panel(ctx, t, c, a):
    if a <= 0:
        return
    x0, y0 = 1010, 300
    text(ctx, "POWER USAGE EFFECTIVENESS", x0, y0, 28, COND, CYAN, a, tracking=0.16)
    # Formula as a fraction.
    fy = y0 + 112
    text(ctx, "PUE  =", x0, fy + 14, 58, HEAD, INK, a)
    fx = x0 + 225
    rise_text(ctx, "Total facility energy", fx, fy - 18, 42, t, c["pue"] + 0.2, SEMI, INK, a=a)
    lw = text_width(ctx, "Total facility energy", SEMI, 42)
    p = ease_out(prog(t, c["pue"] + 0.3, c["pue"] + 0.9))
    rgba(ctx, INK, 0.9 * a)
    ctx.rectangle(fx, fy + 2, lw * p, 2.5)
    ctx.fill()
    rise_text(ctx, "IT energy", fx + lw / 2, fy + 56, 42, t, c["pue"] + 0.6, SEMI, CYAN, "center", a=a)

    # Value over time: 1.0 -> 1.5 (average) -> 1.1 (best).
    v = 1.0 + 0.5 * ease_io(prog(t, c["avg"] - 0.1, c["avg"] + 1.0))
    v -= 0.4 * ease_io(prog(t, c["best"] - 0.1, c["best"] + 1.0))
    bx, by, L = x0, y0 + 300, 500
    bar_a = a * ease_out(prog(t, c["pue"] + 0.8, c["pue"] + 1.4))
    if bar_a <= 0:
        return
    # Readout and its label.
    ra = bar_a * ease_out(prog(t, c["avg"] - 0.2, c["avg"] + 0.4))
    col = mix(AMBER, GREEN, ease_io(prog(t, c["best"], c["best"] + 1.0)))
    text(ctx, f"{v:.2f}", x0, y0 + 250, 104, HEAD, col, ra, glow=0.4)
    la = ra * (1 - ease_in(prog(t, c["best"] - 0.3, c["best"])))
    text(ctx, "INDUSTRY AVERAGE", x0 + 250, y0 + 236, 28, COND, AMBER, la, tracking=0.12)
    lb = bar_a * ease_out(prog(t, c["best"] + 0.3, c["best"] + 0.9))
    text(ctx, "BEST HYPERSCALE SITES", x0 + 250, y0 + 236, 28, COND, GREEN, lb, tracking=0.12)

    rrect(ctx, bx, by, L, 58, 6)
    rgba(ctx, CYAN, 0.85 * bar_a)
    ctx.fill()
    text(ctx, "IT ENERGY", bx + 20, by + 39, 26, COND, (0.02, 0.05, 0.1), bar_a, tracking=0.1)
    ow = (v - 1.0) * L
    if ow > 1:
        rrect(ctx, bx + L + 4, by, ow, 58, 6)
        rgba(ctx, AMBER, 0.9 * bar_a)
        ctx.fill()
    oa = bar_a * clamp((v - 1.02) / 0.1)
    text(ctx, "Overhead: cooling, conversion losses, lighting", bx, by + 100, 24, SEMI, AMBER, oa)
    # Scale from 1.0 (end of the IT bar) to 1.6.
    sy = by + 150
    ctx.set_line_width(1.2)
    rgba(ctx, MUTED, 0.6 * bar_a)
    ctx.move_to(bx + L, sy)
    ctx.line_to(bx + L + 4 + 0.6 * L, sy)
    ctx.stroke()
    for k in range(7):
        val = 1.0 + 0.1 * k
        xx = bx + L + 4 + (val - 1.0) * L if k else bx + L
        rgba(ctx, MUTED, 0.6 * bar_a)
        ctx.move_to(xx, sy - 6)
        ctx.line_to(xx, sy + 6)
        ctx.stroke()
        text(ctx, f"{val:.1f}", xx, sy + 32, 21, SEMI, MUTED, 0.85 * bar_a, "center")
    text(ctx, "PUE", bx + L - 20, sy + 32, 21, COND, MUTED, 0.85 * bar_a, "right", 0.1)
    for val, col2, cue in ((1.5, AMBER, "avg"), (1.1, GREEN, "best")):
        ma = bar_a * ease_out(prog(t, c[cue] + 0.6, c[cue] + 1.2))
        if ma > 0:
            xx = bx + L + 4 + (val - 1.0) * L
            rgba(ctx, col2, ma)
            ctx.move_to(xx, sy - 10)
            ctx.line_to(xx - 9, sy - 24)
            ctx.line_to(xx + 9, sy - 24)
            ctx.close_path()
            ctx.fill()


def draw(ctx, t, S):
    c = S["cues"]
    background(ctx, glow=(CYAN, 0.06, 900), glow_pos=(0.5, 0.55))

    t_b = c["pue"] - 0.4
    u = ease_io(prog(t, t_b, t_b + 1.4))
    iso = M.iso_lerp(ISO_A, ISO_B, u)
    iso.ox += 8 * math.sin(t * 0.25)
    iso.oy += 5 * math.sin(t * 0.21)

    st = {
        "base": ease_out(prog(t, 0.0, 1.0)),
        "grid": prog(t, c["grid"] - 0.3, c["grid"] + 0.6),
        "power": prog(t, c["grid"] + 0.2, c["it"] + 1.0),
        "ups": prog(t, c["ups"] - 0.2, c["ups"] + 0.7),
        "gen": prog(t, c["gen"] - 0.2, c["gen"] + 0.9),
        "it": prog(t, c["it"] - 0.2, c["it"] + 2.0),
        "cool": prog(t, c["cool"] - 0.2, c["cool"] + 1.0),
        "water": prog(t, c["cool"] + 0.4, c["cool"] + 1.6),
        "cold": prog(t, c["cold"] - 0.2, c["cold"] + 0.6),
        "hot": prog(t, c["hot"] - 0.2, c["hot"] + 0.6),
    }
    M.draw_facility(ctx, iso, t, st)

    # Callouts (phase A only).
    ca = 1 - ease_in(prog(t, t_b - 0.4, t_b + 0.2))
    items = [
        ("grid", (2.6, 6.5, 1.6), (-40, -190), "right", "GRID CONNECTION", "High-voltage supply", CYAN),
        ("ups", (6.0, 3.0, 1.7), (-60, -250), "right", "UPS", "Batteries bridge any outage", GREEN),
        ("gen", (5.8, 11.6, 1.2), (-90, 110), "right", "BACKUP GENERATORS", "Long-duration standby", AMBER),
        ("it", (15.5, 10.5, 2.3), (130, 170), "left", "IT EQUIPMENT", "Thousands of servers in racks", CYAN),
        ("cool", (15.5, -1.0, 1.6), (150, -150), "left", "COOLING", "Chillers and air handlers", ICE),
        ("cold", (19.4, 4.4, 0.6), (210, -95), "left", "COLD AISLE", "Air in", ICE),
        ("hot", (19.4, 6.8, 1.5), (230, -10), "left", "HOT AISLE", "Air out", mix(AMBER, RED, 0.4)),
    ]
    for key, anchor, (dx, dy), align, ttl, sub, col in items:
        a = ease_out(prog(t, c[key] - 0.1, c[key] + 0.5)) * ca
        if a <= 0:
            continue
        ax, ay = iso.p(*anchor)
        M.callout(ctx, iso, anchor, ax + dx, ay + dy, ttl, sub, a, col, align)

    eyebrow(ctx, "01", "Inside a data centre", t, 0.3)
    pue_panel(ctx, t, c, ease_out(prog(t, t_b + 0.6, t_b + 1.4)))
    source(ctx, "PUE: Uptime Institute global survey (average ~1.5); hyperscale operator disclosures (~1.1)", t,
           c["pue"] + 1.0, a=ease_out(prog(t, c["pue"], c["pue"] + 1)))
    vignette(ctx, 0.5)
    return {}
