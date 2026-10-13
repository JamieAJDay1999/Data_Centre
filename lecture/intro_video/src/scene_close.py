"""Close: strain or asset, design/site/operate, the three questions, final title."""
from engine import *  # noqa: F401,F403
from engine import W, H
import dc_model as M

ISO = M.fit_iso((-3.5, -2.5, 23, 13.0), (260, 150, 1400, 620))
TURBINES = [(-3.5, 1.0), (-2.0, -2.5), (1.5, -3.0), (21.5, -2.8), (24.5, -0.5)]


def draw(ctx, t, S):
    c = S["cues"]
    dur = S["dur"]
    background(ctx, top=(0.010, 0.018, 0.036), glow=(TEAL, 0.05, 900), glow_pos=(0.5, 0.35))
    q_on = ease_io(prog(t, c["q1"] - 0.6, c["q1"] + 0.4))
    title_on = ease_io(prog(t, c["go"] + 1.6, c["go"] + 2.6))
    fac_a = ease_out(prog(t, 0.0, 1.0)) * (1 - 0.88 * q_on) * (1 - title_on)

    strain = fade(t, c["strain"] - 0.2, c["asset"] + 0.2, 0.4, 0.6)
    asset = ease_out(prog(t, c["asset"] - 0.1, c["asset"] + 1.0))
    iso = M.Iso(ISO.ox, ISO.oy + 6 * math.sin(t * 0.3), ISO.s)

    if fac_a > 0:
        # Turbines behind the facility once it becomes an asset.
        for k, (x, y) in enumerate(TURBINES):
            ta = fac_a * ease_out(prog(asset, k * 0.12, k * 0.12 + 0.5))
            if ta > 0:
                sx, sy = iso.p(x, y, 0)
                turbine(ctx, sx, sy, iso.s * 3.6, t, mix(GREEN, INK, 0.2), ta, 1.0, phase=k)
        st = {k: 1.0 for k in ("base", "grid", "ups", "gen", "it", "cool", "cold", "hot", "power", "water")}
        st["genline"] = 0.0
        ctx.push_group()
        M.draw_facility(ctx, iso, t, st)
        ctx.pop_group_to_source()
        ctx.paint_with_alpha(fac_a)
        # Grid connection: red and stressed, then teal and two-way.
        path = M.power_path(iso)[:5]
        if strain > 0:
            pulse = 0.6 + 0.4 * math.sin(t * 9)
            draw_polyline(ctx, path, 1)
            glow_stroke(ctx, RED, 5, fac_a * strain * pulse, glow=2.5, layers=3)
            rg = cairo.RadialGradient(*iso.p(2.0, 7.0, 1.0), 0, *iso.p(2.0, 7.0, 1.0), 260)
            rg.add_color_stop_rgba(0, RED[0], RED[1], RED[2], 0.18 * strain * fac_a * pulse)
            rg.add_color_stop_rgba(1, RED[0], RED[1], RED[2], 0)
            ctx.set_source(rg)
            ctx.paint()
        if asset > 0:
            draw_polyline(ctx, path, 1)
            glow_stroke(ctx, TEAL, 4, fac_a * asset * 0.8, glow=2.0, layers=2)
            flow_particles(ctx, path, t, GREEN, n=12, speed=0.12, r=4, a=fac_a * asset)
            flow_particles(ctx, path[::-1], t, TEAL, n=8, speed=0.09, r=3, a=fac_a * asset, offset=0.5)

    # Strain vs asset words.
    wa = 1 - ease_in(prog(t, c["design"] - 0.6, c["design"] - 0.1))
    sa = ease_out(prog(t, c["strain"] - 0.1, c["strain"] + 0.5)) * wa
    text(ctx, "A STRAIN ON THE GRID", W / 2 - 60, 840, 54, HEAD, RED, sa * (1 - 0.55 * asset), "right", 0.06)
    aa = ease_out(prog(t, c["asset"] - 0.1, c["asset"] + 0.5)) * wa
    text(ctx, "or", W / 2, 840, 40, SEMI, MUTED, aa, "center")
    text(ctx, "A FLEXIBLE ASSET", W / 2 + 60, 840, 54, HEAD, TEAL, aa, "left", 0.06, glow=0.4)

    # Design · site · operate.
    va = 1 - ease_in(prog(t, c["q1"] - 0.8, c["q1"] - 0.2))
    for k, (word, cue) in enumerate((("DESIGN", "design"), ("SITE", "build"), ("OPERATE", "operate"))):
        p = ease_back(prog(t, c[cue] - 0.05, c[cue] + 0.45), 1.6)
        if p > 0:
            text(ctx, word, W / 2 + (k - 1) * 480, 850 + (1 - clamp(p)) * 20, 72, HEAD,
                 (CYAN, GREEN, GOLD)[k], clamp(p) * va, "center", 0.12, glow=0.4)

    # Three questions.
    qa = 1 - title_on
    for k, (qs, cue, col) in enumerate((("How much flexibility?", "q1", CYAN), ("For how long?", "q2", GREEN),
                                        ("At what cost?", "q3", GOLD))):
        p = ease_out(prog(t, c[cue] - 0.1, c[cue] + 0.6))
        if p > 0:
            text(ctx, qs, W / 2, 390 + k * 150 + (1 - p) * 26, 92, HEAD, col, p * qa, "center", glow=0.4)

    # Final title.
    if title_on > 0:
        cy = H / 2 - 10
        tt = c["go"] + 1.6
        rise_text(ctx, "LECTURE  INTRODUCTION", W / 2, cy - 140, 26, t, tt, COND, CYAN, "center", 0.32, dur=0.8)
        rise_text(ctx, "Low-Carbon & Flexible", W / 2, cy - 22, 122, t, tt + 0.1, HEAD, INK, "center", dur=1.0, dy=30)
        rise_text(ctx, "Data Centres", W / 2, cy + 112, 122, t, tt + 0.25, HEAD, CYAN, "center", dur=1.0, dy=30,
                  glow=0.6)
        lp = ease_io(prog(t, tt + 0.4, tt + 1.6))
        rgba(ctx, CYAN, 0.8 * title_on)
        ctx.set_line_width(2)
        ctx.move_to(W / 2 - 400 * lp, cy + 168)
        ctx.line_to(W / 2 + 400 * lp, cy + 168)
        ctx.stroke()
    vignette(ctx, 0.5)
    return {}
