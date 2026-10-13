"""Drawing primitives, easing, typography and post-processing for the intro video."""
import math
import random

import cairo
import numpy as np
from PIL import Image, ImageFilter

W, H = 1920, 1080
FPS = 30

# ---------------------------------------------------------------- palette
BG = (0.012, 0.020, 0.040)
INK = (0.93, 0.96, 1.00)
MUTED = (0.55, 0.62, 0.74)
DIM = (0.30, 0.36, 0.46)
CYAN = (0.24, 0.84, 1.00)
TEAL = (0.20, 0.95, 0.78)
GREEN = (0.42, 0.95, 0.52)
AMBER = (1.00, 0.68, 0.24)
RED = (1.00, 0.32, 0.30)
MAGENTA = (0.98, 0.36, 0.86)
ICE = (0.55, 0.76, 1.00)
GOLD = (1.00, 0.86, 0.38)
SLATE = (0.07, 0.10, 0.16)

HEAD = "Bahnschrift SemiBold"
BODY = "Bahnschrift Light"
SEMI = "Bahnschrift SemiLight"
COND = "Bahnschrift SemiBold Condensed"
CONDL = "Bahnschrift Condensed"


# ---------------------------------------------------------------- easing
def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def lerp(a, b, t):
    return a + (b - a) * t


def prog(t, a, b):
    """Linear 0..1 progress of t through [a, b]."""
    if b <= a:
        return 1.0 if t >= a else 0.0
    return clamp((t - a) / (b - a))


def smooth(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def ease_out(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def ease_in(x):
    x = clamp(x)
    return x ** 3


def ease_io(x):
    x = clamp(x)
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def ease_expo(x):
    x = clamp(x)
    return 1.0 if x >= 1 else 1 - 2 ** (-10 * x)


def ease_back(x, s=1.70158):
    x = clamp(x)
    return 1 + (s + 1) * (x - 1) ** 3 + s * (x - 1) ** 2


def mix(c1, c2, t):
    return tuple(lerp(a, b, t) for a, b in zip(c1, c2))


def fade(t, a, b, fade_in=0.5, fade_out=0.5):
    """Envelope that rises after a and falls before b."""
    up = ease_out(prog(t, a, a + fade_in)) if fade_in > 0 else (1.0 if t >= a else 0.0)
    down = 1 - ease_in(prog(t, b - fade_out, b)) if fade_out > 0 else (1.0 if t < b else 0.0)
    return up * down


# ---------------------------------------------------------------- canvas
def new_canvas():
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    ctx = cairo.Context(surf)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    return surf, ctx


def rgba(ctx, c, a=1.0):
    ctx.set_source_rgba(c[0], c[1], c[2], a)


def background(ctx, top=(0.020, 0.032, 0.060), bottom=BG, glow=None, glow_pos=(0.5, 0.45)):
    g = cairo.LinearGradient(0, 0, 0, H)
    g.add_color_stop_rgb(0, *top)
    g.add_color_stop_rgb(1, *bottom)
    ctx.set_source(g)
    ctx.paint()
    if glow:
        c, a, r = glow
        rg = cairo.RadialGradient(W * glow_pos[0], H * glow_pos[1], 0, W * glow_pos[0], H * glow_pos[1], r)
        rg.add_color_stop_rgba(0, c[0], c[1], c[2], a)
        rg.add_color_stop_rgba(1, c[0], c[1], c[2], 0)
        ctx.set_source(rg)
        ctx.paint()


def vignette(ctx, strength=0.55):
    rg = cairo.RadialGradient(W / 2, H / 2, H * 0.35, W / 2, H / 2, H * 1.05)
    rg.add_color_stop_rgba(0, 0, 0, 0, 0)
    rg.add_color_stop_rgba(1, 0, 0, 0, strength)
    ctx.set_source(rg)
    ctx.paint()


def rrect(ctx, x, y, w, h, r):
    r = min(r, w / 2, h / 2)
    ctx.new_sub_path()
    ctx.arc(x + w - r, y + r, r, -math.pi / 2, 0)
    ctx.arc(x + w - r, y + h - r, r, 0, math.pi / 2)
    ctx.arc(x + r, y + h - r, r, math.pi / 2, math.pi)
    ctx.arc(x + r, y + r, r, math.pi, 3 * math.pi / 2)
    ctx.close_path()


def panel(ctx, x, y, w, h, a=1.0, r=18, fill=(0.04, 0.07, 0.12), edge=CYAN, edge_a=0.28):
    rrect(ctx, x, y, w, h, r)
    rgba(ctx, fill, 0.78 * a)
    ctx.fill_preserve()
    rgba(ctx, edge, edge_a * a)
    ctx.set_line_width(1.5)
    ctx.stroke()


def glow_stroke(ctx, color, width, a=1.0, glow=3.0, layers=3):
    """Stroke the current path with a soft halo. Path is consumed."""
    path = ctx.copy_path()
    for i in range(layers, 0, -1):
        ctx.new_path()
        ctx.append_path(path)
        rgba(ctx, color, a * 0.10 / i)
        ctx.set_line_width(width + glow * width * i)
        ctx.stroke()
    ctx.new_path()
    ctx.append_path(path)
    rgba(ctx, color, a)
    ctx.set_line_width(width)
    ctx.stroke()


def dot(ctx, x, y, r, color, a=1.0, halo=2.5):
    ctx.new_path()
    if halo:
        rg = cairo.RadialGradient(x, y, 0, x, y, r * halo * 2)
        rg.add_color_stop_rgba(0, color[0], color[1], color[2], 0.45 * a)
        rg.add_color_stop_rgba(1, color[0], color[1], color[2], 0)
        ctx.set_source(rg)
        ctx.arc(x, y, r * halo * 2, 0, 2 * math.pi)
        ctx.fill()
    rgba(ctx, color, a)
    ctx.new_path()
    ctx.arc(x, y, r, 0, 2 * math.pi)
    ctx.fill()


# ---------------------------------------------------------------- text
def font(ctx, family, size):
    ctx.select_font_face(family, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    ctx.set_font_size(size)


def text_width(ctx, s, family, size, tracking=0.0):
    font(ctx, family, size)
    w = ctx.text_extents(s).x_advance
    return w + tracking * size * max(len(s) - 1, 0)


def text(ctx, s, x, y, size, family=BODY, color=INK, a=1.0, align="left", tracking=0.0, glow=0.0):
    """Draw a single line. y is the baseline. tracking is in em."""
    if a <= 0.002 or not s:
        return 0
    font(ctx, family, size)
    w = text_width(ctx, s, family, size, tracking)
    if align == "center":
        x -= w / 2
    elif align == "right":
        x -= w
    if glow > 0:
        rg_a = a * glow
        for k, (spread, alpha) in enumerate(((10, 0.05), (5, 0.08))):
            ctx.save()
            rgba(ctx, color, rg_a * alpha * 3)
            _draw_text_path(ctx, s, x, y, size, tracking)
            ctx.set_line_width(spread)
            ctx.stroke()
            ctx.restore()
    rgba(ctx, color, a)
    ctx.new_path()
    if tracking == 0:
        ctx.move_to(x, y)
        ctx.show_text(s)
    else:
        cx = x
        for ch in s:
            ctx.move_to(cx, y)
            ctx.show_text(ch)
            cx += ctx.text_extents(ch).x_advance + tracking * size
    ctx.new_path()
    return w


def _draw_text_path(ctx, s, x, y, size, tracking):
    ctx.new_path()
    if tracking == 0:
        ctx.move_to(x, y)
        ctx.text_path(s)
    else:
        cx = x
        for ch in s:
            ctx.move_to(cx, y)
            ctx.text_path(ch)
            cx += ctx.text_extents(ch).x_advance + tracking * size


def rise_text(ctx, s, x, y, size, t, t0, family=BODY, color=INK, align="left", tracking=0.0,
              dur=0.6, dy=18, a=1.0, glow=0.0):
    """Text that fades and slides up into place starting at t0."""
    p = ease_out(prog(t, t0, t0 + dur))
    if p <= 0:
        return 0
    return text(ctx, s, x, y + (1 - p) * dy, size, family, color, a * p, align, tracking, glow)


def type_text(ctx, s, x, y, size, t, t0, cps=40, family=BODY, color=INK, a=1.0, align="left",
              tracking=0.0, cursor=True):
    n = int(max(0.0, t - t0) * cps)
    if n <= 0:
        return
    shown = s[:n]
    text(ctx, shown, x, y, size, family, color, a, align, tracking)
    if cursor and n < len(s) + 8 and int(t * 3) % 2 == 0:
        w = text_width(ctx, shown, family, size, tracking)
        rgba(ctx, color, a * 0.8)
        ctx.rectangle(x + w + 4, y - size * 0.75, size * 0.08, size * 0.85)
        ctx.fill()


def eyebrow(ctx, num, label, t, t0=0.2, x=96, y=104, a=1.0):
    """Chapter tag: small numbered tracked label with a rule."""
    p = ease_out(prog(t, t0, t0 + 0.8))
    if p <= 0:
        return
    rgba(ctx, CYAN, 0.9 * p * a)
    ctx.rectangle(x, y - 13, 34 * p, 2.5)
    ctx.fill()
    text(ctx, num, x + 46, y, 21, COND, CYAN, p * a, tracking=0.12)
    text(ctx, label.upper(), x + 84, y, 21, COND, INK, 0.85 * p * a, tracking=0.16)


def title(ctx, s, t, t0=0.35, x=96, y=172, size=58, a=1.0):
    rise_text(ctx, s, x, y, size, t, t0, HEAD, INK, a=a, dur=0.8, dy=24)


def source(ctx, s, t, t0=0.8, a=1.0, x=96, y=H - 46):
    p = ease_out(prog(t, t0, t0 + 1.0))
    text(ctx, s, x, y, 17, SEMI, MUTED, 0.75 * p * a)


def counter(value, decimals=0):
    if decimals == 0:
        return f"{int(round(value)):,}"
    return f"{value:,.{decimals}f}"


# ---------------------------------------------------------------- projections
class Iso:
    """Isometric projection: x to the lower right, y to the lower left, z up."""

    def __init__(self, ox, oy, s):
        self.ox, self.oy, self.s = ox, oy, s
        self.c = math.cos(math.radians(30))
        self.k = math.sin(math.radians(30))

    def p(self, x, y, z=0.0):
        return (self.ox + (x - y) * self.c * self.s, self.oy + (x + y) * self.k * self.s - z * self.s)

    def poly(self, ctx, pts):
        ctx.new_path()
        for i, (x, y, z) in enumerate(pts):
            sx, sy = self.p(x, y, z)
            if i == 0:
                ctx.move_to(sx, sy)
            else:
                ctx.line_to(sx, sy)
        ctx.close_path()

    def box(self, ctx, x, y, z, dx, dy, dz, color, a=1.0, top=1.0, left=0.62, right=0.42,
            edge=None, edge_a=0.0):
        """Solid box; shades top/left(+y face)/right(+x face)."""
        if a <= 0.002:
            return
        faces = (
            ([(x, y + dy, z), (x + dx, y + dy, z), (x + dx, y + dy, z + dz), (x, y + dy, z + dz)], left),
            ([(x + dx, y, z), (x + dx, y + dy, z), (x + dx, y + dy, z + dz), (x + dx, y, z + dz)], right),
            ([(x, y, z + dz), (x + dx, y, z + dz), (x + dx, y + dy, z + dz), (x, y + dy, z + dz)], top),
        )
        for pts, shade in faces:
            self.poly(ctx, pts)
            rgba(ctx, tuple(c * shade for c in color), a)
            ctx.fill_preserve()
            if edge is not None and edge_a > 0:
                rgba(ctx, edge, edge_a * a)
                ctx.set_line_width(1.0)
                ctx.stroke()
            else:
                ctx.new_path()

    def line(self, ctx, a3, b3):
        ctx.move_to(*self.p(*a3))
        ctx.line_to(*self.p(*b3))


def polyline_point(pts, u):
    """Point at fraction u (0..1) of arc length along a 2D polyline."""
    segs = [math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    total = sum(segs) or 1.0
    d = clamp(u) * total
    for i, L in enumerate(segs):
        if d <= L or i == len(segs) - 1:
            f = 0 if L == 0 else d / L
            return (lerp(pts[i][0], pts[i + 1][0], f), lerp(pts[i][1], pts[i + 1][1], f))
        d -= L
    return pts[-1]


def draw_polyline(ctx, pts, u=1.0):
    """Path along the first fraction u of a polyline (for draw-on animation)."""
    ctx.new_path()
    if u <= 0:
        return
    segs = [math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    total = sum(segs) or 1.0
    d = u * total
    ctx.move_to(*pts[0])
    for i, L in enumerate(segs):
        if d >= L:
            ctx.line_to(*pts[i + 1])
            d -= L
        else:
            f = d / L if L else 0
            ctx.line_to(lerp(pts[i][0], pts[i + 1][0], f), lerp(pts[i][1], pts[i + 1][1], f))
            break


def flow_particles(ctx, pts, t, color, n=14, speed=0.25, r=3.2, a=1.0, offset=0.0, halo=2.2):
    """Dots streaming along a polyline."""
    if a <= 0.01:
        return
    for i in range(n):
        u = (i / n + t * speed + offset) % 1.0
        x, y = polyline_point(pts, u)
        edge = min(u / 0.08, (1 - u) / 0.08, 1.0)
        dot(ctx, x, y, r, color, a * edge, halo=halo)


# ---------------------------------------------------------------- icons
def turbine(ctx, x, y, h, t, color=INK, a=1.0, speed=1.0, phase=0.0):
    if a <= 0.01:
        return
    rgba(ctx, color, 0.85 * a)
    ctx.set_line_width(max(1.2, h * 0.035))
    ctx.move_to(x - h * 0.03, y)
    ctx.line_to(x - h * 0.012, y - h)
    ctx.line_to(x + h * 0.012, y - h)
    ctx.line_to(x + h * 0.03, y)
    ctx.close_path()
    ctx.fill()
    hx, hy = x, y - h
    ang = t * speed * 2.2 + phase
    ctx.set_line_width(max(1.0, h * 0.04))
    for k in range(3):
        th = ang + k * 2 * math.pi / 3
        ctx.move_to(hx, hy)
        ctx.line_to(hx + math.cos(th) * h * 0.55, hy + math.sin(th) * h * 0.55)
        ctx.stroke()
    ctx.arc(hx, hy, h * 0.045, 0, 2 * math.pi)
    ctx.fill()


def house(ctx, x, y, s, color, a=1.0, fill=True):
    ctx.new_path()
    ctx.move_to(x - s / 2, y)
    ctx.line_to(x - s / 2, y - s * 0.55)
    ctx.line_to(x, y - s)
    ctx.line_to(x + s / 2, y - s * 0.55)
    ctx.line_to(x + s / 2, y)
    ctx.close_path()
    rgba(ctx, color, a)
    if fill:
        ctx.fill()
    else:
        ctx.set_line_width(1.5)
        ctx.stroke()


def bolt(ctx, x, y, s, color, a=1.0):
    pts = [(0.15, -0.5), (-0.25, 0.08), (0.02, 0.08), (-0.12, 0.5), (0.28, -0.1), (0.0, -0.1)]
    ctx.new_path()
    for i, (px, py) in enumerate(pts):
        (ctx.move_to if i == 0 else ctx.line_to)(x + px * s, y + py * s)
    ctx.close_path()
    rgba(ctx, color, a)
    ctx.fill()


def battery(ctx, x, y, w, h, level, color, a=1.0):
    rrect(ctx, x, y, w, h, h * 0.12)
    rgba(ctx, color, 0.9 * a)
    ctx.set_line_width(3)
    ctx.stroke()
    rgba(ctx, color, 0.9 * a)
    ctx.rectangle(x + w * 0.35, y - h * 0.08, w * 0.3, h * 0.08)
    ctx.fill()
    pad = 7
    fh = (h - 2 * pad) * clamp(level)
    rrect(ctx, x + pad, y + h - pad - fh, w - 2 * pad, fh, 4)
    rgba(ctx, color, 0.75 * a)
    ctx.fill()


# ---------------------------------------------------------------- post-processing
_GRAIN = None
_THERMAL_LUT = None


def _grain():
    global _GRAIN
    if _GRAIN is None:
        rng = np.random.default_rng(7)
        small = rng.normal(0, 1, (8, H // 2, W // 2)).astype(np.float32)
        _GRAIN = [np.asarray(Image.fromarray(g).resize((W, H), Image.BILINEAR)) for g in small]
    return _GRAIN


def thermal_lut():
    global _THERMAL_LUT
    if _THERMAL_LUT is None:
        stops = [(0, (0.0, 0.0, 0.05)), (0.18, (0.18, 0.02, 0.45)), (0.40, (0.75, 0.05, 0.45)),
                 (0.62, (1.0, 0.35, 0.05)), (0.82, (1.0, 0.75, 0.10)), (1.0, (1.0, 1.0, 0.85))]
        lut = np.zeros((256, 3), np.float32)
        for i in range(256):
            v = i / 255
            for (a, ca), (b, cb) in zip(stops, stops[1:]):
                if a <= v <= b:
                    f = (v - a) / (b - a)
                    lut[i] = [lerp(ca[k], cb[k], f) for k in range(3)]
                    break
        _THERMAL_LUT = (lut * 255).astype(np.uint8)
    return _THERMAL_LUT


def surface_rgb(surf):
    surf.flush()
    a = np.ndarray((H, W, 4), np.uint8, surf.get_data())
    return a[:, :, 2::-1].copy()  # BGRA -> RGB


def thermalise(rgb, amount, scan_x=None):
    """Blend toward a thermal-camera look; optional vertical scan edge."""
    if amount <= 0:
        return rgb
    lum = (0.30 * rgb[..., 0] + 0.55 * rgb[..., 1] + 0.15 * rgb[..., 2]).astype(np.float32)
    lum = np.clip(lum * 1.5 + 14, 0, 255).astype(np.uint8)
    th = thermal_lut()[lum].astype(np.float32)
    out = rgb.astype(np.float32)
    if scan_x is None:
        m = amount
        out = out * (1 - m) + th * m
    else:
        xs = np.arange(W, dtype=np.float32)
        m = np.clip((scan_x - xs) / 40.0, 0, 1)[None, :, None] * amount
        out = out * (1 - m) + th * m
        edge = np.exp(-((xs - scan_x) / 6.0) ** 2)[None, :, None] * 160 * amount
        out = out + edge * np.array([1.0, 0.85, 0.6], np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def bloom(rgb, strength=0.9, threshold=120):
    img = Image.fromarray(rgb)
    small = img.resize((W // 4, H // 4), Image.BILINEAR)
    s = np.asarray(small).astype(np.int16) - threshold
    s = np.clip(s * 2, 0, 255).astype(np.uint8)
    s_img = Image.fromarray(s)
    b1 = s_img.filter(ImageFilter.GaussianBlur(5))
    b2 = s_img.resize((W // 16, H // 16), Image.BILINEAR).filter(ImageFilter.GaussianBlur(5))
    g = (np.asarray(b1.resize((W, H), Image.BILINEAR)).astype(np.float32) * 0.6
         + np.asarray(b2.resize((W, H), Image.BILINEAR)).astype(np.float32) * 0.8)
    out = rgb.astype(np.float32) + g * strength
    return np.clip(out, 0, 255).astype(np.uint8)


def grain(rgb, frame, amount=5.0):
    g = _grain()[frame % 8]
    out = rgb.astype(np.float32) + g[..., None] * amount
    return np.clip(out, 0, 255).astype(np.uint8)


def rng_for(seed):
    return random.Random(seed)
