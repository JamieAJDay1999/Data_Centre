"""Render the intro video.

  python render.py still 12.5 out.png          one frame at global time (s)
  python render.py sheet open 12 out.png        contact sheet of a scene
  python render.py bench 40                     time 40 sampled frames
  python render.py video --workers 8            full render -> build/video_noaudio.mp4
"""
import importlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

import engine as E

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
TL = json.loads((BUILD / "timeline.json").read_text())
SCENES = TL["scenes"]
TOTAL = TL["total"]
FPS = E.FPS
NFRAMES = int(math.ceil(TOTAL * FPS))
XF = {"default": 0.8, "toolkit": 0.5}   # crossfade length keyed by the outgoing scene

_mods = {}


def module(sid):
    if sid not in _mods:
        try:
            _mods[sid] = importlib.import_module(f"scene_{sid}")
        except ModuleNotFoundError:
            _mods[sid] = None
    return _mods[sid]


def render_scene(S, t, frame):
    surf, ctx = E.new_canvas()
    mod = module(S["id"])
    fx = {}
    if mod is None:
        E.background(ctx)
        E.text(ctx, S["id"], E.W / 2, E.H / 2, 80, E.HEAD, E.MUTED, 1, "center")
    else:
        fx = mod.draw(ctx, t, S) or {}
    rgb = E.surface_rgb(surf)
    if fx.get("thermal", 0) > 0:
        rgb = E.thermalise(rgb, fx["thermal"], fx.get("scan"))
    return rgb, fx


def frame_at(T, frame):
    for i, S in enumerate(SCENES):
        if T < S["end"] or i == len(SCENES) - 1:
            break
    S = SCENES[i]
    xf_out = XF.get(S["id"], XF["default"])
    blend = None
    if i + 1 < len(SCENES) and T >= S["end"] - xf_out / 2:
        blend = (S, SCENES[i + 1], (T - (S["end"] - xf_out / 2)) / xf_out)
    elif i > 0:
        P = SCENES[i - 1]
        xf_in = XF.get(P["id"], XF["default"])
        if T < S["start"] + xf_in / 2:
            blend = (P, S, (T - (S["start"] - xf_in / 2)) / xf_in)
    if blend:
        A, B, w = blend
        w = E.smooth(w)
        ra, fa = render_scene(A, T - A["start"], frame)
        rb, fb = render_scene(B, T - B["start"], frame)
        rgb = (ra.astype(np.float32) * (1 - w) + rb.astype(np.float32) * w).astype(np.uint8)
        fx = fb if w > 0.5 else fa
    else:
        rgb, fx = render_scene(S, T - S["start"], frame)
    rgb = E.bloom(rgb, fx.get("bloom", 0.85))
    rgb = E.grain(rgb, frame, 4.0)
    # Global fade from black at the very start and to black at the end.
    k = min(1.0, T / 0.4, max(0.0, (TOTAL - T) / 1.2))
    if k < 1:
        rgb = (rgb.astype(np.float32) * k).astype(np.uint8)
    return rgb


def still(T, out):
    Image.fromarray(frame_at(T, int(T * FPS))).save(out)


def sheet(sid, n, out, times=None):
    S = next(s for s in SCENES if s["id"] == sid)
    if times is None:
        times = [S["start"] + 0.2 + (S["dur"] - 0.4) * k / (n - 1) for k in range(n)]
    else:
        times = [S["start"] + x for x in times]
    tw, th = 640, 360
    cols = 3
    rows = math.ceil(len(times) / cols)
    img = Image.new("RGB", (cols * tw, rows * (th + 24)), (10, 10, 10))
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    for k, T in enumerate(times):
        fr = Image.fromarray(frame_at(T, int(T * FPS))).resize((tw, th), Image.LANCZOS)
        x, y = (k % cols) * tw, (k // cols) * (th + 24)
        img.paste(fr, (x, y + 24))
        d.text((x + 6, y + 5), f"{sid}  local {T - S['start']:.2f}s  global {T:.2f}s", fill=(255, 80, 80))
    img.save(out)


def _encode_chunk(args):
    k, f0, f1, path = args
    sys.stdout.flush()
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{E.W}x{E.H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "15",
           "-pix_fmt", "yuv420p", "-threads", "2", "-x264-params", "keyint=60:min-keyint=60",
           str(path)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    for f in range(f0, f1):
        p.stdin.write(frame_at(f / FPS, f).tobytes())
        if (f - f0) % 300 == 0:
            el = time.time() - t0
            print(f"[chunk {k}] {f - f0}/{f1 - f0} frames, {el:.0f}s", flush=True)
    p.stdin.close()
    p.wait()
    return k, time.time() - t0


def video(workers, f_start=0, f_end=None):
    from multiprocessing import Pool
    f_end = f_end or NFRAMES
    chunks_dir = BUILD / "chunks"
    chunks_dir.mkdir(exist_ok=True)
    n = workers * 3
    edges = [f_start + (f_end - f_start) * i // n for i in range(n + 1)]
    jobs = [(i, edges[i], edges[i + 1], chunks_dir / f"c{i:03d}.mp4") for i in range(n)]
    t0 = time.time()
    with Pool(workers) as pool:
        for k, el in pool.imap_unordered(_encode_chunk, jobs):
            print(f"chunk {k} done in {el:.0f}s (wall {time.time() - t0:.0f}s)", flush=True)
    lst = chunks_dir / "list.txt"
    lst.write_text("".join(f"file '{j[3].as_posix()}'\n" for j in jobs))
    out = BUILD / "video_noaudio.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c", "copy", str(out)], check=True)
    print(f"video: {out}  frames {f_end - f_start}  wall {time.time() - t0:.0f}s")


def bench(n):
    import random
    rnd = random.Random(1)
    ts = sorted(rnd.uniform(0, TOTAL) for _ in range(n))
    t0 = time.time()
    per = {}
    for T in ts:
        a = time.time()
        frame_at(T, int(T * FPS))
        sid = next((s["id"] for s in SCENES if T < s["end"]), SCENES[-1]["id"])
        per.setdefault(sid, []).append(time.time() - a)
    el = time.time() - t0
    print(f"{n} frames in {el:.1f}s -> {el / n * 1000:.0f} ms/frame")
    for sid, v in per.items():
        print(f"  {sid:<8} {np.mean(v) * 1000:6.0f} ms  (n={len(v)})")
    print(f"total frames {NFRAMES}; est single-core {NFRAMES * el / n / 60:.1f} min")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "still":
        still(float(sys.argv[2]), sys.argv[3])
    elif cmd == "sheet":
        times = [float(x) for x in sys.argv[5].split(",")] if len(sys.argv) > 5 else None
        sheet(sys.argv[2], int(sys.argv[3]), sys.argv[4], times)
    elif cmd == "bench":
        bench(int(sys.argv[2]))
    elif cmd == "video":
        w = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 8
        video(w)
