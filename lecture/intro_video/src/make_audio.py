"""Procedural score + sound design, mixed under the narration.

Reads build/timeline.json and build/voice.wav, writes build/mix.wav (48 kHz stereo).
"""
import json
import math
import wave
from pathlib import Path

import numpy as np
from scipy.signal import butter, fftconvolve, lfilter, sosfilt

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
SR = 48000
TL = json.loads((BUILD / "timeline.json").read_text())
SC = {s["id"]: s for s in TL["scenes"]}
TOTAL = TL["total"]
N = int((TOTAL + 0.5) * SR)
rng = np.random.default_rng(3)


def midi(m):
    return 440.0 * 2 ** ((m - 69) / 12)


CHORDS = {   # MIDI notes
    "Dm9": [38, 45, 53, 60, 64], "Bbmaj7": [34, 46, 50, 57, 62], "Fmaj7": [41, 48, 52, 57, 60],
    "C/E": [40, 48, 55, 62, 64], "Gm9": [31, 50, 58, 62, 69], "Am7": [33, 52, 55, 60, 64],
    "Dmaj9": [38, 45, 54, 61, 64],
}
PLAN = {   # scene: (progression, arpeggio level, pulse level)
    "open": (["Dm9", "Dm9", "Bbmaj7", "Dm9", "Bbmaj7", "Fmaj7", "Fmaj7"], 0.0, 0.0),
    "anatomy": (["Dm9", "Bbmaj7", "Fmaj7", "C/E"], 0.45, 0.0),
    "scale": (["Dm9", "Bbmaj7", "Fmaj7", "C/E"], 0.6, 0.35),
    "britain": (["Gm9", "Bbmaj7", "Dm9", "C/E"], 0.6, 0.45),
    "prices": (["Dm9", "Am7", "Bbmaj7", "Am7"], 0.45, 0.25),
    "toolkit": (["Bbmaj7", "Fmaj7", "C/E", "Dm9"], 0.7, 0.4),
    "flex": (["Bbmaj7", "Fmaj7", "C/E", "Dm9"], 0.75, 0.5),
    "close": (["Gm9", "Bbmaj7", "Dmaj9", "Dmaj9", "Dmaj9", "Dmaj9"], 0.35, 0.0),
}
CHORD_LEN = 4.0
BEAT = 0.5   # 120 BPM


def table(n_harm=14, soft=5.0):
    x = np.linspace(0, 1, 4096, endpoint=False)
    w = np.zeros_like(x)
    for k in range(1, n_harm + 1):
        w += np.sin(2 * np.pi * k * x) / k * math.exp(-k / soft)
    return w / np.abs(w).max()


SAW = table()


def osc(freq, n, phase0=0.0):
    ph = (phase0 + freq * np.arange(n) / SR) % 1.0
    return SAW[(ph * len(SAW)).astype(np.int32)]


def env_adsr(n, att, rel):
    e = np.ones(n, np.float32)
    a = min(n, int(att * SR))
    r = min(n, int(rel * SR))
    e[:a] = np.linspace(0, 1, a) ** 2
    if r:
        e[-r:] *= np.linspace(1, 0, r) ** 1.5
    return e


def schedule():
    """List of (start, end, chord, arp, pulse) segments over the whole timeline."""
    segs = []
    for sid, (prog, arp, pulse) in PLAN.items():
        s = SC[sid]
        t, k = s["start"], 0
        while t < s["end"] - 0.01:
            e = min(t + CHORD_LEN, s["end"])
            if s["end"] - e < 1.5:
                e = s["end"]
            segs.append((t, e, prog[k % len(prog)], arp, pulse))
            t, k = e, k + 1
    return segs


def pad_layer(segs):
    L = np.zeros(N, np.float32)
    R = np.zeros(N, np.float32)
    for (t0, t1, ch, _, _) in segs:
        att, rel = 1.2, 2.2
        i0 = int(t0 * SR)
        n = int((t1 - t0 + rel) * SR)
        n = min(n, N - i0)
        e = env_adsr(n, att, rel)
        for m in CHORDS[ch][1:]:
            f = midi(m)
            amp = 0.07 if m < 50 else 0.05
            for det, side in ((-7, "L"), (7, "R"), (0, "B")):
                fr = f * 2 ** (det / 1200)
                w = osc(fr, n, rng.random()) * e * amp
                if side in ("L", "B"):
                    L[i0:i0 + n] += w * (0.8 if side == "B" else 1)
                if side in ("R", "B"):
                    R[i0:i0 + n] += w * (0.8 if side == "B" else 1)
        # Sub root.
        f = midi(CHORDS[ch][0])
        sub = np.sin(2 * np.pi * f * np.arange(n) / SR) * e * 0.10
        L[i0:i0 + n] += sub
        R[i0:i0 + n] += sub
    sos = butter(2, 1400, "low", fs=SR, output="sos")
    return sosfilt(sos, L).astype(np.float32), sosfilt(sos, R).astype(np.float32)


def pluck(f, dur=0.45):
    n = int(dur * SR)
    tt = np.arange(n) / SR
    e = np.exp(-tt / 0.16) * (1 - np.exp(-tt / 0.004))
    return ((np.sin(2 * np.pi * f * tt) + 0.25 * np.sin(4 * np.pi * f * tt)) * e).astype(np.float32)


def arp_layer(segs):
    L = np.zeros(N, np.float32)
    R = np.zeros(N, np.float32)
    pattern = [1, 3, 2, 4, 3, 2, 4, 1]
    for (t0, t1, ch, arp, _) in segs:
        if arp <= 0:
            continue
        notes = CHORDS[ch]
        k = int(math.ceil(t0 / (BEAT / 2)))
        while k * BEAT / 2 < t1:
            tt = k * BEAT / 2
            m = notes[pattern[k % 8]] + 12
            p = pluck(midi(m)) * 0.06 * arp * (1.0 if k % 2 == 0 else 0.7)
            i = int(tt * SR)
            j = min(N, i + len(p))
            pan = 0.5 + 0.35 * math.sin(k * 0.9)
            L[i:j] += p[:j - i] * (1 - pan) * 1.4
            R[i:j] += p[:j - i] * pan * 1.4
            k += 1
    # Ping-pong echo (dotted eighth).
    d = int(0.375 * SR)
    for _ in range(3):
        L2, R2 = L.copy(), R.copy()
        L[d:] += R2[:-d] * 0.32
        R[d:] += L2[:-d] * 0.32
    return L, R


def pulse_layer(segs):
    out = np.zeros(N, np.float32)
    n = int(0.35 * SR)
    tt = np.arange(n) / SR
    kick = np.sin(2 * np.pi * (45 + 60 * np.exp(-tt / 0.04)) * tt) * np.exp(-tt / 0.12)
    hat_n = int(0.05 * SR)
    hat = rng.normal(0, 1, hat_n) * np.exp(-np.arange(hat_n) / SR / 0.012)
    hat = sosfilt(butter(2, 7000, "high", fs=SR, output="sos"), hat)
    for (t0, t1, _, _, pl) in segs:
        if pl <= 0:
            continue
        k = int(math.ceil(t0 / BEAT))
        while k * BEAT < t1:
            i = int(k * BEAT * SR)
            j = min(N, i + n)
            out[i:j] += kick[:j - i] * 0.20 * pl
            ih = int((k * BEAT + BEAT / 2) * SR)
            jh = min(N, ih + hat_n)
            if ih < N:
                out[ih:jh] += hat[:jh - ih] * 0.035 * pl
            k += 1
    return out


def reverb(L, R, secs=2.8, wet=0.32):
    n = int(secs * SR)
    tt = np.arange(n) / SR
    decay = np.exp(-tt / (secs / 6.0))
    irL = rng.normal(0, 1, n) * decay
    irR = rng.normal(0, 1, n) * decay
    sos = butter(1, 4500, "low", fs=SR, output="sos")
    irL, irR = sosfilt(sos, irL), sosfilt(sos, irR)
    irL /= np.sqrt((irL ** 2).sum())
    irR /= np.sqrt((irR ** 2).sum())
    wl = fftconvolve(L, irL)[:N]
    wr = fftconvolve(R, irR)[:N]
    return (L + wet * wl).astype(np.float32), (R + wet * wr).astype(np.float32)


# ------------------------------------------------------------------ sfx
def add(buf, t, sig, gain=1.0):
    i = int(t * SR)
    if i >= len(buf) or i + len(sig) <= 0:
        return
    s = sig
    if i < 0:
        s, i = s[-i:], 0
    j = min(len(buf), i + len(s))
    buf[i:j] += s[:j - i] * gain


def whoosh(dur=1.4, rising=True):
    n = int(dur * SR)
    noise = rng.normal(0, 1, n)
    out = np.zeros(n)
    y = 0.0
    for k in range(n):
        u = k / n
        fc = 300 + 5200 * (u if rising else 1 - u) ** 2
        a = 1 - math.exp(-2 * math.pi * fc / SR)
        y += a * (noise[k] - y)
        out[k] = y
    e = np.sin(np.pi * np.linspace(0, 1, n)) ** 2
    return (out * e / (np.abs(out).max() + 1e-9)).astype(np.float32)


def impact():
    n = int(2.8 * SR)
    tt = np.arange(n) / SR
    f = 32 + 70 * np.exp(-tt / 0.25)
    boom = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt / 0.9)
    nz = sosfilt(butter(2, 900, "low", fs=SR, output="sos"), rng.normal(0, 1, n)) * np.exp(-tt / 0.18) * 0.6
    return (boom + nz).astype(np.float32) * 0.8


def riser(dur=2.2):
    n = int(dur * SR)
    tt = np.linspace(0, 1, n)
    nz = sosfilt(butter(2, 2500, "high", fs=SR, output="sos"), rng.normal(0, 1, n))
    return (nz * tt ** 3 * 0.25).astype(np.float32)


def blip(f=1320, dur=0.18, g=0.25):
    n = int(dur * SR)
    tt = np.arange(n) / SR
    return ((np.sin(2 * np.pi * f * tt) + 0.4 * np.sin(2 * np.pi * f * 1.5 * tt))
            * np.exp(-tt / 0.05) * g).astype(np.float32)


def scan_sweep(dur=1.3):
    n = int(dur * SR)
    tt = np.arange(n) / SR
    f = 200 + 1400 * (tt / dur) ** 1.5
    return (np.sin(2 * np.pi * np.cumsum(f) / SR) * np.sin(np.pi * tt / dur) ** 2 * 0.07).astype(np.float32)


def sfx_track():
    L = np.zeros(N, np.float32)
    R = np.zeros(N, np.float32)
    scenes = TL["scenes"]
    for k, s in enumerate(scenes[1:], 1):
        w = whoosh(1.4, rising=k % 2 == 1)
        add(L, s["start"] - 0.9, w, 0.10)
        add(R, s["start"] - 0.85, w, 0.10)
    # Start: status-LED blips.
    for k in range(3):
        add(L, 0.15 + k / 1.1, blip(1760, 0.12, 0.10))
        add(R, 0.15 + k / 1.1, blip(1760, 0.12, 0.10))
    o = SC["open"]
    t_title = o["start"] + o["lines"][-1]["end"] + 0.25
    for buf in (L, R):
        add(buf, t_title - 2.1, riser(2.2), 1.0)
        add(buf, t_title + 0.05, impact(), 0.55)
        add(buf, o["start"] + o["cues"]["heat"] - 0.15, scan_sweep(), 1.0)
    b = SC["britain"]
    c = b["cues"]
    span = max(0.6, c["swales"] - c["zones"])
    for k in range(5):
        tp = b["start"] + c["zones"] + 0.3 + k * span / 4.6 + 0.35
        add(L, tp, blip(988 * 2 ** (k / 12 * 2), 0.2, 0.12))
        add(R, tp, blip(988 * 2 ** (k / 12 * 2), 0.2, 0.12))
    cl = SC["close"]
    t_end = cl["start"] + cl["cues"]["go"] + 1.6
    for buf in (L, R):
        add(buf, t_end - 2.1, riser(2.2), 1.0)
        add(buf, t_end + 0.05, impact(), 0.6)
    return L, R


def duck_env(voice):
    a = np.abs(voice)
    # Envelope follower: fast attack, slow release (block-based for speed).
    blk = 480
    m = a[: len(a) // blk * blk].reshape(-1, blk).max(1)
    env = np.zeros_like(m)
    y = 0.0
    for k, v in enumerate(m):
        y = v if v > y else y * 0.93
        env[k] = y
    env = np.repeat(env, blk)
    env = np.pad(env, (0, len(a) - len(env)), constant_values=0)
    g = 1 - 0.6 * np.clip(env / 0.06, 0, 1)
    return lfilter([0.02], [1, -0.98], g).astype(np.float32)


def read_mono(path):
    with wave.open(str(path)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768
    out = np.zeros(N, np.float32)
    out[:min(N, len(x))] = x[:N]
    return out


def main():
    segs = schedule()
    pL, pR = pad_layer(segs)
    aL, aR = arp_layer(segs)
    pulse = pulse_layer(segs)
    mL, mR = reverb(pL + aL, pR + aR)
    mL += pulse
    mR += pulse
    # Overall music shape: fade in at the start, swell to the titles, fade at the end.
    t = np.arange(N) / SR
    shape = np.clip(t / 2.5, 0, 1) * np.clip((TOTAL + 0.3 - t) / 2.5, 0, 1)
    voice = read_mono(BUILD / "voice.wav")
    voice /= np.abs(voice).max() + 1e-9
    voice *= 0.70
    g = duck_env(voice) * shape
    music_gain = 0.19
    sL, sR = sfx_track()
    L = voice + mL * g * music_gain + sL
    R = voice + mR * g * music_gain + sR
    peak = max(np.abs(L).max(), np.abs(R).max())
    L, R = L / peak * 0.89, R / peak * 0.89
    st = np.stack([L, R], 1)
    with wave.open(str(BUILD / "mix.wav"), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((st * 32767).astype("<i2").tobytes())
    rms = lambda x: 20 * np.log10(np.sqrt((x ** 2).mean()) + 1e-12)
    print(f"mix.wav {TOTAL:.1f}s  voice RMS {rms(voice):.1f} dB  music RMS {rms(mL * g * music_gain):.1f} dB")


if __name__ == "__main__":
    main()
