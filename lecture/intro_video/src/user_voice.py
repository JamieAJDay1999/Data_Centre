"""Edit the presenter's own narration recording into per-line clips.

Input:  build/user/raw.wav (48 kHz mono, converted from the original recording)
Output: build/user/lines/<scene>_<n>.wav and build/user/lines.json
        (per line: clip path, duration, word timings from Whisper on the edited clip)

The edit decision list (EDL) below keeps the best take of every line, removes
false starts and stumbles, and tightens long pauses. Times are approximate
source times in seconds; each boundary is snapped to a silence or energy dip.
"""
import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UB = ROOT / "build" / "user"
SR = 48000

# Boundary modes: "s" onset (search back for silence), "e" offset (search forward
# for silence), "c" cut inside continuous speech (local energy minimum), "x" exact.
EDL = {
    "open": [
        [("seg", 0.85, 7.27, "s", "e")],
        [("seg", 8.10, 15.17, "s", "e")],
        [("seg", 16.50, 20.57, "s", "e")],
    ],
    "anatomy": [
        [("seg", 23.25, 30.26, "s", "e")],
        [("seg", 31.15, 41.68, "s", "e")],
        [("seg", 43.25, 54.02, "s", "e")],
    ],
    "scale": [
        # Drops "and the IAE expects" (mis-spoken acronym): "...in 2025. Roughly double that by 2030..."
        [("seg", 55.85, 62.40, "s", "e"), ("pause", 0.40), ("seg", 64.82, 69.05, "x", "e")],
        [("seg", 70.30, 83.04, "s", "e")],
    ],
    "britain": [
        [("seg", 84.80, 98.09, "s", "e")],
        # Removes "from Lancashire to South Wales" (should be Lanarkshire; the map labels the zones).
        [("seg", 98.95, 105.82, "s", "e"), ("pause", 0.55), ("seg", 106.85, 108.99, "s", "x")],
        [("seg", 111.95, 122.26, "s", "e")],
        # "...and where they run" -> "...and when they run" (word spliced from "when grids are stressed").
        [("seg", 123.65, 125.47, "s", "c"), ("seg", 245.04, 245.29, "c", "c"), ("seg", 125.63, 127.00, "c", "e")],
    ],
    "prices": [
        # Second take only (the first started "This is about, this is...").
        [("seg", 134.25, 141.92, "s", "e")],
        [("seg", 142.90, 152.81, "s", "e")],
        [("seg", 154.35, 157.74, "s", "e")],
    ],
    "toolkit": [
        [("seg", 159.60, 170.39, "s", "e")],
        [("seg", 171.40, 178.69, "s", "e")],
        [("seg", 179.95, 181.20, "s", "e")],
    ],
    "flex": [
        # "...every second." + clean "a job that used to fall to fossil fuel power stations."
        # + retake "With wind and sun, demand must flex too." + "The data centres have three levers."
        [("seg", 183.05, 185.60, "s", "e"), ("pause", 0.35), ("seg", 187.38, 187.70, "c", "e"),
         ("seg", 188.30, 188.66, "s", "e"), ("pause", 0.06), ("seg", 189.20, 191.98, "s", "c"),
         ("pause", 0.45), ("seg", 196.95, 200.34, "s", "e"), ("pause", 0.30), ("seg", 200.95, 202.85, "s", "e")],
        # Retake: "So firstly, the workload. A video call must run now, but ... plentiful,"
        [("seg", 212.25, 213.90, "s", "e"), ("pause", 0.30), ("seg", 214.20, 222.04, "s", "e")],
        [("seg", 223.25, 226.54, "s", "e")],
        [("seg", 226.85, 232.54, "s", "e")],
        [("seg", 233.65, 246.13, "s", "e")],
    ],
    "close": [
        # First take "...a strain on the grid" + second take "or one of its most valuable flexible assets..."
        [("seg", 248.40, 250.49, "s", "e"), ("pause", 0.30), ("seg", 258.45, 264.07, "s", "e")],
        # Adds question rhythm: "...offer? | For how long? | And at what cost?"
        [("seg", 264.95, 266.40, "s", "c"), ("pause", 0.30), ("seg", 266.40, 267.65, "c", "e"),
         ("pause", 0.30), ("seg", 267.80, 269.14, "s", "e")],
        None,   # "Let's find out." was not recorded.
    ],
}

# Caption text for each edited line: what is actually said, in British spelling.
CAPTIONS = {
    "open": ["Every time you ask an AI a question, stream a film, or tap to pay, a building somewhere does the work.",
             "Inside, row after row of servers turn electricity into computation, and almost all of it is turned into heat.",
             "These are data centres, and they're becoming one of the most important loads on the power system."],
    "anatomy": ["Power arrives from the grid, through the UPS: batteries that bridge the gap until generators start, if the grid ever fails.",
                "It feeds thousands of servers packed into racks. Virtually every watt becomes heat, removed by chillers, with servers breathing in from a cold aisle, and out into a hot aisle.",
                "The overhead is captured by PUE: total facility energy, over IT energy. The average is about 1.5. The best sites are around 1.1."],
    "scale": ["Demand is soaring. Data centres use around 485 terawatt-hours in 2025. Roughly double that by 2030: about as much as Japan uses today.",
              "The driver is AI. A conventional rack draws 5–10 kW. An AI rack, well over 100: as much as 400 British homes. That's why liquid cooling is going mainstream."],
    "britain": ["In Britain, data centres already draw around 2.4 gigawatts, clustered around London. In West London, the grid grew so congested that new homes were warned they might wait years for a connection.",
                "Yet developers want to connect more than 50 gigawatts of new data centres: more than Britain's peak demand. And the government has named AI Growth Zones.",
                "Meanwhile, the cleanest power comes from wind in Scotland and the North Sea. On the windiest days, wind farms are paid to switch off, because the network can't carry it all south.",
                "So where we build data centres, and when they run, really matters."],
    "prices": ["This is every quarter-hour of Britain's day-ahead power price in 2025. Each column a day, each row a time of day.",
               "The hot bands are evening peaks, set by gas. The cool stretches are windy hours, and in 672 quarter-hours, prices fell below zero.",
               "A load that could follow this pattern would be enormously valuable."],
    "toolkit": ["So how can we make data centres low-carbon? Efficiency, with free and liquid cooling. Clean supply, from wind and solar, increasingly matched hour by hour.",
                "Heat reuse: in Denmark, one data centre warms thousands of homes, and London is planning a network at Old Oak.",
                "And flexibility."],
    "flex": ["The grid must balance supply and demand every second: a job that used to fall to fossil-fuel power stations. With wind and sun, demand must flex too. And the data centres have three levers.",
             "So firstly, the workload. A video call must run now, but AI training or a backup can often wait, shifting computing to when power is plentiful.",
             "The UPS batteries, which can respond almost instantly.",
             "And the cooling. Chilled water and cold air let chillers ease off, without servers overheating.",
             "It's already happening. In a London trial with National Grid, an AI cluster cut its power by up to 40% on grid signals. And Google has agreed to curtail around a gigawatt of US demand, when grids are stressed."],
    "close": ["Data centres can be a strain on the grid, or one of its most valuable flexible assets. It depends on how we design, site and operate them.",
              "How much flexibility they can offer, for how long, and at what cost?",
              None],
}


# Select a recording: default "user" (first take, EDL above) or e.g. "--take user2".
import sys
TAKE = sys.argv[sys.argv.index("--take") + 1] if "--take" in sys.argv else "user"
TIGHTEN = True
if TAKE != "user":
    import importlib
    _m = importlib.import_module(f"edl_{TAKE}")
    EDL, CAPTIONS, TIGHTEN = _m.EDL, _m.CAPTIONS, False   # later takes keep their pauses
UB = ROOT / "build" / TAKE


def load():
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(UB / "raw.wav"), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32)


X = load()
HOP = int(0.005 * SR)
_n = len(X) // HOP
DB = 20 * np.log10(np.sqrt((X[:_n * HOP].reshape(_n, HOP) ** 2).mean(1)) + 1e-9)
QUIET = -58.0


def frame(t):
    return int(round(t / 0.005))


def snap(t, mode):
    i = frame(t)
    if mode == "s":       # walk back from t until 40 ms of quiet, then start just after it
        k = i
        lo = frame(t - 0.35)
        while k > lo and not (DB[max(0, k - 8):k] < QUIET).all():
            k -= 1
        return max(0.0, k * 0.005 - 0.02)
    if mode == "e":       # walk forward until 40 ms of quiet
        k = i
        hi = frame(t + 0.40)
        while k < hi and not (DB[k:k + 8] < QUIET).all():
            k += 1
        return k * 0.005 + 0.06
    if mode == "c":       # local minimum within +-60 ms
        lo, hi = frame(t - 0.06), frame(t + 0.06)
        return (lo + int(np.argmin(DB[lo:hi]))) * 0.005
    return t


def tighten(y, max_gap=0.55, keep=0.40):
    """Shorten internal pauses longer than max_gap to keep seconds."""
    n = len(y) // HOP
    db = 20 * np.log10(np.sqrt((y[:n * HOP].reshape(n, HOP) ** 2).mean(1)) + 1e-9)
    quiet = db < QUIET + 4
    out, i, last = [], 0, 0
    while i < n:
        if quiet[i]:
            j = i
            while j < n and quiet[j]:
                j += 1
            if i > 0 and j < n and (j - i) * 0.005 > max_gap:
                cut_a = i + int(keep / 2 / 0.005)
                cut_b = j - int(keep / 2 / 0.005)
                out.append(y[last * HOP:cut_a * HOP])
                last = cut_b
            i = j
        else:
            i += 1
    out.append(y[last * HOP:])
    return np.concatenate(out)


def assemble(items):
    """Returns the edited line and the source times of its first onset and last offset."""
    parts, src = [], [None, None]
    fade = int(0.008 * SR)
    for it in items:
        if it[0] == "pause":
            parts.append(np.zeros(int(it[1] * SR), np.float32))
            continue
        if it[0] == "gapfrom":
            src[1] = it[1]
            continue
        _, a, b, ma, mb = it
        a, b = snap(a, ma), snap(b, mb)
        src[0] = a if src[0] is None else src[0]
        src[1] = b
        seg = X[int(a * SR):int(b * SR)].copy()
        ramp = np.linspace(0, 1, fade, dtype=np.float32)
        seg[:fade] *= ramp
        seg[-fade:] *= ramp[::-1]
        parts.append(seg)
    y = np.concatenate(parts)
    return (tighten(y) if TIGHTEN else y), src


CHAIN = ("highpass=f=75,equalizer=f=220:t=q:w=1.0:g=-1.5,equalizer=f=3800:t=q:w=1.2:g=2.0,"
         "deesser=i=0.35,acompressor=threshold=-24dB:ratio=2.5:attack=8:release=140:makeup=1.5,"
         "alimiter=limit=0.95")


def process(y, path):
    tmp = path.with_suffix(".tmp.f32")
    tmp.write_bytes(y.astype(np.float32).tobytes())
    out = subprocess.run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", str(tmp),
                          "-af", CHAIN, "-f", "f32le", "-"], capture_output=True, check=True).stdout
    tmp.unlink()
    z = np.frombuffer(out, np.float32).copy()
    # Level each line to the same voiced RMS.
    n = len(z) // HOP
    fr = z[:n * HOP].reshape(n, HOP)
    rms = np.sqrt((fr ** 2).mean(1))
    voiced = rms[20 * np.log10(rms + 1e-9) > -40]
    target = 10 ** (-19 / 20)
    if len(voiced):
        z *= target / np.sqrt((voiced ** 2).mean())
    z = np.clip(z, -0.98, 0.98)
    import wave
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((z * 32767).astype("<i2").tobytes())
    return len(z) / SR


def main():
    import warnings
    warnings.filterwarnings("ignore")
    import whisper
    model = whisper.load_model("small")
    (UB / "lines").mkdir(exist_ok=True)
    out = {}
    for sid, lines in EDL.items():
        out[sid] = []
        for k, items in enumerate(lines):
            if items is None:
                out[sid].append(None)
                continue
            path = UB / "lines" / f"{sid}_{k}.wav"
            y, src = assemble(items)
            dur = process(y, path)
            r = model.transcribe(str(path), language="en", temperature=0, word_timestamps=True,
                                 condition_on_previous_text=False)
            words = [[w["start"], w["end"] - w["start"], w["word"].strip()]
                     for s in r["segments"] for w in s["words"]]
            out[sid].append({"path": str(path), "dur": dur, "words": words, "caption": CAPTIONS[sid][k],
                             "src": [round(src[0], 3), round(src[1], 3)]})
            print(f"{sid}_{k} {dur:5.2f}s | {r['text'].strip()}")
    (UB / "lines.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
