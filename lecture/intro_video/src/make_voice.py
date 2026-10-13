"""Synthesise narration line by line and build the master timeline.

Outputs (in build/):
  voice/<hash>.mp3, voice/<hash>.json   cached TTS audio + word boundaries
  voice.wav                             full narration track, 48 kHz mono
  timeline.json                         scene windows, line times, cue times
  captions.srt                          subtitle file from word timings
"""
import asyncio
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

import edge_tts

from narration import RATE, SCENES, VOICE

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
VOICE_DIR = BUILD / "voice"
SR = 48000
MARK = re.compile(r"\[\[(\w+)\]\]")


def clean(text):
    return MARK.sub("", text)


def norm(word):
    return re.sub(r"[^a-z0-9]", "", word.lower())


async def synth(text):
    key = hashlib.sha1(f"{VOICE}|{RATE}|{text}".encode()).hexdigest()[:16]
    mp3, meta = VOICE_DIR / f"{key}.mp3", VOICE_DIR / f"{key}.json"
    if mp3.exists() and meta.exists():
        return mp3, json.loads(meta.read_text())
    comm = edge_tts.Communicate(text, VOICE, rate=RATE, boundary="WordBoundary")
    audio, words = b"", []
    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            audio += chunk["data"]
        elif chunk["type"] == "WordBoundary":
            words.append([chunk["offset"] / 1e7, chunk["duration"] / 1e7, chunk["text"]])
    mp3.write_bytes(audio)
    meta.write_text(json.dumps(words))
    return mp3, words


def decode(mp3):
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(mp3), "-ac", "1", "-ar", str(SR),
         "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


NUMSTART = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
            "eight": "8", "nine": "9", "ten": "1", "twenty": "2", "thirty": "3", "forty": "4", "fifty": "5"}


HOMOPHONES = {"cite": "site", "sight": "site"}


def word_match(target, w):
    w = HOMOPHONES.get(w, w)
    if w and target and (target.startswith(w) or w.startswith(target)):
        return True
    return target in NUMSTART and w[:1].isdigit() and w[0] == NUMSTART[target]


def find_cues(text, words, strict=True):
    """Map each [[marker]] to the start time of the word that follows it."""
    cues, search_from = {}, 0
    for m in MARK.finditer(text):
        following = clean(text[m.end():]).split()
        target = norm(following[0]) if following else ""
        for i in range(search_from, len(words)):
            if word_match(target, norm(words[i][2])):
                cues[m.group(1)] = words[i][0]
                search_from = i + 1
                break
        else:
            if strict:
                raise ValueError(f"Cue {m.group(1)!r} not aligned in: {text}")
    return cues


def read_wav(path):
    import wave
    with wave.open(str(path)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768


USER = "--user" in sys.argv
TAKE = sys.argv[sys.argv.index("--take") + 1] if "--take" in sys.argv else "user"
MAX_NATURAL_GAP = 4.0   # ignore longer recorded gaps (retakes etc.)
USER_GAP = 0.15     # extra breathing room between a presenter's lines
FALLBACK = {"lanark": ("zones", 0.9), "swales": ("zones", 1.8)}


def srt_time(t):
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


async def main():
    VOICE_DIR.mkdir(parents=True, exist_ok=True)
    timeline, track = [], []
    user = json.loads((BUILD / TAKE / "lines.json").read_text()) if USER else None
    t = 0.0
    prev_src_end, prev_end_t = None, None   # presenter's previous line: source end, timeline end
    for scene in SCENES:
        s0 = t
        t += scene["lead"]
        lines, cues = [], {}
        for k, (text, gap) in enumerate(scene["lines"]):
            if USER:
                rec = user[scene["id"]][k]
                if rec is None:
                    continue        # line not recorded; its cues get fallbacks below
                pcm = read_wav(rec["path"])
                words = rec["words"]
                shown = rec["caption"]
                gap += USER_GAP
                # Never make a recorded pause shorter: wait at least as long as the presenter did.
                if prev_src_end is not None and "src" in rec:
                    natural = rec["src"][0] - prev_src_end
                    if 0 < natural <= MAX_NATURAL_GAP and t - prev_end_t < natural:
                        t = prev_end_t + natural
            else:
                mp3, words = await synth(clean(text))
                pcm = decode(mp3)
                shown = clean(text)
            dur = len(pcm) / SR
            start_sample = int(round(t * SR))
            track.append((start_sample, pcm))
            for name, ct in find_cues(text, words, strict=not USER).items():
                cues[name] = round(t - s0 + ct, 3)
            lines.append({"start": round(t - s0, 3), "end": round(t - s0 + dur, 3),
                          "text": shown,
                          "words": [[round(t - s0 + w[0], 3), round(w[1], 3), w[2]] for w in words]})
            if USER:
                prev_src_end, prev_end_t = rec["src"][1] if "src" in rec else None, t + dur
            t += dur + gap
        for m in MARK.finditer(" ".join(x[0] for x in scene["lines"])):
            name = m.group(1)
            if name not in cues:
                ref, off = FALLBACK.get(name, (None, 0.5))
                cues[name] = round(cues.get(ref, lines[-1]["end"]) + off, 3)
                print(f"  cue {name!r} not spoken; placed at {cues[name]:.2f}s")
        t += scene["tail"]
        timeline.append({"id": scene["id"], "start": round(s0, 3), "end": round(t, 3),
                         "dur": round(t - s0, 3), "lines": lines, "cues": cues})
        print(f"{scene['id']:<8} {s0:7.2f} -> {t:7.2f}  ({t - s0:5.2f} s)")

    total = t
    out = np.zeros(int(np.ceil(total * SR)) + SR, dtype=np.float32)
    for start, pcm in track:
        out[start:start + len(pcm)] += pcm
    import wave
    with wave.open(str(BUILD / "voice.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(out, -1, 1) * 32767).astype("<i2").tobytes())

    (BUILD / "timeline.json").write_text(json.dumps(
        {"total": round(total, 3), "fps": 30, "scenes": timeline}, indent=1))

    # Captions: original punctuated text, timed by aligned word boundaries, ~2 short lines each.
    srt, cues = [], []
    for scene in timeline:
        for line in scene["lines"]:
            words, tokens = line["words"], line["text"].split()
            times, wi = [], 0
            for tok in tokens:
                tn = norm(tok)
                for j in range(wi, len(words)):
                    wn = norm(words[j][2])
                    if wn and tn and (tn.startswith(wn) or wn.startswith(tn)):
                        times.append((words[j][0], words[j][0] + words[j][1]))
                        wi = j + 1
                        break
                else:
                    times.append(times[-1] if times else (words[0][0], words[0][0] + 0.3))
            chunk = []
            for i, tok in enumerate(tokens):
                chunk.append(i)
                txt = " ".join(tokens[k] for k in chunk)
                brk = tok[-1] in ",.:;?!" and len(txt) > 28
                if i == len(tokens) - 1 or brk or len(txt) > 64:
                    a = scene["start"] + times[chunk[0]][0]
                    b = scene["start"] + times[chunk[-1]][1] + 0.25
                    cues.append([a, b, txt])
                    chunk = []
    for k in range(len(cues) - 1):
        cues[k][1] = min(cues[k][1], cues[k + 1][0] - 0.05)
    for n, (a, b, txt) in enumerate(cues, 1):
        srt.append(f"{n}\n{srt_time(a)} --> {srt_time(b)}\n{txt}\n")
    (BUILD / "captions.srt").write_text("\n".join(srt), encoding="utf-8")
    print(f"TOTAL {total:.2f} s = {int(total // 60)}:{total % 60:05.2f}")


if __name__ == "__main__":
    asyncio.run(main())
