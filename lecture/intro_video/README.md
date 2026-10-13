# Lecture intro video: Low-Carbon & Flexible Data Centres

A ~4-minute narrated, animated introduction to play at the start of the lecture.
Everything is generated from code in `src/`, so wording, timing and visuals can be changed and re-rendered.

## Output

- `output/low_carbon_flexible_data_centres_intro.mp4`: 1920×1080, 30 fps, H.264 + AAC, loudness-normalised
- `output/low_carbon_flexible_data_centres_intro.srt`: optional captions (load alongside the video in VLC/PowerPoint)

## Pipeline

| Step | Script | Output |
|---|---|---|
| Narration script | `src/narration.py` | lines per scene with `[[cue]]` markers |
| Voice + timeline | `src/make_voice.py` | `build/voice.wav`, `build/timeline.json`, `build/captions.srt` |
| Price data | `src/prep_prices.py` | `assets/prices_2025.npy` (from `paper/data/price_timeline.csv`) |
| Score + SFX + mix | `src/make_audio.py` | `build/mix.wav` |
| Frames | `src/render.py video --workers 8` | `build/video_noaudio.mp4` |
| Mux | ffmpeg (below) | `output/*.mp4` |

The voice is Microsoft's `en-GB-RyanNeural` via `edge-tts` (needs internet). Other British options are
`en-GB-ThomasNeural`, `en-GB-SoniaNeural` and `en-GB-LibbyNeural`; change `VOICE` in `narration.py`.
Scenes key their animation to the moment each `[[cue]]` word is spoken, so editing the script
re-times the visuals automatically.

```bash
cd src
python make_voice.py && python make_audio.py
python render.py sheet britain 9 ../build/sheet.png     # preview a scene as a contact sheet
python render.py video --workers 8                       # ~3-5 min on an 8-core CPU
ffmpeg -i ../build/video_noaudio.mp4 -i ../build/mix.wav -map 0:v -map 1:a -c:v copy -c:a aac -b:a 256k \
  -af loudnorm=I=-16:TP=-1.5:LRA=11 -movflags +faststart -shortest ../output/low_carbon_flexible_data_centres_intro.mp4
```

## Sources shown on screen

- IEA, *Key Questions on Energy and AI* (2026): ~485 TWh (2025) → ~950 TWh (2030); AI-focused ~465 TWh by 2030
- Uptime Institute global survey (average PUE ~1.5); hyperscale operator disclosures (~1.1)
- NESO FES 2025 (~2.4 GW / 7.6 TWh connected in GB); NESO evidence to Parliament and Ofgem/NESO 2026 (50+ GW of data-centre connection requests)
- GLA (2022): West London housing connection delays; GOV.UK AI Growth Zones (2025–26)
- LCCC Intermittent Market Reference Price (GB day-ahead), 2025: 672 negative-price quarter-hours
- Meta Odense district heating; OPDC Old Oak & Park Royal heat network
- National Grid / Emerald AI / Nebius London trial (Dec 2025); Google demand-response agreements (~1 GW), via DatacenterDynamics (2026)

Map geometry: Natural Earth 1:10m (public domain), simplified. Wind farms, network routes and data-centre clusters are schematic.
