# MusicVis

Real-time music visualizer that renders an electric energy ball reacting to audio. Search for any song on YouTube, and watch it come alive as a glowing tangle of 3D light trails with bloom, HDR, and perspective depth.

![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![OpenGL 3.3](https://img.shields.io/badge/OpenGL-3.3-green)

<p align="center">
  <a href="https://www.youtube.com/watch?v=UB6VbAJCecY">
    <img src="assets/demo.gif" alt="MusicVis Demo" width="480">
  </a>
  <br>
  <a href="https://www.youtube.com/watch?v=UB6VbAJCecY">Click to watch the full demo with audio on YouTube</a>
</p>

## How It Works

1. **Search** for a song on YouTube directly from the app
2. **Download** — audio is extracted and converted to WAV automatically
3. **Analyze** — the entire track is pre-analyzed at ~60fps for beat detection, spectral features, and EDM-specific patterns (kick/snare/hihat classification, buildups, drops)
4. **Play** — the energy ball visualization reacts in real time to the pre-computed features

The ball is made of 15–25 persistent closed-loop splines orbiting in 3D. Each loop is Catmull-Rom subdivided in 3D space before perspective projection, giving real depth cues — far segments appear thinner and dimmer, close segments glow brighter. When the music is quiet the loops hug a tight sphere; when it gets loud they scatter erratically through the volume interior. Sharp direction changes glow brighter, mimicking long-exposure light photography.

## Rendering Pipeline

```
Loops (CPU) → Line segments with per-segment thickness/brightness
    → HDR framebuffer (MRT: scene + bright pass)
        → Gaussian bloom (ping-pong blur at half res)
            → Composite + tonemapping → screen
```

All rendering is additive-blended with instanced line drawing (up to 50k segments/frame).

## Requirements

- Python 3.12+
- GPU with OpenGL 3.3 support

## Setup

```bash
git clone https://github.com/RomanSlack/MusicVis.git
cd MusicVis
pip install -r requirements.txt
python main.py
```

### Linux

Install system dependencies for audio and OpenGL:

```bash
# Ubuntu/Debian
sudo apt install libasound2-dev portaudio19-dev libgl1-mesa-dev libglu1-mesa-dev

# Fedora
sudo dnf install alsa-lib-devel portaudio-devel mesa-libGL-devel mesa-libGLU-devel
```

If you're using conda and audio devices aren't showing up, the app automatically sets `ALSA_PLUGIN_DIR` to find PipeWire/PulseAudio virtual devices. If you still have issues, make sure PipeWire or PulseAudio is running.

### Windows

No extra system dependencies needed — just `pip install -r requirements.txt` and go.

### macOS

```bash
brew install portaudio
pip install -r requirements.txt
python main.py
```

## Controls

| Control | Action |
|---------|--------|
| Search bar | Type a song name, press Enter |
| Click a result | Starts download + analysis |
| Play/Pause | Bottom bar button |
| Seek | Drag the slider |
| Settings | Right-side panel (bloom, brightness, loop count, energy, etc.) |
| Back | Return to search |

## Settings

**Rendering** — brightness, bloom intensity, bloom tint color

**Shape** — loop count (5–25), noise multiplier, rotation speed, energy multiplier

## Project Structure

```
main.py                          Entry point
src/
  app.py                         GLFW window, state machine, main loop
  audio/
    search.py                    YouTube search via yt-dlp
    downloader.py                Audio download + WAV conversion
    analyzer.py                  Pre-computed feature extraction (librosa)
    player.py                    Playback with sounddevice
    cookie_helper.py             Chrome CDP cookie export
  ui/
    search_panel.py              Search bar + results list
    player_controls.py           Play/pause, seek, time display
    settings_panel.py            Tweakable visualization parameters
  visualization/
    energy_ball.py               3D loop tangle geometry generator
    renderer.py                  OpenGL HDR + bloom pipeline
    shaders/                     GLSL vertex/fragment shaders
assets/
  downloads/                     Cached audio files (gitignored)
```

## Audio Features

The analyzer extracts 15 features per frame:

| Feature | Description |
|---------|-------------|
| `rms` | Overall loudness |
| `bass/mid/treble_energy` | Spectral band energies |
| `kick/snare/hihat_pulse` | Drum hit classification |
| `spectral_centroid` | Brightness of the sound |
| `spectral_bandwidth` | Spread of frequencies |
| `spectral_flux` | Rate of spectral change |
| `onset_strength` | Transient detection |
| `anticipation_factor` | Ramp before a drop (EDM buildups) |
| `explosion_factor` | Decay after a drop |

## License

MIT
