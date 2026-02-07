# MusicVis

Real-time music visualizer that renders an electric energy ball reacting to audio. Search for any song on YouTube, and watch it come alive as a glowing tangle of GPU-generated 3D light trails with multi-scale bloom, anamorphic lens flares, and perspective depth.

![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![OpenGL 4.3](https://img.shields.io/badge/OpenGL-4.3-green)

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

Downloaded songs and analysis results are cached locally — previously played songs load instantly from the library without re-downloading or re-analyzing.

The ball is made of 5–25 persistent closed-loop splines orbiting in 3D, generated entirely on the GPU via compute shaders. Each loop is Catmull-Rom subdivided with 240 segments per loop, with perspective projection giving real depth cues — far segments appear thinner and dimmer, close segments glow brighter. Loops are differentiated by frequency band (bass/mid/treble) with distinct colors driven by spectral centroid. When the music is quiet the loops hug a tight sphere; when it gets loud they scatter erratically through the volume interior.

## Rendering Pipeline

```
Compute shader (spline generation, 240 threads/loop)
    → Instanced line segments with per-segment thickness/brightness/hue/depth
        → HDR framebuffer (MRT: scene + bright pass)
            → Multi-scale bloom (3 scales: 1/2, 1/4, 1/8 res)
                → Anamorphic lens flare (horizontal stretch)
                    → Composite + tonemapping → screen
```

All rendering is additive-blended with instanced line drawing.

## Requirements

- Python 3.12+
- GPU with OpenGL 4.3 support (compute shaders)

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
| Click a result | Starts download + analysis (or plays from cache) |
| Library panel | Browse/play/delete/re-analyze downloaded songs |
| Play/Pause | Bottom bar button |
| Seek | Drag the slider |
| Settings | Right-side panel |
| F11 / F | Toggle fullscreen |
| Escape | Exit fullscreen |
| Back | Return to search |

## Settings

**Presets** — 5 built-in presets (Default, EDM Rave, Chill Lo-fi, Minimal, Cinematic) + save your own as JSON

**Rendering** — brightness, bloom intensity, bloom tint color, anamorphic intensity

**Shape** — loop count (5–25), noise multiplier, rotation speed, energy multiplier

**Effects** — depth fog, waveform ring toggle + opacity

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
    library.py                   Song library, metadata JSON, analysis NPZ cache
  ui/
    search_panel.py              Search bar + results list
    library_panel.py             Library browser with play/delete/re-analyze
    player_controls.py           Play/pause, seek, time display
    settings_panel.py            Tweakable visualization parameters
    preset_manager.py            Built-in + user preset management
  visualization/
    energy_ball.py               GPU compute spline generation + 3D loop tangle
    renderer.py                  OpenGL HDR + multi-scale bloom + anamorphic pipeline
    shaders/
      spline.comp                Compute shader for spline generation
      line.vert / line.frag      Instanced line rendering
      blur.vert / blur.frag      Gaussian blur passes
      composite.frag             HDR composite + tonemapping
      trail.frag                 Trail effect
assets/
  downloads/                     Cached WAV + analysis NPZ files
  presets/                       User-saved preset JSON files
```

## Audio Features

The analyzer extracts 22 features per frame:

| Feature | Description |
|---------|-------------|
| `rms` | Overall loudness |
| `bass/mid/treble_energy` | Spectral band energies |
| `kick/snare/hihat_pulse` | Drum hit classification |
| `spectral_centroid` | Brightness of the sound |
| `spectral_bandwidth` | Spread of frequencies |
| `spectral_flux` | Rate of spectral change |
| `onset_strength` | Transient detection |
| `onset_sharpness` | Percussive transient sharpness |
| `anticipation_factor` | Ramp before a drop (EDM buildups) |
| `explosion_factor` | Decay after a drop |
| `chroma` | 12-bin pitch class distribution |
| `key_index` | Detected musical key (0–11) |
| `section_labels` | Structural segmentation |
| `vocal_presence` | Vocal activity detection |
| `groove_factor` | Rhythmic swing |
| `tempo` | BPM |
| `mel_spectrum` | 64-bin mel spectrogram |

## License

MIT
