"""Audio analysis with librosa - pre-compute all features at ~60fps resolution.

EDM-optimized: sub-band energy, beat classification, local normalization,
onset sharpness, and buildup/drop detection tuned for electronic music.
"""

from dataclasses import dataclass
import numpy as np
import librosa
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d


@dataclass
class AnalysisResult:
    """Pre-computed audio analysis arrays, all at frame resolution."""
    sr: int
    hop_length: int
    duration: float
    n_frames: int

    # Per-frame arrays (length = n_frames)
    rms: np.ndarray              # 0-1 normalized RMS energy
    onset_strength: np.ndarray   # 0-1 locally normalized onset envelope
    beat_pulse: np.ndarray       # 0-1, peaks at beat positions
    spectral_centroid: np.ndarray  # 0-1 normalized
    spectral_bandwidth: np.ndarray  # 0-1 locally normalized
    spectral_flux: np.ndarray    # 0-1 locally normalized

    # Sub-band energy
    bass_energy: np.ndarray      # 0-1, globally normalized (quiet=small)
    mid_energy: np.ndarray       # 0-1, locally normalized
    treble_energy: np.ndarray    # 0-1, locally normalized

    # Beat classification pulses
    kick_pulse: np.ndarray       # 0-1, triangular pulse at kick beats
    snare_pulse: np.ndarray      # 0-1, triangular pulse at snare beats
    hihat_pulse: np.ndarray      # 0-1, continuous treble transient envelope

    # Onset sharpness
    onset_sharpness: np.ndarray  # 0-1, high = percussive transient

    # Climax-related
    anticipation_factor: np.ndarray  # 0-1, ramps up before drop
    explosion_factor: np.ndarray     # 0-1, spikes at drop then decays

    def frame_at(self, time: float) -> int:
        """Get frame index for a given time in seconds."""
        frame = int(time * self.sr / self.hop_length)
        return max(0, min(frame, self.n_frames - 1))

    def get_features(self, time: float) -> dict:
        """Get all features at a given time."""
        f = self.frame_at(time)
        return {
            "rms": float(self.rms[f]),
            "onset_strength": float(self.onset_strength[f]),
            "beat_pulse": float(self.beat_pulse[f]),
            "spectral_centroid": float(self.spectral_centroid[f]),
            "spectral_bandwidth": float(self.spectral_bandwidth[f]),
            "spectral_flux": float(self.spectral_flux[f]),
            "bass_energy": float(self.bass_energy[f]),
            "mid_energy": float(self.mid_energy[f]),
            "treble_energy": float(self.treble_energy[f]),
            "kick_pulse": float(self.kick_pulse[f]),
            "snare_pulse": float(self.snare_pulse[f]),
            "hihat_pulse": float(self.hihat_pulse[f]),
            "onset_sharpness": float(self.onset_sharpness[f]),
            "anticipation_factor": float(self.anticipation_factor[f]),
            "explosion_factor": float(self.explosion_factor[f]),
        }


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Normalize array to 0-1 range (global)."""
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-8:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def _local_normalize(arr: np.ndarray, window_s: float, fps: float) -> np.ndarray:
    """Locally normalize array so quiet sections still show detail.

    Uses a sliding window to find local min/max, then rescales each frame
    relative to its local range. Subtle variations become visible even in
    quiet passages.
    """
    window = max(3, int(window_s * fps))
    smooth_w = max(1, window // 4)
    local_max = uniform_filter1d(maximum_filter1d(arr, window), smooth_w)
    local_min = uniform_filter1d(minimum_filter1d(arr, window), smooth_w)
    denom = local_max - local_min
    denom = np.maximum(denom, 1e-8)
    return np.clip((arr - local_min) / denom, 0.0, 1.0)


def _fit_length(arr: np.ndarray, n_frames: int) -> np.ndarray:
    """Trim or pad array to exactly n_frames."""
    if len(arr) >= n_frames:
        return arr[:n_frames]
    return np.pad(arr, (0, n_frames - len(arr)))


def analyze(audio_path: str, progress_callback=None) -> AnalysisResult:
    """Analyze audio file and return pre-computed features.

    Args:
        audio_path: Path to WAV file
        progress_callback: Optional callable(float) with progress 0.0-1.0
    """
    def report(p):
        if progress_callback:
            progress_callback(p)

    report(0.0)

    # Load audio
    y, sr = librosa.load(audio_path, sr=44100, mono=True)
    duration = len(y) / sr
    hop_length = 735  # ~60fps at 44100 sr
    fps = sr / hop_length

    report(0.05)

    # --- 1A. Mel spectrogram (compute once, reuse for everything) ---
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, hop_length=hop_length, n_mels=128, fmax=16000
    )
    n_frames = S.shape[1]

    report(0.15)

    # --- RMS energy (from mel spectrogram for consistency) ---
    rms_raw = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_raw = _fit_length(rms_raw, n_frames)
    rms = _normalize(rms_raw)

    report(0.2)

    # --- 1B. Sub-band energy ---
    bass_raw = np.mean(S[:15, :], axis=0)       # bins 0-14, ~20-250Hz
    mid_raw = np.mean(S[15:75, :], axis=0)      # bins 15-74, ~250-4kHz
    treble_raw = np.mean(S[75:, :], axis=0)     # bins 75+, ~4kHz+

    # Bass: globally normalized (quiet sections = small ball)
    bass_energy = _normalize(bass_raw)
    # Mid/treble: locally normalized (subtle variations visible in quiet sections)
    mid_energy = _local_normalize(mid_raw, 10.0, fps)
    treble_energy = _local_normalize(treble_raw, 10.0, fps)

    report(0.3)

    # --- Onset strength ---
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset_env = _fit_length(onset_env, n_frames)
    onset_strength = _local_normalize(onset_env, 12.0, fps)

    report(0.35)

    # --- 1F. Onset sharpness ---
    onset_smooth = uniform_filter1d(onset_env, max(1, int(fps * 0.2)))
    sharpness = (onset_env / np.maximum(onset_smooth, 1e-8)) - 1.0
    sharpness = np.clip(sharpness, 0, None)
    onset_sharpness = _normalize(sharpness)

    report(0.4)

    # --- Beat tracking ---
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    beat_pulse = np.zeros(n_frames)
    for bf in beat_frames:
        if bf < n_frames:
            width = 3
            for i in range(max(0, bf - width), min(n_frames, bf + width + 1)):
                dist = abs(i - bf) / width
                beat_pulse[i] = max(beat_pulse[i], 1.0 - dist)

    report(0.5)

    # --- 1D. Beat classification (kick vs snare) + hihat ---
    # Sub-band spectral flux for onset detection per band
    bass_flux = np.sum(np.maximum(0, np.diff(S[:15, :], axis=1)), axis=0)
    bass_flux = np.concatenate([[0.0], bass_flux])
    bass_flux = _fit_length(bass_flux, n_frames)

    mid_flux = np.sum(np.maximum(0, np.diff(S[15:75, :], axis=1)), axis=0)
    mid_flux = np.concatenate([[0.0], mid_flux])
    mid_flux = _fit_length(mid_flux, n_frames)

    treble_flux = np.sum(np.maximum(0, np.diff(S[75:, :], axis=1)), axis=0)
    treble_flux = np.concatenate([[0.0], treble_flux])
    treble_flux = _fit_length(treble_flux, n_frames)

    # Classify each beat as kick or snare
    pulse_width = max(1, int(fps * 0.05))  # ~50ms triangular pulse
    kick_pulse = np.zeros(n_frames)
    snare_pulse = np.zeros(n_frames)

    for bf in beat_frames:
        if bf >= n_frames:
            continue
        if bass_flux[bf] >= mid_flux[bf]:
            # Kick: bass-dominant beat
            for i in range(max(0, bf - pulse_width), min(n_frames, bf + pulse_width + 1)):
                dist = abs(i - bf) / max(1, pulse_width)
                kick_pulse[i] = max(kick_pulse[i], 1.0 - dist)
        else:
            # Snare: mid-dominant beat
            for i in range(max(0, bf - pulse_width), min(n_frames, bf + pulse_width + 1)):
                dist = abs(i - bf) / max(1, pulse_width)
                snare_pulse[i] = max(snare_pulse[i], 1.0 - dist)

    # Hihat: continuous treble transient envelope (not beat-locked)
    hihat_pulse = _local_normalize(treble_flux, 10.0, fps)
    # Peak enhancement: square to emphasize transients, then re-normalize
    hihat_pulse = _normalize(hihat_pulse ** 2)

    report(0.6)

    # --- Spectral centroid ---
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    cent = _fit_length(cent, n_frames)
    spectral_centroid = _normalize(cent)

    report(0.65)

    # --- Spectral bandwidth ---
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]
    bw = _fit_length(bw, n_frames)
    spectral_bandwidth = _local_normalize(bw, 10.0, fps)

    report(0.7)

    # --- Spectral flux (full-spectrum, locally normalized) ---
    full_flux = np.sqrt(
        np.sum(np.maximum(0, np.diff(S, axis=1)) ** 2, axis=0)
    )
    full_flux = np.concatenate([[0.0], full_flux])
    full_flux = _fit_length(full_flux, n_frames)
    spectral_flux = _local_normalize(full_flux, 10.0, fps)

    report(0.8)

    # --- 1E. EDM buildup/drop detection ---
    anticipation_factor, explosion_factor = _detect_climaxes_edm(
        bass_raw, rms_raw, onset_env, cent, n_frames, fps
    )

    report(1.0)

    return AnalysisResult(
        sr=sr,
        hop_length=hop_length,
        duration=duration,
        n_frames=n_frames,
        rms=rms.astype(np.float32),
        onset_strength=onset_strength.astype(np.float32),
        beat_pulse=beat_pulse.astype(np.float32),
        spectral_centroid=spectral_centroid.astype(np.float32),
        spectral_bandwidth=spectral_bandwidth.astype(np.float32),
        spectral_flux=spectral_flux.astype(np.float32),
        bass_energy=bass_energy.astype(np.float32),
        mid_energy=mid_energy.astype(np.float32),
        treble_energy=treble_energy.astype(np.float32),
        kick_pulse=kick_pulse.astype(np.float32),
        snare_pulse=snare_pulse.astype(np.float32),
        hihat_pulse=hihat_pulse.astype(np.float32),
        onset_sharpness=onset_sharpness.astype(np.float32),
        anticipation_factor=anticipation_factor.astype(np.float32),
        explosion_factor=explosion_factor.astype(np.float32),
    )


def _detect_climaxes_edm(
    bass_raw: np.ndarray,
    rms_raw: np.ndarray,
    onset_env: np.ndarray,
    centroid_raw: np.ndarray,
    n_frames: int,
    fps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """EDM-aware buildup/drop detection.

    Buildup score (per-frame composite):
      - Rising spectral centroid (smoothed derivative, clipped positive) x 0.4
      - Rising onset density (onsets/sec increasing) x 0.35
      - Bass cutting out (inverse of bass level) x 0.25

    Drop detection:
      - Frame-to-frame bass energy jump (clipped positive) x 0.6
      - Frame-to-frame RMS energy jump x 0.4
      - Gated by preceding buildup (only count drops after a buildup period)
      - Cluster peaks with 6s minimum gap

    Output curves:
      - anticipation_factor: smoothstep ramp t²(3-2t) over 4s before each drop
      - explosion_factor: exponential decay exp(-4t) over 2s after each drop
    """
    smooth_w = max(3, int(1.0 * fps))  # 1s smoothing

    # --- Buildup score ---
    # 1. Rising spectral centroid
    cent_smooth = uniform_filter1d(centroid_raw, smooth_w)
    cent_deriv = np.gradient(cent_smooth)
    cent_rising = np.clip(cent_deriv, 0, None)
    cent_rising = uniform_filter1d(cent_rising, max(3, int(2.0 * fps)))  # 2s smooth
    if cent_rising.max() > 1e-8:
        cent_rising = cent_rising / cent_rising.max()

    # 2. Rising onset density (rolling count of onsets per second)
    onset_thresh = np.percentile(onset_env, 70)
    onset_binary = (onset_env > onset_thresh).astype(np.float64)
    density_w = max(3, int(2.0 * fps))
    onset_density = uniform_filter1d(onset_binary, density_w)
    density_deriv = np.gradient(uniform_filter1d(onset_density, smooth_w))
    density_rising = np.clip(density_deriv, 0, None)
    if density_rising.max() > 1e-8:
        density_rising = density_rising / density_rising.max()

    # 3. Bass cutting out (inverse bass, high when bass is absent)
    bass_smooth = uniform_filter1d(bass_raw, smooth_w)
    if bass_smooth.max() > 1e-8:
        bass_norm = bass_smooth / bass_smooth.max()
    else:
        bass_norm = np.zeros_like(bass_smooth)
    bass_absent = 1.0 - bass_norm

    # Composite buildup score
    buildup_score = (
        cent_rising * 0.4
        + density_rising * 0.35
        + bass_absent * 0.25
    )
    # Smooth the composite
    buildup_score = uniform_filter1d(buildup_score, max(3, int(2.0 * fps)))

    # --- Drop detection ---
    # Frame-to-frame bass jump
    bass_jump = np.concatenate([[0.0], np.diff(bass_smooth)])
    bass_jump = np.clip(bass_jump, 0, None)
    if bass_jump.max() > 1e-8:
        bass_jump = bass_jump / bass_jump.max()

    # Frame-to-frame RMS jump
    rms_smooth = uniform_filter1d(rms_raw, smooth_w)
    rms_jump = np.concatenate([[0.0], np.diff(rms_smooth)])
    rms_jump = np.clip(rms_jump, 0, None)
    if rms_jump.max() > 1e-8:
        rms_jump = rms_jump / rms_jump.max()

    # Composite drop impulse
    drop_impulse = bass_jump * 0.6 + rms_jump * 0.4

    # Gate by preceding buildup: require buildup_score above threshold
    # in the 2-6 seconds before
    lookback = max(3, int(4.0 * fps))
    buildup_max = maximum_filter1d(buildup_score, lookback)
    # Shift forward so we look at the period *before* each frame
    buildup_preceding = np.roll(buildup_max, lookback // 2)
    buildup_preceding[:lookback] = 0  # no valid lookback at start

    buildup_thresh = np.percentile(buildup_score, 75) if buildup_score.max() > 1e-8 else 0
    gated_drops = drop_impulse * (buildup_preceding > buildup_thresh).astype(np.float64)

    # Find drop peaks with 6s minimum gap
    min_gap = max(1, int(6.0 * fps))
    drop_thresh = np.percentile(gated_drops[gated_drops > 0], 80) if np.any(gated_drops > 0) else 0
    drop_frames = []
    i = 0
    while i < n_frames:
        if gated_drops[i] > drop_thresh:
            cluster_end = min(i + min_gap, n_frames)
            cluster = gated_drops[i:cluster_end]
            peak_offset = np.argmax(cluster)
            drop_frames.append(i + peak_offset)
            i = cluster_end
        else:
            i += 1

    # --- Build anticipation and explosion curves ---
    anticipation = np.zeros(n_frames, dtype=np.float64)
    explosion = np.zeros(n_frames, dtype=np.float64)

    buildup_duration = max(1, int(4.0 * fps))   # 4s ramp before drop
    release_duration = max(1, int(2.0 * fps))    # 2s decay after drop

    for df in drop_frames:
        # Anticipation: smoothstep ramp t²(3-2t) over buildup_duration before drop
        start = max(0, df - buildup_duration)
        for f in range(start, min(df, n_frames)):
            t = (f - start) / max(1, df - start)
            smoothstep = t * t * (3.0 - 2.0 * t)
            anticipation[f] = max(anticipation[f], smoothstep)

        # Explosion: exponential decay exp(-4t) over release_duration after drop
        for f in range(df, min(df + release_duration, n_frames)):
            t = (f - df) / max(1, release_duration)
            anticipation[f] = 0.0  # clear anticipation at/after drop
            explosion[f] = max(explosion[f], np.exp(-4.0 * t))

    return anticipation, explosion
