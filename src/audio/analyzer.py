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

    # A2: Chroma / Key
    chroma: np.ndarray              # (12, n_frames) float32 — pitch class energy
    key_index: int                  # 0-11 (C, C#, D, ... B)

    # A3: Section segmentation
    section_labels: np.ndarray      # (n_frames,) int32 — section ID per frame
    n_sections: int

    # A4: Vocal detection
    vocal_presence: np.ndarray      # (n_frames,) float32 — 0=instrumental, 1=vocal

    # A5: Groove / swing
    groove_factor: float            # scalar 0-1 (0=robotic grid, 1=heavy swing)

    # A6: Tempo
    tempo: float                    # BPM

    # B7: Mel spectrum for waveform ring
    mel_spectrum: np.ndarray        # (64, n_frames) float32 — 64-bin mel spectrogram

    # --- Musical Director features ---

    # Genre classification
    genre_id: int                   # 0=EDM,1=Rock,2=Jazz,3=Classical,4=HipHop,5=Ambient,6=Pop
    genre_confidence: float         # 0-1
    genre_features: np.ndarray      # (7,) float32 — probability per genre

    # Section typing
    section_type: np.ndarray        # (n_frames,) int32 — 0=intro..8=solo
    section_boundaries: np.ndarray  # (N,) int32 — frame indices of section changes
    n_section_boundaries: int

    # Energy trajectory
    energy_trajectory: np.ndarray   # (n_frames,) float32 — -1..+1

    # Valence / Arousal
    valence: np.ndarray             # (n_frames,) float32 — 0=dark, 1=bright
    arousal: np.ndarray             # (n_frames,) float32 — 0=calm, 1=excited

    # Generalized climax detection
    climax_score: np.ndarray        # (n_frames,) float32 — 0-1
    climax_type: np.ndarray         # (n_frames,) int8 — 0=none..5=breakdown_return

    # Look-ahead (pre-computed future awareness)
    lookahead_energy_delta: np.ndarray   # (n_frames,) float32 — -1..+1
    lookahead_section_change: np.ndarray # (n_frames,) float32 — 0-1
    lookahead_climax: np.ndarray         # (n_frames,) float32 — 0-1

    # Rhythmic density
    rhythmic_density: np.ndarray    # (n_frames,) float32 — 0-1

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
            "vocal_presence": float(self.vocal_presence[f]),
            "groove_factor": self.groove_factor,
            "section_id": int(self.section_labels[f]),
            "tempo": self.tempo,
            # Musical Director features
            "genre_id": self.genre_id,
            "genre_confidence": self.genre_confidence,
            "section_type": int(self.section_type[f]),
            "energy_trajectory": float(self.energy_trajectory[f]),
            "valence": float(self.valence[f]),
            "arousal": float(self.arousal[f]),
            "climax_score": float(self.climax_score[f]),
            "climax_type": int(self.climax_type[f]),
            "lookahead_energy_delta": float(self.lookahead_energy_delta[f]),
            "lookahead_section_change": float(self.lookahead_section_change[f]),
            "lookahead_climax": float(self.lookahead_climax[f]),
            "rhythmic_density": float(self.rhythmic_density[f]),
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


def _fit_2d(arr: np.ndarray, n_frames: int) -> np.ndarray:
    """Trim or pad 2D array (n_bins, time) to exactly n_frames columns."""
    if arr.shape[1] >= n_frames:
        return arr[:, :n_frames]
    pad_width = n_frames - arr.shape[1]
    return np.pad(arr, ((0, 0), (0, pad_width)))


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

    report(0.03)

    # --- A1. HPSS ---
    y_harmonic, y_percussive = librosa.effects.hpss(y, margin=2.0)

    report(0.08)

    # --- 1A. Mel spectrogram (compute once, reuse for everything) ---
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, hop_length=hop_length, n_mels=128, fmax=16000
    )
    n_frames = S.shape[1]

    report(0.13)

    # --- RMS energy (from mel spectrogram for consistency) ---
    rms_raw = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_raw = _fit_length(rms_raw, n_frames)
    rms = _normalize(rms_raw)

    report(0.16)

    # --- A6. Tempo + beat tracking (needed for tempo-synced windows) ---
    tempo_val, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    if hasattr(tempo_val, '__len__'):
        tempo_val = float(tempo_val[0]) if len(tempo_val) > 0 else 120.0
    else:
        tempo_val = float(tempo_val)
    if tempo_val < 30.0:
        tempo_val = 120.0

    # Tempo-synced smoothing window (8 beats)
    beat_dur = 60.0 / tempo_val
    window_s = 8.0 * beat_dur

    report(0.20)

    # --- 1B. Sub-band energy ---
    bass_raw = np.mean(S[:15, :], axis=0)       # bins 0-14, ~20-250Hz
    mid_raw = np.mean(S[15:75, :], axis=0)      # bins 15-74, ~250-4kHz
    treble_raw = np.mean(S[75:, :], axis=0)     # bins 75+, ~4kHz+

    # Bass: globally normalized (quiet sections = small ball)
    bass_energy = _normalize(bass_raw)
    # Mid/treble: locally normalized with tempo-synced window
    mid_energy = _local_normalize(mid_raw, window_s, fps)
    treble_energy = _local_normalize(treble_raw, window_s, fps)

    report(0.25)

    # --- Onset strength ---
    onset_env = librosa.onset.onset_strength(y=y_percussive, sr=sr, hop_length=hop_length)
    onset_env = _fit_length(onset_env, n_frames)
    onset_strength = _local_normalize(onset_env, window_s * 1.5, fps)

    report(0.28)

    # --- 1F. Onset sharpness ---
    onset_smooth = uniform_filter1d(onset_env, max(1, int(fps * 0.2)))
    sharpness = (onset_env / np.maximum(onset_smooth, 1e-8)) - 1.0
    sharpness = np.clip(sharpness, 0, None)
    onset_sharpness = _normalize(sharpness)

    report(0.31)

    # --- Beat pulse ---
    beat_pulse = np.zeros(n_frames)
    for bf in beat_frames:
        if bf < n_frames:
            width = 3
            for i in range(max(0, bf - width), min(n_frames, bf + width + 1)):
                dist = abs(i - bf) / width
                beat_pulse[i] = max(beat_pulse[i], 1.0 - dist)

    report(0.35)

    # --- 1D. Beat classification (kick vs snare) + hihat ---
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
            for i in range(max(0, bf - pulse_width), min(n_frames, bf + pulse_width + 1)):
                dist = abs(i - bf) / max(1, pulse_width)
                kick_pulse[i] = max(kick_pulse[i], 1.0 - dist)
        else:
            for i in range(max(0, bf - pulse_width), min(n_frames, bf + pulse_width + 1)):
                dist = abs(i - bf) / max(1, pulse_width)
                snare_pulse[i] = max(snare_pulse[i], 1.0 - dist)

    # Hihat: continuous treble transient envelope
    hihat_pulse = _local_normalize(treble_flux, window_s, fps)
    hihat_pulse = _normalize(hihat_pulse ** 2)

    report(0.40)

    # --- Spectral centroid (from harmonic component) ---
    cent = librosa.feature.spectral_centroid(y=y_harmonic, sr=sr, hop_length=hop_length)[0]
    cent = _fit_length(cent, n_frames)
    spectral_centroid = _normalize(cent)

    report(0.43)

    # --- Spectral bandwidth ---
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]
    bw = _fit_length(bw, n_frames)
    spectral_bandwidth = _local_normalize(bw, window_s, fps)

    report(0.46)

    # --- Spectral flux (full-spectrum, locally normalized) ---
    full_flux = np.sqrt(
        np.sum(np.maximum(0, np.diff(S, axis=1)) ** 2, axis=0)
    )
    full_flux = np.concatenate([[0.0], full_flux])
    full_flux = _fit_length(full_flux, n_frames)
    spectral_flux = _local_normalize(full_flux, window_s, fps)

    report(0.50)

    # --- A2. Chroma / Key (from harmonic component) ---
    chroma_cqt = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, hop_length=hop_length)
    chroma_cqt = _fit_2d(chroma_cqt, n_frames)
    key_index = int(np.argmax(np.mean(chroma_cqt, axis=1)))

    report(0.55)

    # --- A3. Section segmentation ---
    try:
        from sklearn.cluster import SpectralClustering
        # Build recurrence matrix from chroma
        R = librosa.segment.recurrence_matrix(
            chroma_cqt, metric='cosine', mode='affinity', width=9, self=True
        )
        n_clusters = max(2, min(8, int(duration / 20)))
        labels = SpectralClustering(
            n_clusters=n_clusters, affinity='precomputed', random_state=42
        ).fit_predict(R)
        # labels has one entry per spectrogram frame — expand to n_frames if needed
        if len(labels) < n_frames:
            section_labels = np.zeros(n_frames, dtype=np.int32)
            section_labels[:len(labels)] = labels
            section_labels[len(labels):] = labels[-1] if len(labels) > 0 else 0
        else:
            section_labels = labels[:n_frames].astype(np.int32)
        n_sections = n_clusters
    except Exception:
        # Fallback: no segmentation
        section_labels = np.zeros(n_frames, dtype=np.int32)
        n_sections = 1

    report(0.65)

    # --- A4. Vocal detection ---
    vocal_band = S[15:80, :]  # ~300Hz-3kHz
    geo = np.exp(np.mean(np.log(vocal_band + 1e-8), axis=0))
    arith = np.mean(vocal_band, axis=0)
    flatness = geo / (arith + 1e-8)
    vocal_presence = 1.0 - _normalize(flatness)
    vocal_presence = _local_normalize(vocal_presence, window_s, fps)

    report(0.70)

    # --- A5. Groove / swing ---
    if len(beat_frames) > 3:
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
        mean_interval = np.mean(np.diff(beat_times))
        expected = beat_times[0] + mean_interval * np.arange(len(beat_times))
        groove_factor = float(np.clip(
            np.sqrt(np.mean((beat_times - expected[:len(beat_times)])**2)) / 0.05, 0, 1
        ))
    else:
        groove_factor = 0.0

    report(0.75)

    # --- B7. Mel spectrum storage (64-bin for waveform ring) ---
    mel_64 = librosa.feature.melspectrogram(
        y=y, sr=sr, hop_length=hop_length, n_mels=64, fmax=8000
    )
    mel_64 = _fit_2d(mel_64, n_frames)
    mel_max = mel_64.max(axis=1, keepdims=True) + 1e-8
    mel_spectrum = (mel_64 / mel_max).astype(np.float32)

    report(0.80)

    # --- 1E. EDM buildup/drop detection ---
    anticipation_factor, explosion_factor = _detect_climaxes_edm(
        bass_raw, rms_raw, onset_env, cent, n_frames, fps
    )

    report(0.82)

    # --- Genre classification ---
    genre_id, genre_confidence, genre_features = _classify_genre(
        tempo_val, bass_raw, rms_raw, onset_env, cent, chroma_cqt, n_frames, fps
    )

    report(0.84)

    # --- Section type labeling ---
    section_type, section_boundaries_arr, n_section_boundaries = _label_sections(
        section_labels, rms, spectral_centroid, onset_strength, bass_energy,
        vocal_presence, n_frames, fps, genre_id, duration
    )

    report(0.87)

    # --- Generalized climax detection ---
    climax_score, climax_type = _detect_climaxes_general(
        rms, onset_strength, spectral_centroid, bass_energy,
        vocal_presence, spectral_flux, section_type, n_frames, fps, genre_id
    )

    report(0.90)

    # --- Energy trajectory ---
    energy_trajectory = _compute_energy_trajectory(rms_raw, n_frames, fps)

    report(0.92)

    # --- Valence / Arousal ---
    valence_arr, arousal_arr = _compute_valence_arousal(
        chroma_cqt, cent, rms, spectral_flux, tempo_val, n_frames, fps
    )

    report(0.94)

    # --- Rhythmic density ---
    rhythmic_density = _compute_rhythmic_density(onset_env, tempo_val, n_frames, fps)

    report(0.96)

    # --- Look-ahead features ---
    lookahead_energy_delta, lookahead_section_change, lookahead_climax = _compute_lookahead(
        rms, section_boundaries_arr, climax_score, n_frames, fps
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
        chroma=chroma_cqt.astype(np.float32),
        key_index=key_index,
        section_labels=section_labels.astype(np.int32),
        n_sections=n_sections,
        vocal_presence=vocal_presence.astype(np.float32),
        groove_factor=groove_factor,
        tempo=tempo_val,
        mel_spectrum=mel_spectrum,
        # Musical Director features
        genre_id=genre_id,
        genre_confidence=genre_confidence,
        genre_features=genre_features.astype(np.float32),
        section_type=section_type.astype(np.int32),
        section_boundaries=section_boundaries_arr.astype(np.int32),
        n_section_boundaries=n_section_boundaries,
        energy_trajectory=energy_trajectory.astype(np.float32),
        valence=valence_arr.astype(np.float32),
        arousal=arousal_arr.astype(np.float32),
        climax_score=climax_score.astype(np.float32),
        climax_type=climax_type.astype(np.int8),
        lookahead_energy_delta=lookahead_energy_delta.astype(np.float32),
        lookahead_section_change=lookahead_section_change.astype(np.float32),
        lookahead_climax=lookahead_climax.astype(np.float32),
        rhythmic_density=rhythmic_density.astype(np.float32),
    )


def _classify_genre(
    tempo: float,
    bass_raw: np.ndarray,
    rms_raw: np.ndarray,
    onset_env: np.ndarray,
    centroid_raw: np.ndarray,
    chroma: np.ndarray,
    n_frames: int,
    fps: float,
) -> tuple[int, float, np.ndarray]:
    """Classify genre from aggregate audio statistics.

    Returns (genre_id, confidence, probabilities_7).
    Genre IDs: 0=EDM, 1=Rock, 2=Jazz, 3=Classical, 4=Hip-Hop, 5=Ambient, 6=Pop.
    """
    # All features computed as 0-1 normalized values
    total_energy = np.mean(bass_raw) + np.mean(rms_raw) + 1e-8
    bass_ratio = float(np.clip(np.mean(bass_raw) / total_energy, 0, 1))

    rms_p5, rms_p95 = float(np.percentile(rms_raw, 5)), float(np.percentile(rms_raw, 95))
    rms_max = float(rms_raw.max()) + 1e-8
    dynamic_range = float(np.clip((rms_p95 - rms_p5) / rms_max, 0, 1))

    onset_rate = float(np.mean(onset_env > np.percentile(onset_env, 70)))

    cent_max = float(centroid_raw.max()) + 1e-8
    centroid_mean_norm = float(np.clip(np.mean(centroid_raw) / cent_max, 0, 1))
    centroid_cv = float(np.clip(np.std(centroid_raw) / (np.mean(centroid_raw) + 1e-8), 0, 2) / 2.0)

    harmonic_stability = float(np.clip(np.mean(np.max(chroma, axis=0)), 0, 1))
    spectral_var = float(np.clip(np.mean(np.std(chroma, axis=1)) / 0.5, 0, 1))

    if n_frames > 100:
        half = n_frames // 2
        c1 = np.mean(chroma[:, :half], axis=1)
        c2 = np.mean(chroma[:, half:], axis=1)
        n1, n2 = np.linalg.norm(c1), np.linalg.norm(c2)
        chroma_rep = float(np.dot(c1, c2) / (n1 * n2 + 1e-8)) if n1 > 1e-8 and n2 > 1e-8 else 0.5
    else:
        chroma_rep = 0.5

    feat = np.array([
        np.clip(tempo / 200.0, 0, 1),  # tempo normalized
        bass_ratio,
        dynamic_range,
        onset_rate,
        centroid_mean_norm,
        centroid_cv,
        harmonic_stability,
        spectral_var,
        chroma_rep,
    ], dtype=np.float64)

    # Genre prototypes — all values in 0-1 range
    prototypes = np.array([
        # EDM: high tempo, strong bass proportion, moderate dynamics, repetitive
        [0.65, 0.65, 0.35, 0.50, 0.35, 0.25, 0.55, 0.30, 0.85],
        # Rock: medium tempo, balanced bass, wide dynamics, high onset
        [0.55, 0.45, 0.65, 0.55, 0.50, 0.50, 0.45, 0.50, 0.50],
        # Jazz: varied tempo, low bass proportion, high centroid variance
        [0.45, 0.35, 0.50, 0.35, 0.60, 0.70, 0.40, 0.65, 0.30],
        # Classical: lower tempo, low bass, wide dynamics, high harmonic stability
        [0.35, 0.25, 0.75, 0.20, 0.55, 0.55, 0.75, 0.55, 0.40],
        # Hip-Hop: medium tempo, heavy bass proportion, rhythmic
        [0.45, 0.70, 0.40, 0.50, 0.30, 0.25, 0.45, 0.25, 0.70],
        # Ambient: low tempo, low bass, low dynamics, low onset, high harmonic
        [0.25, 0.35, 0.20, 0.10, 0.45, 0.30, 0.70, 0.20, 0.65],
        # Pop: medium tempo, balanced, moderate onset, repetitive
        [0.55, 0.45, 0.45, 0.40, 0.50, 0.40, 0.55, 0.40, 0.75],
    ], dtype=np.float64)

    # Weighted Euclidean distance
    weights = np.array([1.5, 1.5, 1.2, 1.5, 1.0, 1.2, 1.0, 1.0, 1.2])
    distances = np.sqrt(np.sum(weights * (prototypes - feat) ** 2, axis=1))

    # Softmax of negative distances (lower distance = higher prob)
    neg_dist = -distances * 4.0  # temperature scaling
    exp_d = np.exp(neg_dist - np.max(neg_dist))
    probs = exp_d / (exp_d.sum() + 1e-8)

    genre_id = int(np.argmax(probs))
    genre_confidence = float(probs[genre_id])
    return genre_id, genre_confidence, probs.astype(np.float32)


def _label_sections(
    section_labels: np.ndarray,
    rms: np.ndarray,
    centroid: np.ndarray,
    onset: np.ndarray,
    bass: np.ndarray,
    vocal: np.ndarray,
    n_frames: int,
    fps: float,
    genre_id: int,
    duration: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Label section clusters into meaningful types.

    Section types: 0=intro, 1=verse, 2=chorus, 3=bridge, 4=breakdown,
                   5=buildup, 6=drop, 7=outro, 8=solo.
    Returns (section_type_per_frame, boundary_frames, n_boundaries).
    """
    # Find boundaries where cluster ID changes
    changes = np.where(np.diff(section_labels) != 0)[0] + 1
    boundaries = np.concatenate([[0], changes])
    n_boundaries = len(changes)

    # Compute per-section aggregate features
    section_type = np.zeros(n_frames, dtype=np.int32)
    n_sections = len(boundaries)

    for sec_idx in range(n_sections):
        start = boundaries[sec_idx]
        end = boundaries[sec_idx + 1] if sec_idx + 1 < n_sections else n_frames

        seg_rms = float(np.mean(rms[start:end]))
        seg_cent = float(np.mean(centroid[start:end]))
        seg_onset = float(np.mean(onset[start:end]))
        seg_bass = float(np.mean(bass[start:end]))
        seg_vocal = float(np.mean(vocal[start:end]))
        seg_pos = (start + end) / 2.0 / n_frames  # position in song (0-1)

        # Energy trajectory within section
        seg_len = end - start
        if seg_len > int(fps * 2):
            first_q = float(np.mean(rms[start:start + seg_len // 4]))
            last_q = float(np.mean(rms[end - seg_len // 4:end]))
            energy_rising = last_q - first_q
        else:
            energy_rising = 0.0

        # Rule-based labeling
        stype = 1  # default = verse

        if sec_idx == 0 and seg_pos < 0.15 and seg_rms < 0.4:
            stype = 0  # intro
        elif sec_idx == n_sections - 1 and seg_pos > 0.85 and seg_rms < 0.4:
            stype = 7  # outro
        elif genre_id == 0 and seg_bass > 0.6 and seg_rms > 0.6:
            stype = 6  # drop (EDM)
        elif seg_rms < 0.25 and seg_onset < 0.25:
            stype = 4  # breakdown
        elif energy_rising > 0.15 and seg_onset > 0.4:
            stype = 5  # buildup
        elif seg_rms > 0.55 and (seg_vocal > 0.5 or seg_cent > 0.5):
            stype = 2  # chorus
        elif seg_cent > 0.6 and seg_vocal < 0.3:
            stype = 8  # solo
        elif seg_rms < 0.45 and seg_onset < 0.35:
            # Check if this cluster ID appears rarely (bridge)
            cluster_id = section_labels[start]
            cluster_count = np.sum(section_labels == cluster_id)
            if cluster_count < n_frames * 0.15:
                stype = 3  # bridge

        section_type[start:end] = stype

    # Boundary array (frame indices where section changes)
    boundary_arr = changes.astype(np.int32) if len(changes) > 0 else np.zeros(1, dtype=np.int32)

    return section_type, boundary_arr, n_boundaries


def _detect_climaxes_general(
    rms: np.ndarray,
    onset: np.ndarray,
    centroid: np.ndarray,
    bass: np.ndarray,
    vocal: np.ndarray,
    flux: np.ndarray,
    section_type: np.ndarray,
    n_frames: int,
    fps: float,
    genre_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Multi-signal climax scoring with genre-specific weights.

    Returns (climax_score, climax_type) arrays.
    Climax types: 0=none, 1=drop, 2=chorus_peak, 3=crescendo, 4=solo_peak, 5=breakdown_return.
    """
    smooth_w = max(3, int(1.0 * fps))

    # Compute novelty signals (derivative + positive clip + smooth)
    def _novelty(arr):
        d = np.gradient(uniform_filter1d(arr, smooth_w))
        d = np.clip(d, 0, None)
        d = uniform_filter1d(d, max(3, int(0.5 * fps)))
        mx = d.max()
        return d / mx if mx > 1e-8 else d

    energy_nov = _novelty(rms)
    spectral_nov = _novelty(centroid)
    onset_nov = _novelty(onset)
    vocal_nov = _novelty(vocal)
    bass_nov = _novelty(bass)

    # Genre-specific weights: [energy, spectral, onset, vocal, bass]
    genre_weights = {
        0: [0.20, 0.10, 0.20, 0.15, 0.35],  # EDM: bass-heavy
        1: [0.30, 0.15, 0.25, 0.15, 0.15],  # Rock: balanced
        2: [0.20, 0.30, 0.15, 0.10, 0.25],  # Jazz: spectral
        3: [0.40, 0.25, 0.10, 0.10, 0.15],  # Classical: energy
        4: [0.20, 0.10, 0.25, 0.25, 0.20],  # Hip-Hop: vocal+onset
        5: [0.35, 0.30, 0.05, 0.15, 0.15],  # Ambient: energy+spectral
        6: [0.20, 0.15, 0.15, 0.30, 0.20],  # Pop: vocal-driven
    }
    w = genre_weights.get(genre_id, genre_weights[6])

    climax_score = (
        energy_nov * w[0]
        + spectral_nov * w[1]
        + onset_nov * w[2]
        + vocal_nov * w[3]
        + bass_nov * w[4]
    )

    # Normalize to 0-1
    mx = climax_score.max()
    if mx > 1e-8:
        climax_score = climax_score / mx

    # Smooth
    climax_score = uniform_filter1d(climax_score, max(3, int(0.5 * fps)))

    # Determine climax type based on section context
    climax_type = np.zeros(n_frames, dtype=np.int8)
    threshold = 0.3

    for i in range(n_frames):
        if climax_score[i] < threshold:
            continue
        st = section_type[i]
        if st == 6:  # drop
            climax_type[i] = 1
        elif st == 2:  # chorus
            climax_type[i] = 2
        elif st == 5:  # buildup
            climax_type[i] = 3  # crescendo
        elif st == 8:  # solo
            climax_type[i] = 4
        elif st == 4:  # breakdown — if score is high it's a return
            climax_type[i] = 5
        else:
            # Generic climax
            climax_type[i] = 2 if climax_score[i] > 0.6 else 0

    return climax_score, climax_type


def _compute_energy_trajectory(
    rms_raw: np.ndarray, n_frames: int, fps: float
) -> np.ndarray:
    """Smoothed RMS derivative normalized to -1..+1."""
    smooth_w = max(3, int(2.0 * fps))
    smoothed = uniform_filter1d(rms_raw, smooth_w)
    deriv = np.gradient(smoothed)
    # Normalize to -1..+1
    mx = max(abs(deriv.min()), abs(deriv.max()), 1e-8)
    trajectory = np.clip(deriv / mx, -1.0, 1.0)
    return _fit_length(trajectory, n_frames)


def _compute_valence_arousal(
    chroma: np.ndarray,
    centroid_raw: np.ndarray,
    rms: np.ndarray,
    flux: np.ndarray,
    tempo: float,
    n_frames: int,
    fps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute valence (major/minor brightness) and arousal (energy/excitement).

    Valence: correlation with major/minor templates + spectral warmth.
    Arousal: composite of tempo, rms, spectral flux.
    """
    # Major and minor templates (pitch class profiles)
    major = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], dtype=np.float64)
    minor = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0], dtype=np.float64)
    major = major / np.linalg.norm(major)
    minor = minor / np.linalg.norm(minor)

    valence = np.zeros(n_frames, dtype=np.float64)
    for f in range(n_frames):
        c = chroma[:, f].astype(np.float64)
        cn = np.linalg.norm(c)
        if cn < 1e-8:
            valence[f] = 0.5
            continue
        c = c / cn
        # Best correlation across all rotations (key detection)
        best_major = max(np.dot(np.roll(major, k), c) for k in range(12))
        best_minor = max(np.dot(np.roll(minor, k), c) for k in range(12))
        # Major = bright/positive, minor = dark
        valence[f] = 0.5 + (best_major - best_minor) * 0.5

    # Add spectral warmth component
    cent_norm = centroid_raw / (centroid_raw.max() + 1e-8)
    cent_norm = _fit_length(cent_norm, n_frames)
    valence = valence * 0.7 + cent_norm * 0.3
    valence = np.clip(valence, 0.0, 1.0)

    # Smooth
    smooth_w = max(3, int(2.0 * fps))
    valence = uniform_filter1d(valence, smooth_w)

    # Arousal: tempo + rms + flux composite
    tempo_component = np.clip((tempo - 60) / 120.0, 0.0, 1.0)
    rms_component = _fit_length(rms, n_frames)
    flux_component = _fit_length(flux, n_frames)
    arousal = tempo_component * 0.2 + rms_component * 0.5 + flux_component * 0.3
    arousal = np.clip(arousal, 0.0, 1.0)
    arousal = uniform_filter1d(arousal, smooth_w)

    return valence.astype(np.float32), arousal.astype(np.float32)


def _compute_rhythmic_density(
    onset_env: np.ndarray, tempo: float, n_frames: int, fps: float
) -> np.ndarray:
    """Windowed onset count per beat-length window, normalized 0-1."""
    beat_frames = max(1, int(60.0 / tempo * fps))
    threshold = np.percentile(onset_env, 60)
    onset_binary = (onset_env > threshold).astype(np.float64)
    onset_binary = _fit_length(onset_binary, n_frames)
    density = uniform_filter1d(onset_binary, beat_frames)
    mx = density.max()
    if mx > 1e-8:
        density = density / mx
    return np.clip(density, 0.0, 1.0).astype(np.float32)


def _compute_lookahead(
    rms: np.ndarray,
    section_boundaries: np.ndarray,
    climax_score: np.ndarray,
    n_frames: int,
    fps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pre-computed look-ahead features (future awareness from pre-analysis).

    energy_delta: rms 2s ahead minus rms now
    section_change: 0-1 ramp over 3s before each boundary
    climax_proximity: 0-1 ramp over 4s before each climax peak
    """
    # Energy delta: shift rms backward by 2s
    shift = max(1, int(2.0 * fps))
    rms_future = np.roll(rms, -shift)
    rms_future[-shift:] = rms[-shift:]  # clamp end
    energy_delta = np.clip(rms_future - rms, -1.0, 1.0)

    # Section change proximity: 3s ramp before each boundary
    section_change = np.zeros(n_frames, dtype=np.float64)
    ramp_len = max(1, int(3.0 * fps))
    for b in section_boundaries:
        if b <= 0 or b >= n_frames:
            continue
        start = max(0, b - ramp_len)
        for f in range(start, b):
            t = (f - start) / ramp_len
            section_change[f] = max(section_change[f], t)

    # Climax proximity: 4s ramp before each climax peak (score > 0.5)
    climax_prox = np.zeros(n_frames, dtype=np.float64)
    ramp_len_c = max(1, int(4.0 * fps))
    # Find climax peaks
    threshold = 0.5
    in_peak = False
    peak_frames = []
    for i in range(n_frames):
        if climax_score[i] > threshold and not in_peak:
            in_peak = True
            peak_start = i
        elif climax_score[i] <= threshold and in_peak:
            in_peak = False
            # Peak is at the max within this region
            peak_frames.append(peak_start + int(np.argmax(climax_score[peak_start:i])))

    for pf in peak_frames:
        start = max(0, pf - ramp_len_c)
        for f in range(start, pf):
            t = (f - start) / ramp_len_c
            climax_prox[f] = max(climax_prox[f], t)

    return (
        energy_delta.astype(np.float32),
        section_change.astype(np.float32),
        climax_prox.astype(np.float32),
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
      - anticipation_factor: smoothstep ramp t^2(3-2t) over 4s before each drop
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
        # Anticipation: smoothstep ramp t^2(3-2t) over buildup_duration before drop
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
