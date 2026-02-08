"""Optional ML-based audio analysis using essentia ONNX models and allin1.

All functions are designed as drop-in replacements for heuristic functions
in analyzer.py. Each accepts a fallback callable and gracefully degrades
on any failure.

Dependencies (install via requirements-ml.txt):
  - onnxruntime  (ONNX inference for essentia models)
  - allin1       (transformer-based music structure analysis)
  - torch        (required by allin1)
"""

import os
import logging

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Availability check
# ---------------------------------------------------------------------------

_ML_AVAILABLE = {"allin1": False, "onnx": False}


def check_ml_availability() -> dict:
    """Probe for optional ML dependencies. Called once at import time."""
    global _ML_AVAILABLE

    try:
        import onnxruntime  # noqa: F401
        _ML_AVAILABLE["onnx"] = True
        logger.info("[ML] onnxruntime available (version %s)", onnxruntime.__version__)
    except ImportError:
        logger.info("[ML] onnxruntime not installed — using heuristic fallbacks")

    try:
        import allin1  # noqa: F401
        _ML_AVAILABLE["allin1"] = True
        logger.info("[ML] allin1 available")
    except ImportError:
        logger.info("[ML] allin1 not installed — using heuristic section analysis")

    return _ML_AVAILABLE


# Run check at import time
check_ml_availability()


# ---------------------------------------------------------------------------
# 2. Model manager — downloads & caches ONNX models
# ---------------------------------------------------------------------------

_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "assets", "models",
)

_BASE_URL = "https://essentia.upf.edu/models"

# Model registry: name -> (relative URL path, expected filename)
_MODEL_REGISTRY = {
    "effnet": (
        "feature-extractors/discogs-effnet/discogs-effnet-bsdynamic-1.onnx",
        "discogs-effnet-bsdynamic-1.onnx",
    ),
    "genre400": (
        "classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.onnx",
        "genre_discogs400-discogs-effnet-1.onnx",
    ),
    "voice_instrumental": (
        "classification-heads/voice_instrumental/voice_instrumental-discogs-effnet-1.onnx",
        "voice_instrumental-discogs-effnet-1.onnx",
    ),
    "mood_happy": (
        "classification-heads/mood_happy/mood_happy-discogs-effnet-1.onnx",
        "mood_happy-discogs-effnet-1.onnx",
    ),
    "mood_sad": (
        "classification-heads/mood_sad/mood_sad-discogs-effnet-1.onnx",
        "mood_sad-discogs-effnet-1.onnx",
    ),
    "mood_aggressive": (
        "classification-heads/mood_aggressive/mood_aggressive-discogs-effnet-1.onnx",
        "mood_aggressive-discogs-effnet-1.onnx",
    ),
    "mood_relaxed": (
        "classification-heads/mood_relaxed/mood_relaxed-discogs-effnet-1.onnx",
        "mood_relaxed-discogs-effnet-1.onnx",
    ),
}


class ModelManager:
    """Downloads ONNX models on first use, caches InferenceSessions."""

    def __init__(self):
        self._sessions: dict = {}
        os.makedirs(_MODELS_DIR, exist_ok=True)

    def _download(self, model_name: str) -> str:
        """Download model if not cached. Returns local path."""
        url_path, filename = _MODEL_REGISTRY[model_name]
        local_path = os.path.join(_MODELS_DIR, filename)

        if os.path.isfile(local_path):
            return local_path

        url = f"{_BASE_URL}/{url_path}"
        logger.info("[ML] Downloading %s from %s ...", model_name, url)

        import urllib.request
        try:
            urllib.request.urlretrieve(url, local_path)
            size_mb = os.path.getsize(local_path) / (1024 * 1024)
            logger.info("[ML] Downloaded %s (%.1f MB)", model_name, size_mb)
        except Exception as e:
            # Clean up partial download
            if os.path.isfile(local_path):
                os.remove(local_path)
            raise RuntimeError(f"Failed to download {model_name}: {e}") from e

        return local_path

    def get_session(self, model_name: str):
        """Get or create an ONNX InferenceSession for the given model."""
        if model_name in self._sessions:
            return self._sessions[model_name]

        import onnxruntime as ort

        local_path = self._download(model_name)
        sess = ort.InferenceSession(
            local_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self._sessions[model_name] = sess
        logger.info("[ML] Loaded ONNX session: %s", model_name)
        return sess


# Singleton
_model_manager: ModelManager | None = None


def _get_manager() -> ModelManager:
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


# ---------------------------------------------------------------------------
# 3. Effnet embedding extraction
# ---------------------------------------------------------------------------

# Mel spectrogram params matching essentia's TensorflowInputMusiCNN
_EFFNET_SR = 16000
_EFFNET_FRAME = 512
_EFFNET_HOP = 256
_EFFNET_MELS = 96
_EFFNET_PATCH_FRAMES = 128
_EFFNET_PATCH_HOP = 62


def _compute_effnet_mel(y_16k: np.ndarray) -> np.ndarray:
    """Compute mel spectrogram matching effnet's expected input.

    Returns (n_mel_frames, 96) float32 array.
    """
    import librosa

    S = librosa.feature.melspectrogram(
        y=y_16k, sr=_EFFNET_SR,
        n_fft=_EFFNET_FRAME, hop_length=_EFFNET_HOP,
        n_mels=_EFFNET_MELS, fmax=_EFFNET_SR / 2,
        norm="slaney", htk=False,
    )
    # Log-scale matching essentia: log10(1 + 10000 * x)
    S = np.log10(1.0 + S * 10000)
    return S.T.astype(np.float32)  # (time, 96)


def _extract_embeddings(y_16k: np.ndarray) -> np.ndarray:
    """Extract effnet embeddings from 16kHz audio.

    Returns (n_patches, 1280) float32 array.
    """
    mel = _compute_effnet_mel(y_16k)
    n_frames = mel.shape[0]

    # Create patches of 128 frames
    patches = []
    pos = 0
    while pos + _EFFNET_PATCH_FRAMES <= n_frames:
        patch = mel[pos:pos + _EFFNET_PATCH_FRAMES]
        patches.append(patch)
        pos += _EFFNET_PATCH_HOP

    # Handle remainder: pad last patch if needed
    if pos < n_frames and n_frames > _EFFNET_PATCH_FRAMES // 2:
        remaining = mel[pos:]
        pad_len = _EFFNET_PATCH_FRAMES - remaining.shape[0]
        if pad_len > 0:
            remaining = np.pad(remaining, ((0, pad_len), (0, 0)))
        patches.append(remaining[:_EFFNET_PATCH_FRAMES])

    if not patches:
        # Very short audio — single padded patch
        pad_len = _EFFNET_PATCH_FRAMES - n_frames
        padded = np.pad(mel, ((0, max(0, pad_len)), (0, 0)))
        patches.append(padded[:_EFFNET_PATCH_FRAMES])

    # Stack: (n_patches, 128, 96) — model expects rank-3 input
    batch = np.stack(patches, axis=0)

    manager = _get_manager()
    sess = manager.get_session("effnet")
    input_name = sess.get_inputs()[0].name
    # Model has two outputs: [activations(400), embeddings(1280)]
    emb_output_name = sess.get_outputs()[1].name

    # Run in small batches to control memory
    max_batch = 64
    all_embeddings = []
    for i in range(0, len(batch), max_batch):
        chunk = batch[i:i + max_batch]
        emb = sess.run([emb_output_name], {input_name: chunk})[0]
        all_embeddings.append(emb)

    embeddings = np.concatenate(all_embeddings, axis=0)  # (n_patches, 1280)
    return embeddings.astype(np.float32)


def _load_audio_16k(audio_path: str) -> np.ndarray:
    """Load and resample audio to 16kHz mono."""
    import librosa
    y, _ = librosa.load(audio_path, sr=_EFFNET_SR, mono=True)
    return y


# Per-file cache: avoids running effnet 3x during a single analysis
_embedding_cache: dict[str, np.ndarray] = {}


def _get_embeddings_cached(audio_path: str) -> np.ndarray:
    """Get effnet embeddings, using cache if available for this file."""
    if audio_path in _embedding_cache:
        return _embedding_cache[audio_path]
    y_16k = _load_audio_16k(audio_path)
    emb = _extract_embeddings(y_16k)
    _embedding_cache[audio_path] = emb
    return emb


def clear_embedding_cache():
    """Clear the embedding cache after analysis completes."""
    _embedding_cache.clear()


# ---------------------------------------------------------------------------
# 4. Genre classification (400 discogs classes → 7 genre IDs)
# ---------------------------------------------------------------------------

# Mapping from discogs-400 class indices to our 7-genre system.
# Discogs-400 uses a hierarchical "genre---subgenre" naming scheme.
# We map the top-level genre portion.
#
# Our genre IDs: 0=EDM, 1=Rock, 2=Jazz, 3=Classical, 4=Hip-Hop, 5=Ambient, 6=Pop

# Top-level discogs genre → our genre ID
_DISCOGS_TOPLEVEL_MAP = {
    "electronic": 0,
    "techno": 0,
    "house": 0,
    "trance": 0,
    "drum and bass": 0,
    "dubstep": 0,
    "breakbeat": 0,
    "electro": 0,
    "euro house": 0,
    "garage house": 0,
    "hardcore": 0,
    "hardstyle": 0,
    "jungle": 0,
    "progressive trance": 0,
    "tech house": 0,
    "acid house": 0,
    "deep house": 0,
    "minimal": 0,
    "idm": 0,

    "rock": 1,
    "metal": 1,
    "punk": 1,
    "alternative rock": 1,
    "indie rock": 1,
    "hard rock": 1,
    "grunge": 1,
    "post-punk": 1,
    "shoegaze": 1,
    "classic rock": 1,
    "progressive rock": 1,
    "psychedelic rock": 1,
    "garage rock": 1,
    "stoner rock": 1,
    "emo": 1,
    "nu metal": 1,
    "heavy metal": 1,
    "death metal": 1,
    "black metal": 1,
    "thrash": 1,
    "industrial": 1,
    "noise": 1,
    "post-rock": 1,
    "math rock": 1,

    "jazz": 2,
    "swing": 2,
    "bebop": 2,
    "fusion": 2,
    "big band": 2,
    "bossa nova": 2,
    "latin jazz": 2,
    "smooth jazz": 2,
    "free jazz": 2,
    "cool jazz": 2,
    "contemporary jazz": 2,
    "acid jazz": 2,

    "classical": 3,
    "baroque": 3,
    "romantic": 3,
    "opera": 3,
    "modern classical": 3,
    "choral": 3,
    "chamber music": 3,
    "medieval": 3,
    "renaissance": 3,
    "contemporary": 3,
    "impressionist": 3,
    "orchestral": 3,

    "hip hop": 4,
    "hip-hop": 4,
    "rap": 4,
    "trap": 4,
    "gangsta": 4,
    "boom bap": 4,
    "conscious": 4,
    "instrumental hip hop": 4,
    "grime": 4,
    "crunk": 4,
    "drill": 4,
    "g-funk": 4,

    "ambient": 5,
    "downtempo": 5,
    "new age": 5,
    "drone": 5,
    "chillout": 5,
    "dark ambient": 5,
    "space music": 5,
    "field recording": 5,
    "meditation": 5,
    "lo-fi": 5,

    "pop": 6,
    "synth-pop": 6,
    "disco": 6,
    "soul": 6,
    "r&b": 6,
    "funk": 6,
    "country": 6,
    "folk": 6,
    "reggae": 6,
    "ska": 6,
    "blues": 6,
    "gospel": 6,
    "latin": 6,
    "world": 6,
    "afrobeat": 6,
    "dancehall": 6,
    "k-pop": 6,
    "j-pop": 6,
    "ballad": 6,
    "singer-songwriter": 6,
    "chanson": 6,
    "schlager": 6,
    "europop": 6,
}


def _build_discogs400_to_genre7(class_names: list[str]) -> np.ndarray:
    """Build a (400,) int array mapping each discogs class index to a genre ID.

    class_names: list of 400 strings like "Electronic---Techno" or "Rock---Punk".
    Falls back to genre 6 (Pop) for unrecognized classes.
    """
    mapping = np.full(len(class_names), 6, dtype=np.int32)  # default = Pop

    for i, name in enumerate(class_names):
        # Discogs format: "Genre---Subgenre" or just "Genre"
        parts = name.lower().replace("---", "---").split("---")
        # Try subgenre first, then genre
        for part in reversed(parts):
            part = part.strip()
            if part in _DISCOGS_TOPLEVEL_MAP:
                mapping[i] = _DISCOGS_TOPLEVEL_MAP[part]
                break

    return mapping


# Hard-coded discogs-400 top-level genre mapping based on the index structure.
# The first part before "---" is the broad genre.
_DISCOGS_BROAD_GENRE_MAP = {
    "Electronic": 0,
    "Rock": 1,
    "Jazz": 2,
    "Classical": 3,
    "Hip Hop": 4,
    "Ambient": 5,  # subset of Electronic in discogs, but we treat separately
    "Pop": 6,
    "Folk, World, & Country": 6,
    "Funk / Soul": 6,
    "Reggae": 6,
    "Blues": 6,
    "Latin": 6,
    "Stage & Screen": 3,
    "Brass & Military": 3,
    "Non-Music": 6,
    "Children's": 6,
}


def _get_discogs400_class_names() -> list[str] | None:
    """Try to load class names from the metadata JSON, or return None."""
    json_path = os.path.join(_MODELS_DIR, "genre_discogs400-discogs-effnet-1.json")

    # Try to download metadata if not present
    if not os.path.isfile(json_path):
        url = f"{_BASE_URL}/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.json"
        try:
            import urllib.request
            urllib.request.urlretrieve(url, json_path)
        except Exception:
            return None

    try:
        import json
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        # metadata has "classes" key with list of class names
        return meta.get("classes", None)
    except Exception:
        return None


def ml_classify_genre(
    audio_path: str,
    sr: int,
    fallback_fn,
) -> tuple[int, float, np.ndarray]:
    """Classify genre using effnet's 400-class sigmoid activations.

    The effnet model outputs 400 sigmoid probabilities (multi-label)
    alongside the 1280-dim embeddings. We aggregate these into our
    7-genre system by summing per-genre probabilities.

    Returns (genre_id, confidence, probabilities_7).
    Falls back to fallback_fn() on any error.
    """
    try:
        y_16k = _load_audio_16k(audio_path)
        mel = _compute_effnet_mel(y_16k)
        n_mel_frames = mel.shape[0]

        # Create patches
        patches = []
        pos = 0
        while pos + _EFFNET_PATCH_FRAMES <= n_mel_frames:
            patches.append(mel[pos:pos + _EFFNET_PATCH_FRAMES])
            pos += _EFFNET_PATCH_HOP
        if not patches:
            pad_len = _EFFNET_PATCH_FRAMES - n_mel_frames
            padded = np.pad(mel, ((0, max(0, pad_len)), (0, 0)))
            patches.append(padded[:_EFFNET_PATCH_FRAMES])

        batch = np.stack(patches, axis=0)  # (n_patches, 128, 96)

        manager = _get_manager()
        sess = manager.get_session("effnet")
        input_name = sess.get_inputs()[0].name
        act_output_name = sess.get_outputs()[0].name  # activations (400)

        # Get activations from all patches, then average
        all_acts = []
        max_batch = 64
        for i in range(0, len(batch), max_batch):
            chunk = batch[i:i + max_batch]
            acts = sess.run([act_output_name], {input_name: chunk})[0]
            all_acts.append(acts)

        # Activations are sigmoid outputs (multi-label), not logits
        probs_400 = np.mean(np.concatenate(all_acts, axis=0), axis=0)  # (400,)

        # Map 400 classes → 7 genres
        class_names = _get_discogs400_class_names()
        if class_names is not None and len(class_names) == len(probs_400):
            mapping = _build_discogs400_to_genre7(class_names)
        else:
            # Fallback: use broad genre mapping based on index position
            # Discogs-400 is roughly ordered by broad genre
            mapping = np.full(len(probs_400), 6, dtype=np.int32)
            logger.warning("[ML] Could not load discogs-400 class names, using default mapping")

        # Aggregate probabilities per genre
        probs_7 = np.zeros(7, dtype=np.float32)
        for genre_id in range(7):
            mask = mapping == genre_id
            if np.any(mask):
                probs_7[genre_id] = float(np.sum(probs_400[mask]))

        # Normalize
        total = probs_7.sum()
        if total > 1e-8:
            probs_7 /= total

        genre_id = int(np.argmax(probs_7))
        confidence = float(probs_7[genre_id])

        logger.info("[ML] Genre: %d (confidence=%.2f) probs=%s",
                     genre_id, confidence, np.round(probs_7, 3))
        return genre_id, confidence, probs_7

    except Exception as e:
        logger.warning("[ML] Genre classification failed: %s — using fallback", e)
        return fallback_fn()


# ---------------------------------------------------------------------------
# 5. Vocal/instrumental detection
# ---------------------------------------------------------------------------

def ml_detect_vocal(
    audio_path: str,
    sr: int,
    n_frames: int,
    fps: float,
    fallback_fn,
) -> np.ndarray:
    """Detect vocal presence using ONNX model.

    Returns (n_frames,) float32 array of vocal presence [0, 1].
    """
    try:
        embeddings = _get_embeddings_cached(audio_path)
        n_patches = embeddings.shape[0]

        manager = _get_manager()
        sess = manager.get_session("voice_instrumental")
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name

        # Run per-patch for temporal resolution
        vocal_scores = np.zeros(n_patches, dtype=np.float32)
        for i in range(n_patches):
            emb = embeddings[i:i + 1]  # (1, 1280)
            probs = sess.run([output_name], {input_name: emb})[0][0]  # (2,)
            # classes: [instrumental, voice]
            vocal_scores[i] = probs[1]  # voice probability

        # Interpolate patch-level scores to frame resolution
        if n_patches == 1:
            vocal_presence = np.full(n_frames, vocal_scores[0], dtype=np.float32)
        else:
            patch_times = np.linspace(0, 1, n_patches)
            frame_times = np.linspace(0, 1, n_frames)
            vocal_presence = np.interp(frame_times, patch_times, vocal_scores).astype(np.float32)

        logger.info("[ML] Vocal detection: mean=%.2f, max=%.2f",
                     float(np.mean(vocal_presence)), float(np.max(vocal_presence)))
        return vocal_presence

    except Exception as e:
        logger.warning("[ML] Vocal detection failed: %s — using fallback", e)
        return fallback_fn()


# ---------------------------------------------------------------------------
# 6. Valence / Arousal
# ---------------------------------------------------------------------------

def ml_compute_valence_arousal(
    audio_path: str,
    sr: int,
    n_frames: int,
    fps: float,
    fallback_fn,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute valence and arousal using mood classification models.

    Uses 4 mood classifiers as proxies:
      - happy/sad → valence
      - aggressive/relaxed → arousal

    Returns (valence, arousal) each (n_frames,) float32 in [0, 1].
    """
    try:
        embeddings = _get_embeddings_cached(audio_path)
        n_patches = embeddings.shape[0]

        manager = _get_manager()

        # Load all 4 mood sessions
        sess_happy = manager.get_session("mood_happy")
        sess_sad = manager.get_session("mood_sad")
        sess_aggressive = manager.get_session("mood_aggressive")
        sess_relaxed = manager.get_session("mood_relaxed")

        # Get input/output names (same for all classification heads)
        in_name = sess_happy.get_inputs()[0].name
        out_name = sess_happy.get_outputs()[0].name

        happy_scores = np.zeros(n_patches, dtype=np.float32)
        sad_scores = np.zeros(n_patches, dtype=np.float32)
        aggressive_scores = np.zeros(n_patches, dtype=np.float32)
        relaxed_scores = np.zeros(n_patches, dtype=np.float32)

        for i in range(n_patches):
            emb = embeddings[i:i + 1]
            # Class indices: happy=[happy, non_happy], sad=[non_sad, sad],
            #   aggressive=[aggressive, not_aggressive], relaxed=[non_relaxed, relaxed]
            happy_scores[i] = sess_happy.run([out_name], {in_name: emb})[0][0][0]       # happy
            sad_scores[i] = sess_sad.run([out_name], {in_name: emb})[0][0][1]            # sad
            aggressive_scores[i] = sess_aggressive.run([out_name], {in_name: emb})[0][0][0]  # aggressive
            relaxed_scores[i] = sess_relaxed.run([out_name], {in_name: emb})[0][0][1]    # relaxed

        # Valence: happy vs sad (0=sad, 1=happy)
        valence_patch = np.clip(
            0.5 + 0.5 * (happy_scores - sad_scores), 0.0, 1.0
        )
        # Arousal: aggressive vs relaxed (0=relaxed, 1=aggressive)
        arousal_patch = np.clip(
            0.5 + 0.5 * (aggressive_scores - relaxed_scores), 0.0, 1.0
        )

        # Interpolate to frame resolution
        if n_patches == 1:
            valence = np.full(n_frames, valence_patch[0], dtype=np.float32)
            arousal = np.full(n_frames, arousal_patch[0], dtype=np.float32)
        else:
            patch_times = np.linspace(0, 1, n_patches)
            frame_times = np.linspace(0, 1, n_frames)
            valence = np.interp(frame_times, patch_times, valence_patch).astype(np.float32)
            arousal = np.interp(frame_times, patch_times, arousal_patch).astype(np.float32)

        # Light smoothing
        from scipy.ndimage import uniform_filter1d
        smooth_w = max(3, int(2.0 * fps))
        valence = uniform_filter1d(valence, smooth_w).astype(np.float32)
        arousal = uniform_filter1d(arousal, smooth_w).astype(np.float32)

        logger.info("[ML] Valence: mean=%.2f, Arousal: mean=%.2f",
                     float(np.mean(valence)), float(np.mean(arousal)))
        return valence, arousal

    except Exception as e:
        logger.warning("[ML] Valence/arousal failed: %s — using fallback", e)
        return fallback_fn()


# ---------------------------------------------------------------------------
# 7. Structure analysis (allin1)
# ---------------------------------------------------------------------------

# allin1 label → our section type ID
_ALLIN1_LABEL_MAP = {
    "start": 0,    # treat as intro
    "intro": 0,
    "verse": 1,
    "chorus": 2,
    "bridge": 3,
    "break": 4,     # breakdown
    "inst": 1,      # instrumental verse
    "solo": 8,
    "outro": 7,
    "end": 7,       # treat as outro
}


def ml_analyze_structure(
    audio_path: str,
    sr: int,
    hop_length: int,
    n_frames: int,
    fps: float,
    rms: np.ndarray,
    bass_energy: np.ndarray,
    onset_strength: np.ndarray,
    fallback_fn,
) -> tuple[np.ndarray, int, np.ndarray, np.ndarray, int, float]:
    """Analyze music structure using allin1 transformer model.

    Returns (section_labels, n_sections, section_type, section_boundaries, n_boundaries, tempo).
    Falls back to fallback_fn() on any error.
    """
    try:
        import allin1

        logger.info("[ML] Running allin1 structure analysis...")
        result = allin1.analyze(
            audio_path,
            model="harmonix-all",
            include_activations=False,
            include_embeddings=False,
        )

        segments = result.segments
        tempo = float(result.bpm) if result.bpm and result.bpm > 0 else None

        if not segments:
            logger.warning("[ML] allin1 returned no segments — using fallback")
            return fallback_fn()

        # Build per-frame section_labels and section_type arrays
        section_labels = np.zeros(n_frames, dtype=np.int32)
        section_type = np.zeros(n_frames, dtype=np.int32)
        boundaries = []

        for seg_idx, seg in enumerate(segments):
            start_frame = int(seg.start * sr / hop_length)
            end_frame = int(seg.end * sr / hop_length)
            start_frame = max(0, min(start_frame, n_frames - 1))
            end_frame = max(start_frame + 1, min(end_frame, n_frames))

            section_labels[start_frame:end_frame] = seg_idx
            label = seg.label.lower().strip()
            stype = _ALLIN1_LABEL_MAP.get(label, 1)  # default to verse
            section_type[start_frame:end_frame] = stype

            if seg_idx > 0:
                boundaries.append(start_frame)

        n_sections = len(segments)

        # --- Post-process: detect buildups and drops ---
        # allin1 doesn't output EDM-specific buildup/drop labels.
        # Detect them from energy patterns within and between sections.
        _postprocess_buildup_drop(
            section_type, section_labels, boundaries,
            rms, bass_energy, onset_strength, n_frames, fps,
        )

        boundary_arr = np.array(boundaries, dtype=np.int32) if boundaries else np.zeros(1, dtype=np.int32)
        n_boundaries = len(boundaries)

        logger.info("[ML] Structure: %d sections, %d boundaries, tempo=%.1f",
                     n_sections, n_boundaries, tempo or 0.0)

        return section_labels, n_sections, section_type, boundary_arr, n_boundaries, tempo

    except Exception as e:
        logger.warning("[ML] Structure analysis failed: %s — using fallback", e)
        return fallback_fn()


def _postprocess_buildup_drop(
    section_type: np.ndarray,
    section_labels: np.ndarray,
    boundaries: list,
    rms: np.ndarray,
    bass_energy: np.ndarray,
    onset_strength: np.ndarray,
    n_frames: int,
    fps: float,
):
    """In-place modification of section_type to add buildup(5) and drop(6) labels.

    Looks for:
    - Buildup: sections with rising energy slope (especially onset density)
      that precede a high-energy section
    - Drop: sections that follow a buildup or breakdown and have a sudden
      energy increase + strong bass
    """
    from scipy.ndimage import uniform_filter1d

    # Find section start/end from boundaries
    all_starts = [0] + list(boundaries)
    all_ends = list(boundaries) + [n_frames]

    n_secs = len(all_starts)
    if n_secs < 2:
        return

    for i in range(n_secs):
        start = all_starts[i]
        end = all_ends[i]
        seg_len = end - start

        if seg_len < int(fps * 2):
            continue

        seg_rms = rms[start:end]
        seg_onset = onset_strength[start:end]
        seg_bass = bass_energy[start:end]
        current_type = section_type[start]

        # Compute energy slope within section
        q_len = max(1, seg_len // 4)
        first_q_rms = float(np.mean(seg_rms[:q_len]))
        last_q_rms = float(np.mean(seg_rms[-q_len:]))
        energy_slope = last_q_rms - first_q_rms

        first_q_onset = float(np.mean(seg_onset[:q_len]))
        last_q_onset = float(np.mean(seg_onset[-q_len:]))
        onset_slope = last_q_onset - first_q_onset

        # Check if next section is high-energy
        if i + 1 < n_secs:
            next_start = all_starts[i + 1]
            next_end = all_ends[i + 1]
            next_rms = float(np.mean(rms[next_start:next_end]))
            next_bass = float(np.mean(bass_energy[next_start:next_end]))
            seg_rms_mean = float(np.mean(seg_rms))

            # Buildup detection: rising energy + rising onsets + followed by louder section
            if (energy_slope > 0.05 or onset_slope > 0.03) and next_rms > seg_rms_mean + 0.08:
                # Don't override chorus or outro
                if current_type not in (2, 7):
                    section_type[start:end] = 5  # buildup
                    # Also mark the next section as drop if it has strong bass
                    if next_bass > 0.4 and section_type[next_start] not in (2, 7, 8):
                        section_type[next_start:next_end] = 6  # drop

        # Drop detection: sudden energy jump from previous section
        if i > 0:
            prev_start = all_starts[i - 1]
            prev_end = all_ends[i - 1]
            prev_rms = float(np.mean(rms[prev_start:prev_end]))
            seg_rms_mean = float(np.mean(seg_rms))
            seg_bass_mean = float(np.mean(seg_bass))

            contrast = seg_rms_mean - prev_rms
            if (contrast > 0.15 and seg_bass_mean > 0.4
                    and section_type[prev_start] in (4, 5)  # preceded by breakdown or buildup
                    and current_type not in (0, 2, 7, 8)):  # don't override these
                section_type[start:end] = 6  # drop
