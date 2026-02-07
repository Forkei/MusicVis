"""Song library: metadata JSON, analysis NPZ cache, orphan WAV scanning."""

import json
import os
from dataclasses import fields
from datetime import datetime, timezone

import numpy as np

from src.audio.analyzer import AnalysisResult

DOWNLOADS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "assets", "downloads",
)
LIBRARY_JSON = os.path.join(DOWNLOADS_DIR, "library.json")

ANALYSIS_VERSION = 4


class SongLibrary:
    """Manages downloaded songs, metadata, and analysis cache."""

    def __init__(self):
        self._entries: dict[str, dict] = {}
        self._load()
        self._scan_orphans()

    # --- Persistence ---

    def _load(self):
        if os.path.isfile(LIBRARY_JSON):
            try:
                with open(LIBRARY_JSON, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._entries = {e["video_id"]: e for e in data}
            except (json.JSONDecodeError, KeyError):
                self._entries = {}

    def _save(self):
        os.makedirs(DOWNLOADS_DIR, exist_ok=True)
        with open(LIBRARY_JSON, "w", encoding="utf-8") as f:
            json.dump(list(self._entries.values()), f, indent=2)

    # --- Orphan scanning ---

    def _scan_orphans(self):
        """Find WAV files with no library entry and create entries for them."""
        if not os.path.isdir(DOWNLOADS_DIR):
            return

        known_wavs = {e["wav_filename"] for e in self._entries.values()}
        changed = False

        for fname in os.listdir(DOWNLOADS_DIR):
            if not fname.lower().endswith(".wav"):
                continue
            if fname in known_wavs:
                continue

            stem = os.path.splitext(fname)[0]
            video_id = f"local_{stem}"

            # Skip if this local id already exists
            if video_id in self._entries:
                continue

            title = stem.replace("_", " ")
            npz_path = os.path.join(DOWNLOADS_DIR, f"{stem}.analysis.npz")

            self._entries[video_id] = {
                "video_id": video_id,
                "title": title,
                "channel": "",
                "duration": 0,
                "duration_str": "",
                "wav_filename": fname,
                "has_analysis": os.path.isfile(npz_path),
                "added_at": datetime.now(timezone.utc).isoformat(),
            }
            changed = True

        if changed:
            self._save()

    # --- Public API ---

    def get_all(self) -> list[dict]:
        """All entries sorted by added_at descending."""
        return sorted(
            self._entries.values(),
            key=lambda e: e.get("added_at", ""),
            reverse=True,
        )

    def find_by_video_id(self, video_id: str) -> dict | None:
        return self._entries.get(video_id)

    def wav_path(self, entry: dict) -> str:
        return os.path.join(DOWNLOADS_DIR, entry["wav_filename"])

    def register_download(self, song: dict, wav_filename: str):
        """Add entry after a successful download."""
        video_id = song.get("video_id", "")
        if not video_id:
            stem = os.path.splitext(wav_filename)[0]
            video_id = f"local_{stem}"

        self._entries[video_id] = {
            "video_id": video_id,
            "title": song.get("title", os.path.splitext(wav_filename)[0]),
            "channel": song.get("channel", ""),
            "duration": song.get("duration", 0),
            "duration_str": song.get("duration_str", ""),
            "wav_filename": wav_filename,
            "has_analysis": False,
            "added_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save()
        return self._entries[video_id]

    def save_analysis(self, entry: dict, result: AnalysisResult):
        """Save analysis result as compressed NPZ alongside the WAV."""
        stem = os.path.splitext(entry["wav_filename"])[0]
        npz_path = os.path.join(DOWNLOADS_DIR, f"{stem}.analysis.npz")

        data = {"_version": np.array([ANALYSIS_VERSION])}
        for f in fields(result):
            val = getattr(result, f.name)
            if isinstance(val, np.ndarray):
                data[f.name] = val
            else:
                # Scalars stored as 1-element arrays
                data[f.name] = np.array([val])

        np.savez_compressed(npz_path, **data)

        entry["has_analysis"] = True
        if entry["video_id"] in self._entries:
            self._entries[entry["video_id"]]["has_analysis"] = True
            self._save()

    def load_analysis(self, entry: dict) -> AnalysisResult | None:
        """Load cached analysis from NPZ. Returns None on version mismatch or failure."""
        stem = os.path.splitext(entry["wav_filename"])[0]
        npz_path = os.path.join(DOWNLOADS_DIR, f"{stem}.analysis.npz")

        if not os.path.isfile(npz_path):
            return None

        try:
            data = np.load(npz_path)

            # Version check — reject old format
            if "_version" not in data:
                return None
            version = int(data["_version"][0])
            if version != ANALYSIS_VERSION:
                return None

            kwargs = {}
            for f in fields(AnalysisResult):
                arr = data[f.name]
                if f.type == int:
                    kwargs[f.name] = int(arr[0])
                elif f.type == float:
                    kwargs[f.name] = float(arr[0])
                else:
                    kwargs[f.name] = arr
            return AnalysisResult(**kwargs)
        except Exception:
            return None

    def has_cached_analysis(self, entry: dict) -> bool:
        stem = os.path.splitext(entry["wav_filename"])[0]
        npz_path = os.path.join(DOWNLOADS_DIR, f"{stem}.analysis.npz")
        return os.path.isfile(npz_path)

    def delete_song(self, video_id: str):
        """Remove WAV + NPZ + entry."""
        entry = self._entries.get(video_id)
        if not entry:
            return

        wav = os.path.join(DOWNLOADS_DIR, entry["wav_filename"])
        stem = os.path.splitext(entry["wav_filename"])[0]
        npz = os.path.join(DOWNLOADS_DIR, f"{stem}.analysis.npz")

        for path in (wav, npz):
            try:
                os.remove(path)
            except OSError:
                pass

        del self._entries[video_id]
        self._save()

    def delete_analysis(self, video_id: str):
        """Remove NPZ only, mark has_analysis=False."""
        entry = self._entries.get(video_id)
        if not entry:
            return

        stem = os.path.splitext(entry["wav_filename"])[0]
        npz = os.path.join(DOWNLOADS_DIR, f"{stem}.analysis.npz")

        try:
            os.remove(npz)
        except OSError:
            pass

        entry["has_analysis"] = False
        self._save()
