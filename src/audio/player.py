"""Audio playback using sounddevice with precise position tracking."""

import os
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf


class AudioPlayer:
    """Plays audio with callback-based position tracking."""

    def __init__(self):
        self._audio_data: np.ndarray | None = None
        self._sr: int = 44100
        self._channels: int = 2
        self._frame_pos: int = 0
        self._playing: bool = False
        self._stream: sd.OutputStream | None = None
        self._lock = threading.Lock()
        self._duration: float = 0.0

    def load(self, path: str):
        """Load an audio file for playback (WAV, MP3, FLAC, OGG, M4A)."""
        self.stop()
        ext = os.path.splitext(path)[1].lower()
        if ext == ".wav":
            data, sr = sf.read(path, dtype="float32")
        else:
            import librosa
            y, sr = librosa.load(path, sr=None, mono=False)
            # librosa returns (channels, samples) or (samples,) — transpose to (samples, channels)
            if y.ndim == 2:
                data = y.T.astype(np.float32)
            else:
                data = y.astype(np.float32)
        if data.ndim == 1:
            data = np.column_stack([data, data])
        self._audio_data = data
        self._sr = sr
        self._channels = data.shape[1]
        self._duration = len(data) / sr
        self._frame_pos = 0

    def _callback(self, outdata, frames, time_info, status):
        with self._lock:
            if not self._playing or self._audio_data is None:
                outdata[:] = 0
                return

            pos = self._frame_pos
            end = pos + frames
            total = len(self._audio_data)

            if pos >= total:
                outdata[:] = 0
                self._playing = False
                return

            if end > total:
                valid = total - pos
                outdata[:valid] = self._audio_data[pos:total]
                outdata[valid:] = 0
                self._frame_pos = total
                self._playing = False
            else:
                outdata[:] = self._audio_data[pos:end]
                self._frame_pos = end

    def play(self):
        """Start or resume playback."""
        if self._audio_data is None:
            return
        if self._stream is None:
            self._stream = sd.OutputStream(
                samplerate=self._sr,
                channels=self._channels,
                callback=self._callback,
                blocksize=1024,
            )
            self._stream.start()
        self._playing = True

    def pause(self):
        """Pause playback."""
        self._playing = False

    def toggle(self):
        """Toggle play/pause."""
        if self._playing:
            self.pause()
        else:
            self.play()

    def stop(self):
        """Stop playback and release stream."""
        self._playing = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._frame_pos = 0

    def seek(self, time: float):
        """Seek to a time position in seconds."""
        if self._audio_data is None:
            return
        with self._lock:
            frame = int(time * self._sr)
            self._frame_pos = max(0, min(frame, len(self._audio_data)))

    def get_position(self) -> float:
        """Get current playback position in seconds."""
        if self._audio_data is None:
            return 0.0
        return self._frame_pos / self._sr

    def is_playing(self) -> bool:
        return self._playing

    @property
    def duration(self) -> float:
        return self._duration

    def cleanup(self):
        """Clean up resources."""
        self.stop()
