"""YouTube audio download using yt-dlp."""

import glob
import os
import re
import yt_dlp

from src.audio.cookie_helper import COOKIE_FILE, cookies_exist


def sanitize_filename(name: str) -> str:
    """Remove characters that are invalid in filenames."""
    return re.sub(r'[<>:"/\\|?*]', '_', name).strip()


def download(url: str, output_dir: str, progress_callback=None) -> str:
    """Download audio from YouTube URL as WAV.

    Args:
        url: YouTube video URL
        output_dir: Directory to save the WAV file
        progress_callback: Optional callable(float) with progress 0.0-1.0

    Returns:
        Path to the downloaded WAV file.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Add deno to PATH for yt-dlp JS runtime
    deno_dir = os.path.expandvars(r"%USERPROFILE%\.deno\bin")
    if os.path.isdir(deno_dir) and deno_dir not in os.environ["PATH"]:
        os.environ["PATH"] = os.environ["PATH"] + ";" + deno_dir

    def hook(d):
        if progress_callback and d["status"] == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            downloaded = d.get("downloaded_bytes", 0)
            if total > 0:
                progress_callback(downloaded / total)

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
        }],
        "progress_hooks": [hook],
        "quiet": True,
        "no_warnings": True,
        "restrictfilenames": True,
        "remote_components": ["ejs:github"],
    }

    if cookies_exist():
        ydl_opts["cookiefile"] = COOKIE_FILE

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

    title = sanitize_filename(info.get("title", "audio"))

    # Find the output WAV - yt-dlp may have sanitized the filename differently
    wav_pattern = os.path.join(output_dir, "*.wav")
    wav_files = sorted(glob.glob(wav_pattern), key=os.path.getmtime, reverse=True)
    if wav_files:
        wav_path = wav_files[0]
    else:
        raise FileNotFoundError(f"WAV file not found after download in {output_dir}")

    if progress_callback:
        progress_callback(1.0)

    return wav_path
