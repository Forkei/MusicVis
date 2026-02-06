"""YouTube search using yt-dlp."""

import yt_dlp


def search(query: str, max_results: int = 10) -> list[dict]:
    """Search YouTube for songs.

    Returns list of dicts with keys: title, channel, duration, url, video_id
    """
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": "in_playlist",
        "default_search": f"ytsearch{max_results}",
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(query, download=False)

    results = []
    for entry in info.get("entries", []):
        if entry is None:
            continue
        duration = entry.get("duration") or 0
        minutes = int(duration) // 60
        seconds = int(duration) % 60
        results.append({
            "title": entry.get("title", "Unknown"),
            "channel": entry.get("channel") or entry.get("uploader") or "Unknown",
            "duration": duration,
            "duration_str": f"{minutes}:{seconds:02d}",
            "url": entry.get("url") or entry.get("webpage_url") or f"https://www.youtube.com/watch?v={entry.get('id', '')}",
            "video_id": entry.get("id", ""),
        })

    return results
