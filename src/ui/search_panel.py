"""Search panel UI with imgui."""

import os
import platform
import re
import shutil
import threading

from imgui_bundle import imgui


def _ensure_deno_path():
    """Ensure deno is on PATH for yt-dlp JS runtime."""
    if platform.system() == "Windows":
        deno_dir = os.path.expandvars(r"%USERPROFILE%\.deno\bin")
        sep = ";"
    else:
        deno_dir = os.path.expanduser("~/.deno/bin")
        sep = ":"
    if os.path.isdir(deno_dir) and deno_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = os.environ.get("PATH", "") + sep + deno_dir


_YT_URL_RE = re.compile(
    r'(?:https?://)?(?:www\.|m\.|music\.)?(?:youtube\.com/watch\?.*v=|youtu\.be/|youtube\.com/shorts/)([A-Za-z0-9_-]{11})'
)


def _extract_video_id(text: str) -> str | None:
    """Return the 11-char video ID if text looks like a YouTube URL, else None."""
    m = _YT_URL_RE.search(text.strip())
    return m.group(1) if m else None


def _fetch_title(video_id: str, callback):
    """Fetch the video title in a background thread using yt-dlp."""
    _ensure_deno_path()
    try:
        import yt_dlp
        opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "extract_flat": True,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
            title = info.get("title", "YouTube Video")
            callback(video_id, title)
    except Exception:
        pass  # keep the placeholder title


_AUDIO_EXTENSIONS = [
    ("Audio Files", "*.wav *.mp3 *.flac *.ogg *.m4a"),
    ("All Files", "*.*"),
]


def _open_file_dialog() -> str | None:
    """Open a native file dialog and return the selected path, or None."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Import Audio File",
            filetypes=_AUDIO_EXTENSIONS,
        )
        root.destroy()
        return path if path else None
    except Exception:
        return None


class SearchPanel:
    """YouTube search bar and results list."""

    def __init__(self):
        self.query = ""
        self.results: list[dict] = []
        self.selected_index: int = -1
        self.searching = False
        self.error_msg = ""
        self._pending_url_result: dict | None = None
        self.imported_file: str | None = None

    def draw(self) -> dict | None:
        """Draw the search panel. Returns selected result dict or None."""
        selected = None

        imgui.set_next_window_pos((20, 20), imgui.Cond_.once)
        imgui.set_next_window_size((400, 500), imgui.Cond_.once)

        expanded, _ = imgui.begin("Search YouTube", None, imgui.WindowFlags_.no_collapse)
        if expanded:
            # Search input
            imgui.set_next_item_width(-80)
            changed, self.query = imgui.input_text(
                "##search", self.query, imgui.InputTextFlags_.enter_returns_true
            )
            trigger_search = changed

            imgui.same_line()
            if imgui.button("Search") or trigger_search:
                if self.query.strip():
                    video_id = _extract_video_id(self.query)
                    if video_id:
                        # Direct URL submission — build a result dict immediately
                        url = self.query.strip()
                        result = {
                            "url": url,
                            "video_id": video_id,
                            "title": "YouTube Video",
                            "channel": "",
                            "duration_str": "",
                        }
                        self._pending_url_result = result
                        # Fetch real title in the background
                        def _on_title(vid, title):
                            if self._pending_url_result and self._pending_url_result.get("video_id") == vid:
                                self._pending_url_result["title"] = title
                        threading.Thread(
                            target=_fetch_title, args=(video_id, _on_title), daemon=True
                        ).start()
                        selected = result
                    else:
                        self.searching = True
                        self.error_msg = ""

            if self.searching:
                imgui.text("Searching...")

            if self.error_msg:
                imgui.text_colored((1.0, 0.3, 0.3, 1.0), self.error_msg)

            imgui.separator()

            # Hint text and import button
            imgui.text_disabled("Search or paste a YouTube URL")
            imgui.same_line(imgui.get_window_width() - 110)
            if imgui.button("Import File"):
                path = _open_file_dialog()
                if path:
                    self.imported_file = path
            imgui.spacing()

            # Results list
            if self.results:
                for i, r in enumerate(self.results):
                    title = r.get("title", "Unknown")
                    channel = r.get("channel", "")
                    dur = r.get("duration_str", "?:??")

                    label = f"{title}\n  {channel} - {dur}"
                    is_selected = i == self.selected_index

                    if imgui.selectable(f"{title}##{i}", is_selected)[0]:
                        self.selected_index = i
                        selected = r

                    # Show details on same line area
                    imgui.same_line(imgui.get_window_width() - 60)
                    imgui.text_disabled(dur)

                    if channel:
                        imgui.text_disabled(f"  {channel}")
            elif not self.searching:
                pass  # hint text already shown above

        imgui.end()
        return selected
