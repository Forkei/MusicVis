"""Main application: GLFW window + moderngl + imgui + audio integration."""

import os
import sys
import time
import threading
from enum import Enum, auto

import glfw
import moderngl
import numpy as np
from imgui_bundle import imgui
from imgui_bundle.python_backends.glfw_backend import GlfwRenderer as ImGuiGlfwRenderer

from src.audio.search import search as yt_search
from src.audio.downloader import download as yt_download
from src.audio.analyzer import analyze as audio_analyze, AnalysisResult
from src.audio.player import AudioPlayer
from src.audio.cookie_helper import cookies_exist, open_login_browser, export_cookies
from src.visualization.renderer import Renderer
from src.visualization.energy_ball import EnergyBallGenerator
from src.audio.library import SongLibrary
from src.ui.search_panel import SearchPanel
from src.ui.player_controls import PlayerControls
from src.ui.settings_panel import SettingsPanel
from src.ui.library_panel import LibraryPanel


class AppState(Enum):
    SETUP = auto()
    SEARCH = auto()
    DOWNLOADING = auto()
    ANALYZING = auto()
    PLAYING = auto()


DOWNLOADS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "downloads")


class App:
    """Main application class."""

    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height
        self.progress = 0.0
        self.status_msg = ""
        self.error_msg = ""

        # Check if cookies exist, if not go to SETUP
        if cookies_exist():
            self.state = AppState.SEARCH
        else:
            self.state = AppState.SETUP

        # Setup state
        self._setup_browser_proc = None
        self._setup_status = ""

        # Audio
        self.player = AudioPlayer()
        self.analysis: AnalysisResult | None = None
        self.current_song_title = ""

        # Library
        self.library = SongLibrary()
        self.library_panel = LibraryPanel()
        self._current_entry: dict | None = None

        # Background task
        self._bg_thread: threading.Thread | None = None

        # Init GLFW
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        glfw.window_hint(glfw.SAMPLES, 4)

        self.window = glfw.create_window(width, height, "MusicVis - Electric Energy Ball", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)  # vsync

        # moderngl context
        self.ctx = moderngl.create_context()

        # ImGui
        imgui.create_context()
        self.imgui_impl = ImGuiGlfwRenderer(self.window)

        io = imgui.get_io()
        io.config_flags |= imgui.ConfigFlags_.nav_enable_keyboard

        # Style
        style = imgui.get_style()
        imgui.style_colors_dark(style)
        style.window_rounding = 6.0
        style.frame_rounding = 4.0
        style.grab_rounding = 3.0

        # Renderer + generator
        self.renderer = Renderer(self.ctx, width, height)
        self.ball_gen = EnergyBallGenerator()

        # UI panels
        self.search_panel = SearchPanel()
        self.player_controls = PlayerControls()
        self.settings_panel = SettingsPanel()

        # Resize callback
        glfw.set_framebuffer_size_callback(self.window, self._on_resize)

        # Idle features for when no song is loaded
        self._idle_time_start = time.time()

        # Delta time tracking
        self._last_frame_time = time.time()

    def _on_resize(self, window, width, height):
        if width > 0 and height > 0:
            self.width = width
            self.height = height
            self.ctx.viewport = (0, 0, width, height)
            self.renderer.resize(width, height)

    def run(self):
        """Main loop."""
        try:
            while not glfw.window_should_close(self.window):
                glfw.poll_events()
                self.imgui_impl.process_inputs()

                imgui.new_frame()

                self._update()
                self._render()

                imgui.render()
                self.imgui_impl.render(imgui.get_draw_data())

                glfw.swap_buffers(self.window)
        finally:
            self._cleanup()

    def _update(self):
        """Update logic per frame."""
        if self.state == AppState.SETUP:
            self._update_setup()
        elif self.state == AppState.SEARCH:
            self._update_search()
        elif self.state == AppState.DOWNLOADING:
            self._update_downloading()
        elif self.state == AppState.ANALYZING:
            self._update_analyzing()
        elif self.state == AppState.PLAYING:
            self._update_playing()

    def _update_setup(self):
        """Cookie setup flow: opens Chrome for YouTube sign-in."""
        viewport = imgui.get_main_viewport()
        vp_size = viewport.size
        win_w, win_h = 500, 200

        imgui.set_next_window_pos(
            ((vp_size.x - win_w) / 2, (vp_size.y - win_h) / 2),
            imgui.Cond_.always,
        )
        imgui.set_next_window_size((win_w, win_h), imgui.Cond_.always)

        expanded, _ = imgui.begin("YouTube Setup", None, imgui.WindowFlags_.no_collapse)
        if expanded:
            imgui.text_wrapped(
                "YouTube requires authentication to download audio. "
                "Click the button below to open a browser window. "
                "Sign into your Google account, then come back and click 'Done'."
            )
            imgui.spacing()

            if self._setup_browser_proc is None:
                if imgui.button("Open Browser to Sign In", (250, 30)):
                    self._setup_browser_proc = open_login_browser()
                    if self._setup_browser_proc is None:
                        self._setup_status = "Chrome not found. Please install Google Chrome."
                    else:
                        self._setup_status = "Browser opened. Sign into Google, then click 'Done' below."
            else:
                imgui.text(self._setup_status)
                imgui.spacing()

                if imgui.button("Done - I've Signed In", (250, 30)):
                    self._setup_status = "Extracting cookies..."
                    logged_in = export_cookies(self._setup_browser_proc)
                    if logged_in:
                        self._setup_status = "Success! Cookies saved."
                        # Kill browser
                        try:
                            self._setup_browser_proc.terminate()
                        except Exception:
                            pass
                        self._setup_browser_proc = None
                        self.state = AppState.SEARCH
                    elif cookies_exist():
                        self._setup_status = "Cookies saved (not logged in - may not work for all videos)."
                        try:
                            self._setup_browser_proc.terminate()
                        except Exception:
                            pass
                        self._setup_browser_proc = None
                        self.state = AppState.SEARCH
                    else:
                        self._setup_status = "No cookies found. Please sign into YouTube in the browser first."

                imgui.same_line()
                if imgui.button("Skip", (80, 30)):
                    if self._setup_browser_proc:
                        try:
                            self._setup_browser_proc.terminate()
                        except Exception:
                            pass
                    self._setup_browser_proc = None
                    self.state = AppState.SEARCH

            if self._setup_status and "Success" not in self._setup_status:
                imgui.text_colored((1.0, 0.8, 0.3, 1.0), self._setup_status)

        imgui.end()

    def _update_search(self):
        selected = self.search_panel.draw()

        # Library panel
        lib_action = self.library_panel.draw(self.library.get_all())
        if lib_action:
            self._handle_library_action(lib_action)

        # Handle search trigger
        if self.search_panel.searching:
            query = self.search_panel.query
            self.search_panel.searching = False
            self.search_panel.error_msg = ""
            self._search_in_progress = True

            def do_search():
                try:
                    results = yt_search(query)
                    self.search_panel.results = results
                except Exception as e:
                    self.search_panel.error_msg = str(e)
                finally:
                    self._search_in_progress = False

            threading.Thread(target=do_search, daemon=True).start()

        # Show searching state from background thread
        if getattr(self, "_search_in_progress", False):
            self.search_panel.searching = True

        # Handle song selection — check library cache first
        if selected:
            video_id = selected.get("video_id", "")
            existing = self.library.find_by_video_id(video_id) if video_id else None
            if existing:
                self._play_from_library(existing)
            else:
                self._start_download(selected)

    def _play_from_library(self, entry: dict):
        """Play a song from the library, using cached analysis if available."""
        wav = self.library.wav_path(entry)
        if not os.path.isfile(wav):
            self.library.delete_song(entry["video_id"])
            self.search_panel.error_msg = "WAV file missing — entry removed."
            return

        self._current_entry = entry
        self.current_song_title = entry.get("title", "Unknown")

        # Try loading cached analysis
        cached = self.library.load_analysis(entry)
        if cached is not None:
            self.analysis = cached
            self.player.load(wav)
            self.player.play()
            self.state = AppState.PLAYING
        else:
            self._start_analysis(wav)

    def _handle_library_action(self, action: dict):
        """Handle play/delete/reanalyze actions from the library panel."""
        entry = action["entry"]
        act = action["action"]

        if act == "play":
            # Stop current playback if any
            if self.state == AppState.PLAYING:
                self.player.stop()
                self.analysis = None
            self._play_from_library(entry)

        elif act == "delete":
            # Stop if this song is currently playing
            if self._current_entry and self._current_entry.get("video_id") == entry["video_id"]:
                self.player.stop()
                self.analysis = None
                self._current_entry = None
                self.state = AppState.SEARCH
            self.library.delete_song(entry["video_id"])

        elif act == "reanalyze":
            # Stop if this song is currently playing
            if self.state == AppState.PLAYING:
                self.player.stop()
                self.analysis = None
            self.library.delete_analysis(entry["video_id"])
            self._current_entry = entry
            self.current_song_title = entry.get("title", "Unknown")
            wav = self.library.wav_path(entry)
            if os.path.isfile(wav):
                self._start_analysis(wav)
            else:
                self.library.delete_song(entry["video_id"])
                self.search_panel.error_msg = "WAV file missing — entry removed."

    def _start_download(self, song: dict):
        self.state = AppState.DOWNLOADING
        self.progress = 0.0
        self.current_song_title = song.get("title", "Unknown")
        self.status_msg = f"Downloading: {self.current_song_title}"
        self.error_msg = ""

        url = song["url"]

        def do_download():
            try:
                path = yt_download(url, DOWNLOADS_DIR, progress_callback=self._set_progress)
                wav_filename = os.path.basename(path)
                self._current_entry = self.library.register_download(song, wav_filename)
                self._start_analysis(path)
            except Exception as e:
                msg = str(e)
                if "Sign in" in msg or "bot" in msg:
                    self.search_panel.error_msg = "YouTube blocked this video. Try a different result."
                else:
                    self.search_panel.error_msg = f"Download failed: {msg[:120]}"
                self.state = AppState.SEARCH

        self._bg_thread = threading.Thread(target=do_download, daemon=True)
        self._bg_thread.start()

    def _start_analysis(self, wav_path: str):
        self.state = AppState.ANALYZING
        self.progress = 0.0
        self.status_msg = f"Analyzing: {self.current_song_title}"

        def do_analyze():
            try:
                result = audio_analyze(wav_path, progress_callback=self._set_progress)
                self.analysis = result
                if self._current_entry:
                    self.library.save_analysis(self._current_entry, result)
                self.player.load(wav_path)
                self.player.play()
                self.state = AppState.PLAYING
            except Exception as e:
                self.search_panel.error_msg = f"Analysis failed: {str(e)[:120]}"
                self.state = AppState.SEARCH

        self._bg_thread = threading.Thread(target=do_analyze, daemon=True)
        self._bg_thread.start()

    def _set_progress(self, p: float):
        self.progress = max(0.0, min(1.0, p))

    def _update_downloading(self):
        self._draw_progress_overlay()

    def _update_analyzing(self):
        self._draw_progress_overlay()

    def _draw_progress_overlay(self):
        viewport = imgui.get_main_viewport()
        vp_size = viewport.size
        win_w, win_h = 400, 100

        imgui.set_next_window_pos(
            ((vp_size.x - win_w) / 2, (vp_size.y - win_h) / 2),
            imgui.Cond_.always,
        )
        imgui.set_next_window_size((win_w, win_h), imgui.Cond_.always)

        flags = (
            imgui.WindowFlags_.no_title_bar
            | imgui.WindowFlags_.no_resize
            | imgui.WindowFlags_.no_move
        )

        expanded, _ = imgui.begin("##progress", None, flags)
        if expanded:
            imgui.text(self.status_msg)
            imgui.progress_bar(self.progress, (-1, 0))
            if self.error_msg:
                imgui.text_colored((1.0, 0.3, 0.3, 1.0), self.error_msg)
        imgui.end()

    def _update_playing(self):
        # Player controls
        pos = self.player.get_position()
        dur = self.player.duration
        actions = self.player_controls.draw(self.player.is_playing(), pos, dur)

        if actions["toggle"]:
            self.player.toggle()
        if actions["seek"] is not None:
            self.player.seek(actions["seek"])

        # Settings
        self.settings_panel.draw()

        # Library panel (allows switching songs while playing)
        lib_action = self.library_panel.draw(self.library.get_all())
        if lib_action:
            self._handle_library_action(lib_action)

        # Back button
        viewport = imgui.get_main_viewport()
        vp_size = viewport.size
        imgui.set_next_window_pos((20, vp_size.y - 100), imgui.Cond_.always)
        imgui.set_next_window_size((100, 35), imgui.Cond_.always)
        imgui.set_next_window_bg_alpha(0.5)
        flags = (
            imgui.WindowFlags_.no_title_bar
            | imgui.WindowFlags_.no_resize
            | imgui.WindowFlags_.no_move
            | imgui.WindowFlags_.no_scrollbar
        )
        expanded, _ = imgui.begin("##back", None, flags)
        if expanded:
            if imgui.button("Back"):
                self.player.stop()
                self.analysis = None
                self.state = AppState.SEARCH
        imgui.end()

    def _render(self):
        """Render visualization."""
        settings = self.settings_panel.settings
        features = self._get_current_features()

        now = time.time()
        delta_time = min(now - self._last_frame_time, 0.05)
        self._last_frame_time = now

        segments = self.ball_gen.generate(
            self.width, self.height, now, delta_time, features, settings
        )

        # Add flash from onset to settings for renderer
        render_settings = dict(settings)
        render_settings["flash"] = features.get("onset_strength", 0.0)

        self.renderer.render(segments, render_settings, delta_time)

    def _get_current_features(self) -> dict:
        """Get audio features for current playback position."""
        if self.analysis is not None and self.state == AppState.PLAYING:
            pos = self.player.get_position()
            return self.analysis.get_features(pos)

        # Idle animation features — steady tight ball, no pulsing
        t = time.time() - self._idle_time_start
        return {
            "rms": 0.12,
            "onset_strength": 0.0,
            "beat_pulse": 0.0,
            "spectral_centroid": 0.5,
            "spectral_bandwidth": 0.3,
            "spectral_flux": 0.1,
            "bass_energy": 0.12,
            "mid_energy": 0.2,
            "treble_energy": 0.15,
            "kick_pulse": 0.0,
            "snare_pulse": 0.0,
            "hihat_pulse": 0.0,
            "onset_sharpness": 0.0,
            "anticipation_factor": 0.0,
            "explosion_factor": 0.0,
        }

    def _cleanup(self):
        self.player.cleanup()
        self.renderer.cleanup()
        self.imgui_impl.shutdown()
        imgui.destroy_context()
        glfw.terminate()
