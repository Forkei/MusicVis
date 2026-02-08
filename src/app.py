"""Main application: GLFW window + moderngl + imgui + audio integration."""

import multiprocessing
import os
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
from src.audio.analyzer import analyze as audio_analyze, analyze_subprocess, AnalysisResult
from src.audio.player import AudioPlayer
from src.visualization.renderer import Renderer
from src.visualization.energy_ball import EnergyBallGenerator
from src.visualization.director import MusicalDirector
from src.visualization.particles import ParticleSystem, MAX_PARTICLES
from src.audio.library import SongLibrary
from src.ui.search_panel import SearchPanel
from src.ui.player_controls import PlayerControls
from src.ui.settings_panel import SettingsPanel
from src.ui.library_panel import LibraryPanel
from src.ui.preset_manager import PresetManager


class AppState(Enum):
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

        self.state = AppState.SEARCH

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

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
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

        # Renderer + generator + director + particles
        self.renderer = Renderer(self.ctx, width, height)
        self.ball_gen = EnergyBallGenerator(self.ctx)
        self.director = MusicalDirector()
        self.particles = ParticleSystem(self.ctx)

        # UI panels
        self.search_panel = SearchPanel()
        self.player_controls = PlayerControls()
        self.settings_panel = SettingsPanel()
        self.preset_manager = PresetManager()
        self.settings_panel.set_preset_manager(self.preset_manager)

        # Resize callback
        glfw.set_framebuffer_size_callback(self.window, self._on_resize)

        # Key callback for fullscreen toggle (chain with ImGui's)
        self._imgui_key_callback = glfw.set_key_callback(self.window, self._on_key)

        # Fullscreen state
        self._is_fullscreen = False
        self._windowed_pos = glfw.get_window_pos(self.window)
        self._windowed_size = (width, height)

        # Idle features for when no song is loaded
        self._idle_time_start = time.time()
        self._last_directed_settings = {}

        # App start time (for relative time in shaders — 32-bit float precision)
        self._start_time = time.time()

        # Delta time tracking
        self._last_frame_time = time.time()

    def _on_resize(self, window, width, height):
        if width > 0 and height > 0:
            self.width = width
            self.height = height
            self.ctx.viewport = (0, 0, width, height)
            self.renderer.resize(width, height)

    def _on_key(self, window, key, scancode, action, mods):
        # Forward to ImGui's key callback first
        if self._imgui_key_callback:
            self._imgui_key_callback(window, key, scancode, action, mods)

        # Don't handle fullscreen shortcuts when ImGui wants keyboard
        io = imgui.get_io()
        if io.want_capture_keyboard:
            return
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_F11 or (key == glfw.KEY_F and mods == 0):
            self._toggle_fullscreen()
        elif key == glfw.KEY_ESCAPE and self._is_fullscreen:
            self._toggle_fullscreen()

    def _toggle_fullscreen(self):
        if self._is_fullscreen:
            # Exit fullscreen — restore windowed position/size
            x, y = self._windowed_pos
            w, h = self._windowed_size
            glfw.set_window_monitor(self.window, None, x, y, w, h, 0)
            self._is_fullscreen = False
        else:
            # Enter fullscreen — save current position/size
            self._windowed_pos = glfw.get_window_pos(self.window)
            self._windowed_size = glfw.get_window_size(self.window)
            monitor = glfw.get_primary_monitor()
            mode = glfw.get_video_mode(monitor)
            glfw.set_window_monitor(
                self.window, monitor, 0, 0,
                mode.size.width, mode.size.height, mode.refresh_rate
            )
            self._is_fullscreen = True

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
        if self.state == AppState.SEARCH:
            self._update_search()
        elif self.state == AppState.DOWNLOADING:
            self._update_downloading()
        elif self.state == AppState.ANALYZING:
            self._update_analyzing()
        elif self.state == AppState.PLAYING:
            self._update_playing()

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
        self.director.reset()

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
        self._analysis_wav_path = wav_path

        # Run analysis in a separate process to avoid GIL blocking the UI
        self._analysis_progress = multiprocessing.Value('d', 0.0)
        self._analysis_queue = multiprocessing.Queue()
        self._analysis_process = multiprocessing.Process(
            target=analyze_subprocess,
            args=(wav_path, self._analysis_progress, self._analysis_queue),
            daemon=True,
        )
        self._analysis_process.start()

    def _set_progress(self, p: float):
        self.progress = max(0.0, min(1.0, p))

    def _update_downloading(self):
        self._draw_progress_overlay()

    def _update_analyzing(self):
        # Poll progress from the subprocess
        if hasattr(self, '_analysis_progress'):
            self.progress = self._analysis_progress.value

        # Check if result is ready (non-blocking)
        if hasattr(self, '_analysis_queue'):
            import queue
            try:
                status, payload = self._analysis_queue.get_nowait()
            except queue.Empty:
                pass  # still running
            else:
                if status == "ok":
                    self.analysis = payload
                    if self._current_entry:
                        self.library.save_analysis(self._current_entry, payload)
                    self.player.load(self._analysis_wav_path)
                    self.player.play()
                    self.state = AppState.PLAYING
                else:
                    self.search_panel.error_msg = f"Analysis failed: {payload}"
                    self.state = AppState.SEARCH

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

        # Director status overlay
        if self.settings_panel.settings.get("director_enabled", True):
            self.settings_panel.draw_director_status(self._last_directed_settings)

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

        # Pass mel_spectrum frame for waveform ring
        if self.analysis is not None and self.state == AppState.PLAYING:
            pos = self.player.get_position()
            frame = self.analysis.frame_at(pos)
            features["mel_frame"] = self.analysis.mel_spectrum[:, frame]

        now = time.time()
        delta_time = min(now - self._last_frame_time, 0.05)
        self._last_frame_time = now

        # Musical Director: modulate settings based on musical context
        directed_settings = self.director.process(features, settings, delta_time)
        self._last_directed_settings = directed_settings

        # Compute global hue early so ball_gen can use it for ring coloring
        global_hue = features.get("spectral_centroid", 0.5) * 0.85 + 0.05
        global_hue += features.get("key_index", 0) / 12.0 * 0.08  # subtle personality per key
        directed_settings["global_hue"] = global_hue

        seg_buffer, compute_count, ring_segs = self.ball_gen.generate(
            self.width, self.height, now - self._start_time, delta_time, features, directed_settings
        )

        # Add flash and global hue from features to render settings
        render_settings = dict(directed_settings)
        raw_onset = features.get("onset_strength", 0.0)
        sharpness = features.get("onset_sharpness", 0.5)
        render_settings["flash"] = max(0.0, raw_onset - 0.35) / 0.65 * sharpness
        render_settings["global_hue"] = global_hue
        render_settings["arousal"] = features.get("arousal", 0.2)
        render_settings["valence"] = features.get("valence", 0.5)
        render_settings["rms"] = features.get("rms", 0.0)
        render_settings["time"] = now - self._start_time

        # Update particles (use shaken center so particles emit from visual position)
        ball_cx = self.width / 2 + self.ball_gen._shake_offset_x
        ball_cy = self.height / 2 + self.ball_gen._shake_offset_y
        ball_radius = self.ball_gen._current_radius
        self.particles.update(delta_time, features, directed_settings,
                              ball_cx, ball_cy, ball_radius)
        particle_buffer = self.particles.segment_buffer
        particle_count = MAX_PARTICLES

        # Render with or without trails
        trail_decay = directed_settings.get("trail_decay", 0.0)
        if trail_decay > 0.01:
            self.renderer.render_with_trail(
                seg_buffer, compute_count, ring_segs, render_settings,
                trail_decay, delta_time, particle_buffer, particle_count
            )
        else:
            self.renderer.render(
                seg_buffer, compute_count, ring_segs, render_settings,
                delta_time, particle_buffer, particle_count
            )

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
            "vocal_presence": 0.0,
            "groove_factor": 0.0,
            "section_id": 0,
            "tempo": 120.0,
            # Musical Director features (neutral defaults)
            "genre_id": 6,  # Pop (neutral)
            "genre_confidence": 0.0,
            "section_type": 0,
            "energy_trajectory": 0.0,
            "valence": 0.5,
            "arousal": 0.2,
            "climax_score": 0.0,
            "climax_type": 0,
            "lookahead_energy_delta": 0.0,
            "lookahead_section_change": 0.0,
            "lookahead_climax": 0.0,
            "rhythmic_density": 0.0,
        }

    def _cleanup(self):
        self.player.cleanup()
        self.ball_gen.cleanup()
        self.particles.cleanup()
        self.renderer.cleanup()
        self.imgui_impl.shutdown()
        imgui.destroy_context()
        glfw.terminate()
