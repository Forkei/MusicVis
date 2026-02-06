"""OpenGL rendering pipeline: lines -> trail accumulation -> bloom -> composite."""

import os
import math
import numpy as np
import moderngl


SHADER_DIR = os.path.join(os.path.dirname(__file__), "shaders")


def _load_shader(name: str) -> str:
    path = os.path.join(SHADER_DIR, name)
    with open(path, "r") as f:
        return f.read()


class Renderer:
    """Handles the full rendering pipeline with trail persistence and bloom."""

    def __init__(self, ctx: moderngl.Context, width: int, height: int):
        self.ctx = ctx
        self.width = width
        self.height = height
        self._build_shaders()
        self._build_framebuffers(width, height)
        self._build_geometry()

        self._max_segments = 50000
        self._instance_buffer = self.ctx.buffer(reserve=self._max_segments * 6 * 4)

    def _build_shaders(self):
        # Line rendering program
        self.line_prog = self.ctx.program(
            vertex_shader=_load_shader("line.vert"),
            fragment_shader=_load_shader("line.frag"),
        )

        # Blur program
        self.blur_prog = self.ctx.program(
            vertex_shader=_load_shader("blur.vert"),
            fragment_shader=_load_shader("blur.frag"),
        )

        # Composite program
        self.composite_prog = self.ctx.program(
            vertex_shader=_load_shader("blur.vert"),
            fragment_shader=_load_shader("composite.frag"),
        )

    def _build_framebuffers(self, w: int, h: int):
        # HDR scene FBO with MRT (scene + bright pass)
        self.scene_tex = self.ctx.texture((w, h), 4, dtype="f2")
        self.bright_tex = self.ctx.texture((w, h), 4, dtype="f2")
        self.scene_fbo = self.ctx.framebuffer(
            color_attachments=[self.scene_tex, self.bright_tex]
        )

        # Bloom ping-pong at half resolution
        bw, bh = max(1, w // 2), max(1, h // 2)
        self.bloom_texA = self.ctx.texture((bw, bh), 4, dtype="f2")
        self.bloom_texB = self.ctx.texture((bw, bh), 4, dtype="f2")
        self.bloom_fboA = self.ctx.framebuffer(color_attachments=[self.bloom_texA])
        self.bloom_fboB = self.ctx.framebuffer(color_attachments=[self.bloom_texB])

    def _build_geometry(self):
        # Base quad for instanced line rendering: 6 vertices (2 triangles)
        quad = np.array([
            -0.5, -0.5,
             0.5, -0.5,
             0.5,  0.5,
            -0.5, -0.5,
             0.5,  0.5,
            -0.5,  0.5,
        ], dtype="f4")
        self._quad_vbo = self.ctx.buffer(quad)

        # Fullscreen quad for post-processing (-1..1)
        fsq = np.array([
            -1, -1,  1, -1,  1, 1,
            -1, -1,  1,  1, -1, 1,
        ], dtype="f4")
        self._fsq_vbo = self.ctx.buffer(fsq)

        # VAOs for post-processing passes
        self._blur_vao = self.ctx.vertex_array(
            self.blur_prog, [(self._fsq_vbo, "2f", "in_position")]
        )
        self._composite_vao = self.ctx.vertex_array(
            self.composite_prog, [(self._fsq_vbo, "2f", "in_position")]
        )

    def resize(self, width: int, height: int):
        """Resize framebuffers on window resize."""
        if width == self.width and height == self.height:
            return
        self.width = width
        self.height = height

        # Release old
        for obj in [
            self.scene_tex, self.bright_tex, self.scene_fbo,
            self.bloom_texA, self.bloom_texB, self.bloom_fboA, self.bloom_fboB,
        ]:
            obj.release()

        self._build_framebuffers(width, height)

    def render(self, segments: np.ndarray, settings: dict, delta_time: float = 0.016):
        """Render one frame.

        Args:
            segments: (N, 6) float32 array of line segments
            settings: dict with rendering params
            delta_time: time since last frame in seconds
        """
        n_segments = len(segments) if segments is not None else 0

        # --- Pass 1: Render lines to HDR FBO ---
        self.scene_fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)

        if n_segments > 0:
            if n_segments > self._max_segments:
                segments = segments[:self._max_segments]
                n_segments = self._max_segments

            data = segments.astype("f4").tobytes()
            self._instance_buffer.orphan(len(data))
            self._instance_buffer.write(data)

            vao = self.ctx.vertex_array(
                self.line_prog,
                [
                    (self._quad_vbo, "2f", "in_position"),
                    (self._instance_buffer, "2f 2f 1f 1f /i",
                     "in_p0", "in_p1", "in_thickness", "in_brightness"),
                ],
            )

            self.line_prog["u_resolution"].value = (float(self.width), float(self.height))
            self.line_prog["u_flash"].value = settings.get("flash", 0.0)
            self.line_prog["u_base_brightness"].value = settings.get("brightness", 1.2)

            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
            vao.render(moderngl.TRIANGLES, instances=n_segments)
            self.ctx.disable(moderngl.BLEND)
            vao.release()

        # --- Pass 2: Bloom (Gaussian blur on bright pass) ---
        bloom_iterations = 6
        bloom_intensity = settings.get("bloom_intensity", 2.5)
        bloom_tint = settings.get("bloom_tint", (0.3, 0.6, 1.0))

        bw = max(1, self.width // 2)
        bh = max(1, self.height // 2)

        # Copy bright pass to bloom A
        self.bloom_fboA.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.bright_tex.use(0)
        self.blur_prog["u_texture"].value = 0
        self.blur_prog["u_direction"].value = (0.0, 0.0)
        self._blur_vao.render(moderngl.TRIANGLES)

        # Ping-pong blur
        for i in range(bloom_iterations):
            # Horizontal
            self.bloom_fboB.use()
            self.bloom_texA.use(0)
            self.blur_prog["u_texture"].value = 0
            self.blur_prog["u_direction"].value = (1.0 / bw, 0.0)
            self._blur_vao.render(moderngl.TRIANGLES)

            # Vertical
            self.bloom_fboA.use()
            self.bloom_texB.use(0)
            self.blur_prog["u_texture"].value = 0
            self.blur_prog["u_direction"].value = (0.0, 1.0 / bh)
            self._blur_vao.render(moderngl.TRIANGLES)

        # --- Pass 3: Composite to screen ---
        self.ctx.screen.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)

        # Direct scene (no trail persistence)
        self.scene_tex.use(0)
        self.bloom_texA.use(1)
        self.composite_prog["u_scene"].value = 0
        self.composite_prog["u_bloom"].value = 1
        self.composite_prog["u_bloom_intensity"].value = bloom_intensity
        self.composite_prog["u_bloom_tint"].value = bloom_tint

        self._composite_vao.render(moderngl.TRIANGLES)

    def cleanup(self):
        """Release all GPU resources."""
        for obj in [
            self.scene_tex, self.bright_tex, self.scene_fbo,
            self.bloom_texA, self.bloom_texB, self.bloom_fboA, self.bloom_fboB,
            self._quad_vbo, self._fsq_vbo, self._instance_buffer,
            self._blur_vao, self._composite_vao,
        ]:
            try:
                obj.release()
            except Exception:
                pass
