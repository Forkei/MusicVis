"""OpenGL rendering pipeline: lines -> bloom -> composite."""

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
    """Handles the full rendering pipeline with bloom."""

    def __init__(self, ctx: moderngl.Context, width: int, height: int):
        self.ctx = ctx
        self.width = width
        self.height = height
        self._build_shaders()
        self._build_framebuffers(width, height)
        self._build_geometry()

        self._max_segments = 50000
        # Ring-only instance buffer (for CPU-generated waveform ring)
        self._ring_buffer = self.ctx.buffer(reserve=1024 * 8 * 4)

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

        # Multi-scale bloom: 3 scales at 1/2, 1/4, 1/8 resolution
        self.bloom_scales = []
        for divisor in [2, 4, 8]:
            bw, bh = max(1, w // divisor), max(1, h // divisor)
            texA = self.ctx.texture((bw, bh), 4, dtype="f2")
            texB = self.ctx.texture((bw, bh), 4, dtype="f2")
            fboA = self.ctx.framebuffer(color_attachments=[texA])
            fboB = self.ctx.framebuffer(color_attachments=[texB])
            self.bloom_scales.append((texA, texB, fboA, fboB, bw, bh))

        # Anamorphic flare FBO at 1/2 resolution
        aw, ah = max(1, w // 2), max(1, h // 2)
        self.anamorphic_texA = self.ctx.texture((aw, ah), 4, dtype="f2")
        self.anamorphic_texB = self.ctx.texture((aw, ah), 4, dtype="f2")
        self.anamorphic_fboA = self.ctx.framebuffer(color_attachments=[self.anamorphic_texA])
        self.anamorphic_fboB = self.ctx.framebuffer(color_attachments=[self.anamorphic_texB])
        self.anamorphic_w = aw
        self.anamorphic_h = ah

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
        objs = [self.scene_tex, self.bright_tex, self.scene_fbo]
        for texA, texB, fboA, fboB, _, _ in self.bloom_scales:
            objs.extend([texA, texB, fboA, fboB])
        objs.extend([
            self.anamorphic_texA, self.anamorphic_texB,
            self.anamorphic_fboA, self.anamorphic_fboB,
        ])
        for obj in objs:
            obj.release()

        self._build_framebuffers(width, height)

    def render(self, segment_buffer: moderngl.Buffer, compute_count: int,
               ring_segments: np.ndarray | None, settings: dict,
               delta_time: float = 0.016):
        """Render one frame.

        Args:
            segment_buffer: SSBO containing compute-generated segments (used as instance VBO)
            compute_count: Number of segments from compute shader
            ring_segments: Optional (N, 8) float32 array of ring segments from CPU
            settings: dict with rendering params
            delta_time: time since last frame in seconds
        """
        # --- Pass 1: Render lines to HDR FBO ---
        self.scene_fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)

        bloom_tint = settings.get("bloom_tint", (0.3, 0.6, 1.0))
        depth_fog = settings.get("depth_fog", 0.4)

        # Set shared uniforms
        self.line_prog["u_resolution"].value = (float(self.width), float(self.height))
        self.line_prog["u_flash"].value = settings.get("flash", 0.0)
        self.line_prog["u_base_brightness"].value = settings.get("brightness", 1.2)
        self.line_prog["u_global_hue"].value = settings.get("global_hue", 0.5)
        self.line_prog["u_depth_fog"].value = depth_fog
        self.line_prog["u_fog_color"].value = bloom_tint

        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE

        # Render compute-generated segments from SSBO
        if compute_count > 0 and segment_buffer is not None:
            vao = self.ctx.vertex_array(
                self.line_prog,
                [
                    (self._quad_vbo, "2f", "in_position"),
                    (segment_buffer, "2f 2f 1f 1f 1f 1f /i",
                     "in_p0", "in_p1", "in_thickness", "in_brightness",
                     "in_hue", "in_depth"),
                ],
            )
            vao.render(moderngl.TRIANGLES, instances=compute_count)
            vao.release()

        # Render ring segments (CPU-generated)
        if ring_segments is not None and len(ring_segments) > 0:
            n_ring = len(ring_segments)
            data = ring_segments.astype("f4").tobytes()
            needed = n_ring * 8 * 4
            if self._ring_buffer.size < needed:
                self._ring_buffer.release()
                self._ring_buffer = self.ctx.buffer(reserve=needed)
            self._ring_buffer.orphan(needed)
            self._ring_buffer.write(data)

            ring_vao = self.ctx.vertex_array(
                self.line_prog,
                [
                    (self._quad_vbo, "2f", "in_position"),
                    (self._ring_buffer, "2f 2f 1f 1f 1f 1f /i",
                     "in_p0", "in_p1", "in_thickness", "in_brightness",
                     "in_hue", "in_depth"),
                ],
            )
            ring_vao.render(moderngl.TRIANGLES, instances=n_ring)
            ring_vao.release()

        self.ctx.disable(moderngl.BLEND)

        # --- Pass 2: Multi-scale bloom ---
        bloom_iterations = 4
        bloom_intensity = settings.get("bloom_intensity", 2.5)

        for scale_idx, (texA, texB, fboA, fboB, bw, bh) in enumerate(self.bloom_scales):
            fboA.use()
            self.ctx.viewport = (0, 0, bw, bh)
            self.ctx.clear(0.0, 0.0, 0.0, 1.0)
            self.bright_tex.use(0)
            self.blur_prog["u_texture"].value = 0
            self.blur_prog["u_direction"].value = (0.0, 0.0)
            self._blur_vao.render(moderngl.TRIANGLES)

            for i in range(bloom_iterations):
                fboB.use()
                texA.use(0)
                self.blur_prog["u_texture"].value = 0
                self.blur_prog["u_direction"].value = (1.0 / bw, 0.0)
                self._blur_vao.render(moderngl.TRIANGLES)

                fboA.use()
                texB.use(0)
                self.blur_prog["u_texture"].value = 0
                self.blur_prog["u_direction"].value = (0.0, 1.0 / bh)
                self._blur_vao.render(moderngl.TRIANGLES)

        # --- Pass 2b: Anamorphic lens flare ---
        anamorphic_intensity = settings.get("anamorphic_flare", 0.3)
        aw, ah = self.anamorphic_w, self.anamorphic_h

        self.anamorphic_fboA.use()
        self.ctx.viewport = (0, 0, aw, ah)
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.bloom_scales[0][0].use(0)
        self.blur_prog["u_texture"].value = 0
        self.blur_prog["u_direction"].value = (0.0, 0.0)
        self._blur_vao.render(moderngl.TRIANGLES)

        for i in range(8):
            self.anamorphic_fboB.use()
            self.anamorphic_texA.use(0)
            self.blur_prog["u_texture"].value = 0
            self.blur_prog["u_direction"].value = (2.0 / aw, 0.0)
            self._blur_vao.render(moderngl.TRIANGLES)

            self.anamorphic_fboA.use()
            self.anamorphic_texB.use(0)
            self.blur_prog["u_texture"].value = 0
            self.blur_prog["u_direction"].value = (2.0 / aw, 0.0)
            self._blur_vao.render(moderngl.TRIANGLES)

        # --- Pass 3: Composite to screen ---
        self.ctx.viewport = (0, 0, self.width, self.height)
        self.ctx.screen.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)

        self.scene_tex.use(0)
        self.bloom_scales[0][0].use(1)
        self.bloom_scales[1][0].use(2)
        self.bloom_scales[2][0].use(3)
        self.anamorphic_texA.use(4)

        self.composite_prog["u_scene"].value = 0
        self.composite_prog["u_bloom_0"].value = 1
        self.composite_prog["u_bloom_1"].value = 2
        self.composite_prog["u_bloom_2"].value = 3
        self.composite_prog["u_anamorphic"].value = 4
        self.composite_prog["u_bloom_intensity"].value = bloom_intensity
        self.composite_prog["u_anamorphic_intensity"].value = anamorphic_intensity
        self.composite_prog["u_bloom_tint"].value = bloom_tint

        self._composite_vao.render(moderngl.TRIANGLES)

    def cleanup(self):
        """Release all GPU resources."""
        objs = [
            self.scene_tex, self.bright_tex, self.scene_fbo,
            self._quad_vbo, self._fsq_vbo, self._ring_buffer,
            self._blur_vao, self._composite_vao,
            self.anamorphic_texA, self.anamorphic_texB,
            self.anamorphic_fboA, self.anamorphic_fboB,
        ]
        for texA, texB, fboA, fboB, _, _ in self.bloom_scales:
            objs.extend([texA, texB, fboA, fboB])
        for obj in objs:
            try:
                obj.release()
            except Exception:
                pass
