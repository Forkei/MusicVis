#version 430 core

in float v_brightness;
in float v_cross;
in vec2 v_screen_pos;
in float v_hue;
in float v_depth;

layout(location = 0) out vec4 frag_color;
layout(location = 1) out vec4 bright_color;

uniform vec2 u_resolution;
uniform float u_flash;           // 0-1 extra flash from onsets
uniform float u_base_brightness; // Global brightness multiplier
uniform float u_global_hue;      // From spectral centroid (0-1)
uniform float u_depth_fog;       // 0-1 fog intensity
uniform vec3 u_fog_color;        // Fog color (matches bloom tint)
uniform float u_saturation;      // Director-driven saturation
uniform float u_rms;             // Audio RMS energy (0-1)

// HSV to RGB conversion
vec3 hsv2rgb(float h, float s, float v) {
    h = fract(h) * 6.0;
    float c = v * s;
    float x = c * (1.0 - abs(mod(h, 2.0) - 1.0));
    float m = v - c;
    vec3 rgb;
    if      (h < 1.0) rgb = vec3(c, x, 0.0);
    else if (h < 2.0) rgb = vec3(x, c, 0.0);
    else if (h < 3.0) rgb = vec3(0.0, c, x);
    else if (h < 4.0) rgb = vec3(0.0, x, c);
    else if (h < 5.0) rgb = vec3(x, 0.0, c);
    else              rgb = vec3(c, 0.0, x);
    return rgb + m;
}

void main() {
    float dist = abs(v_cross);

    // Very soft, fluid Gaussian falloff (electric glow look)
    float core = exp(-dist * dist * 3.0);
    float glow = exp(-dist * dist * 0.8);

    // Hue-based coloring: global centroid hue + per-instance offset
    float hue = fract(u_global_hue + v_hue);

    // Saturation: tonal certainty base + depth + flash
    float sat = u_saturation + v_depth * 0.1 + u_flash * 0.05;
    // Energy-driven HSV value: quiet=dim, loud=bright
    float hsv_value = 0.55 + u_rms * 0.45;
    vec3 base_color = hsv2rgb(hue, clamp(sat, 0.3, 1.0), hsv_value);

    // Blend toward bright version of the same hue for the core (preserves color)
    float hot_sat = clamp(sat * 0.65, 0.2, 0.7);
    vec3 hot_color = hsv2rgb(hue, hot_sat, hsv_value);
    vec3 line_color = mix(base_color, hot_color, core * 0.4);
    vec3 color = line_color * glow * v_brightness;

    // Depth fog: far segments (low depth) add subtle ambient glow
    float fog = pow(1.0 - v_depth, 2.0) * u_depth_fog;
    color = mix(color, u_fog_color * 0.5, fog);

    // Apply flash
    color *= 1.0 + u_flash * 2.0;
    color *= u_base_brightness;

    float alpha = glow * v_brightness;
    frag_color = vec4(color, alpha);

    // Bright pass for bloom: only bloom the truly bright parts
    float lum = dot(color, vec3(0.2126, 0.7152, 0.0722));
    float bright = smoothstep(0.5, 1.5, lum);
    bright_color = vec4(color * bright, 1.0);
}
