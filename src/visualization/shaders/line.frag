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
    vec3 base_color = hsv2rgb(hue, 0.8, 1.0);

    // Blend with white for the core (hot center of each line)
    vec3 line_color = mix(base_color, vec3(1.0), core * 0.7);
    vec3 color = line_color * glow * v_brightness;

    // Depth fog: far segments (low depth) fade toward fog color
    float fog = pow(1.0 - v_depth, 2.0) * u_depth_fog;
    color = mix(color, u_fog_color * 0.3, fog);

    // Apply flash
    color *= 1.0 + u_flash * 2.0;
    color *= u_base_brightness;

    float alpha = glow * v_brightness;
    frag_color = vec4(color, alpha);

    // Bright pass for bloom (lower threshold for more glow)
    float lum = dot(color, vec3(0.2126, 0.7152, 0.0722));
    float bright = max(0.0, lum - 0.3) / (lum + 0.001);
    bright_color = vec4(color * bright, 1.0);
}
