#version 330 core

in float v_brightness;
in float v_cross;
in vec2 v_screen_pos;

layout(location = 0) out vec4 frag_color;
layout(location = 1) out vec4 bright_color;

uniform vec2 u_resolution;
uniform float u_flash;           // 0-1 extra flash from onsets
uniform float u_base_brightness; // Global brightness multiplier

void main() {
    float dist = abs(v_cross);

    // Very soft, fluid Gaussian falloff (electric glow look)
    float core = exp(-dist * dist * 3.0);
    float glow = exp(-dist * dist * 0.8);

    // Radial distance from screen center (0 at center, 1 at edge)
    vec2 center = u_resolution * 0.5;
    float radial_dist = length(v_screen_pos - center) / (min(u_resolution.x, u_resolution.y) * 0.5);
    radial_dist = clamp(radial_dist, 0.0, 1.0);

    // 4-stop radial gradient: white(0) -> cyan(0.3) -> blue(0.7) -> indigo(1.0)
    vec3 white  = vec3(1.0, 1.0, 1.0);
    vec3 cyan   = vec3(0.53, 0.8, 1.0);    // #88CCFF
    vec3 blue   = vec3(0.0, 0.5, 1.0);     // #0080FF
    vec3 indigo = vec3(0.18, 0.0, 0.53);   // #2E0087

    vec3 radial_color;
    if (radial_dist < 0.3) {
        radial_color = mix(white, cyan, radial_dist / 0.3);
    } else if (radial_dist < 0.7) {
        radial_color = mix(cyan, blue, (radial_dist - 0.3) / 0.4);
    } else {
        radial_color = mix(blue, indigo, (radial_dist - 0.7) / 0.3);
    }

    // Blend radial color with white for the core (hot center of each line)
    vec3 line_color = mix(radial_color, white, core * 0.7);
    vec3 color = line_color * glow * v_brightness;

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
