#version 430 core

in vec2 v_uv;
out vec4 frag_color;

uniform float u_time;
uniform float u_arousal;
uniform vec3 u_bloom_tint;
uniform vec2 u_resolution;
uniform float u_bg_boost;   // Extra intensity from director (calm sections)
uniform float u_valence;    // 0=dark/cool, 1=bright/warm — color temperature
uniform float u_rms;        // Current energy level

// Simple hash-based noise (no texture needed)
float hash(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f); // smoothstep

    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));

    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float fbm(vec2 p) {
    float value = 0.0;
    float amplitude = 0.5;
    for (int i = 0; i < 4; i++) {
        value += amplitude * noise(p);
        p *= 2.0;
        amplitude *= 0.5;
    }
    return value;
}

void main() {
    vec2 uv = v_uv;
    float aspect = u_resolution.x / u_resolution.y;

    // Slow animated coordinate
    vec2 pos = vec2(uv.x * aspect, uv.y) * 3.0;
    pos += u_time * 0.02; // very slow drift

    // Two FBM layers with different motion for depth
    float n1 = fbm(pos + vec2(u_time * 0.01, u_time * 0.015));
    float n2 = fbm(pos * 1.5 - vec2(u_time * 0.008, u_time * 0.012));

    float nebula = n1 * 0.6 + n2 * 0.4;

    // Radial falloff: stronger at edges, subtle at center (donut shape)
    vec2 center = uv - 0.5;
    float radial = length(center);
    float ring = smoothstep(0.1, 0.5, radial) * smoothstep(0.9, 0.5, radial);

    // Intensity driven by arousal + director boost + quiet fill
    float quiet_fill = max(0.0, 1.0 - u_rms * 3.0) * 0.02;  // More visible when quiet
    float intensity = (u_arousal * 0.03 + u_bg_boost + quiet_fill) * ring;

    // Color: modulate bloom tint
    vec3 color = u_bloom_tint * nebula * intensity;

    // Subtle color temperature shift from valence (warm/cool)
    vec3 warm = vec3(1.0, 0.7, 0.4);
    vec3 cool = vec3(0.4, 0.7, 1.0);
    vec3 temp_tint = mix(cool, warm, u_valence);
    color = mix(color, color * temp_tint, 0.3);

    frag_color = vec4(color, 1.0);
}
