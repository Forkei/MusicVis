#version 430 core

in vec2 v_uv;
out vec4 frag_color;

uniform sampler2D u_scene;
uniform sampler2D u_bloom_0;       // 1/2 res (tight glow)
uniform sampler2D u_bloom_1;       // 1/4 res (medium spread)
uniform sampler2D u_bloom_2;       // 1/8 res (wide diffusion)
uniform sampler2D u_anamorphic;    // horizontal streak
uniform float u_bloom_intensity;
uniform float u_anamorphic_intensity;
uniform vec3 u_bloom_tint;

void main() {
    vec3 scene = texture(u_scene, v_uv).rgb;

    // Bloom: boost wider scales for softer, more atmospheric glow
    vec3 bloom = texture(u_bloom_0, v_uv).rgb * 0.4
               + texture(u_bloom_1, v_uv).rgb * 0.35
               + texture(u_bloom_2, v_uv).rgb * 0.25;
    vec3 flare = texture(u_anamorphic, v_uv).rgb;

    // Color-preserving tint: lerp toward tint color instead of multiplying
    // This keeps original bloom colors while gently shifting toward the tint
    bloom = mix(bloom, bloom * u_bloom_tint * 2.0, 0.3);
    flare = mix(flare, flare * u_bloom_tint * 2.0, 0.3);

    // Subtle radial background gradient (dark ambient, not pure black)
    vec2 center = v_uv - 0.5;
    float radial = length(center);
    vec3 bg_color = u_bloom_tint * 0.02 * (1.0 - radial * 1.2);
    bg_color = max(bg_color, vec3(0.0));

    // Combine: background + scene + bloom + flare
    vec3 color = bg_color + scene + bloom * u_bloom_intensity + flare * u_anamorphic_intensity;

    // ACES-inspired tonemap (better color preservation than Reinhard)
    // Attempt to preserve saturation during compression
    vec3 x = color;
    color = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14);
    color = clamp(color, 0.0, 1.0);

    // Gamma correction
    color = pow(color, vec3(1.0 / 2.2));

    // Vignette: stronger to frame the center
    vec2 q = v_uv - 0.5;
    color *= 1.0 - dot(q, q) * 0.7;

    frag_color = vec4(color, 1.0);
}
