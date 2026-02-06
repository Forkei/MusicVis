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
    vec3 bloom = texture(u_bloom_0, v_uv).rgb * 0.5
               + texture(u_bloom_1, v_uv).rgb * 0.35
               + texture(u_bloom_2, v_uv).rgb * 0.15;
    vec3 flare = texture(u_anamorphic, v_uv).rgb;

    // Apply tint
    bloom *= u_bloom_tint;
    flare *= u_bloom_tint;

    // Combine
    vec3 color = scene + bloom * u_bloom_intensity + flare * u_anamorphic_intensity;

    // Reinhard tonemap
    color = color / (color + 1.0);

    // Gamma correction
    color = pow(color, vec3(1.0 / 2.2));

    // Subtle vignette
    vec2 q = v_uv - 0.5;
    color *= 1.0 - dot(q, q) * 0.5;

    frag_color = vec4(color, 1.0);
}
