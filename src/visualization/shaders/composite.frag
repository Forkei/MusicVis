#version 330 core

in vec2 v_uv;
out vec4 frag_color;

uniform sampler2D u_scene;
uniform sampler2D u_bloom;
uniform float u_bloom_intensity;
uniform vec3 u_bloom_tint;

void main() {
    vec3 scene = texture(u_scene, v_uv).rgb;
    vec3 bloom = texture(u_bloom, v_uv).rgb;

    // Apply blue tint to bloom
    bloom *= u_bloom_tint;

    // Additive blend
    vec3 color = scene + bloom * u_bloom_intensity;

    // Reinhard tonemap
    color = color / (color + 1.0);

    // Gamma correction
    color = pow(color, vec3(1.0 / 2.2));

    // Subtle vignette
    vec2 q = v_uv - 0.5;
    color *= 1.0 - dot(q, q) * 0.5;

    frag_color = vec4(color, 1.0);
}
