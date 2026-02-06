#version 430 core

in vec2 v_uv;
out vec4 frag_color;

uniform sampler2D u_prev_trail;
uniform sampler2D u_current_scene;
uniform float u_decay;

void main() {
    vec3 prev = texture(u_prev_trail, v_uv).rgb;
    vec3 current = texture(u_current_scene, v_uv).rgb;

    // Fade previous trail and add current frame
    vec3 result = prev * u_decay + current;

    frag_color = vec4(result, 1.0);
}
