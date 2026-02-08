#version 430 core

in vec2 v_uv;
out vec4 frag_color;

uniform sampler2D u_prev_trail;
uniform sampler2D u_current_scene;
uniform float u_decay;
uniform float u_dt;

void main() {
    vec3 prev = texture(u_prev_trail, v_uv).rgb;
    vec3 current = texture(u_current_scene, v_uv).rgb;

    // Frame-rate independent decay: pow(decay, dt * 60) normalizes to 60fps baseline
    float adjusted_decay = pow(u_decay, u_dt * 60.0);
    vec3 result = prev * adjusted_decay + current;

    frag_color = vec4(result, 1.0);
}
