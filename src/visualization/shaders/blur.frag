#version 430 core

in vec2 v_uv;
out vec4 frag_color;

uniform sampler2D u_texture;
uniform vec2 u_direction;  // (1/width, 0) or (0, 1/height)

void main() {
    // 9-tap Gaussian blur
    float weights[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

    vec3 result = texture(u_texture, v_uv).rgb * weights[0];

    for (int i = 1; i < 5; i++) {
        vec2 offset = u_direction * float(i);
        result += texture(u_texture, v_uv + offset).rgb * weights[i];
        result += texture(u_texture, v_uv - offset).rgb * weights[i];
    }

    frag_color = vec4(result, 1.0);
}
