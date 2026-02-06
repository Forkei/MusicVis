#version 330 core

// Per-vertex (base quad: 2 triangles making a unit quad)
in vec2 in_position;  // (-0.5..0.5, -0.5..0.5)

// Per-instance (one per line segment)
in vec2 in_p0;         // Start point
in vec2 in_p1;         // End point
in float in_thickness; // Line width in screen coords
in float in_brightness;// 0-1

out float v_brightness;
out float v_cross;     // -1..1 across the line width
out vec2 v_screen_pos; // Pixel-space position for radial gradient

uniform vec2 u_resolution;

void main() {
    vec2 dir = in_p1 - in_p0;
    float len = length(dir);
    if (len < 0.0001) {
        gl_Position = vec4(-10.0, -10.0, 0.0, 1.0);
        return;
    }

    vec2 tangent = dir / len;
    vec2 normal = vec2(-tangent.y, tangent.x);

    // in_position.x: 0..1 along the segment, in_position.y: -0.5..0.5 across
    vec2 pos = in_p0 + tangent * (in_position.x + 0.5) * len
             + normal * in_position.y * in_thickness;

    v_screen_pos = pos;

    // Convert to clip space (-1..1)
    vec2 ndc = pos / u_resolution * 2.0 - 1.0;

    gl_Position = vec4(ndc, 0.0, 1.0);
    v_brightness = in_brightness;
    v_cross = in_position.y * 2.0;  // -1..1
}
