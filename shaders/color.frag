#version 330

// Decide color and depth of the rasterized pixels in each shape
in vec3 frag_color;
out vec4 out_color;

void main() {
    out_color = vec4(frag_color, 1.0);
}
