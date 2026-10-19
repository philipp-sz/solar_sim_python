#version 330

// Applied on every vertex, transform vertices to final Screenspace
in vec3 in_pos;
uniform vec3 u_color;
uniform mat4 u_mvp;
out vec3 frag_color;

void main() {
    gl_Position = u_mvp * vec4(in_pos, 1.0);
    frag_color = u_color;
}
