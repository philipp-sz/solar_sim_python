#version 330

in vec3 in_pos;
uniform mat4 u_sky_mvp;
out vec3 v_dir;


void main() {
    gl_Position = (u_sky_mvp * vec4(in_pos, 1.0)).xyww;
    v_dir = in_pos;
}
