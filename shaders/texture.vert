#version 330

in vec3 in_pos;
in vec2 in_uv;
uniform mat4 u_mvp;
uniform mat4 u_model;
out vec2 i_uv;
out vec3 v_normal;
out vec3 v_rel;

void main() {
    gl_Position = u_mvp * vec4(in_pos, 1.0);
    v_normal = normalize(mat3(u_model) * in_pos); // cut out translation before transformation
    v_rel = mat3(u_model) * in_pos; // position relative to the centre of the body, in AU
    i_uv = in_uv;
}
