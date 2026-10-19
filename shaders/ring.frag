#version 330

in vec2 i_uv;
in vec3 v_rel;
uniform sampler2D u_texture;
uniform vec3 u_normal;
uniform vec3 u_light_dir;
uniform float u_sphere_radius;
out vec4 out_color;

void main() {
    float b = dot(v_rel, u_light_dir);
    float h = sqrt(max(dot(v_rel, v_rel) - b * b, 0.0));
    float lit = b < 0.0 ? smoothstep(0.97 * u_sphere_radius, 1.03 * u_sphere_radius, h) : 1.0;
    vec4 ring = texture(u_texture, i_uv);
    float diffuse = abs(dot(normalize(u_normal), u_light_dir));
    out_color = vec4(ring.rgb * (0.1 + 0.9 * diffuse * lit), ring.a);
}
