#version 330

in vec3 v_dir;
uniform sampler2D u_sky;
uniform float u_brightness;
uniform mat3 u_ecl_to_gal;
out vec4 out_color;
const float PI = 3.14159265;

void main() {
    vec3 d = normalize(u_ecl_to_gal * v_dir);
    float lon = atan(d.y, d.x);
    float lat = asin(clamp(d.z, -1.0, 1.0));
    vec2 uv = vec2(0.5 - lon / (2.0 * PI), 0.5 + lat / PI);
    out_color = vec4(texture(u_sky, uv).rgb * u_brightness, 1.0);
}
