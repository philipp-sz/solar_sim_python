#version 330

in vec2 i_uv;
in vec3 v_normal;
in vec3 v_rel;
uniform sampler2D u_texture;
uniform sampler2D u_night_texture;
uniform sampler2D u_clouds_texture;
uniform sampler2D u_ring_texture;
uniform vec3 u_light_dir;
uniform float u_lit; // is illuminated (1) or glows on its own (0)
uniform float u_has_night; // earth night side (1) or day side (0)
uniform float u_clouds_on;
uniform float u_ring_shadow_on; // 1 only for the body that carries the ring
uniform vec3 u_ring_normal; // normal of the ring plane in world space
uniform vec2 u_ring_radii; // inner and outer radius of the ring in AU
out vec4 out_color;

void main() {
    vec3 n = normalize(v_normal);

    float ring_light = 1.0; // share of the sunlight that passes the ring
    if (u_ring_shadow_on > 0.5) {
        float ln = dot(u_light_dir, u_ring_normal);
        if (abs(ln) > 1e-4) { // ray to the Sun is not parallel to the ring plane
            float t = -dot(v_rel, u_ring_normal) / ln; // distance along the ray to the ring plane
            if (t > 0.0) { // the ring plane lies on the Sun side of this point
                float rho = length(v_rel + t * u_light_dir); // distance of the crossing point from the planet centre
                float u = (rho - u_ring_radii.x) / (u_ring_radii.y - u_ring_radii.x);
                if (u > 0.0 && u < 1.0) ring_light = 1.0 - texture(u_ring_texture, vec2(u, 0.5)).a;
            }
        }
    }
    float diffuse = max(dot(n, u_light_dir), 0.0) * ring_light;

    vec3 day = texture(u_texture, i_uv).rgb;
    float alpha = texture(u_texture, i_uv).a;
    vec3 night = texture(u_night_texture, i_uv).rgb;
    float cloud = texture(u_clouds_texture, i_uv).r * u_clouds_on;
    day = mix(day, vec3(1.0), cloud);
    night = night * (1.0 - cloud);

    vec3 lit = day * (0.1 + 0.9 * diffuse);
    float day_amount = smoothstep(0.0, 0.25, diffuse);
    vec3 earth = mix(night, lit, day_amount);
    vec3 surface = mix(lit, earth, u_has_night);
    out_color = vec4(mix(day, surface, u_lit), alpha);
}
