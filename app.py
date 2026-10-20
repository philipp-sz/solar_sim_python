from typing import Any, cast

import numpy as np
import moderngl
import moderngl_window as mglw

import config as cfg
from utils import load_texture, load_program
from calculations.calc_graphics import look_at, perspective, calc_points, calc_translation, calc_scaling, rotation_x, rotation_z, calc_ring, calc_ecliptic_to_galactic
from calculations.calc_physics import correct_center_of_mass, step_velocity_verlet, calc_accelerations, calc_differences
from calculations.bodies import (masses, positions, velocities, body_colors,
                                display_radii, names, real_radii, orbital_periods,
                                rotation_periods, axial_tilts, texture_switchable,
                                day_night_textures)


class SolarSystemApp(mglw.WindowConfig):
    """Window that shows the simulated solar system seen by an orbit camera controlled with the mouse."""

    title = cfg.WINDOW_TITLE
    window_size = cfg.WINDOW_SIZE
    gl_version = cfg.GL_VERSION
    aspect_ratio = None # type: ignore

    def __init__(self, **kwargs: Any) -> None:
        """Create the window, the GPU objects and the camera state."""
        super().__init__(**kwargs)
        self.ctx.enable(moderngl.DEPTH_TEST) # Discard fragments that lie behind an already drawn surface
        self.num_bodies = len(names)
        self.num_trail_points = cfg.TRAIL_POINTS
        self.ring_body = names.index(cfg.RING_BODY_NAME)
        self.number_keys: dict[int, int | None] = {self.wnd.keys.NUMBER_0: None}
        self.number_keys.update({self.wnd.keys.NUMBER_0 + i: i - 1 for i in range(1, 10)})

        self._create_program() # Compile GLSL shaders
        self._init_uniforms()
        self._create_mesh() # Create vbo, ibo, vao
        self._init_camera()
        self._init_bodies()
        self._init_trail()

    def _create_program(self) -> None:
        """Compile the colour program (trails) and the texture program (bodies and ring) from the shaders folder."""
        self.color_prog = load_program(self.ctx, "color")
        self.texture_prog = load_program(self.ctx, "texture")
        self.ring_prog = load_program(self.ctx, "ring")
        self.sky_prog = load_program(self.ctx, "sky")

    def _init_uniforms(self) -> None:
        """Look up the shader uniforms once per program and set the ones that never change."""
        self._init_color_uniforms()
        self._init_texture_uniforms()
        self._init_ring_uniforms()
        self._init_sky_uniforms()

    def _init_color_uniforms(self) -> None:
        """Uniforms of the colour program (orbit trails)."""
        self.color_mvp = cast(moderngl.Uniform, self.color_prog["u_mvp"])
        self.color_rgb = cast(moderngl.Uniform, self.color_prog["u_color"])

    def _init_texture_uniforms(self) -> None:
        """Uniforms of the texture program (bodies); the texture slots never change."""
        self.texture_prog["u_texture"] = cfg.TEXTURE_UNIT_MAIN
        self.texture_prog["u_night_texture"] = cfg.TEXTURE_UNIT_NIGHT
        self.texture_prog["u_clouds_texture"] = cfg.TEXTURE_UNIT_CLOUDS
        self.texture_prog["u_ring_texture"] = cfg.TEXTURE_UNIT_RING
        self.tex_mvp = cast(moderngl.Uniform, self.texture_prog["u_mvp"])
        self.tex_model = cast(moderngl.Uniform, self.texture_prog["u_model"])
        self.tex_light_dir = cast(moderngl.Uniform, self.texture_prog["u_light_dir"])
        self.tex_lit = cast(moderngl.Uniform, self.texture_prog["u_lit"])
        self.tex_has_night = cast(moderngl.Uniform, self.texture_prog["u_has_night"])
        self.tex_clouds_on = cast(moderngl.Uniform, self.texture_prog["u_clouds_on"])
        self.tex_ring_shadow_on = cast(moderngl.Uniform, self.texture_prog["u_ring_shadow_on"])
        self.tex_ring_normal = cast(moderngl.Uniform, self.texture_prog["u_ring_normal"])
        self.tex_ring_radii = cast(moderngl.Uniform, self.texture_prog["u_ring_radii"])

    def _init_ring_uniforms(self) -> None:
        """Uniforms of the ring program."""
        self.ring_prog["u_texture"] = cfg.TEXTURE_UNIT_MAIN
        self.ring_mvp = cast(moderngl.Uniform, self.ring_prog["u_mvp"])
        self.ring_model = cast(moderngl.Uniform, self.ring_prog["u_model"])
        self.ring_normal = cast(moderngl.Uniform, self.ring_prog["u_normal"])
        self.ring_light_dir = cast(moderngl.Uniform, self.ring_prog["u_light_dir"])
        self.ring_sphere_radius = cast(moderngl.Uniform, self.ring_prog["u_sphere_radius"])

    def _init_sky_uniforms(self) -> None:
        """Uniforms of the sky program; the brightness and the ecliptic to galactic rotation never change."""
        self.sky_prog["u_sky"] = cfg.TEXTURE_UNIT_MAIN
        self.sky_prog["u_brightness"] = cfg.SKY_BRIGHTNESS
        cast(moderngl.Uniform, self.sky_prog["u_ecl_to_gal"]).write(calc_ecliptic_to_galactic().T.tobytes())
        self.sky_mvp = cast(moderngl.Uniform, self.sky_prog["u_sky_mvp"])

    def _init_textures(self, vertices: np.ndarray, textures: np.ndarray) -> None:
        """Create the textured sphere vertex array and load all body textures."""
        self.texture_idx = np.zeros(self.num_bodies, dtype=int)
        texture_input = np.hstack((vertices, textures))
        self.texture_vbo = self.ctx.buffer(texture_input.tobytes())
        self.texture_vao = self.ctx.vertex_array(
            self.texture_prog,
            [(self.texture_vbo, "3f 2f", "in_pos", "in_uv")],
            index_buffer=self.ibo,
            index_element_size=4 # Bytes per entry in index buffer
        )

        self.textures: list[list[moderngl.Texture]] = []
        for name in names:
            dir_name = cfg.BODY_TEXTURE_DIR / name.lower().replace("+", "_")
            self.textures.append([load_texture(self.ctx, file, components=3) for file in sorted(dir_name.glob("*.jpg"))])

    def _init_sky(self) -> None:
        """Create the sky sphere (positions only) and load the sky map."""
        vertices, indices, _ = calc_points(cfg.SPHERE_STACKS, cfg.SPHERE_SLICES)
        self.sky_texture = load_texture(self.ctx, cfg.SKY_TEXTURE_PATH, components=3)

        self.sky_vbo = self.ctx.buffer(vertices.tobytes())
        self.sky_ibo = self.ctx.buffer(indices.tobytes())
        self.sky_vao = self.ctx.vertex_array(
            self.sky_prog,
            [(self.sky_vbo, "3f", "in_pos")],
            index_buffer=self.sky_ibo,
            index_element_size=4
        )

    def _init_ring(self) -> None:
        """Upload the ring mesh and load the RGBA ring texture."""
        ring_input, indices = calc_ring(cfg.RING_INNER, cfg.RING_SEGMENTS)
        self.ring_vbo = self.ctx.buffer(ring_input.tobytes())
        self.ring_ibo = self.ctx.buffer(indices.tobytes())
        self.ring_vao = self.ctx.vertex_array(
            self.ring_prog,
            [(self.ring_vbo, "3f 2f", "in_pos", "in_uv")],
            index_buffer=self.ring_ibo,
            index_element_size=4
        )
        self.ring_texture = load_texture(self.ctx, cfg.RING_TEXTURE_PATH)

    def _init_earth_overlay(self) -> None:
        """Load the Earth cloud texture."""
        self.earth_cloud_texture = load_texture(self.ctx, cfg.CLOUD_TEXTURE_PATH, components=3)

    def _create_mesh(self) -> None:
        """Upload the sphere mesh to the GPU and describe its layout in a vertex array."""
        vertices, indices, textures = calc_points(cfg.SPHERE_STACKS, cfg.SPHERE_SLICES)
        self.ibo = self.ctx.buffer(indices.tobytes()) # Index Buffer Object: Indices that reference vertices

        self._init_textures(vertices, textures)
        self._init_earth_overlay()
        self._init_ring()
        self._init_sky()

    def _init_camera(self) -> None:
        """Set the start state of the orbit camera (angles in radians, distance in AU)."""
        self.az = np.deg2rad(cfg.CAMERA_START_AZIMUTH)
        self.el = np.deg2rad(cfg.CAMERA_START_ELEVATION)
        self.dist = cfg.CAMERA_START_DIST
        self.max_dist = cfg.CAMERA_MAX_DIST # Cap scrolling out
        self.near = cfg.CAMERA_NEAR_FACTOR * self.dist
        self.far = cfg.CAMERA_FAR
        self.follow: int | None = None # Index of the body the camera follows
        self.up = np.array(cfg.CAMERA_UP, dtype=np.float32)

    def _init_bodies(self) -> None:
        """Set the start state of the simulation: positions, velocities, accelerations and time stepping."""
        self.pos, self.vel = correct_center_of_mass(positions, velocities, masses)
        _differences = calc_differences(self.pos)
        self.acc = calc_accelerations(_differences, masses)
        self.dt = cfg.DT # in days
        self.time_factor = cfg.TIME_FACTOR # days per second
        self.dt_budget = 0.
        self.max_steps = cfg.MAX_STEPS_PER_FRAME # max steps per image
        self.real_radii_mode = False
        self.radii = display_radii
        self.spin_angles = np.zeros(self.num_bodies)
        self.max_spin_rate = cfg.MAX_SPIN_RATE # degrees per second
        self.cap_spin = cfg.SPIN_CAP_ENABLED
        self.inspect_spin_rate = cfg.INSPECT_SPIN_RATE  # degrees per second, same for every body
        self.inspect_mode = False
        self.speed = self.time_factor
        self.clouds_on = False

    def _init_trail(self) -> None:
        """Create the ring buffers, sampling timers and GPU buffers for the orbit trails of all bodies."""
        self.trail_histories = np.zeros((self.num_bodies, self.num_trail_points, 3), dtype=np.float32)
        self.trail_heads = np.zeros((self.num_bodies,), dtype=int)
        self.trail_counts = np.zeros((self.num_bodies,), dtype=int)
        self.trail_timers = np.zeros((self.num_bodies,))
        self.trail_timers[0] = np.inf # Skip creating points for sun
        self.trail_intervals = np.maximum(self.dt, orbital_periods / self.num_trail_points) # Interval between points of one trail

        self.trail_vbos = [
            self.ctx.buffer(reserve=self.num_trail_points * 3 * 4, dynamic=True)
            for _ in range(self.num_bodies)
        ]
        self.trail_vaos = [
            self.ctx.vertex_array(self.color_prog, [(vbo, "3f", "in_pos")])
            for vbo in self.trail_vbos
        ]

    def _add_point(self, i: int, point: np.ndarray) -> None:
        """Store a position in the ring buffer of body i, overwriting its oldest point when the buffer is full."""
        self.trail_histories[i][self.trail_heads[i]] = point
        self.trail_heads[i] = (self.trail_heads[i] + 1) % self.num_trail_points
        self.trail_counts[i] = min(self.trail_counts[i] + 1, self.num_trail_points)

    def _ordered_points(self, i: int) -> np.ndarray:
        """Return the stored trail points of body i from oldest to newest."""
        if self.trail_counts[i] < self.num_trail_points:
            return self.trail_histories[i][:self.trail_counts[i]]
        return np.roll(self.trail_histories[i], -self.trail_heads[i], axis=0) # Move oldest point back to the first index

    def on_mouse_drag_event(self, x: int, y: int, dx: int, dy: int) -> None:
        """Rotate the camera around the target while a mouse button is held."""
        self.az += dx * cfg.MOUSE_SENSITIVITY
        self.el += dy * cfg.MOUSE_SENSITIVITY
        self.el = max(np.deg2rad(-cfg.CAMERA_MAX_ELEVATION), min(self.el, np.deg2rad(cfg.CAMERA_MAX_ELEVATION)))

    def on_mouse_scroll_event(self, x_offset: float, y_offset: float) -> None:
        """Zoom: every mouse wheel notch changes the distance to the target by 10 percent."""
        self.dist *= cfg.ZOOM_STEP ** y_offset

    def on_key_event(self, key: int, action: Any, modifiers: Any) -> None:
        """Keys 1-9 select the body the camera follows, 0 goes back to the origin, R toggles real and exaggerated radii."""
        if action != self.wnd.keys.ACTION_PRESS:
            return
        if key in self.number_keys:
            self.follow = self.number_keys[key]
        if key == self.wnd.keys.R:
            self.real_radii_mode = not self.real_radii_mode
        if key == self.wnd.keys.C:
            self.cap_spin = not self.cap_spin
        if key == self.wnd.keys.T and self.follow is not None and texture_switchable[self.follow]:
            self.texture_idx[self.follow] = (self.texture_idx[self.follow]+1) % len(self.textures[self.follow])
        if key == self.wnd.keys.W:
            self.clouds_on = not self.clouds_on

    def _advance_simulation(self, frame_time: float) -> None:
        """Run as many fixed-size Verlet steps as the time budget of this frame allows."""
        self.dt_budget += self.speed * frame_time
        steps = 0
        while self.dt_budget >= self.dt and steps < self.max_steps:
            self.pos, self.vel, self.acc = step_velocity_verlet(self.pos, self.vel, self.acc, masses, self.dt)
            self.dt_budget -= self.dt
            steps += 1

            self.trail_timers -= self.dt
            for i in range(self.num_bodies):
                if self.trail_timers[i] <= 0:
                    self._add_point(i, self.pos[i])
                    self.trail_timers[i] += self.trail_intervals[i]

    def _update_spins(self, frame_time: float) -> None:
        """Advance the spin angle of every body, optionally limited to a maximum rotation speed."""
        if self.inspect_mode:
            delta = np.full(self.num_bodies, np.deg2rad(self.inspect_spin_rate) * frame_time)
        else:
            days = self.speed * frame_time
            delta = 2 * np.pi * days / rotation_periods # type: ignore
            if self.cap_spin:
                delta = np.minimum(delta, np.deg2rad(self.max_spin_rate) * frame_time)
        self.spin_angles = (self.spin_angles + delta) % (2 * np.pi)

    def _update_camera(self) -> None:
        """Compute the view and projection matrices from the orbit camera state and limit the distance."""
        self.radii = display_radii if not self.real_radii_mode else real_radii
        max_radius = np.max(self.radii) if self.follow is None else self.radii[self.follow]
        min_dist = self.near + max_radius
        self.dist = max(min_dist, min(self.dist, self.max_dist))

        target = np.zeros(3) if self.follow is None else self.pos[self.follow]
        eye = target + self.dist * np.array((np.cos(self.el)*np.cos(self.az),
                                            np.cos(self.el)*np.sin(self.az),
                                            np.sin(self.el)))
        self.V = look_at(eye, target, self.up) # Move Camera to origin and rotate so that it looks along -z

        aspect = self.wnd.aspect_ratio
        self.near = cfg.CAMERA_NEAR_FACTOR * self.dist
        self.P = perspective(cfg.CAMERA_FOV, aspect, self.near, self.far) # Project Objects so that more distant ones look smaller

    def _calc_light_dirs(self):
        d = self.pos[0] - self.pos
        norms = np.linalg.norm(d, axis=1, keepdims=True)
        return np.divide(d, norms, out=np.zeros_like(d), where=norms > 0)

    def _ring_normal(self) -> np.ndarray:
        """Return the normal of the ring plane in world space as a float32 vector (the ring is tilted like its body)."""
        Rx = rotation_x(np.deg2rad(axial_tilts[self.ring_body]))
        return (Rx @ np.array([0, 0, 1, 0], dtype=np.float32))[:3].astype(np.float32)

    def _draw_ring(self) -> None:
        """Draw the Saturn ring with alpha blending, after all opaque objects."""
        light_dirs = self._calc_light_dirs()
        i = self.ring_body

        Rx = rotation_x(np.deg2rad(axial_tilts[i]))
        M = calc_translation(self.pos[i]) @ Rx @ calc_scaling(cfg.RING_SCALE * self.radii[i])
        mvp = self.P @ self.V @ M
        self.ring_mvp.write(mvp.T.tobytes())
        self.ring_model.write(M.T.tobytes())
        self.ring_sphere_radius.write(self.radii[i].astype(np.float32).tobytes())

        self.ring_normal.write(self._ring_normal().tobytes())
        self.ring_light_dir.write(light_dirs[i].astype(np.float32).tobytes())

        self.ring_texture.use(location=cfg.TEXTURE_UNIT_MAIN)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.ring_vao.render()
        self.ctx.disable(moderngl.BLEND)

    def _draw_sky(self) -> None:
        """Draw the sky first: view matrix without translation, no depth test."""
        view_rot = self.V.copy()
        view_rot[:3, 3] = 0
        self.sky_mvp.write((self.P @ view_rot).T.tobytes())
        self.sky_texture.use(location=cfg.TEXTURE_UNIT_MAIN)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.sky_vao.render()
        self.ctx.enable(moderngl.DEPTH_TEST)

    def _draw_bodies(self) -> None:
        """Draw every body as the shared sphere mesh with its own model matrix and colour."""
        light_dirs = self._calc_light_dirs()

        ring_outer = cfg.RING_SCALE * self.radii[self.ring_body]
        self.tex_ring_radii.value = (float(cfg.RING_INNER * ring_outer), float(ring_outer))
        self.tex_ring_normal.write(self._ring_normal().tobytes())
        self.ring_texture.use(location=cfg.TEXTURE_UNIT_RING)

        for i in range(self.num_bodies):

            Rx = rotation_x(np.deg2rad(axial_tilts[i]))
            Rz = rotation_z(self.spin_angles[i])
            M = calc_translation(self.pos[i]) @ Rx @ Rz @ calc_scaling(self.radii[i])

            mvp = self.P @ self.V @ M
            self.tex_mvp.write(mvp.T.tobytes())
            self.tex_model.write(M.T.tobytes())

            self.tex_light_dir.write(light_dirs[i].astype(np.float32).tobytes())
            self.tex_lit.value = float(i != 0)
            self.tex_has_night.value = float(day_night_textures[i])
            self.tex_clouds_on.value = float(day_night_textures[i] and self.clouds_on)
            self.tex_ring_shadow_on.value = float(i == self.ring_body)
            self.textures[i][self.texture_idx[i]].use(location=cfg.TEXTURE_UNIT_MAIN)
            self.textures[i][-1].use(location=cfg.TEXTURE_UNIT_NIGHT)
            self.earth_cloud_texture.use(location=cfg.TEXTURE_UNIT_CLOUDS)

            self.texture_vao.render()

    def _draw_trails(self) -> None:
        """Draw the orbit trail of every body except the Sun as a line strip in the colour of the body."""
        vp = self.P @ self.V
        self.color_mvp.write(vp.T.tobytes())
        for i in range(1, self.num_bodies):
            point = self._ordered_points(i)
            self.trail_vbos[i].write(point)
            self.color_rgb.write(body_colors[i])

            num_points = len(point)
            if num_points >= 2:
                self.trail_vaos[i].render(mode=moderngl.LINE_STRIP, vertices=num_points)

    def on_render(self, time: float, frame_time: float) -> None:
        """Draw one frame: advance the simulation, update the camera and draw all bodies."""
        self.ctx.clear(*cfg.BACKGROUND_COLOR)

        self.inspect_mode = self.wnd.is_key_pressed(self.wnd.keys.SPACE)
        self.speed = 0.0 if self.inspect_mode else self.time_factor

        self._advance_simulation(frame_time)
        self._update_spins(frame_time)
        self._update_camera()
        self._draw_sky()
        self._draw_bodies()
        self._draw_trails()
        self._draw_ring()
