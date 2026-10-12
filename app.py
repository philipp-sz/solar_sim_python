from pathlib import Path
from typing import Any, cast
from PIL import Image

import numpy as np
import moderngl
import moderngl_window as mglw

from calculations.calc_graphics import look_at, perspective, calc_points, calc_translation, calc_scaling
from calculations.bodies import masses, positions, velocities, body_colors, display_radii, names, real_radii, orbital_periods
from calculations.calc_physics import correct_center_of_mass, step_velocity_verlet, calc_accelerations, calc_differences


class Sphere(mglw.WindowConfig):
    """Window that shows the simulated solar system seen by an orbit camera controlled with the mouse."""

    title = "Solar System Simulation"
    window_size = (800, 600)
    gl_version = (3, 3)
    aspect_ratio = None # type: ignore

    def __init__(self, **kwargs: Any) -> None:
        """Create the window, the GPU objects and the camera state."""
        super().__init__(**kwargs)
        self.num_bodies = len(names)
        self.num_trail_points = 200
        self.number_keys: dict[int, int | None] = {self.wnd.keys.NUMBER_0: None}
        self.number_keys.update({self.wnd.keys.NUMBER_0 + i: i - 1 for i in range(1, 10)})

        self._create_program() # Compile GLSL shaders
        self._init_uniforms()
        self._create_mesh() # Create vbo, ibo, vao
        self._init_camera()
        self._init_bodies()
        self._init_trail()

    def _create_program(self) -> None:
        """Compile the vertex and fragment shader into one GPU program."""
        self.prog = self.ctx.program(
            # Applied on every vertex, transform vertices to final Screenspace
            vertex_shader="""
                #version 330
                in vec3 in_pos;
                uniform vec3 u_color;
                uniform mat4 u_mvp;
                out vec3 frag_color;
                void main() {
                    gl_Position = u_mvp * vec4(in_pos, 1.0);
                    frag_color = u_color;
                }
            """,
            # Decide color and depth of the rasterized pixels in each shape
            fragment_shader="""
                #version 330
                in vec3 frag_color;
                out vec4 out_color;
                void main() {
                    out_color = vec4(frag_color, 1.0);
                }
            """
        )

        self.texture_prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec3 in_pos;
                in vec2 in_uv;
                uniform mat4 u_mvp;
                uniform mat4 u_model;
                out vec2 i_uv;
                out vec3 v_normal;
                void main() {
                    gl_Position = u_mvp * vec4(in_pos, 1.0);
                    v_normal = normalize(mat3(u_model) * in_pos); // cut out translation before transformation
                    i_uv = in_uv;
                }
            """,
            fragment_shader="""
                #version 330
                in vec2 i_uv;
                uniform sampler2D u_texture;
                in vec3 v_normal;
                uniform vec3 u_light_dir;
                uniform float u_lit; // is illuminated (1) or glows on its own (0)
                out vec4 out_color;
                void main() {
                    vec3 n = normalize(v_normal);
                    float diffuse = max(dot(n, u_light_dir), 0.0);
                    float light = mix(1.0, 0.1 + 0.9 * diffuse, u_lit);
                    out_color = texture(u_texture, i_uv) * light;
                }
            """
        )

    def _init_uniforms(self) -> None:
        """Look up the shader uniforms once and keep typed references to them."""
        self.u_mvp = cast(moderngl.Uniform, self.prog["u_mvp"])
        self.u_color = cast(moderngl.Uniform, self.prog["u_color"])
        self.u_tex_mvp = cast(moderngl.Uniform, self.texture_prog["u_mvp"])
        self.u_texture = cast(moderngl.Uniform, self.texture_prog["u_texture"])
        self.u_texture.value = 0 # read texture from position 0
        self.u_model = cast(moderngl.Uniform, self.texture_prog["u_model"])
        self.u_light_dir = cast(moderngl.Uniform, self.texture_prog["u_light_dir"])
        self.u_lit = cast(moderngl.Uniform, self.texture_prog["u_lit"])

    def _init_textures(self, vertices, textures) -> None:
        texture_input = np.hstack((vertices, textures))
        self.texture_vbo = self.ctx.buffer(texture_input.tobytes())
        self.texture_vao = self.ctx.vertex_array(
            self.texture_prog,
            [(self.texture_vbo, "3f 2f", "in_pos", "in_uv")],
            index_buffer=self.ibo,
            index_element_size=4 # Bytes per entry in index buffer
        )

        path = Path(__file__).parent / "assets" / "textures" / "bodies"
        self.textures: list[moderngl.Texture] = []
        for i, name in enumerate(names):
            name = name.lower().replace("+", "_")

            dir_name = path / name
            variant = sorted(dir_name.glob("*.jpg"))[0]
            file_name = dir_name / variant
            img = Image.open(file_name).convert("RGB")
            img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM) # OpenGL expects Bottom img row first

            self.textures.append(self.ctx.texture(img.size, 3, img.tobytes()))
            self.textures[i].filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            self.textures[i].build_mipmaps()

    def _create_mesh(self) -> None:
        """Upload the sphere mesh to the GPU and describe its layout in a vertex array."""
        vertices, indices, textures = calc_points(32, 64)

        self.vbo = self.ctx.buffer(vertices.tobytes()) # Vertex Buffer Object: Reserve storage in VRAM
        self.ibo = self.ctx.buffer(indices.tobytes()) # Index Buffer Object: Indices that reference vertices
        self.vao = self.ctx.vertex_array( # Vertex Array Object: Stores information about how an object is built
            self.prog,
            [(self.vbo, "3f", "in_pos")],
            index_buffer=self.ibo,
            index_element_size=4
        )

        self._init_textures(vertices, textures)

    def _init_camera(self) -> None:
        """Set the start state of the orbit camera (angles in radians, distance in AU)."""
        self.az = np.deg2rad(33.69)
        self.el = np.deg2rad(47.97)
        self.dist = 10.
        self.max_dist = 100. # Cap scrolling out
        self.near = 0.01 * self.dist
        self.far = 200.
        self.follow: int | None = None # Index of the body the camera follows
        self.up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def _init_bodies(self) -> None:
        """Set the start state of the simulation: positions, velocities, accelerations and time stepping."""
        self.pos, self.vel = correct_center_of_mass(positions, velocities, masses)
        _differences = calc_differences(self.pos)
        self.acc = calc_accelerations(_differences, masses)
        self.dt = 1. # in days
        self.time_factor = 60. # days per second
        self.dt_budget = 0.
        self.max_steps = 200 # max steps per image
        self.real_radii_mode = False
        self.radii = display_radii

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
            self.ctx.vertex_array(self.prog, [(vbo, "3f", "in_pos")])
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
        self.az += dx * 0.01
        self.el += dy * 0.01
        self.el = max(np.deg2rad(-89), min(self.el, np.deg2rad(89)))

    def on_mouse_scroll_event(self, x_offset: float, y_offset: float) -> None:
        """Zoom: every mouse wheel notch changes the distance to the target by 10 percent."""
        self.dist *= 1.1 ** y_offset

    def on_key_event(self, key: int, action: Any, modifiers: Any) -> None:
        """Keys 1-9 select the body the camera follows, 0 goes back to the origin, R toggles real and exaggerated radii."""
        if action != self.wnd.keys.ACTION_PRESS:
            return
        if key in self.number_keys:
            self.follow = self.number_keys[key]
        if key == self.wnd.keys.R:
            self.real_radii_mode = not self.real_radii_mode

    def _advance_simulation(self, frame_time: float) -> None:
        """Run as many fixed-size Verlet steps as the time budget of this frame allows."""
        self.dt_budget += self.time_factor * frame_time
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
        self.near = 0.01 * self.dist
        self.P = perspective(60, aspect, self.near, self.far) # Project Objects so that more distant ones look smaller

    def _draw_bodies(self) -> None:
        """Draw every body as the shared sphere mesh with its own model matrix and colour."""
        for i in range(self.num_bodies):
            M = calc_translation(self.pos[i]) @ calc_scaling(self.radii[i])
            mvp = self.P @ self.V @ M
            self.u_tex_mvp.write(mvp.T.tobytes())
            self.textures[i].use(location=0) # write texture to position 0 on GPU
            self.u_model.write(M.T.tobytes())

            d = self.pos[0] - self.pos[i] 
            if i != 0:
                self.u_lit.value = 1.0
                light_dir = d / np.linalg.norm(d)
                self.u_light_dir.write(np.array(light_dir, dtype=np.float32).tobytes())
            else:
                self.u_lit.value = 0.0

            self.texture_vao.render()

    def _draw_trails(self) -> None:
        """Draw the orbit trail of every body except the Sun as a line strip in the colour of the body."""
        vp = self.P @ self.V
        self.u_mvp.write(vp.T.tobytes())
        for i in range(1, self.num_bodies):
            point = self._ordered_points(i)
            self.trail_vbos[i].write(point)
            self.u_color.write(body_colors[i])

            num_points = len(point)
            if num_points >= 2:
                self.trail_vaos[i].render(mode=moderngl.LINE_STRIP, vertices=num_points)

    def on_render(self, time: float, frame_time: float) -> None:
        """Draw one frame: advance the simulation, update the camera and draw all bodies."""
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.clear(0.1, 0.1, 0.1, 1.0)

        self._advance_simulation(frame_time)
        self._update_camera()
        self._draw_bodies()
        self._draw_trails()
