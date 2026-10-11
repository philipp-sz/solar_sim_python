from typing import Any

import numpy as np
import moderngl
import moderngl_window as mglw

from calculations.calc_graphics import look_at, perspective, calc_points, calc_translation, calc_scaling, RADIUS
from calculations.bodies import masses, positions, velocities, body_colors, display_radii, names
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
        self.number_keys: dict[int, int | None] = {self.wnd.keys.NUMBER_0: None}
        self.number_keys.update({self.wnd.keys.NUMBER_0 + i: i - 1 for i in range(1, 10)})

        self._create_program() # Compile GLSL shaders
        self._create_mesh() # Create vbo, ibo, vao
        self._init_camera()
        self._init_bodies()

        self.on_resize(self.wnd.width, self.wnd.height)

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

    def _create_mesh(self) -> None:
        """Upload the sphere mesh to the GPU and describe its layout in a vertex array."""
        vertices, indices = calc_points(32, 64)

        self.vbo = self.ctx.buffer(vertices.tobytes()) # Vertex Buffer Object: Reserve storage in VRAM
        self.ibo = self.ctx.buffer(indices.tobytes()) # Index Buffer Object: Indices that reference vertices
        self.vao = self.ctx.vertex_array( # Vertex Array Object: Stores information about how an object is built
            self.prog,
            [(self.vbo, "3f", "in_pos")],
            index_buffer=self.ibo,
            index_element_size=4
        )

    def _init_camera(self) -> None:
        """Set the start state of the orbit camera (angles in radians, distance in AU)."""
        self.az = np.deg2rad(33.69)
        self.el = np.deg2rad(47.97)
        self.dist = 10.
        self.max_dist = 100. # Cap scrolling out
        self.near = 0.1
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
        self.budget = 0.
        self.max_steps = 200 # max steps per image

    def on_mouse_drag_event(self, x: int, y: int, dx: int, dy: int) -> None:
        """Rotate the camera around the target while a mouse button is held."""
        self.az += dx * 0.01
        self.el += dy * 0.01
        self.el = max(np.deg2rad(-89), min(self.el, np.deg2rad(89)))

    def on_mouse_scroll_event(self, x_offset: float, y_offset: float) -> None:
        """Zoom: every mouse wheel notch changes the distance to the target by 10 percent."""
        self.dist *= 1.1 ** y_offset

        max_radius = np.max(display_radii) if self.follow is None else display_radii[self.follow]
        min_dist = self.near + max_radius
        self.dist = max(min_dist, min(self.dist, self.max_dist))

    def on_key_event(self, key: int, action: Any, modifiers: Any) -> None:
        """Choose which body the camera follows: keys 1-9 select a body, key 0 goes back to the origin."""
        if action != self.wnd.keys.ACTION_PRESS:
            return
        if key in self.number_keys:
            self.follow = self.number_keys[key]

    def on_resize(self, width: int, height: int) -> None:
        """Recompute the projection matrix, because it depends on the aspect ratio of the window."""
        aspect = self.wnd.aspect_ratio
        self.P = perspective(60, aspect, self.near, self.far) # Project Objects so that more distant ones look smaller

    def on_render(self, time: float, frame_time: float) -> None:
        """Draw one frame: advance the simulation, update the camera and draw all bodies."""
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.clear(0.1, 0.1, 0.1, 1.0)

        self.budget += self.time_factor * frame_time
        steps = 0
        while self.budget >= self.dt and steps < self.max_steps:
            self.pos, self.vel, self.acc = step_velocity_verlet(self.pos, self.vel, self.acc, masses, self.dt)
            self.budget -= self.dt
            steps += 1

        target = np.zeros(3) if self.follow is None else self.pos[self.follow]

        eye = target + self.dist * np.array((np.cos(self.el)*np.cos(self.az),
                                            np.cos(self.el)*np.sin(self.az),
                                            np.sin(self.el)))
        self.V = look_at(eye, target, self.up) # Move Camera to origin and rotate so that it looks along -z

        for i in range(len(names)):
            M = calc_translation(self.pos[i]) @ calc_scaling(display_radii[i])
            mvp = self.P @ self.V @ M
            self.prog["u_mvp"].write(mvp.T.tobytes()) # type: ignore
            self.prog["u_color"].write(body_colors[i]) # type: ignore

            self.vao.render()
