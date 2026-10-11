from typing import Any

import numpy as np
import moderngl
import moderngl_window as mglw

from calculations.calc_graphics import look_at, perspective, calc_points, RADIUS


class Sphere(mglw.WindowConfig):
    """Window that shows a sphere seen by an orbit camera controlled with the mouse."""

    title = "Index Buffer + Depth Test"
    window_size = (800, 600)
    gl_version = (3, 3)
    aspect_ratio = None # type: ignore

    def __init__(self, **kwargs: Any) -> None:
        """Create the window, the GPU objects and the camera state."""
        super().__init__(**kwargs)

        self._create_program() # Compile GLSL shaders
        self._create_mesh() # Create vbo, ibo, vao
        self._init_camera()

        self.on_resize(self.wnd.width, self.wnd.height)

    def _create_program(self) -> None:
        """Compile the vertex and fragment shader into one GPU program."""
        self.prog = self.ctx.program(
            # Applied on every vertex, transform vertices to final Screenspace
            vertex_shader="""
                #version 330
                in vec3 in_pos;
                uniform mat4 u_mvp;
                in vec3 in_color;
                out vec3 v_color;
                void main() {
                    gl_Position = u_mvp * vec4(in_pos, 1.0);
                    v_color = in_color;
                }
            """,
            # Decide color and depth of the rasterized pixels in each shape
            fragment_shader="""
                #version 330
                in vec3 v_color;
                out vec4 f_color;
                void main() {
                    f_color = vec4(v_color, 1.0);
                }
            """
        )

    def _create_mesh(self) -> None:
        """Upload the sphere mesh to the GPU and describe its layout in a vertex array."""
        positions, indices = calc_points(32, 64)
        colors = positions * 0.5 + 0.5
        vertices = np.hstack([positions, colors]).astype(np.float32)

        self.vbo = self.ctx.buffer(vertices.tobytes()) # Vertex Buffer Object: Reserve storage in VRAM
        self.ibo = self.ctx.buffer(indices.tobytes()) # Index Buffer Object: Indices that reference vertices
        self.vao = self.ctx.vertex_array( # Vertex Array Object: Stores information about how an object is built
            self.prog,
            [(self.vbo, "3f 3f", "in_pos", "in_color")],
            index_buffer=self.ibo,
            index_element_size=4
        )

    def _init_camera(self) -> None:
        """Set the start state of the orbit camera (angles in radians, distance in AU)."""
        self.az = np.deg2rad(33.69)
        self.el = np.deg2rad(47.97)
        self.dist = 5.385
        self.max_dist = 50 # Cap scrolling out
        self.near = 0.1
        self.far = 100
        self.target = np.zeros(3, dtype=np.float32)
        self.up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def on_mouse_drag_event(self, x: int, y: int, dx: int, dy: int) -> None:
        """Rotate the camera around the target while a mouse button is held."""
        self.az += dx * 0.01
        self.el += dy * 0.01
        self.el = max(np.deg2rad(-89), min(self.el, np.deg2rad(89)))

    def on_mouse_scroll_event(self, x_offset: float, y_offset: float) -> None:
        """Zoom: every mouse wheel notch changes the distance to the target by 10 percent."""
        self.dist *= 1.1 ** y_offset

        min_dist = self.near + np.max(RADIUS)
        self.dist = max(min_dist, min(self.dist, self.max_dist))

    def on_resize(self, width: int, height: int) -> None:
        """Recompute the projection matrix, because it depends on the aspect ratio of the window."""
        aspect = self.wnd.aspect_ratio
        self.P = perspective(60, aspect, self.near, self.far) # Project Objects so that more distant ones look smaller

    def on_render(self, time: float, frame_time: float) -> None:
        """Draw one frame: update the camera and draw the sphere."""
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.clear(0.1, 0.1, 0.1, 1.0)

        eye = self.target + self.dist * np.array((np.cos(self.el)*np.cos(self.az),
                                            np.cos(self.el)*np.sin(self.az),
                                            np.sin(self.el)))
        self.V = look_at(eye, self.target, self.up) # Move Camera to origin and rotate so that it looks along -z

        mvp = self.P @ self.V
        self.prog["u_mvp"].write(mvp.T.tobytes())
        self.vao.render()
