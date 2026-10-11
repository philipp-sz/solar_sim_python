import numpy as np
import moderngl
import moderngl_window as mglw

from calc_graphics import look_at, perspective, calc_points, rotation_z, RADIUS


class Sphere(mglw.WindowConfig):
    title = "Index Buffer + Depth Test"
    window_size = (800, 600)
    gl_version = (3, 3)
    aspect_ratio = None # type: ignore

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.prog = self.ctx.program(
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
            fragment_shader="""
                #version 330
                in vec3 v_color;
                out vec4 f_color;
                void main() {
                    f_color = vec4(v_color, 1.0);
                }
            """
        )

        positions, indices = calc_points(32, 64)
        colors = positions * 0.5 + 0.5
        vertices = np.hstack([positions, colors]).astype(np.float32)

        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.ibo = self.ctx.buffer(indices.tobytes())
        self.vao = self.ctx.vertex_array(
            self.prog,
            [(self.vbo, "3f 3f", "in_pos", "in_color")],
            index_buffer=self.ibo,
            index_element_size=4
        )

        self.az = np.deg2rad(33.69)
        self.el = np.deg2rad(47.97)
        self.dist = 5.385
        self.near = 0.1
        self.target = np.zeros(3, dtype=np.float32)
        self.up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # One RGB colour per body, same order as in physics_calc.py (Mars is index 3!)
        self.body_colors = np.array([
            [1.00, 0.85, 0.30],  # 0 Sun
            [0.60, 0.60, 0.60],  # 1 Mercury
            [0.90, 0.80, 0.60],  # 2 Venus
            [0.80, 0.35, 0.20],  # 3 Mars
            [0.75, 0.60, 0.45],  # 4 Jupiter
            [0.90, 0.85, 0.60],  # 5 Saturn
            [0.60, 0.85, 0.90],  # 6 Uranus
            [0.20, 0.30, 0.80],  # 7 Neptune
            [0.20, 0.50, 0.80],  # 8 Earth+Moon
        ], dtype=np.float32)

        self.on_resize(self.wnd.width, self.wnd.height)

    def on_mouse_drag_event(self, x, y, dx, dy):
        self.az += dx * 0.01
        self.el += dy * 0.01
        self.el = max(np.deg2rad(-89), min(self.el, np.deg2rad(89)))

    def on_mouse_scroll_event(self, x_offset, y_offset):
        self.dist *= 1.1 ** y_offset

        min_dist = self.near + np.max(RADIUS)
        self.dist = max(min_dist, min(self.dist, 50))

    def on_resize(self, width, height):
        aspect = self.wnd.aspect_ratio
        self.P = perspective(60, aspect, self.near, 100)

    def on_render(self, time: float, frame_time:float):
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.clear(0.1, 0.1, 0.1, 1.0)

        eye = self.target + self.dist * np.array((np.cos(self.el)*np.cos(self.az),
                                            np.cos(self.el)*np.sin(self.az),
                                            np.sin(self.el)))
        self.V = look_at(eye, self.target, self.up)

        mvp = self.P @ self.V
        self.prog["u_mvp"].write(mvp.T.tobytes())
        self.vao.render()

if __name__ == "__main__":
    mglw.run_window_config(Sphere)