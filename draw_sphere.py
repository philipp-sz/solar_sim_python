import numpy as np
import moderngl
import moderngl_window as mglw

from calc_graphics import look_at, perspective, calc_points, rotation_z


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

        self.on_resize(self.wnd.width, self.wnd.height)
        self.V = look_at(np.array([3.0, 2.0, 4.0], dtype=np.float32),
                         np.zeros(3, dtype=np.float32),
                         np.array([0.0, 0.0, 1.0], dtype=np.float32))

    def on_resize(self, width, height):
        aspect = self.wnd.aspect_ratio
        self.P = perspective(60, aspect, 0.1, 100)

    def on_render(self, time: float, frame_time:float):
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.clear(0.1, 0.1, 0.1, 1.0)

        mvp = self.P @ self.V @ rotation_z(time)
        self.prog["u_mvp"].write(mvp.T.tobytes())
        self.vao.render()

if __name__ == "__main__":
    mglw.run_window_config(Sphere)