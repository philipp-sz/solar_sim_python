import math
import numpy as np
import moderngl_window as mglw


# class SimpleWindow(mglw.WindowConfig):
#     title = "Simple Window"
#     window_size = (800, 600)
#     gl_version = (3, 3)

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.alpha = 1.0

#     def change_color_sine(self, time: float):
#         r = 0.5 * math.sin(time) + 0.5
#         g = 0.5 * math.sin(time + math.pi / 2) + 0.5
#         b = 0.5 * math.sin(time + math.pi) + 0.5
#         self.color = (r, g, b, self.alpha)

#     def on_render(self, time: float, frame_time: float):
#         self.change_color_sine(time)
#         self.ctx.clear(*self.color)


class Triangle(mglw.WindowConfig):
    title = "Triangle Window"
    window_size = (800, 600)
    gl_version = (3, 3)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_pos;
                in vec3 in_color;
                out vec3 v_color;
                void main() {
                    gl_Position = vec4(in_pos, 0.0, 1.0);
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

        vertices = np.array([
            -0.6, -0.6,  1.0, 0.0, 0.0,
            0.6, -0.6,   0.0, 1.0, 0.0,
            0.0,  0.6,   0.0, 0.0, 1.0,
        ], dtype=np.float32)

        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.vao = self.ctx.vertex_array(
            self.prog, [(self.vbo, "2f 3f", "in_pos", "in_color")]
        )

    def on_render(self, time: float, frame_time:float):
        self.ctx.clear(0.1, 0.1, 0.1, 1.0)
        self.vao.render()

if __name__ == "__main__":
    mglw.run_window_config(Triangle)