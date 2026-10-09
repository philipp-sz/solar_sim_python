import numpy as np
import moderngl_window as mglw


def rotation_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    R = np.eye(4, dtype=np.float32)
    R[0, 0], R[0, 1] = c, -s
    R[1, 0], R[1, 1] = s, c
    return R

class Triangle(mglw.WindowConfig):
    title = "Triangle Window"
    window_size = (800, 600)
    gl_version = (3, 3)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.prog = self.ctx.program(
            vertex_shader="""
                #version 330
                uniform mat4 u_transform;
                in vec2 in_pos;
                in vec3 in_color;
                out vec3 v_color;
                void main() {
                    gl_Position = u_transform * vec4(in_pos, 0.0, 1.0);
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
        R = rotation_z(time)
        self.prog["u_transform"].write(R.T.tobytes())
        self.vao.render()

if __name__ == "__main__":
    mglw.run_window_config(Triangle)