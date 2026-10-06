import math
import moderngl_window as mglw


class SimpleWindow(mglw.WindowConfig):
    title = "Simple Window"
    window_size = (800, 600)
    gl_version = (3, 3)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.alpha = 1.0

    def change_color_sine(self, time: float):
        r = 0.5 * math.sin(time) + 0.5
        g = 0.5 * math.sin(time + math.pi / 2) + 0.5
        b = 0.5 * math.sin(time + math.pi) + 0.5
        self.color = (r, g, b, self.alpha)

    def on_render(self, time: float, frame_time: float):
        self.change_color_sine(time)
        self.ctx.clear(*self.color)

if __name__ == "__main__":
    mglw.run_window_config(SimpleWindow)