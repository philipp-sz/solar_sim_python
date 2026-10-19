from pathlib import Path

import moderngl
from PIL import Image

import config as cfg


IMAGE_MODES = {3: "RGB", 4: "RGBA"}


def load_texture(ctx: moderngl.Context, path: Path, components: int = 4) -> moderngl.Texture:
    """Load an image file as a mipmapped GPU texture with 3 (RGB) or 4 (RGBA) components."""
    if components not in IMAGE_MODES:
        raise ValueError(f"load_texture: components must be 3 or 4, got {components}")
    img = Image.open(path).convert(IMAGE_MODES[components])
    img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM) # OpenGL expects the bottom image row first
    texture = ctx.texture(img.size, components, img.tobytes())
    texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
    texture.build_mipmaps() # Without mipmaps this filter gives a black texture
    return texture


def load_program(ctx: moderngl.Context, name: str) -> moderngl.Program:
    """Compile the GPU program from shaders/<name>.vert and shaders/<name>.frag."""
    vertex_shader = (cfg.SHADER_DIR / f"{name}.vert").read_text(encoding="utf-8")
    fragment_shader = (cfg.SHADER_DIR / f"{name}.frag").read_text(encoding="utf-8")
    return ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
