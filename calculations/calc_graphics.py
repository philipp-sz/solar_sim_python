import numpy as np
from numpy.typing import NDArray


Matrix = NDArray[np.float32]
Vector = NDArray[np.floating]


def calc_vertices(num_stacks: int, num_slices: int) -> tuple[list[tuple[float, float, float]], list[tuple[float, float]]]:
    """Return positions (x, y, z) and texture coordinates (u, v) of a unit UV sphere (radius 1, centred at the origin) as a (num_stacks+1) x (num_slices+1) grid, row by row."""
    vertices = []
    textures = []
    for i in range(num_stacks + 1):
        v = 1 - i / num_stacks
        theta = np.pi * i / num_stacks # Angle from positive z downwards for each horizontal stack
        for j in range(num_slices + 1):
            u = j / num_slices
            textures.append((u, v))

            phi = 2 * np.pi * j / num_slices # Angle in XY plane around z axis for each vertical slice
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            vertices.append((x, y, z))
    return vertices, textures


def calc_indices(num_stacks: int, num_slices: int) -> list[tuple[int, int, int]]:
    """Return the triangles of the sphere as index triples into the vertex list (counter-clockwise seen from outside)."""
    indices = []
    for i in range(num_stacks):
        for j in range(num_slices):
            # Each grid cell (square) is split into two triangles.
            top_left = i * (num_slices + 1) + j # (i, j)
            top_right = top_left + 1 # (i, j + 1)
            bottom_left = (i + 1) * (num_slices + 1) + j # (i + 1, j)
            bottom_right = bottom_left + 1 # (i + 1, j + 1)

            indices.append((top_left, bottom_left, top_right))
            indices.append((top_right, bottom_left, bottom_right))
    return indices


def calc_points(num_stacks: int, num_slices: int) -> tuple[NDArray[np.float32], NDArray[np.uint32], NDArray[np.float32]]:
    """Return the sphere mesh as GPU-ready arrays: float32 vertices (N, 3), uint32 indices (T, 3) and float32 texture coordinates (N, 2)."""
    vertices, textures = calc_vertices(num_stacks, num_slices)
    indices = calc_indices(num_stacks, num_slices)

    vertices_array = np.array(vertices, dtype=np.float32)
    textures_array = np.array(textures, dtype=np.float32)
    indices_array = np.array(indices, dtype=np.uint32)
    return vertices_array, indices_array, textures_array


def calc_ecliptic_to_galactic() -> Matrix:
    """Return the 3x3 rotation matrix that turns ecliptic directions into galactic directions."""
    eps = np.deg2rad(23.4392911)
    R = np.array([[-0.0548755604, -0.8734370902, -0.4838350155],
                [+0.4941094279, -0.4448296300, +0.7469822445],
                [-0.8676661490, -0.1980763734, +0.4559837762]])
    E = np.array([[1, 0, 0], 
                [0, np.cos(eps), np.sin(eps)], 
                [0, -np.sin(eps), np.cos(eps)]])
    return (R @ E.T).astype(np.float32)


def calc_ring(inner: float, segments: int) -> tuple[NDArray[np.float32], NDArray[np.uint32]]:
    """Return a flat ring (outer radius 1, inner radius `inner`) in the xy plane as float32 vertices (2N, 3), uint32 indices (2N, 3) and float32 texture coordinates (2N, 2)."""
    angles = 2 * np.pi * np.arange(segments) / segments
    c, s = np.cos(angles), np.sin(angles)

    vertices = np.zeros((2 * segments, 5), dtype=np.float32)
    vertices[0::2, 0], vertices[0::2, 1] = inner * c, inner * s # inner radius scaled down
    vertices[1::2, 0], vertices[1::2, 1] = c, s
    vertices[1::2, 3], vertices[:, 4] = 1.0, 0.5

    k = np.arange(segments)
    k_next = (k + 1) % segments
    inner_k, outer_k = 2 * k, 2 * k + 1
    inner_n, outer_n = 2 * k_next, 2 * k_next + 1

    tri_a = np.stack([inner_k, outer_k, inner_n], axis=1)
    tri_b = np.stack([outer_k, outer_n, inner_n], axis=1)
    indices = np.concatenate([tri_a, tri_b]).astype(np.uint32)
    return vertices, indices


def calc_translation(translation_vector: Vector) -> Matrix:
    """Return the 4x4 matrix that moves points by the given (x, y, z) vector."""
    translation_matrix = np.eye(4, dtype=np.float32)
    translation_matrix[:-1, -1] = translation_vector # offset goes into the last column, rows 0..2
    return translation_matrix


def calc_scaling(scalar: float | Vector) -> Matrix:
    """Return the 4x4 matrix that scales points by a factor (or per axis by a vector) around the origin."""
    scaling_matrix = np.eye(4, dtype=np.float32)
    np.fill_diagonal(scaling_matrix, scalar)
    scaling_matrix[-1, -1] = 1.0  # fill_diagonal also overwrote w; it must stay 1
    return scaling_matrix


def rotation_z(angle: float) -> Matrix:
    """Return the 4x4 matrix that rotates points counter-clockwise around the z axis (angle in radians)."""
    c, s = np.cos(angle), np.sin(angle)
    R = np.eye(4, dtype=np.float32)
    R[0, 0], R[0, 1] = c, -s
    R[1, 0], R[1, 1] = s, c
    return R


def rotation_x(angle: float) -> Matrix:
    """Return the 4x4 matrix that rotates points counter-clockwise around the x axis (angle in radians)."""
    c, s = np.cos(angle), np.sin(angle)
    R = np.eye(4, dtype=np.float32)
    R[1, 1], R[1, 2] = c, -s
    R[2, 1], R[2, 2] = s, c
    return R


def look_at(eye: Vector, target: Vector, up: Vector) -> Matrix:
    """Return the view matrix that puts the camera at `eye` looking at `target`, with `up` as the upward direction."""
    f = target - eye # Direction of view
    f = f / np.linalg.norm(f)

    r = np.cross(f, up) # Right vector, perpendicular to f and up
    r_norm = np.linalg.norm(r)
    if r_norm < 1e-3:
        raise ValueError(f"look_at: view direction is parallel to up (|f x up| = {r_norm}), right vector undefined. Change eye, target or up.")
    r = r / r_norm
    u = np.cross(r, f) # Recalibrated up vector, now pointing upwards perpendicular to f

    R_t = np.eye(4, dtype=np.float32)
    R_t[0, :3] = r
    R_t[1, :3] = u
    R_t[2, :3] = -f

    T_inv = np.eye(4, dtype=np.float32)
    T_inv[0:-1, -1] = -eye

    V = R_t @ T_inv
    return V


def perspective(fovy: float, aspect: float, near: float, far: float) -> Matrix:
    """Return the perspective projection matrix (fovy in degrees)."""
    fovy_rad = np.radians(fovy)
    f = 1.0 / np.tan(fovy_rad / 2.0)

    A = -(far + near) / (far - near)
    B = -(2.0 * far * near) / (far - near)

    P = np.zeros((4, 4), dtype=np.float32)

    P[0, 0] = f / aspect  # dividing by aspect keeps shapes undistorted in wide windows
    P[1, 1] = f
    P[2, 2] = A
    P[2, 3] = B
    P[3, 2] = -1.0
    return P