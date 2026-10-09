import numpy as np


CENTER = np.array([0.0, 0.0, 0.0], dtype=np.float64)
RADIUS = np.array([1.0, 1.0, 1.0], dtype=np.float64)


def calc_vertices(num_stacks, num_slices):
    vertices = []
    for i in range(num_stacks + 1):
        theta = np.pi * i / num_stacks
        for j in range(num_slices + 1):
            phi = 2 * np.pi * j / num_slices
            x = RADIUS[0] * np.sin(theta) * np.cos(phi) + CENTER[0]
            y = RADIUS[1] * np.sin(theta) * np.sin(phi) + CENTER[1]
            z = RADIUS[2] * np.cos(theta) + CENTER[2]
            vertices.append((x, y, z))
    return vertices


def calc_indices(num_stacks, num_slices):
    indices = []
    for i in range(num_stacks):
        for j in range(num_slices):
            top_left = i * (num_slices + 1) + j # (i, j)
            top_right = top_left + 1 # (i, j + 1)
            bottom_left = (i + 1) * (num_slices + 1) + j # (i + 1, j)
            bottom_right = bottom_left + 1 # (i + 1, j + 1)

            indices.append((top_left, bottom_left, top_right))
            indices.append((top_right, bottom_left, bottom_right))
    return indices


def calc_points(num_stacks, num_slices):
    vertices = calc_vertices(num_stacks, num_slices)
    indices = calc_indices(num_stacks, num_slices)
    return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)


def calc_translation(translation_vector):
    translation_matrix = np.eye(4, dtype=np.float32)
    translation_matrix[:-1, -1] = translation_vector
    return translation_matrix


def calc_scaling(scalar):
    scaling_matrix = np.eye(4, dtype=np.float32)
    np.fill_diagonal(scaling_matrix, scalar)
    scaling_matrix[-1, -1] = 1.0
    return scaling_matrix


def rotation_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    R = np.eye(4, dtype=np.float32)
    R[0, 0], R[0, 1] = c, -s
    R[1, 0], R[1, 1] = s, c
    return R


def look_at(eye, target, up):
    f = target - eye # Direction of view
    f = f / np.linalg.norm(f)

    r = np.cross(f, up) # Right vector, perpendicular to f and up
    r_norm = np.linalg.norm(r)
    if r_norm < 1e-3:
        raise ValueError("look_at: view direction is parallel to up (|f x up| = 3.2e-05), right vector undefined. Change eye, target or up.")
    r = r / r_norm
    u = np.cross(r, f) # Recalibrated up vector, perpendicular to r and f

    R_t = np.eye(4, dtype=np.float32)
    R_t[0, :3] = r
    R_t[1, :3] = u
    R_t[2, :3] = -f

    T_inv = np.eye(4, dtype=np.float32)
    T_inv[0:-1, -1] = -eye

    V = R_t @ T_inv
    return V


def perspective(fovy, aspect, near, far):
    fovy_rad = np.radians(fovy)
    f = 1.0 / np.tan(fovy_rad / 2.0) # Scale factor for x and y
    A = -(far + near) / (far - near)
    B = -(2.0 * far * near) / (far - near)

    P = np.zeros((4, 4), dtype=np.float32)

    P[0, 0] = f / aspect
    P[1, 1] = f
    P[2, 2] = A
    P[2, 3] = B
    P[3, 2] = -1.0
    return P