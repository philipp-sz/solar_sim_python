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




def look_at(eye, target, up):
    f = target - eye # Direction of view
    f = f / np.linalg.norm(f)
    dot = np.dot(f, up)
    if np.abs(dot) > 0.99:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    r = np.cross(f, up) # Right vector, perpendicular to f and up
    r = r / np.linalg.norm(r)
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






def check_mvp(vertices, M, V, P):
    verts = np.asarray(vertices, dtype=np.float64)
    homo = np.hstack([verts, np.ones((len(verts), 1))])
    clip = homo @ (P @ V @ M).T
    ndc = clip[:, :3] / clip[:, 3:4]

    center = P @ V @ M @ np.array([0.0, 0.0, 0.0, 1.0])
    center = center[:3] / center[3]

    print(f"w-Bereich       : {clip[:,3].min():.4f} .. {clip[:,3].max():.4f}")
    print(f"x-Bereich (NDC) : {ndc[:,0].min():+.4f} .. {ndc[:,0].max():+.4f}  Spanne {np.ptp(ndc[:,0]):.4f}")
    print(f"y-Bereich (NDC) : {ndc[:,1].min():+.4f} .. {ndc[:,1].max():+.4f}  Spanne {np.ptp(ndc[:,1]):.4f}")
    print(f"z-Bereich (NDC) : {ndc[:,2].min():+.4f} .. {ndc[:,2].max():+.4f}")
    inside = np.all(np.abs(ndc[:, :2]) <= 1.0, axis=1).sum()
    print(f"Mittelpunkt     : x={center[0]:+.4f}  y={center[1]:+.4f}")
    print(f"im Bild         : {inside} von {len(ndc)} Vertices")
    print(f"Spannenverhaeltnis y/x : {np.ptp(ndc[:,1])/np.ptp(ndc[:,0]):.4f}")

vertices = calc_vertices(16, 32)
M = calc_translation(np.array([5., 0., 0.])) @ calc_scaling(0.1)
V = look_at(np.array([0., 0., 15.]), np.zeros(3), np.array([0., 1., 0.]))
P = perspective(60, 16/9, 0.1, 100)
check_mvp(vertices, M, V, P)