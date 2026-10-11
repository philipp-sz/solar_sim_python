import numpy as np
from numpy.typing import NDArray


# All arrays below use this order: from the Sun outwards (Earth+Moon is one body).
names: list[str] = [
    "Sun",
    "Mercury",
    "Venus",
    "Earth+Moon",
    "Mars",
    "Jupiter",
    "Saturn",
    "Uranus",
    "Neptune",
]

# Masses in solar masses
masses: NDArray[np.float64] = np.array([
    1.0,                   # Sun
    1.6601375118e-07,      # Mercury
    2.4478382878e-06,      # Venus
    3.0404326541e-06,      # Earth+Moon
    3.2271560376e-07,      # Mars
    9.5479191522e-04,      # Jupiter
    2.8588567272e-04,      # Saturn
    4.3662437359e-05,      # Uranus
    5.1513897250e-05,      # Neptune
], dtype=np.float64)

# Positions [x, y, z] in AU
positions: NDArray[np.float64] = np.array([
    [-0.0030666110, -0.0055330208,  0.0001316113],  # Sun
    [-0.2182679157, -0.4147406355, -0.0135716708],  # Mercury
    [ 0.0858106384, -0.7272954031, -0.0149124450],  # Venus
    [-0.1773363691,  0.9622526536,  0.0000747448],  # Earth+Moon
    [ 0.3375130624, -1.3925350382, -0.0372856147],  # Mars
    [-1.6970696243,  4.9233497025,  0.0175575505],  # Jupiter
    [ 9.5042767549,  0.2522054276, -0.3828039312],  # Saturn
    [ 9.8772419323, 16.7944818132, -0.0655924885],  # Uranus
    [29.8690628179,  0.5134001901, -0.6989326245],  # Neptune
], dtype=np.float64)

# Velocities [vx, vy, vz] in AU/day
velocities: NDArray[np.float64] = np.array([
    [ 0.0000071761,  0.0000001778, -0.0000001359],  # Sun
    [ 0.0192390283, -0.0117374484, -0.0027232926],  # Mercury
    [ 0.0199457363,  0.0024005437, -0.0011176076],  # Venus
    [-0.0172044191, -0.0031134758,  0.0000001405],  # Earth+Moon
    [ 0.0141265880,  0.0045403810, -0.0002512212],  # Mars
    [-0.0072234398, -0.0021013620,  0.0001703655],  # Jupiter
    [-0.0004559085,  0.0055648500, -0.0000786252],  # Saturn
    [-0.0034192266,  0.0018105541,  0.0000510180],  # Uranus
    [-0.0000746197,  0.0031572846, -0.0000632994],  # Neptune
], dtype=np.float64)

# One RGB colour per body (values 0..1). Placeholder until textures are used.
body_colors: NDArray[np.float32] = np.array([
    [1.00, 0.85, 0.30],  # Sun
    [0.60, 0.60, 0.60],  # Mercury
    [0.90, 0.80, 0.60],  # Venus
    [0.20, 0.50, 0.80],  # Earth+Moon
    [0.80, 0.35, 0.20],  # Mars
    [0.75, 0.60, 0.45],  # Jupiter
    [0.90, 0.85, 0.60],  # Saturn
    [0.60, 0.85, 0.90],  # Uranus
    [0.20, 0.30, 0.80],  # Neptune
], dtype=np.float32)

# Display radii in AU. Exaggerated: the real radii are far below one pixel.
display_radii: NDArray[np.float64] = np.array([
    0.15,  # Sun
    0.04,  # Mercury
    0.05,  # Venus
    0.05,  # Earth+Moon
    0.05,  # Mars
    0.12,  # Jupiter
    0.10,  # Saturn
    0.08,  # Uranus
    0.08,  # Neptune
], dtype=np.float64)

real_radii: NDArray[np.float64] = np.array([
    4.6505e-03,  # Sun
    1.6308e-05,  # Mercury
    4.0454e-05,  # Venus
    4.2588e-05,  # Earth+Moon
    2.2657e-05,  # Mars
    4.6733e-04,  # Jupiter
    3.8926e-04,  # Saturn
    1.6953e-04,  # Uranus
    1.6459e-04,  # Neptune
], dtype=np.float64)

orbital_periods: NDArray[np.float64] = np.array([
    0.0,        # Sun
    87.97,      # Mercury
    224.70,     # Venus
    365.26,     # Earth+Moon
    686.98,     # Mars
    4332.59,    # Jupiter
    10759.22,   # Saturn
    30688.5,    # Uranus
    60182.0,    # Neptune
], dtype=np.float64)
