import numpy as np
import matplotlib.pyplot as plt


G = 2.959122082855911e-4

# Masses in Solar masses
masses = np.array([
    1.0,                   # Sun
    1.6601375118e-07,      # Mercury
    2.4478382878e-06,      # Venus
    3.2271560376e-07,      # Mars
    9.5479191522e-04,      # Jupiter
    2.8588567272e-04,      # Saturn
    4.3662437359e-05,      # Uranus
    5.1513897250e-05,      # Neptune
    3.0404326541e-06       # Earth+Moon
], dtype=np.float64)

# Positions [x, y, z] in AU
positions = np.array([
    [-0.0030666110, -0.0055330208,  0.0001316113],  # Sun
    [-0.2182679157, -0.4147406355, -0.0135716708],  # Mercury
    [ 0.0858106384, -0.7272954031, -0.0149124450],  # Venus
    [ 0.3375130624, -1.3925350382, -0.0372856147],  # Mars
    [-1.6970696243,  4.9233497025,  0.0175575505],  # Jupiter
    [ 9.5042767549,  0.2522054276, -0.3828039312],  # Saturn
    [ 9.8772419323, 16.7944818132, -0.0655924885],  # Uranus
    [29.8690628179,  0.5134001901, -0.6989326245],  # Neptune
    [-0.1773363691,  0.9622526536,  0.0000747448]   # Earth+Moon
], dtype=np.float64)

# Velocities [vx, vy, vz] in AU/day
velocities = np.array([
    [ 0.0000071761,  0.0000001778, -0.0000001359],  # Sun
    [ 0.0192390283, -0.0117374484, -0.0027232926],  # Mercury
    [ 0.0199457363,  0.0024005437, -0.0011176076],  # Venus
    [ 0.0141265880,  0.0045403810, -0.0002512212],  # Mars
    [-0.0072234398, -0.0021013620,  0.0001703655],  # Jupiter
    [-0.0004559085,  0.0055648500, -0.0000786252],  # Saturn
    [-0.0034192266,  0.0018105541,  0.0000510180],  # Uranus
    [-0.0000746197,  0.0031572846, -0.0000632994],  # Neptune
    [-0.0172044191, -0.0031134758,  0.0000001405]   # Earth+Moon
], dtype=np.float64)

def calc_differences(positions):
    positions_i = positions[:, None, :]
    positions_j = positions[None, :, :]
    differences = positions_j - positions_i
    return differences

def calc_accelerations(differences, masses):
    distances = np.linalg.norm(differences, axis=-1)
    distances_cubed = np.where(distances == 0, np.inf, distances ** 3)
    masses_j = masses[None, :, None]
    return G * np.sum(masses_j * (differences / distances_cubed[..., None]), axis=1)

def step_euler(pos, vel, acc, masses, dt):
    pos_next = pos + vel * dt
    acc_next = calc_accelerations(calc_differences(pos_next), masses)
    vel_next = vel + acc * dt
    return pos_next, vel_next, acc_next

def step_velocity_verlet(pos, vel, acc, masses, dt):
    pos_next = pos + vel * dt + 0.5 * acc * dt**2
    acc_next = calc_accelerations(calc_differences(pos_next), masses)
    vel_next = vel + 0.5 * (acc + acc_next) * dt
    return pos_next, vel_next, acc_next

def get_energy(positions, velocities, masses):
    normed_vels = np.linalg.norm(velocities, axis=1)
    differences = calc_differences(positions)
    rel_positions = np.linalg.norm(differences, axis=-1)
    kinetic_energy = 0.5 * np.sum(masses * normed_vels**2)
    np.fill_diagonal(rel_positions, np.inf)
    potential_energy = -G * np.sum(masses[:, None] * masses[None, :] / rel_positions) / 2
    energy = kinetic_energy + potential_energy
    return kinetic_energy, potential_energy, energy

def plot_energy_error(runs, log=True, title=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, (times, energies) in runs.items():
        energies = np.asarray(energies)
        rel = (energies - energies[0]) / abs(energies[0])
        ax.plot(np.asarray(times) / 365.25, np.abs(rel) if log else rel,
                label=label, lw=1)
    ax.set_xlabel("Zeit [Jahre]")
    ax.set_ylabel("|ΔE / E₀|" if log else "ΔE / E₀")
    ax.set_yscale("log") if log else ax.axhline(0, color="black", lw=0.8)
    ax.grid(alpha=0.3); ax.legend(); ax.set_title(title)
    fig.tight_layout(); plt.show()

def simulate(step, pos, vel, masses, dt, n_steps):
    pos, vel = pos.copy(), vel.copy()
    acc = calc_accelerations(calc_differences(pos), masses)
    times = np.empty(n_steps + 1)
    energies = np.empty(n_steps + 1)
    times[0], energies[0] = 0.0, get_energy(pos, vel, masses)[-1]
    for k in range(n_steps):
        pos, vel, acc = step(pos, vel, acc, masses, dt)
        times[k + 1] = (k + 1) * dt
        energies[k + 1] = get_energy(pos, vel, masses)[-1]
    return times, energies

differences = calc_differences(positions)
accelerations = calc_accelerations(differences, masses)
R = sum(positions * masses[:, None]) / masses.sum()
V = sum(velocities * masses[:, None]) / sum(masses[:, None])
positions = positions - R[None, :]
velocities = velocities - V

T = 365.25
n = 365
dt = T / n
integrators = {"Euler": step_euler, "Velocity Verlet": step_velocity_verlet}
runs = {name: simulate(f, positions, velocities, masses, dt, n)
        for name, f in integrators.items()}
plot_energy_error(runs)
