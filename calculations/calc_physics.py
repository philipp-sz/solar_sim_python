from typing import Callable

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.float64]
StepFunction = Callable[[Array, Array, Array, Array, float], tuple[Array, Array, Array]]

G = 2.959122082855911e-4


def calc_differences(positions: Array) -> Array:
    """Return all pairwise difference vectors, shape (N, N, 3); entry [i, j] points from body i to body j."""
    positions_i = positions[:, None, :]
    positions_j = positions[None, :, :]
    differences = positions_j - positions_i
    return differences


def calc_accelerations(differences: Array, masses: Array) -> Array:
    """Return the Newtonian gravitational acceleration of every body caused by all other bodies, shape (N, 3)."""
    distances = np.linalg.norm(differences, axis=-1)
    distances_cubed = np.where(distances == 0, np.inf, distances ** 3)
    masses_j = masses[None, :, None]
    return G * np.sum(masses_j * (differences / distances_cubed[..., None]), axis=1)


def step_euler(pos: Array, vel: Array, acc: Array, masses: Array, dt: float) -> tuple[Array, Array, Array]:
    """Advance the system by one explicit Euler step; return new positions, velocities and accelerations."""
    pos_next = pos + vel * dt
    acc_next = calc_accelerations(calc_differences(pos_next), masses)
    vel_next = vel + acc * dt
    return pos_next, vel_next, acc_next


def step_velocity_verlet(pos: Array, vel: Array, acc: Array, masses: Array, dt: float) -> tuple[Array, Array, Array]:
    """Advance the system by one velocity-Verlet step; return new positions, velocities and accelerations."""
    pos_next = pos + vel * dt + 0.5 * acc * dt**2
    acc_next = calc_accelerations(calc_differences(pos_next), masses)
    vel_next = vel + 0.5 * (acc + acc_next) * dt
    return pos_next, vel_next, acc_next


def get_energy(positions: Array, velocities: Array, masses: Array) -> tuple[float, float, float]:
    """Return kinetic, potential and total energy of the system."""
    normed_vels = np.linalg.norm(velocities, axis=1) # speed of each body
    differences = calc_differences(positions)
    rel_positions = np.linalg.norm(differences, axis=-1)
    kinetic_energy = 0.5 * np.sum(masses * normed_vels**2)
    np.fill_diagonal(rel_positions, np.inf)
    potential_energy = -G * np.sum(masses[:, None] * masses[None, :] / rel_positions) / 2
    energy = kinetic_energy + potential_energy
    return kinetic_energy, potential_energy, energy


def correct_center_of_mass(positions: Array, velocities: Array, masses: Array) -> tuple[Array, Array]:
    """Shift positions and velocities so that the centre of mass is at the origin and at rest."""
    R = sum(positions * masses[:, None]) / masses.sum()
    V = sum(velocities * masses[:, None]) / sum(masses[:, None])
    positions = positions - R[None, :]
    velocities = velocities - V
    return positions, velocities
