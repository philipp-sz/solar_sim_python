import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

from calculations.bodies import masses, positions, velocities
from calculations.calc_physics import (calc_accelerations,
                                       calc_differences,
                                       correct_center_of_mass,
                                       step_euler,
                                       step_velocity_verlet,
                                       get_energy,
                                       StepFunction)


def plot_energy_error(runs: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]],
                      log: bool = True, title: str | None = None) -> None:
    """Plot the relative energy error over time for each integrator run."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, (times, energies) in runs.items():
        energies = np.asarray(energies)
        rel = (energies - energies[0]) / abs(energies[0])
        ax.plot(np.asarray(times) / 365.25, np.abs(rel) if log else rel, # 365.25 days per year: x axis in years
                label=label, lw=1)
    ax.set_xlabel("Time [years]")
    ax.set_ylabel("|ΔE / E₀|" if log else "ΔE / E₀")
    ax.set_yscale("log") if log else ax.axhline(0, color="black", lw=0.8)
    ax.grid(alpha=0.3); ax.legend(); ax.set_title(title)
    fig.tight_layout(); plt.show()

def simulate(step: StepFunction, pos: NDArray[np.float64], vel: NDArray[np.float64],
             masses: NDArray[np.float64], dt: float, n_steps: int
             ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Run `n_steps` steps of the given integrator and return the times and the total energy at each step."""
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


if __name__ == "__main__":
    positions, velocities = correct_center_of_mass(positions, velocities, masses)
    T = 365.25 # Simulated time in days (one Julian year)
    n = 365 # Number of steps, so dt is about one day
    dt = T / n
    integrators: dict[str, StepFunction] = {"Euler": step_euler, "Velocity Verlet": step_velocity_verlet}
    runs = {name: simulate(f, positions, velocities, masses, dt, n)
            for name, f in integrators.items()}
    plot_energy_error(runs)
