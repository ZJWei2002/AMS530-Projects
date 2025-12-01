from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from mpi4py import MPI


try:
    # Matplotlib is only required on rank 0 when generating plots.
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - plotting is optional in headless envs
    plt = None  # type: ignore


COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()


@dataclass
class SimulationParams:
    N: int = 15_000          # total number of particles
    L: float = 50.0          # domain is [0, L] x [0, L]
    M: int = 5               # number of sub-boxes per dimension (5x5 grid)
    rc: float = 3.0          # cutoff radius for Lennard-Jones potential
    seed: int = 12345        # base RNG seed


def compute_subbox_counts(case: int, params: SimulationParams) -> np.ndarray:
    """
    Compute the number of particles in each sub-box for a given case.

    Returns
    -------
    counts : ndarray of shape (M, M), dtype=int
        counts[alpha, beta] is n(alpha+1, beta+1) in the problem statement.
    """
    N, M = params.N, params.M

    alpha = np.arange(1, M + 1, dtype=float)
    beta = np.arange(1, M + 1, dtype=float)
    A, B = np.meshgrid(alpha, beta, indexing="ij")

    if case == 1:
        # Case 1: weights proportional to (α + β)
        weights = A + B
    elif case == 2:
        # Case 2: weights proportional to |α − β|
        weights = np.abs(A - B)
    elif case == 3:
        # Case 3: weights proportional to α * β
        weights = A * B
    else:
        raise ValueError(f"Unsupported case: {case}")

    total_w = weights.sum()
    if total_w <= 0:
        raise RuntimeError("Sum of weights is non-positive; cannot assign particles.")

    # Real-valued expected counts.
    counts_float = N * weights / total_w

    # Integer counts via floor + redistribution of remaining particles.
    counts_int = np.floor(counts_float).astype(int)
    deficit = N - int(counts_int.sum())

    if deficit > 0:
        # Give extra particles to cells with the largest fractional part.
        remainders = (counts_float - counts_int).ravel()
        order = np.argsort(remainders)[::-1]  # descending
        for idx in order[:deficit]:
            counts_int.flat[idx] += 1
    elif deficit < 0:
        # Remove particles from cells with the smallest fractional part,
        # while keeping counts non-negative.
        remainders = (counts_float - counts_int).ravel()
        order = np.argsort(remainders)  # ascending
        removed = 0
        for idx in order:
            if removed == -deficit:
                break
            if counts_int.flat[idx] > 0:
                counts_int.flat[idx] -= 1
                removed += 1

    assert counts_int.sum() == N, "Particle counts do not sum to N."
    return counts_int


def generate_particle_positions(
    case: int, params: SimulationParams
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate particle positions for a given case.

    This is performed only on rank 0 and then broadcast.

    Returns
    -------
    positions : ndarray, shape (N, 2)
        Cartesian coordinates (x, y) of all particles.
    subbox_indices : ndarray, shape (N, 2), dtype=int
        Integer indices (alpha, beta) in {1,..,M}^2 for each particle.
    """
    N, L, M = params.N, params.L, params.M
    dx = L / M

    counts = compute_subbox_counts(case, params)

    rng = np.random.default_rng(params.seed + case)

    positions_list = []
    subbox_list = []

    for ai in range(M):        # 0-based index for alpha
        for bj in range(M):    # 0-based index for beta
            n_box = counts[ai, bj]
            if n_box == 0:
                continue

            x_min = ai * dx
            x_max = (ai + 1) * dx
            y_min = bj * dx
            y_max = (bj + 1) * dx

            xs = rng.uniform(x_min, x_max, size=n_box)
            ys = rng.uniform(y_min, y_max, size=n_box)
            positions_box = np.stack((xs, ys), axis=1)

            positions_list.append(positions_box)
            # Store 1-based (α, β) indices to match the problem statement.
            subbox_list.append(np.column_stack([
                np.full(n_box, ai + 1, dtype=int),
                np.full(n_box, bj + 1, dtype=int),
            ]))

    positions = np.concatenate(positions_list, axis=0)
    subbox_indices = np.concatenate(subbox_list, axis=0)

    assert positions.shape == (N, 2), "Generated wrong number of particles."

    return positions, subbox_indices


def distribute_particles_indices(N: int) -> np.ndarray:
    """
    Determine which particle indices belong to this rank (particle decomposition).

    We use a simple contiguous block decomposition of the index range [0, N).
    """
    counts = [N // SIZE + (1 if r < N % SIZE else 0) for r in range(SIZE)]
    displs = np.cumsum([0] + counts[:-1])

    start = displs[RANK]
    stop = start + counts[RANK]
    return np.arange(start, stop, dtype=int)


def compute_local_energies(
    positions: np.ndarray, idx_local: np.ndarray, params: SimulationParams
) -> np.ndarray:
    """
    Compute E_i for all particles owned by this rank.

    For each local particle i, we sum V_ij over all j != i and then multiply
    by 1/2, as defined in the project handout:

        E_i = (1/2) * sum_{j ≠ i} V_ij

    The Lennard-Jones potential is truncated at rc:

        V_ij = 1/r^12 - 2/r^6,  r < rc
              0,               r >= rc

    Implementation details
    ----------------------
    - We avoid sqrt by working with r^2 and the identities
          r^6  = (r^2)^3
          1/r^6 = (1/r^2)^3
          1/r^12 = (1/r^6)^2
    - Complexity per rank is O(N_local * N). For N = 15,000 and 25 ranks,
      N_local ≈ 600, giving about 9e6 pair checks per case per rank.
    """
    N = positions.shape[0]
    rc2 = params.rc * params.rc

    E_local = np.zeros(len(idx_local), dtype=np.float64)

    for k, i in enumerate(idx_local):
        xi, yi = positions[i]
        Ei = 0.0
        for j in range(N):
            if j == i:
                continue
            dx = xi - positions[j, 0]
            dy = yi - positions[j, 1]
            r2 = dx * dx + dy * dy
            if r2 >= rc2 or r2 == 0.0:
                continue

            inv_r2 = 1.0 / r2
            inv_r6 = inv_r2 ** 3
            Vij = (inv_r6 ** 2) - 2.0 * inv_r6
            Ei += Vij

        E_local[k] = 0.5 * Ei

    return E_local


def gather_energies(
    idx_local: np.ndarray, E_local: np.ndarray, N: int
) -> np.ndarray | None:
    """
    Gather local energies from all ranks and assemble the global E array on rank 0.
    """
    all_indices = COMM.gather(idx_local, root=0)
    all_energies = COMM.gather(E_local, root=0)

    if RANK != 0:
        return None

    E_global = np.empty(N, dtype=np.float64)
    for indices, energies in zip(all_indices, all_energies):
        E_global[indices] = energies
    return E_global


def plot_case_results(
    case: int,
    params: SimulationParams,
    positions: np.ndarray,
    energies: np.ndarray,
    timings: np.ndarray,
) -> None:
    """
    Produce the required plots for a given case (on rank 0).
    """
    if plt is None:
        # Headless or matplotlib missing; just skip plotting.
        return

    # Scatter heatmap of per-particle energy using the full range of E_i.
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(
        positions[:, 0],
        positions[:, 1],
        c=energies,
        s=5,
        cmap="viridis",
    )
    plt.colorbar(sc, label=r"$E_i$")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Project 5.1 - Case {case}: per-particle energy")
    plt.xlim(0.0, params.L)
    plt.ylim(0.0, params.L)
    plt.tight_layout()
    plt.savefig(f"project5_case{case}_energy_heatmap.png", dpi=200)
    plt.close()

    # Bar plot of per-rank timings.
    ranks = np.arange(timings.size)
    plt.figure(figsize=(6, 4))
    plt.bar(ranks, timings, color="tab:blue", edgecolor="black")
    mean_t = float(timings.mean())
    plt.axhline(mean_t, color="red", linestyle="--", linewidth=1.5,
                label=f"mean = {mean_t:.2f} s")
    # Zoom the y-axis to highlight small variations between ranks.
    t_min = float(timings.min())
    t_max = float(timings.max())
    padding = max(0.05 * (t_max - t_min), 0.05)
    plt.ylim(t_min - padding, t_max + padding)
    plt.xlabel("MPI rank")
    plt.ylabel("wall-time (s)")
    plt.title(f"Project 5.1 - Case {case}: per-rank timings")
    plt.grid(axis="y", alpha=0.3)
    plt.legend(loc="upper right", fontsize="small")
    plt.tight_layout()
    plt.savefig(f"project5_case{case}_timings.png", dpi=200)
    plt.close()


def run_case(case: int, params: SimulationParams) -> None:
    """Run a single test case end-to-end."""
    if RANK == 0:
        positions, subbox_indices = generate_particle_positions(case, params)
    else:
        positions = None
        subbox_indices = None

    # Broadcast positions to all ranks.
    if RANK == 0:
        buf = positions
    else:
        buf = None
    buf = COMM.bcast(buf, root=0)
    positions = buf

    # Each rank determines which particles it owns.
    idx_local = distribute_particles_indices(params.N)

    # Time just the energy computation on each rank.
    COMM.Barrier()
    t0 = MPI.Wtime()
    E_local = compute_local_energies(positions, idx_local, params)
    COMM.Barrier()
    t1 = MPI.Wtime()
    local_time = t1 - t0

    # Gather timings on rank 0.
    timings = COMM.gather(local_time, root=0)

    # Gather energies on rank 0.
    E_global = gather_energies(idx_local, E_local, params.N)

    if RANK == 0:
        timings_arr = np.array(timings, dtype=float)
        # Save raw timings to a text file for later use in the report.
        np.savetxt(
            f"project5_case{case}_timings.txt",
            np.column_stack([np.arange(timings_arr.size), timings_arr]),
            header="rank   time_seconds",
            fmt="%3d %.6f",
        )
        # Also save per-particle data in case it is useful in post-processing.
        np.savetxt(
            f"project5_case{case}_particle_data.txt",
            np.column_stack([positions, E_global]),
            header="x   y   E_i",
        )

        plot_case_results(case, params, positions, E_global, timings_arr)


def main() -> None:
    if SIZE < 1:
        raise RuntimeError("MPI communicator must contain at least one rank.")

    params = SimulationParams()

    if RANK == 0:
        print(f"[Project 5.1] Running with {SIZE} MPI ranks.")

    for case in (1, 2, 3):
        if RANK == 0:
            print(f"\n=== Case {case} ===")
        run_case(case, params)
        COMM.Barrier()

    if RANK == 0:
        print("\nAll cases completed.")


if __name__ == "__main__":
    main()


