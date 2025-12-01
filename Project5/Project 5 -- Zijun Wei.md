## Please check the link below to see the programming part of this Project. The repository includes:

## Link: https://github.com/ZJWei2002/AMS530-Projects/tree/main/Project5

## Project Structure
The `Project5` folder contains the following files and directories:

```
Project5/
├── problem5_1.py                  # Main MPI implementation for Problem 5.1
├── project5_case1_particle_data.txt
├── project5_case1_timings.txt
├── project5_case1_energy_heatmap.png
├── project5_case1_timings.png
├── project5_case2_particle_data.txt
├── project5_case2_timings.txt
├── project5_case2_energy_heatmap.png
├── project5_case2_timings.png
├── project5_case3_particle_data.txt
├── project5_case3_timings.txt
├── project5_case3_energy_heatmap.png
├── project5_case3_timings.png
├── Parallel-Computing-Project5-2025.pdf
└── Parallel-Computing-Project5-GradeSheet-2025.pdf
```

---

## Problem 5.1

### Problem Description ![[Pasted image 20251201004708.png]]

---

## Part 1: Algorithm Description

### 1. Parallel Decomposition Strategy

This implementation uses particle decomposition over MPI ranks:

- Number of MPI ranks = 25.
- Particles are globally indexed from 0 to \(N-1\).
- Each rank \(p\) owns a contiguous block of particle indices of nearly equal size:
  - `distribute_particles_indices` divides the range `[0, N)` so that each rank gets either ⌊N/P⌋ or ⌈N/P⌉ particles.
- Owned particles: rank \(p\) computes \(E_i\) only for its local indices \(i\), but it needs positions of all particles to evaluate interactions.

This design achieves a quasi-uniform workload because the cost per owned particle is roughly proportional to the number of neighbors within the cutoff radius, and every rank owns approximately the same number of i-particles, regardless of spatial distribution.

We broadcast the complete position array to all ranks once per case so that each process can compute its local energies independently.

### 2. Particle Distributions for the Three Cases

The function `compute_subbox_counts(case, params)` computes integer particle counts `n(α, β)` for each sub-box:
- It first forms a weight matrix according to the formula for each case:
  - Case 1: `weights = α + β`
  - Case 2: `weights = |α − β|`
  - Case 3: `weights = α * β`
- It then scales these weights so that the sum of all counts equals exactly N = 15,000:
  - real-valued expected counts are computed as `counts_float = N * weights / weights.sum()`
  - integer counts are obtained via floor + redistribution:
    - start with `floor(counts_float)`
    - distribute any remaining particles to sub-boxes with the largest fractional parts.

The function `generate_particle_positions(case, params)` then:
- Computes sub-box bounds in the 50 × 50 box (each sub-box has side length 10).
- For each sub-box, samples uniformly distributed \((x, y)\) positions using `rng.uniform` within its rectangular bounds.
- Concatenates all samples into a global position array of shape `(N, 2)`.

### 3. Lennard–Jones Potential and Energy Computation

For each local particle \(i\) owned by a rank, the code computes:

$$
E_i = \dfrac{1}{2} \sum_{j \ne i} V_{ij}, \quad
V_{ij} =
\begin{cases}
\dfrac{1}{r_{ij}^{12}} - \dfrac{2}{r_{ij}^{6}}, & r_{ij} < r_c = 3, \\
0, & r_{ij} \ge r_c,
\end{cases}
$$
where \(r_{ij}^2 = (x_i - x_j)^2 + (y_i - y_j)^2\).

Implementation details:
- We work with squared distance \(r_{ij}^2\) to avoid expensive square roots:
  - `r2 = dx*dx + dy*dy`
  - if `r2 >= rc2`, then `Vij = 0`.
  - otherwise, we compute `inv_r2 = 1 / r2`, `inv_r6 = inv_r2 ** 3`,
    and `Vij = (inv_r6 ** 2) - 2.0 * inv_r6`.
- Self-interactions (`j == i`) are skipped.
- Each rank loops over all j = 0..N-1 for each of its local i-particles.

The per-rank complexity is approximately O(N_local × N).
With N = 15,000 and P = 25, we have N_local ≈ 600, so each rank evaluates on the order of 9 million pair checks per case.

### 4. MPI Communication Pattern

- Broadcast positions:
  - Rank 0 generates all positions and broadcasts the full `(N, 2)` array to all ranks using `COMM.bcast`.
- Local computation:
  - Each rank computes energies `E_local` for its own particles without further communication.
- Gather results:
  - Indices and local energies are gathered on rank 0 via `COMM.gather` and assembled into the full array `E_global`.
- Timing collection:
  - Each rank records its local wall-time using `MPI.Wtime()`.
  - Rank 0 gathers these times into a list and saves them to `project5_caseX_timings.txt`.

### 5. Execution Commands

To run all three cases with P = 25 MPI ranks:

```bash
cd Project5
mpiexec -np 25 python problem5_1.py
```

The script will:
- Print a brief progress log on rank 0:
  - `[Project 5.1] Running with 25 MPI ranks.`
  - `=== Case 1 ===`, `=== Case 2 ===`, `=== Case 3 ===`
  - `All cases completed.`
- Generate text and image files for each case in the `Project5` directory.

---
## Part 2: Results

### 1. Energy Visualizations

For each case, we generate a 2D scatter-heatmap of per-particle energy \(E_i\).
Each point represents a particle at position \((x, y)\), colored by its computed energy.

#### Case 1: Per-Particle Energy
![[project5_case1_energy_heatmap.png]]

Observations:
- Particle density gradually increases from the bottom-left corner towards the top-right corner.
- Because the full energy range is used for the color scale, a small number of extremely high-energy particles (from very close pairs) stretches the color bar up to about 10^30.
- As a result, most particles with much smaller energies appear in dark purple, and only a few points in the densest upper-right region reach noticeably brighter colors.

#### Case 2: Per-Particle Energy

![[project5_case2_energy_heatmap.png]]

Observations:
- The diagonal sub-boxes with alpha = beta are empty, creating an obvious staircase pattern of empty squares across the domain.
- Densities increase as we move away from the diagonal, consistent with n(alpha, beta) being proportional to |alpha − beta|.
- Using the complete energy range, the color bar extends to about 10^34, again dominated by a few extreme-energy particles.
- Most particles remain in the dark-purple range, but regions farther from the empty diagonal have slightly brighter hues, reflecting their higher typical interaction energies.

#### Case 3: Per-Particle Energy

![[project5_case3_energy_heatmap.png]]

Observations:
- The density is lowest near the bottom-left corner and highest near the top-right corner, matching the product-based distribution n(alpha, beta) proportional to alpha · beta.
- Compared to Case 1, the density gradient is more extreme, leading to a pronounced clustering in sub-boxes with large (alpha, beta) indices.
- With the full (unclipped) energy range, the maximum reaches roughly 10^36, indicating extremely close pairs in the most crowded regions.
- This again causes most particles to appear in dark colors, but the top-right corner, where density and neighbor counts are highest, still shows relatively brighter points corresponding to larger typical energies.

### 2. Per-Rank Timing Results

For each case, rank 0 writes a text file `project5_caseX_timings.txt` containing the execution time of the energy computation on each rank.

#### Case 1: Timings

![[project5_case1_timings 3.png]]

From `project5_case1_timings.txt`:
- Times per rank are tightly clustered around 32.0 seconds.
- Minimum time ≈ 32.01 s, maximum time ≈ 32.04 s, so the total spread is only about 0.03 s.

#### Case 2: Timings

![[project5_case2_timings 1.png]]

From `project5_case2_timings.txt`:
- Times per rank are clustered around 29.0 seconds.
- Minimum time ≈ 28.88 s, maximum time ≈ 29.00 s, giving a spread of roughly 0.12 s (well below 0.5% of the mean).

#### Case 3: Timings

![[project5_case3_timings 1.png]]

From `project5_case3_timings.txt`:
- Times per rank are clustered around 29.9 seconds.
- Minimum time ≈ 29.82 s, maximum time ≈ 29.90 s, for a spread of about 0.08 s (well under 0.3% of the mean).
- Comparing cases, Cases 2 and 3 are slightly faster than Case 1, consistent with small differences in neighbor counts and memory behavior, but all three show similarly tight timing bands.

---

## Part 3: Performance and Load-Balancing Analysis

### 1. Load Balance

Because we divide particles evenly by index across ranks, each core is responsible for approximately 600 particles:
- The timing plots for all three cases show that all 25 ranks finish within a very narrow time band.
- This is strong evidence that particle decomposition alone is sufficient to achieve the requested quasi-load balance, even when the spatial distribution is highly non-uniform (Cases 2 and 3).

In particular:
- Case 2 has entire diagonal sub-boxes empty, yet per-rank runtimes remain uniform.
- Case 3 has a strong density gradient, but runtimes are still nearly identical.

This demonstrates that:
- The cost per owned particle is dominated by the total number of pair checks with the global set of particles.
- Since every rank processes a similar number of i-particles, the computational work is evenly distributed.

### 2. Computational Cost and Scaling

For each case:
- Each rank processes about 600 i-particles.
- For each i-particle, we loop over all N = 15,000 particles to test which pairs fall within the cutoff.
- This yields O(N² / P) total work, i.e., O(N² / 25).

 The absolute runtimes (≈29–32 seconds per case on 25 cores) are consistent with:
- ~9 million pair checks per rank per case.
- Additional overhead for evaluating the Lennard–Jones potential and accumulating energies.

### 3. Possible Improvements

If further optimization were required, potential extensions include:
- Neighbor lists / cell lists:
  - Avoid testing all N particles for each i by only checking nearby particles, reducing complexity towards \(O(N)\) instead of \(O(N^2)\).
- Hybrid particle + spatial decomposition:
  - Decompose the spatial domain and restrict pair searches to neighboring cells.
- Vectorization and NumPy acceleration:
  - Vectorize inner loops over `j` to exploit SIMD and efficient array operations.

---

## Conclusions
- Implemented a parallel Lennard–Jones energy computation for \(N = 15000\) particles in a 2D box using MPI with particle decomposition.
- Correctly realized all three specified non-uniform particle distributions (Cases 1–3) and generated the required energy visualizations and per-core timing data.
- Experimental timings show excellent load balance across 25 cores for all cases, fulfilling the quasi-load-balance requirement of the assignment.
- The project demonstrates that a simple particle-based decomposition, combined with efficient broadcasting and gathering, can provide both correctness and strong parallel performance for this kind of all-pairs interaction problem.


