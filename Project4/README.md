# Problem 4.1
## Problem Description![[Pasted image 20251114203510.png]]

---
## Part 1: Surface Area Computation

### 1. Algorithm Description
The surface area computation uses a parallel grid-based numerical integration method with gradient computation for implicit surfaces.

#### Key Concepts
- **Implicit Surface**: The heart is defined by the equation (x² + (3/2 y)² + z² - 1)³ - (x² + (1/50 y)²) z³ = 0
- **Domain Decomposition**: The 3D grid is divided along the x-axis across processors
- **Gradient-Based Integration**: Uses numerical gradient to compute surface area elements
- **Adaptive Projection**: Chooses projection with largest gradient component to avoid numerical instability

#### Algorithm Steps
For a grid of size n_grid × n_grid × n_grid over bounding box [-2, 2]³:

1. **Domain Decomposition**: 
   - Divide grid along x-axis: each processor handles slice [start_idx, end_idx)
   - Load balancing: remainder points distributed to first processors

2. **For each grid point (i, j, k) in processor's slice**:
   - Compute coordinates: x = x_min + (i + 0.5) × dx
   - Evaluate heart equation: f_val = heart_equation(x, y, z)
   
3. **If |f_val| < surface_threshold (near surface)**:
   - Compute numerical gradient using finite differences:
     - fx = [f(x+h,y,z) - f(x-h,y,z)] / (2h)
     - fy = [f(x,y+h,z) - f(x,y-h,z)] / (2h)
     - fz = [f(x,y,z+h) - f(x,y,z-h)] / (2h)
   - Compute gradient magnitude: |∇f| = √(fx² + fy² + fz²)
   - Find max gradient component: max(|fx|, |fy|, |fz|)
   - Compute surface area element:
     - If |fx| is largest: dS = (|∇f| / |fx|) × dy × dz
     - If |fy| is largest: dS = (|∇f| / |fy|) × dx × dz
     - If |fz| is largest: dS = (|∇f| / |fz|) × dx × dy
   - Accumulate: surface_area_local += dS

4. **Parallel Reduction**:
   - total_surface_area = MPI_Reduce(surface_area_local, SUM, root=0)
   - total_flops = MPI_Reduce(flop_count, SUM, root=0)

#### Communication Pattern
- **Minimal communication**: Each processor works independently on its slice
- **Single reduction**: Only one MPI_Reduce operation at the end
- **No data dependencies**: No communication needed during computation

#### Time Complexity
- **Computation**: O(n_grid³ / P) per processor
- **Communication**: O(1) - single reduction operation
- **Overall**: O(n_grid³ / P) with minimal communication overhead

#### Execution Commands
To compute surface area for different processor counts:

```bash
# Compute surface area with P=1
mpiexec -np 1 python problem4_1.py surface 300

# Compute surface area with P=4
mpiexec -np 4 python problem4_1.py surface 300

# Compute surface area with P=16
mpiexec -np 16 python problem4_1.py surface 300
```

To run all experiments automatically and generate results table:

```bash
python run_experiments_4_1.py
```

This script runs experiments for P ∈ {1, 4, 16} and generates the results file `results_4_1.txt`.

---
### 2. Results(from `results_4_1.txt`)

![[Pasted image 20251114211814.png]]
#### Computed Values
**Surface Area:**
- P=1: 202.932103
- P=4: 202.932103
- P=16: 202.932103

**FLOPs per Core:**
- P=1: 511,257,800 FLOPs/core
- P=4: 127,814,450 FLOPs/core
- P=16: 31,953,612 FLOPs/core

---
### 3. FLOPs Analysis

#### FLOPs Scaling Analysis
**Surface Area Computation:**
- P=1: 511,257,800 FLOPs/core (baseline)
- P=4: 127,814,450 FLOPs/core (ratio: 511.3M / 127.8M = 4.00×)
- P=16: 31,953,612 FLOPs/core (ratio: 511.3M / 32.0M = 15.98×)

#### Key Observations
- **Perfect parallel scaling**: FLOP counts per core decrease by exactly factor of P
- **Ideal load distribution**: Each processor performs approximately 1/P of the total work
- **Consistent results**: Identical computed values across all processor counts confirm correctness
- **Efficiency**: E(P) ≈ 100% for all processor counts, indicating perfect parallel efficiency
- **Minimal communication overhead**: Single reduction operation has negligible impact on FLOP counts

---
### 4. Performance Analysis
#### Performance Trends
For fixed grid size (n_grid = 300), FLOP counts per core scale inversely with processor count:
- **P=1**: 511.3M FLOPs/core
- **P=4**: 127.8M FLOPs/core (exactly 1/4)
- **P=16**: 32.0M FLOPs/core (exactly 1/16)

Computed surface area remains constant across all processor counts:
- **P=1**: 202.932103
- **P=4**: 202.932103
- **P=16**: 202.932103

#### Conclusions
Surface area computation demonstrates excellent parallel scaling with near-perfect efficiency. The algorithm achieves ideal speedup because communication overhead is minimal (single reduction) and work is evenly distributed. Results are identical across all processor counts, confirming both correctness and numerical stability of the parallel implementation.

---
---
## Part 2: Mass Computation

### 1. Algorithm Description
The mass computation uses parallel volume integration over the heart volume with time-dependent density.

#### Key Concepts
- **Volume Integration**: Mass = ∫∫∫ ρ(x, y, z; t) dV over the heart volume
- **Time-Dependent Density**: ρ(x, y, z; t) = ρ₀ × [(1 + sin(2πft)) / 2]
- **Inside/Outside Test**: Uses sign of heart equation (f < 0 means inside)
- **Domain Decomposition**: Same x-axis decomposition as surface area computation

#### Algorithm Steps
For a grid of size n_grid × n_grid × n_grid over bounding box [-2, 2]³:

1. **Domain Decomposition**: 
   - Same as surface area: divide grid along x-axis
   - Each processor handles slice [start_idx, end_idx)

2. **For each grid point (i, j, k) in processor's slice**:
   - Compute coordinates: x = x_min + (i + 0.5) × dx
   - Check if inside heart: is_inside_heart(x, y, z)
     - Returns true if heart_equation(x, y, z) < 0

3. **If inside heart**:
   - Compute density at time t: ρ = rho0 × [(1 + sin(2πft)) / 2]
   - Volume element: dV = dx × dy × dz
   - Accumulate: mass_local += ρ × dV

4. **Parallel Reduction**:
   - total_mass = MPI_Reduce(mass_local, SUM, root=0)
   - total_flops = MPI_Reduce(flop_count, SUM, root=0)

#### Communication Pattern
- **Minimal communication**: Each processor works independently
- **Single reduction**: Only one MPI_Reduce operation at the end
- **No data dependencies**: No communication during computation

#### Time Complexity
- **Computation**: O(n_grid³ / P) per processor, but only processes interior points
- **Communication**: O(1) - single reduction operation
- **Overall**: O(n_grid³ / P) with minimal communication overhead

#### Execution Commands
To compute mass for different processor counts:

```bash
# Compute mass with P=1
mpiexec -np 1 python problem4_1.py mass 300

# Compute mass with P=4
mpiexec -np 4 python problem4_1.py mass 300

# Compute mass with P=16
mpiexec -np 16 python problem4_1.py mass 300
```

To run all experiments automatically and generate results table:

```bash
python run_experiments_4_1.py
```

This script runs experiments for P ∈ {1, 4, 16} and generates the results file `results_4_1.txt`.

---
### 2. Results (from `results_4_1.txt`)

![[Pasted image 20251114212310.png]]
#### Computed Values
**Mass (t=1):**
- P=1: 1.658121
- P=4: 1.658121
- P=16: 1.658121

**FLOPs per Core:**
- P=1: 444,591,360 FLOPs/core
- P=4: 111,147,840 FLOPs/core
- P=16: 27,786,960 FLOPs/core

---
### 3. FLOPs Analysis

#### FLOPs Scaling Analysis
**Mass Computation:**
- P=1: 444,591,360 FLOPs/core (baseline)
- P=4: 111,147,840 FLOPs/core (ratio: 444.6M / 111.1M = 4.00×)
- P=16: 27,786,960 FLOPs/core (ratio: 444.6M / 27.8M = 16.00×)

#### Key Observations
- **Perfect parallel scaling**: FLOP counts per core decrease by exactly factor of P
- **Ideal load distribution**: Each processor performs approximately 1/P of the total work
- **Consistent results**: Identical computed values across all processor counts confirm correctness
- **Efficiency**: E(P) = 100% for all processor counts, indicating perfect parallel efficiency
- **Minimal communication overhead**: Single reduction operation has negligible impact on FLOP counts

---
### 4. Performance Analysis
#### Performance Trends
For fixed grid size (n_grid = 300), FLOP counts per core scale inversely with processor count:
- **P=1**: 444.6M FLOPs/core
- **P=4**: 111.1M FLOPs/core (exactly 1/4)
- **P=16**: 27.8M FLOPs/core (exactly 1/16)

Computed mass remains constant across all processor counts:
- **P=1**: 1.658121
- **P=4**: 1.658121
- **P=16**: 1.658121

**Density Verification:**
- At t=1, f=60 Hz: sin(2π × 60 × 1) = sin(120π) = 0
- Therefore: ρ = ρ₀ × (1 + 0) / 2 = 0.5
- Mass = 1.658121 → Volume = 1.658121 / 0.5 = 3.316242

#### Conclusions
Mass computation demonstrates perfect parallel scaling with 100% efficiency. The algorithm achieves ideal speedup because communication overhead is minimal (single reduction) and work is evenly distributed. Results are identical across all processor counts, confirming both correctness and numerical stability. The computed mass value (1.658121) is consistent with the analytical density value (0.5) at t=1, providing additional verification of correctness.
