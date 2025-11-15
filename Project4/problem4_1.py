#!/usr/bin/env python3
import numpy as np
from mpi4py import MPI
import sys
import time

flop_count = 0

def count_flops(operations):
    """Count floating point operations"""
    global flop_count
    flop_count += operations

def heart_equation(x, y, z):
    """
    Evaluate the heart surface equation.
    Returns value at (x, y, z). Surface is where this equals 0.
    """
    term1 = (x**2 + (1.5 * y)**2 + z**2 - 1)**3
    term2 = (x**2 + (0.02 * y)**2) * z**3
    result = term1 - term2
    count_flops(15)  # Approximate: 2 mults, 1 add, 1 sub, 1 pow(3) = ~15 ops
    return result

def is_inside_heart(x, y, z):
    """Check if point is inside the heart (where equation < 0)"""
    value = heart_equation(x, y, z)
    count_flops(1)  # comparison
    return value < 0

def density(x, y, z, t, rho0=1.0, f=60.0):
    """
    Compute instantaneous density at point (x, y, z) at time t.
    ρ(x, y, z; t) = ρ₀(x, y, z) [(1 + sin(2πft)) / 2]
    """
    sin_arg = 2 * np.pi * f * t
    sin_val = np.sin(sin_arg)
    density_val = rho0 * (1 + sin_val) / 2
    count_flops(5)  # 2 mults, 1 add, 1 sin, 1 div
    return density_val

def compute_surface_area_parallel_v2(comm, rank, size, n_grid=200):
    """
    Surface area computation using parallel grid-based integration.
    Uses the formula: Surface Area = ∫∫ |∇f| / max(|fx|, |fy|, |fz|) dA
    where we integrate over the projection with the largest gradient component.
    """
    global flop_count
    flop_count = 0
    
    x_min, x_max = -2.0, 2.0
    y_min, y_max = -2.0, 2.0
    z_min, z_max = -2.0, 2.0
    
    dx = (x_max - x_min) / n_grid
    dy = (y_max - y_min) / n_grid
    dz = (z_max - z_min) / n_grid
    
    points_per_proc = n_grid // size
    remainder = n_grid % size
    start_idx = rank * points_per_proc + min(rank, remainder)
    end_idx = start_idx + points_per_proc + (1 if rank < remainder else 0)
    
    surface_area_local = 0.0
    h = 1e-6
    surface_threshold = 0.05
    
    for i in range(start_idx, end_idx):
        for j in range(n_grid):
            for k in range(n_grid):
                x = x_min + (i + 0.5) * dx
                y = y_min + (j + 0.5) * dy
                z = z_min + (k + 0.5) * dz
                
                f_val = heart_equation(x, y, z)
                
                if abs(f_val) < surface_threshold:
                    fx = (heart_equation(x + h, y, z) - heart_equation(x - h, y, z)) / (2 * h)
                    fy = (heart_equation(x, y + h, z) - heart_equation(x, y - h, z)) / (2 * h)
                    fz = (heart_equation(x, y, z + h) - heart_equation(x, y, z - h)) / (2 * h)
                    
                    grad_mag = np.sqrt(fx**2 + fy**2 + fz**2)
                    # Note: heart_equation FLOPs already counted in function calls
                    # Additional: 6 subtractions, 3 divisions, 3 squares, 2 additions, 1 sqrt
                    count_flops(6 + 3 + 3 + 2 + 1)  # ~15 FLOPs for gradient computation (beyond function evals)
                    
                    if grad_mag > 1e-10:
                        max_grad_comp = max(abs(fx), abs(fy), abs(fz))
                        
                        if max_grad_comp > 1e-10:
                            if abs(fx) == max_grad_comp:
                                dS = (grad_mag / abs(fx)) * dy * dz
                            elif abs(fy) == max_grad_comp:
                                dS = (grad_mag / abs(fy)) * dx * dz
                            else:
                                dS = (grad_mag / abs(fz)) * dx * dy
                            
                            surface_area_local += dS
                            count_flops(5)  # 3 abs, 1 max comparison, 1 division, 2 mults, 1 add
    
    total_surface_area = comm.reduce(surface_area_local, op=MPI.SUM, root=0)
    total_flops = comm.reduce(flop_count, op=MPI.SUM, root=0)
    
    if rank == 0:
        flops_per_core = total_flops / size
        return total_surface_area, flops_per_core
    else:
        return None, None

def compute_mass_parallel(comm, rank, size, t=1.0, n_grid=200, rho0=1.0, f=60.0):
    """
    Compute total mass at time t using parallel volume integration.
    Mass = ∫∫∫ ρ(x, y, z; t) dV over the heart volume
    """
    global flop_count
    flop_count = 0
    
    x_min, x_max = -2.0, 2.0
    y_min, y_max = -2.0, 2.0
    z_min, z_max = -2.0, 2.0
    
    dx = (x_max - x_min) / n_grid
    dy = (y_max - y_min) / n_grid
    dz = (z_max - z_min) / n_grid
    
    points_per_proc = n_grid // size
    remainder = n_grid % size
    start_idx = rank * points_per_proc + min(rank, remainder)
    end_idx = start_idx + points_per_proc + (1 if rank < remainder else 0)
    
    mass_local = 0.0
    
    for i in range(start_idx, end_idx):
        for j in range(n_grid):
            for k in range(n_grid):
                x = x_min + (i + 0.5) * dx
                y = y_min + (j + 0.5) * dy
                z = z_min + (k + 0.5) * dz
                
                if is_inside_heart(x, y, z):
                    rho = density(x, y, z, t, rho0, f)
                    dV = dx * dy * dz
                    mass_local += rho * dV
                    count_flops(4)  # 3 mults for dV, 1 mult for rho*dV, 1 add
    
    total_mass = comm.reduce(mass_local, op=MPI.SUM, root=0)
    total_flops = comm.reduce(flop_count, op=MPI.SUM, root=0)
    
    if rank == 0:
        flops_per_core = total_flops / size
        return total_mass, flops_per_core
    else:
        return None, None

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if len(sys.argv) < 2:
        if rank == 0:
            print("Usage: mpiexec -np P python problem4_1.py <task> [n_grid]")
            print("  task: 'surface' or 'mass' or 'both'")
            print("  n_grid: grid resolution (default: 300)")
        return
    
    task = sys.argv[1].lower()
    n_grid = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    
    if rank == 0:
        print(f"Problem 4.1 - Processors: {size}, Grid: {n_grid}×{n_grid}×{n_grid}")
        print("=" * 60)
    
    results = {}
    
    if task in ['surface', 'both']:
        if rank == 0:
            print("\nComputing surface area...")
        
        start_time = time.time()
        surface_area, flops_per_core = compute_surface_area_parallel_v2(comm, rank, size, n_grid)
        elapsed_time = time.time() - start_time
        
        if rank == 0:
            print(f"Surface Area: {surface_area:.6f}")
            print(f"Surface Area FLOPs per core: {flops_per_core:.0f}")
            print(f"Time: {elapsed_time:.6f} seconds")
            results['surface_area'] = surface_area
            results['surface_flops'] = flops_per_core
    
    if task in ['mass', 'both']:
        if rank == 0:
            print("\nComputing mass at t=1...")
        
        start_time = time.time()
        mass, flops_per_core = compute_mass_parallel(comm, rank, size, t=1.0, n_grid=n_grid)
        elapsed_time = time.time() - start_time
        
        if rank == 0:
            print(f"Total Mass: {mass:.6f}")
            print(f"Mass FLOPs per core: {flops_per_core:.0f}")
            print(f"Time: {elapsed_time:.6f} seconds")
            results['mass'] = mass
            results['mass_flops'] = flops_per_core
    
    if rank == 0:
        print("\n" + "=" * 60)
        print("Results Summary:")
        if 'surface_area' in results:
            print(f"  Surface Area: {results['surface_area']:.6f}")
            print(f"  Surface Area FLOPs/core: {results['surface_flops']:.0f}")
        if 'mass' in results:
            print(f"  Mass (t=1): {results['mass']:.6f}")
            print(f"  Mass FLOPs/core: {results['mass_flops']:.0f}")

if __name__ == "__main__":
    main()

