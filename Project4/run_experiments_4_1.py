#!/usr/bin/env python3
import subprocess
import sys
import os
import numpy as np
from collections import defaultdict

MPI_CMD = os.environ.get('MPI_CMD', 'mpiexec')
PROCESSORS = [1, 4, 16]
N_GRID = 300

def format_number(num):
    if num is None:
        return "-"
    if isinstance(num, float):
        if num >= 1e6:
            return f"{num:,.0f}"
        else:
            return f"{num:.6f}"
    return f"{num:,}"

def run_experiment(P, task, n_grid=N_GRID):
    try:
        import time
        start_time = time.time()
        result = subprocess.run(
            [MPI_CMD, '-np', str(P), 'python', 'problem4_1.py', task, str(n_grid)],
            capture_output=True,
            text=True,
            timeout=300
        )
        elapsed_time = time.time() - start_time
        
        if result.returncode != 0:
            return None
        
        output = result.stdout
        results = {}
        lines = output.split('\n')
        
        if task in ['surface', 'both']:
            for i, line in enumerate(lines):
                if 'Surface Area:' in line and 'FLOPs' not in line:
                    try:
                        value = float(line.split(':')[1].strip())
                        results['surface_area'] = value
                    except:
                        pass
                if 'Surface Area FLOPs per core:' in line or ('FLOPs per core:' in line and i > 0 and 'Surface' in lines[i-1]):
                    try:
                        value = float(line.split(':')[1].strip())
                        results['surface_flops'] = value
                    except:
                        pass
        
        if task in ['mass', 'both']:
            for i, line in enumerate(lines):
                if ('Total Mass:' in line or 'Mass (t=1):' in line) and 'FLOPs' not in line:
                    try:
                        value = float(line.split(':')[1].strip())
                        results['mass'] = value
                    except:
                        pass
                if 'Mass FLOPs per core:' in line or ('FLOPs per core:' in line and i > 0 and 'Mass' in lines[i-1]):
                    try:
                        value = float(line.split(':')[1].strip())
                        results['mass_flops'] = value
                    except:
                        pass
        
        if results:
            results['elapsed_time'] = elapsed_time
            return results
        else:
            return None
            
    except subprocess.TimeoutExpired:
        return None
    except Exception as e:
        return None

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    surface_results = {}
    mass_results = {}
    
    for P in PROCESSORS:
        result = run_experiment(P, 'surface')
        if result and 'surface_area' in result:
            surface_results[P] = {
                'area': result['surface_area'],
                'flops': result.get('surface_flops', 0),
                'time': result.get('elapsed_time', 0)
            }
    
    surface_row_file = "Surface area (FLOPs)"
    for P in PROCESSORS:
        if P in surface_results:
            flops = surface_results[P]['flops']
            surface_row_file += f"  {flops:.0f}"
        else:
            surface_row_file += "  -"
    
    sa_row_file = "Surface area (value)"
    for P in PROCESSORS:
        if P in surface_results:
            sa_row_file += f"  {surface_results[P]['area']:.6f}"
        else:
            sa_row_file += "  -"
    
    for P in PROCESSORS:
        result = run_experiment(P, 'mass')
        if result and 'mass' in result:
            mass_results[P] = {
                'mass': result['mass'],
                'flops': result.get('mass_flops', 0),
                'time': result.get('elapsed_time', 0)
            }
    
    mass_row_file = "Heart mass (FLOPs)"
    for P in PROCESSORS:
        if P in mass_results:
            flops = mass_results[P]['flops']
            mass_row_file += f"  {flops:.0f}"
        else:
            mass_row_file += "  -"
    
    mass_val_row_file = "Heart mass (value)"
    for P in PROCESSORS:
        if P in mass_results:
            mass_val_row_file += f"  {mass_results[P]['mass']:.6f}"
        else:
            mass_val_row_file += "  -"
    
    with open('results_4_1.txt', 'w') as f:
        f.write("Problem 4.1 Results\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("PART 1: Surface Area Computation\n")
        f.write("=" * 60 + "\n\n")
        f.write("Results:\n")
        f.write(f"{'':<20} {'P=1':<15} {'P=4':<15} {'P=16':<15}\n")
        f.write("-" * 60 + "\n")
        f.write(surface_row_file + "\n")
        f.write(sa_row_file + "\n")
        
        f.write("\n\n" + "=" * 60 + "\n")
        f.write("PART 2: Mass Computation\n")
        f.write("=" * 60 + "\n\n")
        f.write("Results:\n")
        f.write(f"{'':<20} {'P=1':<15} {'P=4':<15} {'P=16':<15}\n")
        f.write("-" * 60 + "\n")
        f.write(mass_row_file + "\n")
        f.write(mass_val_row_file + "\n")
        
        f.write("\n\n" + "=" * 60 + "\n")
        f.write("OVERALL SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write("Results:\n")
        f.write(f"{'':<20} {'P=1':<15} {'P=4':<15} {'P=16':<15}\n")
        f.write("-" * 60 + "\n")
        f.write(surface_row_file + "\n")
        f.write(sa_row_file + "\n")
        f.write(mass_row_file + "\n")
        f.write(mass_val_row_file + "\n")

if __name__ == "__main__":
    main()

