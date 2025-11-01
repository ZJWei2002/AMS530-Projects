import subprocess
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

MPI_CMD = os.environ.get('MPI_CMD', 'mpiexec')
PROCESSORS = [4, 16, 64]
MATRIX_SIZES = [256, 1024, 4096]
DEFAULT_TIMEOUT = 60

def run_fox_experiment(P, N, timeout=None):
    sqrt_P = int(np.sqrt(P))
    block_size = N // sqrt_P
    
    if timeout is None:
        timeout = DEFAULT_TIMEOUT
    
    print(f"  Running Fox: P={P}, N={N}, block_size={block_size}...", end=" ")
    
    try:
        result = subprocess.run(
            [MPI_CMD, '-np', str(P), 'python', 'fox_multiply.py', 
             str(block_size), 'test'],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode == 0:
            # Extract time from output
            for line in result.stdout.split('\n'):
                if 'Execution time:' in line:
                    time_str = line.split(':')[1].strip().split()[0]
                    time_val = float(time_str)
                    print(f"Time: {time_val:.6f}s")
                    return time_val
            print("X (time not found)")
            return None
        else:
            print(f"X Error: {result.stderr[:100]}")
            return None
    except subprocess.TimeoutExpired:
        print(f"X Timeout ({timeout}s exceeded)")
        return None
    except Exception as e:
        print(f"X Exception: {str(e)[:50]}")
        return None

def run_strassen_experiment(P, N, timeout=None):
    print(f"  Running Strassen: P={P}, N={N}...", end=" ")
    
    try:
        result = subprocess.run(
            [MPI_CMD, '-np', str(P), 'python', 'strassen_multiply.py', 
             str(N), 'test'],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode == 0:
            # Extract time from output
            for line in result.stdout.split('\n'):
                if 'Execution time:' in line:
                    time_str = line.split(':')[1].strip().split()[0]
                    time_val = float(time_str)
                    print(f"Time: {time_val:.6f}s")
                    return time_val
            print("X (time not found)")
            return None
        else:
            print(f"X Error: {result.stderr[:100]}")
            return None
    except subprocess.TimeoutExpired:
        print(f"X Timeout ({timeout}s exceeded)")
        return None
    except Exception as e:
        print(f"X Exception: {str(e)[:50]}")
        return None

def collect_timing_data():
    print("=" * 80)
    print("PERFORMANCE TESTING AND TIMING DATA COLLECTION")
    print("=" * 80)
    
    results = {
        'fox': {},
        'strassen': {}
    }
    
    print("\n" + "=" * 80)
    print("FOX'S ALGORITHM")
    print("=" * 80)
    
    for P in PROCESSORS:
        for N in MATRIX_SIZES:
            time_val = run_fox_experiment(P, N)
            if time_val is not None:
                results['fox'][(P, N)] = time_val
    
    print("\n" + "=" * 80)
    print("STRASSEN'S ALGORITHM")
    print("=" * 80)
    
    if os.path.exists('strassen_multiply.py'):
        for P in PROCESSORS:
            for N in MATRIX_SIZES:
                time_val = run_strassen_experiment(P, N)
                if time_val is not None:
                    results['strassen'][(P, N)] = time_val
    else:
        print("  Strassen algorithm not yet implemented. Skipping...")
    
    print("\n" + "=" * 80)
    print("SAVING TIMING DATA")
    print("=" * 80)
    
    os.makedirs('results', exist_ok=True)
    
    with open('results/timing_results.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PROBLEM 3.1: TIMING RESULTS FOR ALL (P, N) COMBINATIONS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("FOX'S ALGORITHM\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'P':<6} {'N':<8} {'Time (seconds)':<15}\n")
        f.write("-" * 80 + "\n")
        for (P, N), t in sorted(results['fox'].items()):
            f.write(f"{P:<6} {N:<8} {t:<15.6f}\n")
        
        f.write("\nSTRASSEN'S ALGORITHM\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'P':<6} {'N':<8} {'Time (seconds)':<15}\n")
        f.write("-" * 80 + "\n")
        if results['strassen']:
            for (P, N), t in sorted(results['strassen'].items()):
                f.write(f"{P:<6} {N:<8} {t:<15.6f}\n")
        else:
            f.write("(Not implemented yet)\n")
    
    print("  Results saved to: results/timing_results.txt")
    
    return results

def calculate_speedup(times, base_P=None):
    speedups = {}
    
    if base_P is None:
        available_Ps = sorted(set(P for (P, N) in times.keys()))
        if not available_Ps:
            return speedups
        base_P = available_Ps[0]
    
    for N in MATRIX_SIZES:
        base_time = times.get((base_P, N), None)
        if base_time is None:
            continue
        
        speedups_N = {}
        for P in PROCESSORS:
            time_P = times.get((P, N), None)
            if time_P is not None and time_P > 0:
                speedups_N[P] = base_time / time_P
        if speedups_N:
            speedups[N] = speedups_N
    
    return speedups

def plot_speedup_curves(results):
    print("\n" + "=" * 80)
    print("TASK 5: GENERATING SPEEDUP CURVES")
    print("=" * 80)
    
    if not results['fox']:
        print("  No Fox results to plot. Run experiments first.")
        return
    
    fox_speedups = calculate_speedup(results['fox'])
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, N in enumerate(MATRIX_SIZES):
        ax = axes[idx]
        
        if N not in fox_speedups:
            ax.text(0.5, 0.5, f'No data for N={N}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Matrix Size: {N}×{N}', fontsize=12, fontweight='bold')
            continue
        
        P_vals = sorted(fox_speedups[N].keys())
        speedup_vals = [fox_speedups[N][p] for p in P_vals]
        ax.plot(P_vals, speedup_vals, 'o-', label="Fox's Algorithm", 
               linewidth=2, markersize=8, color='blue')
        
        ideal_P = [4, 16, 64]
        ideal_speedup = [1, 4, 16]
        ax.plot(ideal_P, ideal_speedup, 'r--', label='Ideal Speedup', 
               linewidth=1.5, alpha=0.7)
        
        ax.set_xlabel('Number of Processors', fontsize=11)
        ax.set_ylabel('Speedup', fontsize=11)
        ax.set_title(f'Matrix Size: {N}×{N}', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)
        ax.set_yscale('log', base=2)
        ax.set_xticks([4, 16, 64])
        ax.set_xticklabels(['4', '16', '64'])
    
    plt.suptitle('Speedup Curves for Fox\'s Algorithm', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/speedup_curves_fox.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: results/speedup_curves_fox.png")
    
    if results['strassen']:
        strassen_speedups = calculate_speedup(results['strassen'])
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, N in enumerate(MATRIX_SIZES):
            ax = axes[idx]
            
            if N not in strassen_speedups:
                ax.text(0.5, 0.5, f'No data for N={N}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Matrix Size: {N}×{N}', fontsize=12, fontweight='bold')
                continue
            
            P_vals = sorted(strassen_speedups[N].keys())
            speedup_vals = [strassen_speedups[N][p] for p in P_vals]
            ax.plot(P_vals, speedup_vals, 's-', label="Strassen's Algorithm", 
                   linewidth=2, markersize=8, color='green')
            
            ideal_P = [4, 16, 64]
            ideal_speedup = [1, 4, 16]
            ax.plot(ideal_P, ideal_speedup, 'r--', label='Ideal Speedup', 
                   linewidth=1.5, alpha=0.7)
            
            ax.set_xlabel('Number of Processors', fontsize=11)
            ax.set_ylabel('Speedup', fontsize=11)
            ax.set_title(f'Matrix Size: {N}×{N}', fontsize=12, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log', base=2)
            ax.set_yscale('log', base=2)
            ax.set_xticks([4, 16, 64])
            ax.set_xticklabels(['4', '16', '64'])
        
        plt.suptitle('Speedup Curves for Strassen\'s Algorithm', 
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/speedup_curves_strassen.png', dpi=300, bbox_inches='tight')
        print(f"  Saved: results/speedup_curves_strassen.png")
    
    plt.close('all')

def main():
    print("=" * 80)
    print("PROBLEM 3.1: PARALLEL MATRIX MULTIPLICATION")
    print("TASKS 2, 3, 4, 5: Implementation, Testing, Data Collection, Plotting")
    print("=" * 80)
    print(f"\nUsing MPI launcher: {MPI_CMD}")
    print(f"Working directory: {os.getcwd()}\n")
    
    results = collect_timing_data()
    plot_speedup_curves(results)
    
    print("\n" + "=" * 80)
    print("ALL TASKS COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - results/timing_results.txt")
    print("  - results/speedup_curves_fox.png")
    print("  - results/speedup_curves_strassen.png")
    print()

if __name__ == "__main__":
    main()

