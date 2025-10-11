#!/usr/bin/env python3
from mpi4py import MPI
import numpy as np
import time
import sys
import os

def MY_Global_Min_Loc(local_array: np.ndarray, N: int, root: int = 0, comm: MPI.Comm | None = None):
    if comm is None:
        comm = MPI.COMM_WORLD

    rank = comm.Get_rank()
    size = comm.Get_size()

    # Single process case
    if size == 1:
        return np.full(N, rank, dtype=int) if rank == root else None

    min_vals  = local_array.astype(np.float64, copy=True)
    min_owners = np.full(N, rank, dtype=int)

    # Rotate ranks so that the chosen root behaves like vrank 0.
    # Physical rank r -> virtual rank vrank = (r - root) mod size
    # Virtual partner -> physical partner: partner = (partner_vrank + root) mod size
    vrank = (rank - root) % size

    step = 1
    while step < size:
        if (vrank & step) != 0:
            partner_v = vrank ^ step
            partner = (partner_v + root) % size

            comm.Send([min_vals, MPI.DOUBLE], dest=partner, tag=100)
            comm.Send([min_owners, MPI.INT],  dest=partner, tag=101)
            break

        else:
            partner_v = vrank ^ step
            if partner_v < size:
                partner = (partner_v + root) % size

                recv_vals   = np.empty(N, dtype=min_vals.dtype)
                recv_owners = np.empty(N, dtype=min_owners.dtype)

                comm.Recv([recv_vals,   MPI.DOUBLE], source=partner, tag=100)
                comm.Recv([recv_owners, MPI.INT],    source=partner, tag=101)

                better = recv_vals < min_vals
                ties   = recv_vals == min_vals

                if np.any(better):
                    min_vals[better]   = recv_vals[better]
                    min_owners[better] = recv_owners[better]

                if np.any(ties):
                    lower = ties & (recv_owners < min_owners)
                    if np.any(lower):
                        min_owners[lower] = recv_owners[lower]
        step <<= 1

    return min_owners if vrank == 0 else None

def generate_test_arrays(P, N, filename = "arrays.txt"):
    # generate arrays using random number from 0 to 1000
    np.random.seed(42)
    
    arrays = []
    for p in range(P):
        array = np.random.randint(0, 1000, N)
        arrays.append(array)
    
    # Save arrays to file
    with open(filename, 'w') as f:
        for array in arrays:
            f.write(' '.join(map(str, array)) + '\n')
    
    return arrays

def load_test_arrays(filename = "arrays.txt"):
    # load test arrays from file
    arrays = []
    with open(filename, 'r') as f:
        for line in f:
            array = np.array(list(map(int, line.strip().split())))
            arrays.append(array)
    return arrays

def calculate_correct_answer(arrays):
    # Find the correct result without using parallelization
    P = len(arrays)
    N = len(arrays[0])
    
    result = np.zeros(N, dtype=int)
    
    for i in range(N):
        min_value = float('inf')
        min_rank = 0
        
        for rank in range(P):
            if arrays[rank][i] < min_value:
                min_value = arrays[rank][i]
                min_rank = rank
        
        result[i] = min_rank
    
    return result

def test_tie_case():
    """Test with tie case where multiple processes have same minimum values"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if size >= 2:
        N = 4
        local_array = np.zeros(N, dtype=int)
        
        # Create test case with ties
        if rank == 0:
            local_array = np.array([5, 3, 7, 1])
        elif rank == 1:
            local_array = np.array([5, 3, 2, 1])
        else:
            # For ranks > 1, use dummy values
            local_array = np.array([10 + rank] * N)
        
        # Call MY_Global_Min_Loc
        result = MY_Global_Min_Loc(local_array, N, root=0, comm=comm)
        
        # Verify result on root
        if rank == 0:
            print(f"\n=== Tie Case Test ===")
            print(f"Arrays assigned to processors:")
            print(f"  Process 0: [5 3 7 1]")
            print(f"  Process 1: [5 3 2 1]")
            print(f"\nGot: {result}")
            
            # For ties case, verify that the result is valid
            # Position 0: min(5,5) = 5, can be rank 0 or 1
            # Position 1: min(3,3) = 3, can be rank 0 or 1  
            # Position 2: min(7,2) = 2, must be rank 1
            # Position 3: min(1,1) = 1, can be rank 0 or 1
            valid_result = (result[2] == 1) and (result[0] in (0,1)) and (result[1] in (0,1)) and (result[3] in (0,1))
            
            return {
                'name': 'Tie Case Test',
                'input': 'P = 2, N = 4 with arrays [5,3,7,1], [5,3,2,1]',
                'expected': 'Any valid rank for ties (position 2 must be rank 1)',
                'got': result.tolist(),
                'passed': valid_result
            }
    
    return None

def test_example():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if size >= 3:
        N = 4
        local_array = np.zeros(N, dtype=int)
        
        # Initialize test data based on example from problem
        if rank == 0:
            local_array = np.array([1, 9, 3, 4])
        elif rank == 1:
            local_array = np.array([5, 6, 7, 2])
        elif rank == 2:
            local_array = np.array([9, 8, 6, 1])
        else:
            # For ranks > 2, use dummy values
            local_array = np.array([1000 + rank * N + i for i in range(N)])
        
        # Call MY_Global_Min_Loc
        result = MY_Global_Min_Loc(local_array, N, root=0, comm=comm)
        
        # Verify result on root
        if rank == 0:
            expected = np.array([0, 1, 0, 2])
            is_correct = np.array_equal(result, expected)
            
            print(f"\n=== Example from Problem ===")
            print(f"Arrays assigned to processors:")
            print(f"  Process 0: [1 9 3 4]")
            print(f"  Process 1: [5 6 7 2]")
            print(f"  Process 2: [9 8 6 1]")
            print(f"\nExpected: {expected}")
            print(f"Got:      {result}\n")
            
            return {
                'name': 'Example from Problem Description',
                'input': 'P = 3, N = 4 with arrays [1,9,3,4], [5,6,7,2], [9,8,6,1]',
                'expected': expected.tolist(),
                'got': result.tolist(),
                'passed': is_correct
            }
    
    return None

def verify_with_generated_data():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    N = 10
    
    # Generate test data
    if rank == 0:
        P = size
        arrays = generate_test_arrays(P, N)
        correct_answer = calculate_correct_answer(arrays)
        
        
        # Save correct answer for verification
        np.savetxt("correct_answer.txt", correct_answer, fmt='%d')
        
        # Distribute arrays to processes
        for p in range(P):
            if p != 0:  # Send to other processes
                comm.send(arrays[p], dest = p, tag = 0)
        local_array = arrays[0]  # Root keeps its own array
        
    else:
        # Receive array from root
        local_array = comm.recv(source = 0, tag = 0)
        correct_answer = None
    
    result = MY_Global_Min_Loc(local_array, N, root = 0, comm = comm)
    
    # Verify result
    if rank == 0:
        print(f"\n=== Random Array Verification ===")
        print(f"Arrays assigned to processors:")
        for p in range(P):
            print(f"  Process {p}: {arrays[p]}")
        print(f"\nExpected: {correct_answer}")
        print(f"Got:      {result}\n")
        
        is_correct = np.array_equal(result, correct_answer)
        
        return {
            'name': 'Generated Data Test',
            'input': f'P = {size}, N = {N} with random arrays',
            'expected': correct_answer.tolist(),
            'got': result.tolist(),
            'passed': is_correct
        }
    
    return None

def run_performance_test(P, N):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Generate test data
    if rank == 0:
        arrays = generate_test_arrays(P, N, f"arrays_P{P}_N{N}.txt")
        
        # Distribute arrays to processes
        for p in range(P):
            if p != 0:  # Send to other processes
                comm.send(arrays[p], dest=p, tag=0)
        local_array = arrays[0]  # Root keeps its own array
        
    else:
        # Receive array from root
        local_array = comm.recv(source=0, tag=0)
    
    # Synchronize before timing
    comm.Barrier()
    
    # Measure execution time using MPI_Wtime
    start_time = MPI.Wtime()
    result = MY_Global_Min_Loc(local_array, N, root=0, comm=comm)
    end_time = MPI.Wtime()
    
    execution_time = end_time - start_time
    
    # Root process reports timing
    if rank == 0:
        print(f"P={size}, N={N}: {execution_time:.6f} seconds")
        return execution_time
    return None

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print(f"Running with {size} processes")
    
    # Test correctness with example from problem descritpion
    example_test_result = test_example()
    
    # Test correctness with tie case
    tie_test_result = test_tie_case()
    
    # Test correctness with generated data
    generated_test_result = verify_with_generated_data()
    
    # Performance tests with different N values
    if rank == 0:
        print("=== Performance Tests ===")
    
    test_sizes = [10, 100, 1000, 10000]
    timing_results = []
    
    for N in test_sizes:
        timing = run_performance_test(size, N)
        if timing is not None:
            timing_results.append((N, timing))
    
    if rank == 0:
        # Write results to file
        with open("results.txt", "w") as f:
            f.write(f"Running with {size} processes\n\n")
            f.write("=== Correctness Tests ===\n")
            
            # Example from problem description
            if example_test_result:
                f.write(f"{example_test_result['name']}\n")
                f.write(f"Input: {example_test_result['input']}\n")
                f.write("Arrays assigned to processors:\n")
                f.write("  Process 0: [1 9 3 4]\n")
                f.write("  Process 1: [5 6 7 2]\n")
                f.write("  Process 2: [9 8 6 1]\n")
                f.write(f"\nExpected: {example_test_result['expected']}\n")
                f.write(f"Got:      {example_test_result['got']}\n\n")
            
            # Tie case test
            if tie_test_result:
                f.write(f"{tie_test_result['name']}\n")
                f.write(f"Input: {tie_test_result['input']}\n")
                f.write("Arrays assigned to processors:\n")
                f.write("  Process 0: [5 3 7 1]\n")
                f.write("  Process 1: [5 3 2 1]\n")
                f.write(f"\nExpected: {tie_test_result['expected']}\n")
                f.write(f"Got: {tie_test_result['got']}\n\n")
            
            # Generated data test
            if generated_test_result:
                f.write(f"{generated_test_result['name']}\n")
                f.write(f"Input: {generated_test_result['input']}\n")
                
                # Load and display the arrays used in the test
                try:
                    arrays = load_test_arrays("arrays.txt")
                    f.write("Arrays assigned to processors:\n")
                    for p in range(size):
                        f.write(f"  Process {p}: {arrays[p]}\n")
                except:
                    f.write("Arrays: Generated test data\n")
                
                f.write(f"\nExpected: {generated_test_result['expected']}\n")
                f.write(f"Got:      {generated_test_result['got']}\n\n")
            
            f.write("=== Performance Tests ===\n")
            f.write("Timing Results:\n")
            for N, time_val in timing_results:
                f.write(f"P={size}, N={N}: {time_val:.6f} seconds\n")
            
if __name__ == "__main__":
    main()
