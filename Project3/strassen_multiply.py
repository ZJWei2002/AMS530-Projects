import numpy as np
from mpi4py import MPI
import sys
import time

BASE_CASE_SIZE = 1024

def standard_multiply(A, B):
    return A @ B

def strassen_serial(A, B):
    n = A.shape[0]
    
    if n <= BASE_CASE_SIZE:
        return standard_multiply(A, B)
    
    mid = n // 2
    A11, A12 = A[:mid, :mid], A[:mid, mid:]
    A21, A22 = A[mid:, :mid], A[mid:, mid:]
    B11, B12 = B[:mid, :mid], B[:mid, mid:]
    B21, B22 = B[mid:, :mid], B[mid:, mid:]
    
    M1 = strassen_serial(A11 + A22, B11 + B22)
    M2 = strassen_serial(A21 + A22, B11)
    M3 = strassen_serial(A11, B12 - B22)
    M4 = strassen_serial(A22, B21 - B11)
    M5 = strassen_serial(A11 + A12, B22)
    M6 = strassen_serial(A21 - A11, B11 + B12)
    M7 = strassen_serial(A12 - A22, B21 + B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    C = np.zeros((n, n), dtype=np.float64)
    C[:mid, :mid] = C11
    C[:mid, mid:] = C12
    C[mid:, :mid] = C21
    C[mid:, mid:] = C22
    
    return C

def strassen_multiply(A, B, comm, rank, size):
    n = A.shape[0]
    
    if n <= BASE_CASE_SIZE:
        return standard_multiply(A, B)
    
    mid = n // 2
    A11, A12 = A[:mid, :mid], A[:mid, mid:]
    A21, A22 = A[mid:, :mid], A[mid:, mid:]
    B11, B12 = B[:mid, :mid], B[:mid, mid:]
    B21, B22 = B[mid:, :mid], B[mid:, mid:]
    
    num_product_processors = min(7, size)
    products = [None] * 7
    
    for i in range(7):
        product_owner = i % num_product_processors
        if rank == product_owner:
            if i == 0:
                products[i] = strassen_serial(A11 + A22, B11 + B22)
            elif i == 1:
                products[i] = strassen_serial(A21 + A22, B11)
            elif i == 2:
                products[i] = strassen_serial(A11, B12 - B22)
            elif i == 3:
                products[i] = strassen_serial(A22, B21 - B11)
            elif i == 4:
                products[i] = strassen_serial(A11 + A12, B22)
            elif i == 5:
                products[i] = strassen_serial(A21 - A11, B11 + B12)
            elif i == 6:
                products[i] = strassen_serial(A12 - A22, B21 + B22)
    
    for i in range(7):
        product_owner = i % num_product_processors
        if products[i] is None:
            products[i] = np.empty((mid, mid), dtype=np.float64)
        comm.Bcast(products[i], root=product_owner)
    
    M1, M2, M3, M4, M5, M6, M7 = products
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    C = np.zeros((n, n), dtype=np.float64)
    C[:mid, :mid] = C11
    C[:mid, mid:] = C12
    C[mid:, :mid] = C21
    C[mid:, mid:] = C22
    
    return C

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if len(sys.argv) != 3:
        if rank == 0:
            print("Usage: mpiexec -np P python strassen_multiply.py N test_type")
        return
    
    N = int(sys.argv[1])
    # sys.argv[2] (test_type) is accepted but unused - kept for command-line compatibility
    
    if N < 2:
        if rank == 0:
            print(f"Error: Matrix size ({N}) must be at least 2!")
        return
    
    if rank == 0:
        print(f"Strassen's Algorithm - Processors: {size}, Matrix Size: {N}×{N}")
    
    num_product_processors = min(7, size)
    compute_group = rank < num_product_processors
    
    compute_comm = comm.Split(0 if compute_group else MPI.UNDEFINED, rank)
    
    start = time.time()
    
    if compute_group:
        if rank == 0:
            np.random.seed(42)
            A = np.random.uniform(-1.0, 1.0, (N, N))
            B = np.random.uniform(-1.0, 1.0, (N, N))
            
            for dest in range(1, num_product_processors):
                comm.Send(A, dest=dest, tag=0)
                comm.Send(B, dest=dest, tag=1)
        else:
            A = np.empty((N, N), dtype=np.float64)
            B = np.empty((N, N), dtype=np.float64)
            comm.Recv(A, source=0, tag=0)
            comm.Recv(B, source=0, tag=1)
        
        C = strassen_multiply(A, B, compute_comm, rank, num_product_processors)
    else:
        C = None
    
    comm.Barrier()
    elapsed = time.time() - start
    
    if rank == 0:
        print(f"Execution time: {elapsed:.6f} seconds")

if __name__ == "__main__":
    main()

