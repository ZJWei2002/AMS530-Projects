import numpy as np
from mpi4py import MPI
import sys
import time

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if len(sys.argv) != 3:
        if rank == 0:
            print("Usage: mpiexec -np P python fox_multiply.py block_size test_type")
            print("  block_size: Size of each block per processor")
            print("  test_type: 'test' or any label")
        return
    
    block_size = int(sys.argv[1])
    
    sqrt_P = int(np.sqrt(size))
    if sqrt_P * sqrt_P != size:
        if rank == 0:
            print(f"Error: Number of processors ({size}) must be a perfect square!")
        return
    
    total_N = block_size * sqrt_P
    
    if rank == 0:
        print(f"Fox's Algorithm - Processors: {size}, Matrix Size: {total_N}*{total_N}")
        print(f"Block size per processor: {block_size}*{block_size}")
    
    seed = 42 + rank
    np.random.seed(seed)
    
    A_local = np.random.uniform(-1.0, 1.0, (block_size, block_size))
    B_local = np.random.uniform(-1.0, 1.0, (block_size, block_size))
    
    row = rank // sqrt_P
    col = rank % sqrt_P
    
    row_comm = comm.Split(row, col)
    col_comm = comm.Split(col, row)
    
    C_local = np.zeros((block_size, block_size), dtype=np.float64)
    B_temp = B_local.copy()
    
    A_broadcast = np.empty((block_size, block_size), dtype=np.float64)
    B_new = np.empty((block_size, block_size), dtype=np.float64)
    
    start = time.time()
    
    for step in range(sqrt_P):
        source_col = (row + step) % sqrt_P
        
        if col == source_col:
            A_broadcast[:] = A_local
        
        row_comm.Bcast(A_broadcast, root=source_col)
        C_local += A_broadcast @ B_temp
        
        if step < sqrt_P - 1:
            send_rank = row * sqrt_P + ((col - 1 + sqrt_P) % sqrt_P)
            recv_rank = row * sqrt_P + ((col + 1) % sqrt_P)
            comm.Sendrecv(sendbuf=B_temp, dest=send_rank, recvbuf=B_new, source=recv_rank)
            B_temp, B_new = B_new, B_temp
    
    comm.Barrier()
    elapsed = time.time() - start
    
    if rank == 0:
        print(f"Execution time: {elapsed:.6f} seconds")

if __name__ == "__main__":
    main()

