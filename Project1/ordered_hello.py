from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

for i in range(size):
    if rank == i:
        print(f"Hello from Processor {rank} of {size}")
    comm.Barrier()  # ensure only one process prints at a time