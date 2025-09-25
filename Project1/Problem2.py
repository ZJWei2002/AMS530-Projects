from mpi4py import MPI

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

if rank == 0:
    print(f"Hello from Processor {rank}")
    if size > 1:
        MPI.COMM_WORLD.send(None, dest = rank + 1)
else:
    MPI.COMM_WORLD.recv(source = rank - 1)
    print(f"Hello from Processor {rank}")
    if rank < size-1:
        MPI.COMM_WORLD.send(None, dest = rank + 1)