# Problem 2.1: Algorithm Descriptions

## MPI_Bcast

**Purpose**: Broadcast data from a root process to all other processes in the communicator.

**Algorithm**: Use a tree-based approach where the root sends data to its children, and each process forwards the data to its own children. For P processes, create a binary tree structure where process 0 is the root. The root sends data to processes 1 and 2, process 1 sends to processes 3 and 4, and so on. This creates log₂(P) communication steps.

**Scalability**: Communication time grows logarithmically with the number of processes, making it efficient for large process counts. However, bandwidth becomes a bottleneck for large data sizes since all processes receive the same amount of data.

## MPI_Scatter

**Purpose**: Distribute distinct portions of data from a root process to each of the other processes.

**Algorithm**: The root process splits its data into P equal chunks and sends each chunk to a different process using MPI_Send. Each non-root process receives its assigned chunk using MPI_Recv. The root can send chunks sequentially to processes 0, 1, 2, ..., P-1, or use a tree-based distribution where the root sends larger chunks to intermediate processes that then redistribute to their subtrees.

**Scalability**: Memory usage scales linearly with data size per process, but communication time depends on the distribution pattern. A linear approach has O(P) communication steps, while a tree approach reduces this to O(log P) steps at the cost of more complex implementation.

## MPI_Allgather

**Purpose**: Gather data from all processes and distribute the complete collected result back to all processes.

**Algorithm**: Use a two-phase approach. First, gather data to the root using a tree-based collection where each process sends its data up the tree. Second, broadcast the complete data from root to all processes using a tree-based distribution. Alternatively, use a ring algorithm where each process sends its data around the ring, accumulating data from all processes in P-1 steps.

**Scalability**: The ring algorithm has O(P) communication steps but good bandwidth utilization. The tree-based approach has O(log P) steps but may create bandwidth bottlenecks at higher levels of the tree, especially for large data sizes.

## MPI_Alltoall

**Purpose**: Each process sends a distinct block of data to every other process and simultaneously receives a distinct block from every other process.

**Algorithm**: Implement using P-1 communication rounds. In round i, each process sends data to process (rank + i) mod P and receives data from process (rank - i) mod P. This ensures each process communicates with every other process exactly once. Use MPI_Sendrecv to handle the simultaneous send and receive operations.

**Scalability**: Communication time scales as O(P) with the number of processes, and each process must handle P-1 communication rounds. Memory requirements grow quadratically with P since each process must store data blocks for all other processes, making it expensive for large process counts.

## MPI_Reduce

**Purpose**: Combine data from all processes using an operation (sum, min, max, etc.) and send the final result to a single root process.

**Algorithm**: Use a tree-based reduction where each process sends its data up the tree, and intermediate processes combine received data with their own before forwarding. The root process performs the final reduction. For example, in a binary tree, leaves send data to their parents, parents combine data and send to their parents, continuing until the root receives the final result.

**Scalability**: Communication steps scale as O(log P), making it efficient for large process counts. However, the reduction operation itself may become a bottleneck for complex operations or large data sizes, as intermediate processes must perform computations on accumulated data.
