Three simple N-Queens solvers: sequential, using C++11 task-based parallelism, and CUDA.

The CUDA solver recursively fills a table with initial states on the CPU, then solves each of these in a separate thread in the GPU. There are probably smarter ways to go about this.
