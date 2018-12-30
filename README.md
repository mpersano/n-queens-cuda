Three simple N-Queens solvers:

* serial
* using C++11 task-based parallelism
* using CUDA

The CUDA solver recursively fills a table with initial states on the CPU, then solves each of these in a separate thread in the GPU. It's probably much slower than it should be.
