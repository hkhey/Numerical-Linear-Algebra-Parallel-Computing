from mpi4py import MPI
import numpy as np
import time

# Matrix-vector multiplication function
def matrix_vector_mult(A, x):
    m = A.shape[0]
    n = A.shape[1]
    b = np.zeros((m,))
    for i in range(m):
        for j in range(n):
            b[i] += A[i,j]*x[j]
    return b
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

m = 1000
n = 1000
A = np.random.rand(m, n)
x = np.random.rand(n,)

sendcounts = [m//size]*size
sendcounts[-1] += m % size
senddispls = [sum(sendcounts[:i]) for i in range(size)]
local_A = np.zeros((sendcounts[rank], n))
comm.Scatterv([A, sendcounts, senddispls, MPI.DOUBLE], local_A, root=0)

sendcounts = [n//size]*size
sendcounts[-1] += n % size
senddispls = [sum(sendcounts[:i]) for i in range(size)]
local_x = np.zeros(sendcounts[rank])
comm.Scatterv([x, sendcounts, senddispls, MPI.DOUBLE], local_x, root=0)

local_b = matrix_vector_mult(local_A, local_x)

recvcounts = [m//size]*size
recvcounts[-1] += m % size
recvdispls = [sum(recvcounts[:i]) for i in range(size)]
b = np.zeros((m,))
comm.Gatherv(local_b, [b, recvcounts, recvdispls, MPI.DOUBLE], root=0)

if rank == 0:
    start_time = time.time()
    b_dot = np.dot(A, x)
    dot_time = time.time() - start_time
    error = np.linalg.norm(b-b_dot)
    print("CPU time of parallel multiplication using {} processes is {:.6f}".format(size, dot_time))
    print("The error comparing to the dot product is :", error)

