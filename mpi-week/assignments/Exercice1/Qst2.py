from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print("I am process ", rank, " out of ", size, " processes")

MPI.Finalize()

