# NUMIO

## Usage
First, build all numio versions (one per IO backend) with ```make``` or build a specific one by executing ```make numio-<io_backend>```.
To execute a numio version, issue a call of the form ```mpirun -n 2 ./numio-<backend> -m iter=20,size=10,pert=1 -w freq=5,path=run.out -r freq=18``` or similar (see below for more detail).

### Parameters
The current parameters are : ```-m iter=<iter>,size=<size>,pert=<pert> [-w freq=<freq>,path=/path/to/file[,imm][,nofilesync][,pattern=<pat>]] [-r freq=<freq>[,path=/path/to/readfile]] [-c freq=<freq>,size=<size>] [-t <threads>] [-l]```. Optional parameters are enclosed in ```[]```.
- ```-m``` (Matrix options):
  - iter: Controls number of iterations that the Jacobi method will be applied to the matrix
  - size: Controls the matrix size (square matrix)
  - pert: Choice of perturbation function. (1: $f(x,y) = 0$,
2: $f(x,y) = 2 * \pi^2 * sin(\pi * x) * sin(\pi * y)$)
- ```-w``` (Write options):
  - freq: Controls write frequency. 0 means no writing, 1 means write the matrix every iteration, 2 means every second iteration and so on
  - path: Path to the matrix output file
  - imm: Provide this flag if the writes should be sync'd immediately and not overlapped with computation
  - nofilesync: Provide this flag to disable all filesyncs
  - pattern: A pattern can be provided for irregular write accesses, consisting of integers separated with colons. An example pattern could look like ```4:0:5```. Now the first third of write calls would be quadrupled by writing the same matrix to 4 different files, the second third of write calls would be skipped, and the final third calls would write to 5 different files back-to-back.
- ```-r``` (Read options):
  - freq: Controls read frequency. The input numbers have the same meaning as for writing.
  - path: File to read from. Make sure that the dimensions of the data in the file match the current matrix. If you do not provide a path to a file while also going for a read frequency above 0, the data that is being written will also be used for reading. Therefore you need to ensure that the first write happens before the first read by making sure that ```readfreq > writefreq```.
- ```-c``` (Fake MPI collective communication options):
  - freq: Controls the frequency of the collective communication calls. The numbers have the same meaning as with the other frequencies before.
  - size: The size of the buffer in KB used for the collective communication. Specifically, this will be the size of the send buffer per process, the collective call in use is ```MPI_Allgather```.
- ```-l```: Controls whether each process writes to a single file or one process-local file (Default: one global file; Setting this flag: one file per process)
- ```-t```: Controls number of threads per process (Default/Recommended: 1)



### Dependencies
Every version requires some MPI implementation.
Depending on the I/O backend that you want to use, the involved library is required as well. Meaning, if you want to be able to use every available backend, you will need:
- NetCDF (built with PnetCDF underneath)
- HDF5
- ADIOS2

The numio Makefile relies on ```pkg-config``` for HDF5 and on the internal config tools of NetCDF and ADIOS2, so make sure that those are available at the time of compiling the executables.

## Available backends
The currently available IO backends are:

- POSIX: Writing with pwrite, followed by fsync.

- MPI IO Blocking: Writing with MPI_File_write_at.

- MPI IO Nonblocking: Writing with MPI_File_iwrite_at, completion is waited for one iteration before the next write call occurs.

- MPI IO Split collectives: Writing with MPI_File_write_at_all_begin, the matching end call occurs one iteration before the next write.

- NetCDF (using PnetCDF and NetCDF5 files)

- ADIOS2 (using BP3 engine)

- HDF5

## Things to keep in mind
The communication and computation are currently separated into two distinct phases every iteration.
The write calls are always flushed to disk one iteration before the next write call, unless specified otherwise by parameters.
The report on CPU usage is only accurate if each process uses just a single thread.
Reported write sizes exclude any metadata accesses or file creation etc.
ADIOS2 currently does not work with reading and writing to the same file simultaneously (WIP, looking at the different engines).
To measure throughput, two points in time are used: the first being right before the first write in an iteration occurs, and the second one is right after the wait for completion (if necessary) and the file sync (or where it would be, if file syncs are disabled). Keep this in mind when evaluating throughputs.
