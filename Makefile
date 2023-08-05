CC=mpicc

CFLAGS  = -std=c11 -Wall -Wextra -Wpedantic -O3 -g -fopenmp -iquoteinclude
LDFLAGS = $(CFLAGS)
LDLIBS  = -Wl,--no-as-needed,-ldl -lm

#Set this to the respective root of the library on your system manually if pkg-config doesn't find it.
HDF5_ROOT=$(shell pkg-config --variable=prefix hdf5)

TGTS = numio-posix numio-mpisync numio-mpiasync numio-mpisplitcoll numio-hdf5 numio-netcdf numio-adios2
.PRECIOUS: %.o

all: $(TGTS)

#Adios2
numio-adios2: numio-adios2.o IO/adios2.o helperfunctions.o argparse.o
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS) $(shell adios2-config --c-libs)

numio-adios2.o: numio.c
	$(CC) $(shell adios2-config --c-flags) $(CFLAGS) -DADIOS2 -c -o $@ $^

IO/adios2.o: IO/adios2.c
	$(CC) $(shell adios2-config --c-flags) $(CFLAGS) -DADIOS2 -c -o $@ $^

#NetCDF
numio-netcdf: numio-netcdf.o IO/netcdf.o helperfunctions.o argparse.o
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS) $(shell nc-config --libs) -Wl,-rpath,$(shell nc-config --libdir)

numio-netcdf.o: numio.c
	$(CC) $(CFLAGS) -DNETCDF -c -o $@ $^

IO/netcdf.o: IO/netcdf.c
	$(CC) $(shell nc-config --cflags) $(CFLAGS) -DNETCDF -c -o $@ $^

#HDF5
numio-hdf5: numio-hdf5.o IO/hdf5.o helperfunctions.o argparse.o
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS) -L$(HDF5_ROOT)/lib -Wl,-rpath,$(HDF5_ROOT)/lib -lhdf5

numio-hdf5.o: numio.c
	$(CC) -I$(HDF5_ROOT)/include $(CFLAGS) -DHDF5 -c -o $@ $^

IO/hdf5.o: IO/hdf5.c
	$(CC) -I$(HDF5_ROOT)/include $(CFLAGS) -DHDF5 -c -o $@ $^

#MPI Split collectives
numio-mpisplitcoll: numio-mpisplitcoll.o IO/mpisplitcoll.o helperfunctions.o argparse.o
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

numio-mpisplitcoll.o: numio.c
	$(CC) $(CFLAGS) -DMPI_IO_SPLITCOLL -c -o $@ $^

IO/mpisplitcoll.o: IO/mpisplitcoll.c
	$(CC) $(CFLAGS) -DMPI_IO_SPLITCOLL -c -o $@ $^

#MPI Async
numio-mpiasync: numio-mpiasync.o IO/mpiasync.o helperfunctions.o argparse.o
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

numio-mpiasync.o: numio.c
	$(CC) $(CFLAGS) -DMPI_ASYNC -c -o $@ $^

IO/mpiasync.o: IO/mpiasync.c
	$(CC) $(CFLAGS) -DMPI_ASYNC -c -o $@ $^


#The rest
numio-%: numio.o IO/%.o helperfunctions.o argparse.o
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $^


clean:
	$(RM) $(TGTS) *.o IO/*.o
