#ifndef IOFUNCTIONS_H
#define IOFUNCTIONS_H

#include <inttypes.h>
#include <mpi.h>
#ifdef HDF5
#include <hdf5.h>
#endif
#ifdef ADIOS2
#include <adios2_c.h>
#endif
#include "numiostructs.h"


#ifdef HDF5
struct hdf5
{
	hid_t file;
	hid_t dset;
};
#endif

struct netcdf
{
	int ncid;
	int varid;
};

#ifdef ADIOS2
struct adios2
{
	adios2_engine* engine;
	adios2_variable* variable;
};
#endif

union FileDescriptor{
	int posix;
	MPI_File mpi;
	struct netcdf netcdf;
	#ifdef HDF5
	struct hdf5 hdf5;
	#endif
	#ifdef ADIOS2
	struct adios2 adios2;
	#endif
};

void printBackend();

union FileDescriptor openFile(uint64_t rank, struct options* options);

void closeFile(union FileDescriptor* fd);

uint64_t writeMatrixToFile(double const* M, uint64_t const N, struct options const* options, struct process_data* process_data, union FileDescriptor* fd, size_t const curr_fd);

#ifdef MPI_IO_SPLITCOLL
void waitForCompletion(double const* M, union FileDescriptor const* fd, struct process_data const* process_data);
#elif MPI_ASYNC
void waitForCompletion(struct process_data* process_data);
#endif

void fileSync(union FileDescriptor const* fd);

union FileDescriptor openFileForRead(char const* path);

uint64_t readMatrixFromFile(double* M, uint64_t const N, struct options const* options, struct process_data const* process_data, union FileDescriptor* fd);

#endif