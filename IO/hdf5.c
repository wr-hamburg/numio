#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <hdf5.h>
#include "helperfunctions.h"
#include "iofunctions.h"

void printBackend()
{
    printf("HDF5");
}

union FileDescriptor openFile(uint64_t rank, struct options* options)
{
    union FileDescriptor fd;
    MPI_Comm comm;

    if (options->file_per_process == PROCESS_LOCAL_FILE)
    {
        comm = MPI_COMM_SELF;
    }
    else
    {
        comm = MPI_COMM_WORLD;
    }

    hid_t initial_plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fclose_degree(initial_plist_id, H5F_CLOSE_STRONG);
    H5Pset_fapl_mpio(initial_plist_id, comm, MPI_INFO_NULL);
    fd.hdf5.file = H5Fcreate(options->path_to_write_file, H5F_ACC_TRUNC, H5P_DEFAULT, initial_plist_id);

    if (!fd.hdf5.file)
    {
        fprintf(stderr, "Rank %"PRIu64" couldn't open file %s\n", rank, options->path_to_write_file);
        exit(1);
    }

    H5Pclose(initial_plist_id);

    return fd;
}

void closeFile(union FileDescriptor* fd)
{
    H5Fclose(fd->hdf5.file);
}

static void calculateOffsets(uint64_t const N, struct process_data const* process_data, struct options const* options,
                             uint64_t* size, int write_file, hid_t* filespace, hid_t* memspace, hid_t* plist_id)
{
    uint64_t rank = process_data->rank;
    uint64_t last_rank = process_data->world_size - 1;

    hsize_t lines_to_write = (process_data->num_lines_with_halo - 1);
    hsize_t previous_lines = process_data->global_start;
    
    if (rank != last_rank && rank != 0)
    {
        lines_to_write--;
    }

    hsize_t count[2];
    count[0] = lines_to_write;
    count[1] = N;
    *size = count[0] * count[1] * H5Tget_size(H5T_NATIVE_DOUBLE);
    
    hsize_t dims[2];
    hsize_t offset[2];

    if (options->file_per_process == PROCESS_LOCAL_FILE && write_file)
    {
        dims[0] = count[0];
        offset[0] = 0;
    }
    else
    {
        dims[0] = N;
        offset[0] = previous_lines;
    }
    dims[1] = N;
    offset[1] = 0;

    *filespace = H5Screate_simple(2, dims, NULL);    
    *memspace = H5Screate_simple(2, count, NULL);

    *plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(*plist_id, H5FD_MPIO_COLLECTIVE);

    H5Sselect_hyperslab(*filespace, H5S_SELECT_SET, offset, NULL, count, NULL);
}

uint64_t writeMatrixToFile(double const* M, uint64_t const N, struct options const* options,struct process_data* process_data, union FileDescriptor* fd, size_t const curr_fd)
{
    static int initialized_offsets = 0;
    static int initialized_fds = -1;
    static uint64_t size_of_single_write;

    static hid_t filespace;
    static hid_t memspace;
    static hid_t plist_id;
    
    if (initialized_fds < (int)curr_fd)
    {
        if (!initialized_offsets)
        {
            calculateOffsets(N, process_data, options, &size_of_single_write, 1, &filespace, &memspace, &plist_id);
            initialized_offsets = 1;
        }

        fd->hdf5.dset = H5Dcreate(fd->hdf5.file, "data", H5T_NATIVE_DOUBLE, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        initialized_fds++;
    }

    herr_t status = H5Dwrite(fd->hdf5.dset, H5T_NATIVE_DOUBLE, memspace, filespace, plist_id, M+process_data->address_offset);

    if (status < 0)
    {
        fprintf(stderr, "Write failed at rank %ld, exiting...\n", process_data->rank);
        exit(1);
    }

    return size_of_single_write;
}

void fileSync(union FileDescriptor const* fd)
{
    if (H5Fflush(fd->hdf5.file, H5F_SCOPE_LOCAL) < 0)
    {
        fprintf(stderr, "Flush failed, exiting...\n");
        exit(1);
    }
}

union FileDescriptor openFileForRead(char const* path)
{
    union FileDescriptor fd;
    hid_t initial_plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fclose_degree(initial_plist_id, H5F_CLOSE_STRONG);
    H5Pset_fapl_mpio(initial_plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);
    fd.hdf5.file = H5Fopen(path, H5F_ACC_RDWR, initial_plist_id);
    if (!fd.hdf5.file)
    {
        fprintf(stderr, "Couldn't open file %s for reading\n", path);
        exit(1);
    }

    H5Pclose(initial_plist_id);

    return fd;
}

uint64_t readMatrixFromFile(double* M, uint64_t const N, struct options const* options, struct process_data const* process_data, union FileDescriptor* fd)
{
    static int initialized = 0;
    static uint64_t size_of_single_read;

    static hid_t filespace;
    static hid_t memspace;
    static hid_t plist_id;

    if (!initialized)
    {
        calculateOffsets(N, process_data, options, &size_of_single_read, 0, &filespace, &memspace, &plist_id);
        fd->hdf5.dset = H5Dopen(fd->hdf5.file, "data", H5P_DEFAULT);
        initialized = 1;
    }
    
    herr_t status = H5Dread(fd->hdf5.dset, H5T_NATIVE_DOUBLE, memspace, filespace, plist_id, M+process_data->address_offset);

    if (status < 0)
    {
        fprintf(stderr, "Read failed at rank %ld, exiting...\n", process_data->rank);
        exit(1);
    }

    return size_of_single_read;
}
