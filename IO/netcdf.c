#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <netcdf.h>
#include <netcdf_par.h>
#include "iofunctions.h"
#include "helperfunctions.h"

#define HANDLENETCDF(x)\
                do { int retval = (x);\
                if (retval) {\
                    fprintf(stderr, "Error: %s\n", nc_strerror(retval));exit(1);}\
                }while(0)\

void printBackend()
{
    printf("NetCDF");
}

union FileDescriptor openFile(uint64_t rank, struct options* options)
{
    union FileDescriptor fd;

    if (options->file_per_process == PROCESS_LOCAL_FILE)
    {
        HANDLENETCDF(nc_create(options->path_to_write_file, NC_CLOBBER | NC_64BIT_DATA, &fd.netcdf.ncid));
    }
    else
    {
        HANDLENETCDF(nc_create_par(options->path_to_write_file, NC_CLOBBER | NC_64BIT_DATA, MPI_COMM_WORLD, MPI_INFO_NULL, &fd.netcdf.ncid));
    }

    //HANDLENETCDF(nc_enddef(fd.ncid));
    //HANDLENETCDF(nc_redef(fd.ncid));
    return fd;
}

void closeFile(union FileDescriptor* fd)
{
    HANDLENETCDF(nc_close(fd->netcdf.ncid));
}

static void calculateOffsets(uint64_t const N, struct process_data const* process_data, uint64_t* size, size_t start[2], size_t count[2])
{
    uint64_t rank = process_data->rank;
    uint64_t last_rank = process_data->world_size - 1;
    
    count[0] = process_data->num_lines_with_halo - 1;
    count[1] = N;
    
    if (rank != last_rank && rank != 0)
    {
        count[0]--;
    }
        
    start[0] = process_data->global_start;
    start[1] = 0;
    *size = count[0] * count[1] * nctypelen(NC_DOUBLE);
}

uint64_t writeMatrixToFile(double const* M, uint64_t const N, struct options const* options, struct process_data* process_data, union FileDescriptor* fd, size_t const curr_fd)
{
    static int initialized_offsets = 0;
    static int initialized_fds = -1;
    static int dimids[2];
    static size_t start[2], count[2];
    static uint64_t write_size;

    if (initialized_fds < (int)curr_fd)
    {        
        if (!initialized_offsets)
        {
            calculateOffsets(N, process_data, &write_size, start, count);
            initialized_offsets = 1;
        }
        
        if (options->file_per_process == PROCESS_LOCAL_FILE)
        {
            HANDLENETCDF(nc_def_dim(fd->netcdf.ncid, "x", count[0], &dimids[0]));
            HANDLENETCDF(nc_def_dim(fd->netcdf.ncid, "y", N, &dimids[1]));
            HANDLENETCDF(nc_def_var(fd->netcdf.ncid, "data", NC_DOUBLE, 2, dimids, &(fd->netcdf.varid)));        
        }
        else
        {
            HANDLENETCDF(nc_def_dim(fd->netcdf.ncid, "x", N, &dimids[0]));
            HANDLENETCDF(nc_def_dim(fd->netcdf.ncid, "y", N, &dimids[1]));
            HANDLENETCDF(nc_def_var(fd->netcdf.ncid, "data", NC_DOUBLE, 2, dimids, &(fd->netcdf.varid)));
            HANDLENETCDF(nc_var_par_access(fd->netcdf.ncid, fd->netcdf.varid, NC_COLLECTIVE));
        }
        HANDLENETCDF(nc_enddef(fd->netcdf.ncid));
        initialized_fds++;
    }

    if (options->file_per_process == PROCESS_LOCAL_FILE)
    {
        HANDLENETCDF(nc_put_var_double(fd->netcdf.ncid, fd->netcdf.varid, M+process_data->address_offset));
    }
    else
    {
        HANDLENETCDF(nc_put_vara_double(fd->netcdf.ncid, fd->netcdf.varid, start, count, M+process_data->address_offset));
    }

    return write_size;
}

void fileSync(union FileDescriptor const* fd)
{
    HANDLENETCDF(nc_sync(fd->netcdf.ncid));
}

union FileDescriptor openFileForRead(char const* path)
{
    union FileDescriptor fd;
    HANDLENETCDF(nc_open_par(path, NC_NOWRITE, MPI_COMM_WORLD, MPI_INFO_NULL, &fd.netcdf.ncid));
    return fd;
}

uint64_t readMatrixFromFile(double* M, uint64_t const N, struct options const* options, struct process_data const* process_data, union FileDescriptor* fd)
{
    static int initialized = 0;
    static size_t start[2], count[2];
    static uint64_t read_size;

    if (!initialized)
    {
        calculateOffsets(N, process_data, &read_size, start, count);
        HANDLENETCDF(nc_inq_varid(fd->netcdf.ncid, "data", &(fd->netcdf.varid)));
        initialized = 1;
    }
    HANDLENETCDF(nc_get_vara_double(fd->netcdf.ncid, fd->netcdf.varid, start, count, M + process_data->address_offset));
    return read_size;
}