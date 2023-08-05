#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <adios2_c.h>
#include "helperfunctions.h"
#include "iofunctions.h"

#define CHECKADIOSERROR(x)\
                do { int retval = (x);\
                if (retval) {\
                    fprintf(stderr, "Error: %d\n", retval);exit(1);}\
                }while(0)\

void printBackend()
{
    printf("ADIOS2");
}

void checkHandlerForNull(const void *handler)
{
    if (handler == NULL)
    {
        fprintf(stderr, "Error creating handler, exiting...\n");
        exit(1);
    }
}

//Needed for most functions in both reading and writing, needs to be closed exactly once at the end
static adios2_adios *adios = NULL;
static int existing_engines = 0;

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

    if (adios == NULL)
    {
        adios = adios2_init_mpi(comm);
        checkHandlerForNull(adios);
    }

    adios2_io* io = adios2_declare_io(adios, "Write");
    checkHandlerForNull(io);

    fd.adios2.engine = adios2_open(io, options->path_to_write_file, adios2_mode_write);
    checkHandlerForNull(fd.adios2.engine);

    existing_engines++;

    return fd;
}

void closeFile(union FileDescriptor* fd)
{
    CHECKADIOSERROR(adios2_close(fd->adios2.engine));
    existing_engines--;
    if (existing_engines == 0)
        CHECKADIOSERROR(adios2_finalize(adios));
}

static void calculateOffsets(uint64_t const N, struct process_data const* process_data, struct options const* options, adios2_io* io, adios2_variable** variable,
                             uint64_t* size, size_t global_dims[2], size_t start[2], size_t count[2], int write)
{
    uint64_t rank = process_data->rank;
    uint64_t last_rank = process_data->world_size - 1;

    count[0] = process_data->num_lines_with_halo - 1;
    count[1] = N;

    if (rank != last_rank && rank != 0)
    {
        count[0]--;
    }

    if (options->file_per_process == PROCESS_LOCAL_FILE)
    {
        global_dims[0] = count[0];
        start[0] = 0;
    }
    else
    {
        global_dims[0] = N;
        start[0] = process_data->global_start;
    }
    global_dims[1] = N;
    start[1] = 0;

    if (write)
    {
        io = adios2_at_io(adios, "Write");
        checkHandlerForNull(io);
        *variable = adios2_define_variable(io, "matrix", adios2_type_double, 2, global_dims, start, count, adios2_constant_dims_true);
        checkHandlerForNull(*variable);
    }
    else
    {
        io = adios2_at_io(adios, "Read");
        checkHandlerForNull(io);
        *variable = adios2_inquire_variable(io, "matrix");
        checkHandlerForNull(*variable);
    }

    //Don't think there is a function in the C bindings that lets you query the size of their datatypes,
    //but it's always 8 Bytes for doubles (according to their docs).
    *size = count[0] * count[1] * 8;
}

uint64_t writeMatrixToFile(double const* M, uint64_t const N, struct options const* options,struct process_data* process_data, union FileDescriptor* fd, size_t const curr_fd)
{
    static int initialized = 0;
    static adios2_variable* variable;
    static size_t global_dims[2], start[2], count[2];
    static uint64_t write_size;
    static adios2_io* io;

    if (!initialized)
    {
        calculateOffsets(N, process_data, options, io, &variable, &write_size, global_dims, start, count, 1);
        initialized = 1;
    }

    CHECKADIOSERROR(adios2_put(fd->adios2.engine, variable, M+process_data->address_offset, adios2_mode_deferred));

    return write_size;
}

void fileSync(union FileDescriptor const* fd)
{
    CHECKADIOSERROR(adios2_perform_data_write(fd->adios2.engine));
}

union FileDescriptor openFileForRead(char const* path)
{
    union FileDescriptor fd;
    if (adios == NULL)
    {
        adios = adios2_init_mpi(MPI_COMM_WORLD);
        checkHandlerForNull(adios);
    }

    adios2_io* io = adios2_declare_io(adios, "Read");
    checkHandlerForNull(io);

    fd.adios2.engine = adios2_open(io, path, adios2_mode_read);
    checkHandlerForNull(fd.adios2.engine);
    existing_engines++;
    return fd;
}

uint64_t readMatrixFromFile(double* M, uint64_t const N, struct options const* options, struct process_data const* process_data, union FileDescriptor* fd)
{
    static int initialized = 0;
    static adios2_variable* variable;
    static size_t global_dims[2], start[2], count[2];
    static uint64_t read_size;
    static adios2_io* io;

    if (!initialized)
    {
        calculateOffsets(N, process_data, options, io, &variable, &read_size, global_dims, start, count, 0);
        initialized = 1;
    }

    CHECKADIOSERROR(adios2_get(fd->adios2.engine, variable, M+process_data->address_offset, adios2_mode_deferred));
    CHECKADIOSERROR(adios2_perform_gets(fd->adios2.engine));

    return read_size;
}
