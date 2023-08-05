#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <limits.h>
#include "helperfunctions.h"
#include "iofunctions.h"

void printBackend()
{
    printf("MPI Split collectives");
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

    if (MPI_File_open(comm, options->path_to_write_file, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &(fd.mpi)) != MPI_SUCCESS)
    {
        fprintf(stderr, "Rank %"PRIu64" couldn't open file %s\n", rank, options->path_to_write_file);
        exit(1);
    }

    return fd;
}

void closeFile(union FileDescriptor* fd)
{
    MPI_File_close(&fd->mpi);
}

static void calculateOffsets(uint64_t const N, struct options const* options, struct process_data const* process_data, uint64_t* offset_file, uint64_t* size)
{
    uint64_t const rank = process_data->rank;
    uint64_t const last_rank = process_data->world_size - 1;

    uint64_t lines_to_write = (process_data->num_lines_with_halo - 1);

    if (rank != last_rank && rank != 0)
    {
        lines_to_write--;
    }

    *size = lines_to_write * N;

    if (options->file_per_process == PROCESS_LOCAL_FILE)
    {
        *offset_file = 0;
    }
    else
    {
        *offset_file = process_data->global_start * N * sizeof(double);
    }
}

uint64_t writeMatrixToFile(double const* M, uint64_t const N, struct options const* options, struct process_data* process_data, union FileDescriptor* fd, size_t const curr_fd)
{
    static int initialized = 0;
    static uint64_t size_of_write;
    static uint64_t offset_for_write;
    int size_of_single_write = 0;
    uint64_t elements_written = 0;

    if (!initialized)
    {
        calculateOffsets(N, options, process_data, &offset_for_write, &size_of_write);
        initialized = 1;
    }

    while (elements_written != size_of_write)
    {
        if (size_of_write - elements_written > INT_MAX)
        {
            size_of_single_write = INT_MAX;
        }
        else
        {
            size_of_single_write = size_of_write - elements_written;
            process_data->offset_for_final_write = elements_written;
        }

        if (MPI_File_write_at_all_begin(fd->mpi, offset_for_write + elements_written*sizeof(double), M + process_data->address_offset + elements_written, size_of_single_write, MPI_DOUBLE) != MPI_SUCCESS)
        {
            fprintf(stderr, "Write failed at rank %"PRIu64", exiting...\n", process_data->rank);
            exit(1);
        }

        //Only one split collective can remain open per file, according to MPI specs
        if (elements_written + size_of_single_write != size_of_write)
        {
            MPI_Status status;
            MPI_File_write_at_all_end(fd->mpi, M + process_data->address_offset + elements_written, &status);

            int count;
            MPI_Get_count(&status, MPI_DOUBLE, &count);
            elements_written += count;
        }
        else
        {
            elements_written += size_of_single_write;
        }
    }

    process_data->expected_write_size = elements_written;
    return elements_written * sizeof(double);
}

void waitForCompletion(double const* M, union FileDescriptor const* fd, struct process_data const* process_data)
{
    MPI_Status status;
    MPI_File_write_at_all_end(fd->mpi, M + process_data->address_offset + process_data->offset_for_final_write, &status);

    int count;
    MPI_Get_count(&status, MPI_DOUBLE, &count);

    uint64_t written_elements = process_data->offset_for_final_write + count;

    if (written_elements != process_data->expected_write_size)
    {
        fprintf(stderr, "Write failed at rank %"PRIu64", written elements: %"PRIu64", supposed write size: %"PRIu64"\n", process_data->rank, written_elements, process_data->expected_write_size);
        exit(1);
    }
}

void fileSync(union FileDescriptor const* fd)
{
    MPI_File_sync(fd->mpi);
}

union FileDescriptor openFileForRead(char const* path)
{
    union FileDescriptor fd;
    if (MPI_File_open(MPI_COMM_WORLD, path, MPI_MODE_RDONLY, MPI_INFO_NULL, &(fd.mpi)) != MPI_SUCCESS)
    {
        fprintf(stderr, "Couldn't open file %s\n", path);
        exit(1);
    }
    return fd;
}

uint64_t readMatrixFromFile(double* M, uint64_t const N, struct options const* options, struct process_data const* process_data, union FileDescriptor* fd)
{
    static int initialized = 0;
    static uint64_t size_of_read;
    static uint64_t offset_for_read;
    int size_of_single_read = 0;
    uint64_t elements_read = 0;

    if (!initialized)
    {
        calculateOffsets(N, options, process_data, &offset_for_read, &size_of_read);
        initialized = 1;
    }

    while (elements_read != size_of_read)
    {
        if (size_of_read - elements_read > INT_MAX)
        {
            size_of_single_read = INT_MAX;
        }
        else
        {
            size_of_single_read = size_of_read - elements_read;
        }

        MPI_Status status;
        if (MPI_File_read_at(fd->mpi, offset_for_read + elements_read*sizeof(double), M + process_data->address_offset + elements_read, size_of_single_read, MPI_DOUBLE, &status) != MPI_SUCCESS)
        {
            fprintf(stderr, "Read failed at rank %"PRIu64", exiting...\n", process_data->rank);
            exit(1);
        }

        int count;
        MPI_Get_count(&status, MPI_DOUBLE, &count);
        if (count != size_of_single_read)
        {
            fprintf(stderr, "Read failed at rank %"PRIu64", read elements: %d, supposed read size: %d\n", process_data->rank, count, size_of_single_read);
            exit(1);
        }

        elements_read += count;
    }

    return elements_read * sizeof(double);
}