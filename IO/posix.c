#define _GNU_SOURCE

#include <unistd.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <sys/stat.h>
#include "iofunctions.h"
#include "helperfunctions.h"

void printBackend()
{
    printf("POSIX");
}

union FileDescriptor openFile(uint64_t rank, struct options* options)
{
    union FileDescriptor fd;
    fd.posix = open(options->path_to_write_file, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH);
    if (fd.posix == -1)
    {
        fprintf(stderr, "Rank %"PRIu64" couldn't open file %s\n", rank, options->path_to_write_file);
        exit(1);
    }

    return fd;
}

void closeFile(union FileDescriptor* fd)
{
    close(fd->posix);
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

    *size = lines_to_write * N * sizeof(double);

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
    uint64_t number_of_bytes_written = 0;

    if (!initialized)
    {
        calculateOffsets(N, options, process_data, &offset_for_write, &size_of_write);
        initialized = 1;
    }

    while (size_of_write != number_of_bytes_written)
    {
        ssize_t res = pwrite(fd->posix, M + process_data->address_offset + number_of_bytes_written/sizeof(double), size_of_write - number_of_bytes_written, offset_for_write + number_of_bytes_written);
        if (res == -1)
        {
            fprintf(stderr, "Write failed at rank %"PRIu64", exiting...\n", process_data->rank);
            exit(1);
        }
        number_of_bytes_written += res;
    }

    return number_of_bytes_written;
}

void fileSync(union FileDescriptor const* fd)
{
    fsync(fd->posix);
}

union FileDescriptor openFileForRead(char const* path)
{
    union FileDescriptor fd;
    fd.posix = open(path, O_RDONLY);
    if (fd.posix == -1)
    {
        fprintf(stderr, "A process couldn't open file %s\n", path);
        exit(1);
    }
    return fd;
}

uint64_t readMatrixFromFile(double* M, uint64_t const N, struct options const* options, struct process_data const* process_data, union FileDescriptor* fd)
{
    static int initialized = 0;
    static uint64_t size_of_read;
    static uint64_t offset_for_read;
    uint64_t number_of_bytes_read = 0;

    if (!initialized)
    {
        calculateOffsets(N, options, process_data, &offset_for_read, &size_of_read);
        initialized = 1;
    }

    while (size_of_read != number_of_bytes_read)
    {
        ssize_t res = pread(fd->posix, M + process_data->address_offset + number_of_bytes_read/sizeof(double), size_of_read - number_of_bytes_read, offset_for_read + number_of_bytes_read);
        if (res == -1)
        {
            fprintf(stderr, "Read failed at rank %"PRIu64", exiting...\n", process_data->rank);
            exit(1);
        }
        number_of_bytes_read += res;
    }

    return number_of_bytes_read;
}
