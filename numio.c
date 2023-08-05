/*
 * numio - I/O Benchmark
 * Copyright (C) BLINDED
 * Copyright (C) BLINDED
 * Copyright (C) BLINDED
 * Copyright (C) BLINDED
 * Copyright (C) BLINDED
 * Copyright (C) BLINDED
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>
#include <malloc.h>
#include <string.h>
#include <sys/time.h>
#include <mpi.h>
#include <omp.h>
#include "numiostructs.h"
#include "iofunctions.h"
#include "helperfunctions.h"
#include "argparse.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct timeval start_time;
struct timeval comp_time;

static void
initVariables(struct calculation_arguments* arguments, struct calculation_results* results, struct options const* options, struct process_data* process_data, struct metrics* metrics)
{
    uint64_t const rank = process_data->rank;
    uint64_t const world_size = process_data->world_size;
    uint64_t num_lines;

    if(options->lines < world_size || options->lines < 9)
    {
        if (rank == 0)
        fprintf(stderr, "Too many processes for too few lines, aborting\n");
        exit(1);
    }

    omp_set_dynamic(0);
    omp_set_num_threads(options->threads);

    metrics->bytes_written = 0;
    metrics->bytes_read = 0;
    metrics->time_spent_copying = 0;
    metrics->time_spent_reading = 0;
    metrics->time_spent_writing = 0;
    metrics->read_operations = 0;
    metrics->write_operations = 0;

    arguments->h = 1.0 / (options->lines - 1);

    results->m = 0;
    results->stat_iteration = 0;
    results->stat_precision = 0;

    num_lines = options->lines / world_size;
    uint64_t rest = options->lines % world_size;

    if(rest > rank)
    {
        num_lines = num_lines + 1;
    }

    if(rank < rest)
    {
        process_data->global_start = rank * num_lines;
    }
    else
    {
        process_data->global_start = (num_lines + 1) * rest + (rank - rest) * num_lines;
    }

    process_data->global_end = process_data->global_start + num_lines - 1;

    process_data->num_lines_with_halo = num_lines;
    if(world_size > 1)
    {
        if (rank == 0 || rank == world_size - 1)
        {
            process_data->num_lines_with_halo += 1;
        }
        else
        {
            process_data->num_lines_with_halo += 2;
        }
    }

    process_data->address_offset = options->lines;
    if (rank == 0)
    {
        process_data->address_offset = 0;
    }
}


static void freeMatrices(struct calculation_arguments const* arguments)
{
    free(arguments->M);
    free(arguments->matrix_copy);
}


static void* allocateMemory(size_t size)
{
    void* p;

    if ((p = malloc(size)) == NULL)
    {
        fprintf(stderr, "Allocation problem! (%" PRIu64 " bytes were requested)\n", size);
        exit(1);
    }

    return p;
}


static void allocateMatrices(struct calculation_arguments* arguments, struct options const* options, struct process_data const* process_data)
{
    uint64_t const N = options->lines;
    uint64_t const num_lines_with_halo = process_data->num_lines_with_halo;

    arguments->M = allocateMemory(2 * N * num_lines_with_halo * sizeof(double));
    arguments->matrix_copy = allocateMemory(N * num_lines_with_halo * sizeof(double));
}


static void initMatrices(struct calculation_arguments const* arguments, struct options const* options, struct process_data const* process_data)
{
    uint64_t g, i, j, last_line;

    uint64_t const N = options->lines;
    double const h = arguments->h;
    uint64_t const num_lines_with_halo = process_data->num_lines_with_halo;
    uint64_t const rank = process_data->rank;
    uint64_t const world_size = process_data->world_size;

    typedef double(*matrix)[num_lines_with_halo][N];
    matrix Matrix = (matrix)arguments->M;

    //Ensure that the last process actually initializes its last line
    if(rank == world_size - 1)
    {
        last_line = num_lines_with_halo;
    }
    else
    {
        last_line = num_lines_with_halo - 1;
    }

    for (g = 0; g < 2; g++)
    {
        if(rank == 0)
        {
            i = 0;
        }
        else
        {
            i = 1;
        }

        for (; i < last_line; i++)
        {
            for (j = 0; j < N; j++)
            {
                Matrix[g][i][j] = 0.0;
            }
        }
    }

    //initialize borders, depending on function (function 2: nothing to do)
    if (options->inf_func == FUNC_F0)
    {
        //initialize vertical border
        for (g = 0; g < 2; g++)
        {
            uint64_t offset_index = rank == 0 ? 0 : 1;

            for (i = 0; i < last_line - offset_index; i++)
            {
                Matrix[g][i+offset_index][0] = 1.0 - (h * (i + process_data->global_start));
                Matrix[g][i+offset_index][N-1] = h * (i + process_data->global_start);
            }
        }

        //initialize first row
        if (rank == 0)
        {
            for (g = 0; g < 2; g++)
            {
                for (j = 0; j < N; j++)
                {
                    Matrix[g][0][j] = 1.0 - (h * j);
                }
            }
        }

        //initialize last row
        if (rank == world_size - 1)
        {
            for (g = 0; g < 2; g++)
            {
                for (j = 0; j < N; j++)
                {
                    Matrix[g][num_lines_with_halo - 1][j] = h * j;
                }
            }
        }
    }
}

static void
calculate(struct calculation_arguments const* arguments, struct calculation_results* results, struct options const* options, struct process_data* process_data,
            union FileDescriptor* write_files, union FileDescriptor* read_file, struct metrics* metrics)
{
    uint64_t i, j;
    int m1, m2;
    double star;
    double residuum;
    double maxresiduum;
    double global_maxresiduum;

    uint64_t const N = options->lines;
    double const h = arguments->h;

    uint64_t const rank = process_data->rank;
    uint64_t const predecessor = rank - 1;
    uint64_t const successor = rank + 1;
    uint64_t const last_rank = process_data->world_size - 1;
    uint64_t const iterations_between_writes = options->write_frequency;
    uint64_t const iterations_between_reads = options->read_frequency;
    uint64_t num_lines_with_halo = process_data->num_lines_with_halo;
    uint64_t current_write_pattern = 0;
    double pattern_phase_length = options->pattern_arr_len != 0 ? ((1.0*options->term_iteration) / options->pattern_arr_len) : 0.0;

    MPI_Request requests[4] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL};
    MPI_Status statuses[4];
    struct timeval write_start;
    struct timeval write_completion;

    double pih = 0.0;
    double fpisin = 0.0;

    uint64_t term_iteration = options->term_iteration;

    typedef double (*matrix)[num_lines_with_halo][N];
    matrix Matrix = (matrix)arguments->M;

    double* matrix_copy = arguments->matrix_copy;

    m1 = 0;
    m2 = 1;

    if (options->inf_func == FUNC_FPISIN)
    {
        pih = M_PI * h;
        fpisin = 0.25 * (2 * M_PI * M_PI) * h * h;
    }

    while (term_iteration > 0)
    {
        maxresiduum = 0;
        results->stat_iteration++;

        //Fake MPI collective communication
        if (options->comm_frequency != 0 && results->stat_iteration % options->comm_frequency == 0)
        {
            size_t elements_in_buffer = (options->comm_size_in_kb * 1000) / sizeof(char);
            char* buf = calloc(sizeof(char), elements_in_buffer);
            char* recv_buf = calloc(sizeof(char), elements_in_buffer * process_data->world_size);
            if (!buf || !recv_buf)
            {
                fprintf(stderr, "Memory allocation for fake MPI collective comms failed, aborting...\n");
                exit(1);
            }
            MPI_Allgather(buf, elements_in_buffer, MPI_CHAR, recv_buf, elements_in_buffer, MPI_CHAR, MPI_COMM_WORLD);
            free(buf);
            free(recv_buf);
        }

        if (iterations_between_reads != 0 && results->stat_iteration % iterations_between_reads == 0)
        {
            struct timeval read_start;
            struct timeval read_completion;

            //Need to ensure that something has been written to the file already if we are reading from
            //the same file that is being written to, otherwise opening won't work because detecting the
            //format wouldn't work. (In theory, calling enddef/redef after file creation should work,
            //but then the NetCDF variable can't be found anymore, regardless of how much has been written)
            //TODO look further into this
            #ifdef NETCDF
            static int init = 0;
            if (!init && options->path_to_read_file == NULL)
            {
                *read_file = openFileForRead(options->path_to_write_file);
            }
            init=1;
            #endif
            gettimeofday(&read_start, NULL);
            metrics->bytes_read += readMatrixFromFile((double*)Matrix[m2], N, options, process_data, read_file);
            gettimeofday(&read_completion, NULL);
            metrics->read_operations++;

            metrics->time_spent_reading += (read_completion.tv_sec - read_start.tv_sec) * 1000000 + (read_completion.tv_usec - read_start.tv_usec);
            Matrix = (matrix)arguments->M;
        }

        if (rank != last_rank)
        {
            MPI_Isend(Matrix[m2][num_lines_with_halo - 2], N, MPI_DOUBLE, successor, 0, MPI_COMM_WORLD, &requests[0]);
            MPI_Irecv(Matrix[m2][num_lines_with_halo - 1], N, MPI_DOUBLE, successor, 0, MPI_COMM_WORLD, &requests[1]);
        }
        if (rank != 0)
        {
            MPI_Isend(Matrix[m2][1], N, MPI_DOUBLE, predecessor, 0, MPI_COMM_WORLD, &requests[2]);
            MPI_Irecv(Matrix[m2][0], N, MPI_DOUBLE, predecessor, 0, MPI_COMM_WORLD, &requests[3]);
        }

        MPI_Waitall(4, requests, statuses);

        #pragma omp parallel for default(shared) reduction(max: maxresiduum) private(residuum, star, j)
        for (i = 1; i < num_lines_with_halo-1; i++)
        {
            double fpisin_i = 0.0;

            if (options->inf_func == FUNC_FPISIN)
            {
                double adapted_i = rank == 0 ? (double)i : (double)(i + process_data->global_start - 1);
                fpisin_i = fpisin * sin(pih * adapted_i);
            }

            for (j = 1; j < N - 1; j++)
            {
                star = 0.25 * (Matrix[m2][i - 1][j] + Matrix[m2][i][j - 1] + Matrix[m2][i][j + 1] + Matrix[m2][i + 1][j]);

                if (options->inf_func == FUNC_FPISIN)
                {
                    star += fpisin_i * sin(pih * (double)j);
                }

                if (term_iteration == 1)
                {
                    residuum = Matrix[m2][i][j] - star;
                    residuum = fabs(residuum);
                    maxresiduum = (residuum < maxresiduum) ? maxresiduum : residuum;
                }

                Matrix[m1][i][j] = star;
            }
        }

        if (iterations_between_writes != 0 && results->stat_iteration % iterations_between_writes == 0)
        {
            size_t writes = 1;
            if (options->pattern_arr_len != 0)
            {
                if (((current_write_pattern + 1) * pattern_phase_length) < results->stat_iteration)
                {
                    current_write_pattern++;
                }

                writes = options->pattern_array[current_write_pattern];
            }

            if (!options->immediate_sync_write)
            {
                struct timeval copy_start;
                struct timeval copy_completion;

                gettimeofday(&copy_start, NULL);
                memcpy(matrix_copy, (double*)Matrix[m1], num_lines_with_halo * N * sizeof(double));
                gettimeofday(&copy_completion, NULL);
                metrics->time_spent_copying += (copy_completion.tv_sec - copy_start.tv_sec) * 1000000 + (copy_completion.tv_usec - copy_start.tv_usec);
            }
            else
            {
                matrix_copy = (double*)Matrix[m1];
            }

            gettimeofday(&write_start, NULL);
            for (size_t curr_fd = 0; curr_fd < writes; curr_fd++)
            {
                metrics->bytes_written += writeMatrixToFile(matrix_copy, N, options, process_data, &write_files[curr_fd], curr_fd);
                metrics->write_operations++;
            }

            if (options->immediate_sync_write)
            {
                #if MPI_ASYNC
                waitForCompletion(process_data);
                #endif

                for (size_t curr_fd = 0; curr_fd < writes; curr_fd++)
                {
                    #if MPI_IO_SPLITCOLL
                    waitForCompletion(matrix_copy, &write_files[curr_fd], process_data);
                    #endif
                    fileSync(&write_files[curr_fd]);
                }
                gettimeofday(&write_completion, NULL);
                metrics->time_spent_writing += (write_completion.tv_sec - write_start.tv_sec) * 1000000 + (write_completion.tv_usec - write_start.tv_usec);
            }
        }

        if (term_iteration == 1)
        {
            MPI_Allreduce(&maxresiduum, &global_maxresiduum, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            results->stat_precision = global_maxresiduum;
        }

        i  = m1;
        m1 = m2;
        m2 = i;

        term_iteration--;

        //Completion needs to occur before the next write is started.
        if (!options->immediate_sync_write && metrics->bytes_written != 0 && (results->stat_iteration % iterations_between_writes  == (uint64_t)iterations_between_writes - 1 || iterations_between_writes == 1 || term_iteration == 0))
        {
            #if MPI_ASYNC
            waitForCompletion(process_data);
            #endif

            size_t writes = 1;
            if (options->pattern_arr_len != 0)
            {
                writes = options->pattern_array[current_write_pattern];
            }

            for (size_t curr_fd = 0; curr_fd < writes; curr_fd++)
            {
                #if MPI_IO_SPLITCOLL
                waitForCompletion(matrix_copy, &write_files[curr_fd], process_data);
                #endif

                if (!options->no_file_sync)
                {
                    fileSync(&write_files[curr_fd]);
                }
            }

            gettimeofday(&write_completion, NULL);
            metrics->time_spent_writing += (write_completion.tv_sec - write_start.tv_sec) * 1000000 + (write_completion.tv_usec - write_start.tv_usec);
        }
    }

    results->m = m2;
}


static void
displayOptions(struct calculation_results const* results, struct options const* options)
{
    uint64_t const N = options->lines;
    double time = (comp_time.tv_sec - start_time.tv_sec) + (comp_time.tv_usec - start_time.tv_usec) * 1e-6;
    uint64_t num_matrices = options->immediate_sync_write ? 2 : 3;
    double mem_in_mb = N * N * sizeof(double) * num_matrices / 1000.0 / 1000.0;

    printf("Utilized backend:\t");
    printBackend();
    printf("\nTime taken:\t\t%.3lf s\n", time);
    printf("Mem used (lower bound):\t%.2lf MB\n", mem_in_mb);

    printf("Method for calculation:\t");
    printf("Jacobi");
    printf("\n");

    printf("Lines:\t\t\t%" PRIu64 "\n", options->lines);

    printf("Perturbation function:\t");
    if (options->inf_func == FUNC_F0)
    {
        printf("f(x,y) = 0");
    }
    else if (options->inf_func == FUNC_FPISIN)
    {
        printf("f(x,y) = 2 * pi^2 * sin(pi * x) * sin(pi * y)");
    }
    printf("\n");

    printf("Number of iterations:\t%" PRIu64 "\n", results->stat_iteration);
    printf("Norm of error:\t\t%e\n", results->stat_precision);
    printf("\n");
}


static void
displayMatrix(struct calculation_arguments const* arguments, struct calculation_results const* results, struct options const* options, struct process_data const* process_data)
{
    uint64_t const N = options->lines;
    uint64_t x, y;
    uint64_t const num_lines_with_halo = process_data->num_lines_with_halo;

    typedef double(*matrix)[num_lines_with_halo][N];
    matrix Matrix = (matrix)arguments->M;
    int m = results->m;

    MPI_Status status;

    uint64_t rank = process_data->rank;
    uint64_t from = process_data->global_start;
    uint64_t to = process_data->global_end;

    if (rank == 0)
    printf("Matrix:\n");

    for (y = 0; y < 9; y++)
    {
        //Want to make sure to print the first and last line every time
        uint64_t line;
        if (y == 8)
        {
            line = options->lines - 1;
        }
        else
        {
            line = round(y * (options->lines / 9.0));
        }

        if (rank == 0)
        {
            //check whether this line belongs to rank 0
            if (line > to)
            {
                MPI_Recv(Matrix[m][0], N, MPI_DOUBLE, MPI_ANY_SOURCE, 42 + y, MPI_COMM_WORLD, &status);
            }
        }
        else
        {
            if (line >= from && line <= to)
            {
                MPI_Send(Matrix[m][line - from + 1], N, MPI_DOUBLE, 0, 42 + y, MPI_COMM_WORLD);
            }
        }

        if (rank == 0)
        {
            for (x = 0; x < 9; x++)
            {
                uint64_t col;
                if (x == 8)
                {
                    col = options->lines - 1;
                }
                else
                {
                    col = round(x * (options->lines / 9.0));
                }

                if (line >= from && line <= to)
                {
                    printf("%7.4f", Matrix[m][line][col]);
                }
                else
                {
                    printf("%7.4f", Matrix[m][0][col]);
                }
            }
            printf("\n");
        }
    }
    fflush(stdout);
}


static void displayMetrics(struct process_data const* process_data, struct metrics const* metrics)
{
    uint64_t sum_bytes_written;
    MPI_Reduce(&metrics->bytes_written, &sum_bytes_written, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

    uint64_t sum_bytes_read;
    MPI_Reduce(&metrics->bytes_read, &sum_bytes_read, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

    double sum_cpu_usage;
    MPI_Reduce(&metrics->cpu_usage, &sum_cpu_usage, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    uint64_t sum_copying_time;
    MPI_Reduce(&metrics->time_spent_copying, &sum_copying_time, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

    uint64_t sum_write_operations;
    MPI_Reduce(&metrics->write_operations, &sum_write_operations, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

    uint64_t sum_read_operations;
    MPI_Reduce(&metrics->read_operations, &sum_read_operations, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

    uint64_t avg_write_time;
    MPI_Reduce(&metrics->time_spent_writing, &avg_write_time, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    avg_write_time /= process_data->world_size;

    uint64_t avg_read_time;
    MPI_Reduce(&metrics->time_spent_reading, &avg_read_time, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    avg_read_time /= process_data->world_size;

    if (process_data->rank == 0)
    {
        double time_spent_writing_in_s = avg_write_time * 1e-6;
        double time_spent_reading_in_s = avg_read_time * 1e-6;
        double total_throughput_writing = time_spent_writing_in_s != 0 ? sum_bytes_written / time_spent_writing_in_s : 0.0;
        double total_throughput_reading = time_spent_reading_in_s != 0 ? sum_bytes_read / time_spent_reading_in_s : 0.0;
        double copying_time = sum_copying_time * 1e-6;

        printf("Total write operations across all processes: %"PRIu64"\n", sum_write_operations);
        if (sum_bytes_written < 1000000)
        {
            printf("Bytes written in total: %"PRIu64" B\n", sum_bytes_written);
            printf("Total throughput in writing: %.2lf B/s\n", total_throughput_writing);
        }
        else
        {
            printf("Bytes written in total: %.2lf MB\n", sum_bytes_written / 1000000.0);
            printf("Total throughput in writing: %.2lf MB/s\n", total_throughput_writing / 1000000.0);
        }
        printf("Time spent copying matrices (avg/process): %.3lf s\n\n", copying_time / process_data->world_size);

        printf("Total read operations across all processes: %"PRIu64"\n", sum_read_operations);
        if (sum_bytes_read < 1000000)
        {
            printf("Bytes read in total: %"PRIu64" B\n", sum_bytes_read);
            printf("Total throughput in reading: %.2lf B/s\n", total_throughput_reading);
        }
        else
        {
            printf("Bytes read in total: %.2lf MB\n", sum_bytes_read / 1000000.0);
            printf("Total throughput in reading: %.2lf MB/s\n", total_throughput_reading / 1000000.0);
        }

        printf("Average CPU/core usage per process: %.2lf %%\n\n", sum_cpu_usage / process_data->world_size);
    }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    struct options               options;
    struct calculation_arguments arguments;
    struct calculation_results   results;
    struct process_data          process_data;
    struct metrics               metrics;

    process_data.rank = rank;
    process_data.world_size = size;

    parseArgs(argc, argv, &options);
    if (rank == 0)
        checkArgs(&options);

    union FileDescriptor write_files [options.num_files_per_proc];
    union FileDescriptor read_file;
    if (options.write_frequency != 0)
    {
        //Filename will have the respective rank appended to it, alongside the char 'l', if process local files are necessary
		if (options.file_per_process == PROCESS_LOCAL_FILE)
        {
            char* tmp = NULL;
            transformFilename(rank, &tmp, options.path_to_write_file, 'l');
            options.path_to_write_file = tmp;
        }

        //If pattern is supplied (else-case), the filename will have a number alongside char 'r' appended before opening.
        if (options.num_files_per_proc == 1)
        {
            write_files[0] = openFile(rank, &options);
        }
        else
        {
            size_t path_str_length = strlen(options.path_to_write_file);
            char* original_filename = malloc(sizeof(char) * (path_str_length + 1));
            strcpy(original_filename, options.path_to_write_file);
            options.path_to_write_file = NULL;

            //Going from highest to lowest here to ensure that the filename for the first file
            //will be in path_to_write_file at the end. This file will always be written to (except for pattern 0).
            //Due to this, the potential read without specified path will want to read from that file, as it has
            //actual guaranteed content, as opposed to the other files.
            for (int i = options.num_files_per_proc - 1; i >= 0; i--)
            {
                transformFilename(i, &options.path_to_write_file, original_filename, 'r');
				write_files[i] = openFile(rank, &options);
            }
            free(original_filename);
        }
    }

    if (options.read_frequency != 0)
    {
        if (options.path_to_read_file != NULL)
        {
            read_file = openFileForRead(options.path_to_read_file);
        }
        #ifndef NETCDF
        else
        {
            read_file = openFileForRead(options.path_to_write_file);
        }
        #endif
    }

    initVariables(&arguments, &results, &options, &process_data, &metrics);

    allocateMatrices(&arguments, &options, &process_data);
    initMatrices(&arguments, &options, &process_data);

    if(rank == 0)
    {
        gettimeofday(&start_time, NULL);
    }
    calculate(&arguments, &results, &options, &process_data, write_files, &read_file, &metrics);
    MPI_Barrier(MPI_COMM_WORLD);

    getCpuUsage(&metrics);
    if(rank == 0)
    {
        gettimeofday(&comp_time, NULL);
        displayOptions(&results, &options);
    }
    displayMetrics(&process_data, &metrics);
    displayMatrix(&arguments, &results, &options, &process_data);

    freeMatrices(&arguments);
    free(options.pattern_array);
    #if defined MPI_ASYNC
    free(process_data.request_arr);
    #endif

    if (options.write_frequency != 0)
    {
        for (uint64_t i = 0; i < options.num_files_per_proc; i++)
            closeFile(&write_files[i]);
        if (options.file_per_process == PROCESS_LOCAL_FILE || options.num_files_per_proc != 1)
            free(options.path_to_write_file);
    }
    if (options.write_frequency != 0 && options.path_to_read_file != NULL)
    {
        closeFile(&read_file);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
