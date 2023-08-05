#define _XOPEN_SOURCE 500

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <math.h>
#include "argparse.h"


static _Noreturn void usage()
{
    printf("Usage: numio-<backend> -m iter=<iter>,size=<size>,pert=<pert> [-w freq=<freq>,path=/path/to/file[,imm][,nofilesync][,pattern=<pat>]]");
    printf("[-r freq=<freq>[,path=/path/to/readfile]] [-c freq=<freq>,size=<size>] [-t <threads>] [-l]\n");
    printf("\n");
    printf("Optional parameters are enclosed in []\n");
    printf("-m (Matrix options):\n");
    printf("  - iter: number of iterations to go through: (1 .. %d)\n", MAX_ITERATION);
    printf("  - size: number of matrix lines (9 .. ). Note that the matrix is square\n");
    printf("  - pert: perturbation/interference function (1 .. 2)\n");
    printf("                 %1d: f(x,y) = 0\n", FUNC_F0);
    printf("                 %1d: f(x,y) = 2 * pi^2 * sin(pi * x) * sin(pi * y)\n", FUNC_FPISIN);
    printf("-w (Write options):\n");
    printf("  - freq: number of iterations between writing (0: no writing, 1: every iteration, 2: every second iter... )\n");
    printf("  - path: path to the matrix output file\n");
    printf("  - imm: Provide this flag if the writes should be sync'd immediately and not overlapped with computation\n");
    printf("  - nofilesync: Provide this flag to disable all filesyncs\n");
    printf("  - pattern: A pattern can be provided for irregular write accesses, consisting of integers separated with colons. See README for an example\n");
    printf("-r (Read options):\n");
    printf("  - freq: number of iterations between reading (numbers with same effect as in write frequency)\n");
    printf("  - path: path to file that is used for reading (parameter is optional and only necessary if writefreq >= readfreq)\n");
    printf("-c (Fake MPI collective communication):\n");
    printf("  - freq: number of iterations between the fake collective communication (numbers with same effect as previous frequencies)\n");
    printf("  - size: size of the buffer used in the collective communication in KB\n");
    printf("-t: Number of threads (Default/Recommended is 1)\n");
    printf("-l: per-process file for writing (0/default: one file, 1: one file per process)\n");
    printf("  If this is provided, the name of the write files will have the respective rank appended to them\n");
    printf("\n");
    printf("Example: mpirun -n 2 ./numio-<backend> -m iter=20,size=10,pert=1 -w freq=5,path=run.out,pattern=4:0:5 -r freq=18 \n");
    exit(EXIT_FAILURE);
}

unsigned long handleStrtoul(const char option, const char* arg)
{
    errno = 0;
    unsigned long num;
    char* endptr;

    num = strtoul(arg, &endptr, 0);
    if (errno != 0)
    {
        perror("strtoul");
        exit(EXIT_FAILURE);
    }
    if (*endptr != '\0' && endptr != arg)
    {
        fprintf(stderr, "Can't parse -%c properly\n", option);
        usage();
    }
    if (endptr == arg)
    {
        fprintf(stderr, "No digits found in -%c\n", option);
        usage();
    }
    return num;
}

void parsePattern(const char option, char* value, struct options* options)
{
    options->pattern_arr_len = 1;
    char* token;

    for(size_t i = 0; *(value + i) != '\0'; i++)
    {
        if (*(value + i) == ':')
        options->pattern_arr_len++;
    }
    options->pattern_array = realloc(options->pattern_array, options->pattern_arr_len * sizeof(uint64_t));

    token = strtok(value, ":");
    for (size_t i = 0; i < options->pattern_arr_len; i++)
    {
        if (token == NULL)
        {
            fprintf(stderr, "Unexpected error while parsing pattern\n");
            usage();
        }
        options->pattern_array[i] = handleStrtoul(option, token);
        token = strtok(NULL, ":");

        //The highest number in pattern defines the necessary number of files, since every write in a cycle will go to a different file
        if (options->pattern_array[i] > options->num_files_per_proc)
        {
            options->num_files_per_proc = options->pattern_array[i];
        }
    }
}


void processMatrixOptions(const char option, char* opt_str, struct options* options)
{
    int ret_subopt;
    char* value;
    char* tokens[] = {"size", "iter", "pert", NULL};

    while (*opt_str != '\0')
    {
        ret_subopt = getsubopt(&opt_str, tokens, &value);
        switch (ret_subopt)
        {
            case 0:
            options->lines = handleStrtoul(option, value);
            break;

            case 1:
            options->term_iteration = handleStrtoul(option, value);
            break;

            case 2:
            options->inf_func = handleStrtoul(option, value);
            break;

            default:
            fprintf(stderr, "Unrecognized option %s in %c\n", value, option);
            usage();
        }
    }
}

void processRWOptions(const char option, char* opt_str, struct options* options)
{
    int ret_subopt;
    char* value;
    char* tokens[] = {"freq", "path", "imm", "pattern", "nofilesync", NULL};

    while (*opt_str != '\0')
    {
        ret_subopt = getsubopt(&opt_str, tokens, &value);
        switch (ret_subopt)
        {
            case 0:
            if (option == 'w')
                options->write_frequency = handleStrtoul(option, value);
            else
                options->read_frequency = handleStrtoul(option, value);
            break;

            case 1:
            if (option == 'w')
                options->path_to_write_file = value;
            else
                options->path_to_read_file = value;
            break;

            case 2:
            if (option == 'w')
            {
                options->immediate_sync_write = 1;
            }
            else
            {
                fprintf(stderr, "imm is not available for reads, as they are always sync'd immediately\n");
                usage();
            }
            break;

            case 3:
            if (option == 'w')
            {
                parsePattern(option, value, options);
            }
            else
            {
                fprintf(stderr, "pattern is not available for reads as of now\n");
                usage();
            }
            break;

            case 4:
            if (option == 'w')
            {
                options->no_file_sync = 1;
            }
            else
            {
                fprintf(stderr, "nofilesync is not available for reads, as they are always sync'd immediately\n");
                usage();
            }
            break;

            default:
            fprintf(stderr, "Unrecognized option %s in %c\n", value, option);
            usage();
        }
    }
}

void processCommOptions(const char option, char* opt_str, struct options* options)
{
    int ret_subopt;
    char* value;
    char* tokens[] = {"freq", "size", NULL};

    while (*opt_str != '\0')
    {
        ret_subopt = getsubopt(&opt_str, tokens, &value);
        switch (ret_subopt)
        {
            case 0:
            options->comm_frequency = handleStrtoul(option, value);
            break;

            case 1:
            options->comm_size_in_kb = handleStrtoul(option, value);
            break;

            default:
            fprintf(stderr, "Unrecognized option %s in %c\n", value, option);
            usage();
        }
    }
}

void parseArgs(int argc, char** argv, struct options* options)
{
    int option;

    options->write_frequency = 0;
    options->path_to_write_file = NULL;
    options->read_frequency = 0;
    options->path_to_read_file = NULL;
    options->file_per_process = 0;
    options->threads = 1;
    options->lines = 0;
    options->inf_func = 0;
    options->immediate_sync_write = 0;
    options->no_file_sync = 0;
    options->num_files_per_proc = 1;
    options->pattern_arr_len = 0;
    options->pattern_array = malloc(0);
    options->comm_frequency = 0;
    options->comm_size_in_kb = 0;

    while ((option = getopt(argc, argv, ":w:t:m:r:c:l")) != -1)
    {
        switch (option)
        {
            //process local file
            case 'l':
            options->file_per_process = PROCESS_LOCAL_FILE;
            break;

            //Matrix-related options (size, iterations, perturbation func)
            case 'm':
            processMatrixOptions(option, optarg, options);
            break;

            //Read and write options respectively (frequency, path)
            case 'r':
            case 'w':
            processRWOptions(option, optarg, options);
            break;

            //faked MPI collective comms options
            case 'c':
            processCommOptions(option, optarg, options);
            break;

            //Thread num
            case 't':
            options->threads = handleStrtoul(option, optarg);
            break;

            case ':':
            fprintf(stderr, "Option -%c requires an argument\n", optopt);
            usage();

            default:
            fprintf(stderr, "Unexpected option -%c\n", optopt);
            usage();
        }
    }

    if (argc != optind)
    {
        fprintf(stderr, "Unexpected extra option %s, aborting...\n", argv[optind]);
        usage();
    }
}

static int readWriteConflict(struct options const* options)
{
    int reads_more_frequent = options->read_frequency <= options->write_frequency;

    //If the pattern starts with one or more 0's, ensure that the first actual write still occurs before the first read
    if (options->pattern_arr_len != 0)
    {
        uint64_t first_index_of_nonzero_in_pattern = 0;
        for (uint64_t i = 0; i < options->pattern_arr_len; i++)
        {
            if (options->pattern_array[i] != 0)
            {
                first_index_of_nonzero_in_pattern = i;
                break;
            }
        }

        double pattern_phase_length =  (1.0*options->term_iteration) / options->pattern_arr_len;
        uint64_t empty_iterations = floor(pattern_phase_length * first_index_of_nonzero_in_pattern);
        uint64_t first_iteration_with_write = empty_iterations - (empty_iterations % options->write_frequency) + options->write_frequency;

        if (first_iteration_with_write >= options->read_frequency)
        {
            reads_more_frequent = 1;
        }
    }
    return reads_more_frequent || options->write_frequency == 0;
}

void checkArgs(struct options const* options)
{
    if (options->inf_func != FUNC_F0 && options->inf_func != FUNC_FPISIN)
    {
        fprintf(stderr, "Faulty input for perturbation function\n");
        usage();
    }

    if (options->term_iteration < 1 || options->term_iteration > MAX_ITERATION)
    {
        fprintf(stderr, "Faulty input for number of iterations\n");
        usage();
    }

    if (options->path_to_read_file == NULL && options->read_frequency != 0 && readWriteConflict(options))
    {
        fprintf(stderr, "Provide a read file if reads are more frequent than writes, otherwise the first read will break\n");
        usage();
    }

    if (options->lines < 9)
    {
        fprintf(stderr, "Faulty input for number of lines\n");
        usage();
    }

    if (options->path_to_write_file == NULL && options->write_frequency != 0)
    {
        fprintf(stderr, "Provide a path to which the output matrix will be written\n");
        usage();
    }

    if (options->immediate_sync_write && options->no_file_sync)
    {
        fprintf(stderr, "Immediate file sync and no file sync don't match\n");
        usage();
    }
}
