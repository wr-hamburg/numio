#ifndef NUMIOSTRUCTS_H
#define NUMIOSTRUCTS_H

#include <inttypes.h>
#include <mpi.h>

//For checks regarding file_per_process
#define PROCESS_LOCAL_FILE 1

struct process_data
{
	uint64_t rank;
	uint64_t world_size;
	uint64_t num_lines_with_halo;   /* number of lines + the halo lines for the communication */
	uint64_t global_start;          /* global index of the first line of the locally managed matrix */
	uint64_t global_end;            /* global index of the last line of the locally managed matrix */
	uint64_t address_offset;        /* Address offset within own matrix when writing to exclude halo lines */
	#if defined MPI_ASYNC
	MPI_Request* request_arr;       /* to wait for completion */
	uint64_t request_arr_len;
	#endif
	#if defined MPI_IO_SPLITCOLL
	uint64_t offset_for_final_write;
	#endif
	#if defined MPI_ASYNC || defined MPI_IO_SPLITCOLL
	uint64_t expected_write_size;   /* comparison for equality upon completion */
	#endif
};

struct calculation_arguments
{
	double   h;            /* length of a space between two lines */
	double*  M;            /* two matrices with real values */
	double*  matrix_copy;  /* copy of single matrix for IO */
};

struct calculation_results
{
	uint64_t m;
	uint64_t stat_iteration; /* number of current iteration */
	double   stat_precision; /* actual precision of all processes in iteration */
};

struct options
{
	uint64_t write_frequency;      /* number of iterations between writes */
	uint64_t read_frequency;       /* number of iterations between reads */
	uint64_t lines;                /* matrix size */
	uint64_t inf_func;             /* inference function */
	uint64_t term_iteration;       /* terminate if iteration number reached */
	uint64_t threads;              /* number of threads per process */
	uint64_t file_per_process;     /* 0 if every process should write to same global file, else 1 */
	uint64_t immediate_sync_write; /* 1 if the writes should be sync'd immediately, else 0 */
	uint64_t no_file_sync;         /* 1 if no file syncs should occur at all, else 0 */
	char* path_to_read_file;       /* path to read file (if supplied) */
	char* path_to_write_file;
	uint64_t* pattern_array;       /* array holding the pattern values */
	uint64_t pattern_arr_len;
	uint64_t num_files_per_proc;   /* number of files necessary for the supplied pattern (max in pattern essentially) */
	uint64_t comm_frequency;       /* number of iterations between fake mpi collective communication */
	uint64_t comm_size_in_kb;      /* buffer size for fake collective comms*/
};

struct metrics
{
	uint64_t bytes_written;
	uint64_t bytes_read;
	double cpu_usage;
	uint64_t time_spent_copying;
	uint64_t time_spent_writing;
	uint64_t time_spent_reading;
	uint64_t write_operations;
	uint64_t read_operations;
};

#endif