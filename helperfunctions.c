#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <inttypes.h>
#include <string.h>
#include "helperfunctions.h"

/*************************************************/
/* To get the CPU usage, /proc/self/stat         */
/* and /proc/uptime are used to get the relevant */
/* measurements per process for later reduction. */
/*************************************************/
void getCpuUsage(struct metrics* metrics)
{
    char buf[128];
    unsigned long utime, stime, cutime, cstime;
    long long unsigned int starttime;
    double uptime;
    long hertz = sysconf(_SC_CLK_TCK);
    int ret;
    FILE* stat_file = fopen("/proc/self/stat", "r");
    if (stat_file == NULL)
    {
        printf("Problem opening /proc/self/stat, aborting...\n");
        exit(1);
    }
    //skipping over unnecessary values
    for (int i = 0; i < 13; i++)
    {
        ret = fscanf(stat_file, "%s", buf);
        if (ret == EOF)
        {
            fprintf(stderr, "Problem reading /proc/self/stat, aborting...\n");
            exit(1);
        }
    }
    ret = fscanf(stat_file, "%lu %lu %lu %lu", &utime, &stime, &cutime, &cstime);
    if (ret == EOF)
    {
        fprintf(stderr, "Problem reading /proc/self/stat, aborting...\n");
        exit(1);
    }

    for (int i = 0; i < 4; i++)
    {
        ret = fscanf(stat_file, "%s", buf);
        if (ret == EOF)
        {
            fprintf(stderr, "Problem reading /proc/self/stat, aborting...\n");
            exit(1);
        }
    }
    ret = fscanf(stat_file, "%llu", &starttime);
    if (ret == EOF)
    {
        fprintf(stderr, "Problem reading /proc/self/stat, aborting...\n");
        exit(1);
    }
    fclose(stat_file);

    FILE* uptime_file = fopen("/proc/uptime", "r");
    if (uptime_file == NULL)
    {
        fprintf(stderr, "Problem opening /proc/uptime, aborting...\n");
        exit(1);
    }
    ret = fscanf(uptime_file, "%lf", &uptime);
    if (ret == EOF)
    {
        fprintf(stderr, "Problem reading /proc/uptime, aborting...\n");
        exit(1);
    }
    fclose(uptime_file);

    unsigned long total = utime + stime + cutime + cstime;
    double seconds = uptime - ((double) starttime / hertz);
    metrics->cpu_usage = 100 * (((double)total / hertz) / seconds);
}

void transformFilename(int num, char** path, char const* original_path, char extension)
{
    int ret;
    int num_as_char_length = snprintf(NULL, 0, "%d", num);
    char sep = '_';
    int max_length = 3 + num_as_char_length + strlen(original_path);
    *path = realloc(*path, sizeof(char) * max_length);
    
    ret = snprintf(*path, max_length, "%s%c%c%d", original_path, sep, extension, num);

    if (ret != max_length-1 || (*path)[max_length-1] != '\0')
    {
        fprintf(stderr, "Filename conversion for -pattern and/or -l option failed, aborting...\n");
        exit(EXIT_FAILURE);
    }
}