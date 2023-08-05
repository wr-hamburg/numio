#ifndef HELPERFUNCTIONS_H
#define HELPERFUNCTIONS_H

#include <inttypes.h>
#include "iofunctions.h"

void getCpuUsage(struct metrics* metrics);

void transformFilename(int num, char** path, char const* original_path, char extension);

#endif