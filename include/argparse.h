#ifndef ARGPARSE_H
#define ARGPARSE_H

#define MAX_ITERATION     200000
#define FUNC_F0           1
#define FUNC_FPISIN       2

#include "numiostructs.h"

void parseArgs(int argc, char** argv, struct options* options);

void checkArgs(struct options const* options);

#endif