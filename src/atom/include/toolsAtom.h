#ifndef TOOLSATOM_H
#define TOOLSATOM_H

#include "isddftAtom.h"
#include "isddft.h"
#include "tools.h"
#include <stdio.h>

void Sort_Int(int *X, int len, int *Xsorted, int *Ind);
void printColMajor(double *Mat, int n_rows, int n_cols);

/**
* @brief Flip contents of every column in Matrix for column major storage
*/
void flipContents(double *Mat, int n_rows, int n_cols);

#endif