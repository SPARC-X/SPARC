#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <float.h>
#include <assert.h>

#include "toolsAtom.h"
#include "isddftAtom.h"
#include "tools.h"

#define min(x,y) ((x)<(y)?(x):(y))
#define max(x,y) ((x)>(y)?(x):(y))

/**
 * @brief   Sort an array in ascending order and return original index.
 */
void Sort_Int(int *X, int len, int *Xsorted, int *Ind) {
    int i;
    Sortstruct *sortobj = (Sortstruct *)malloc(len * sizeof(Sortstruct));
    if (sortobj == NULL) {
        printf("\nMemory allocation failed!\n");
        exit(EXIT_FAILURE);
    }
    for (i = 0; i < len; i++) {
        sortobj[i].value = X[i];
        sortobj[i].index = i;
    }
    qsort(sortobj, len, sizeof(sortobj[0]), cmp); // cmp is the compare func
    for (i = 0; i < len; i++) {
        Xsorted[i] = sortobj[i].value;
        Ind[i] = sortobj[i].index;
    }
    free(sortobj);
}

/**
* @brief To print in row x col format using a col major format storage
*/
void printColMajor(double *Mat, int n_rows, int n_cols) {
    for (int row = 0; row < n_rows; row++) {
        for (int col = 0; col < n_cols; col++) {
            printf("%11.8f ", Mat[col*n_rows + row]);
        }
        printf("\n");
    }
}

/**
* @brief Flip contents of every column in Matrix for column major storage
*/
void flipContents(double *Mat, int n_rows, int n_cols) {
    for (int col = 0; col < n_cols; col++) {
        for (int row = 0; row < n_rows / 2; row++) {
            int opposite_row = (n_rows - 1) - row;
            double temp = Mat[col * n_rows + row];
            Mat[col * n_rows + row] = Mat[col * n_rows + opposite_row];
            Mat[col * n_rows + opposite_row] = temp;
        }
    }
}