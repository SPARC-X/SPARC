#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#ifdef USE_MKL
    #define MKL_Complex16 double _Complex
    #include <mkl.h>
#else
    #include <cblas.h>
    #include <lapacke.h>
#endif


#include "mlff_types.h"
#include "isddft.h"
#include "covariance_matrix.h"
#include "cyclix_mlff_tools.h"

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))

void get_img_cyclix(double L2, double temp_tol, int *img_ny, int *img_py){
    *img_ny = 0;
    *img_py = floor(2*M_PI/L2+temp_tol) - 1; // all atoms and images in cyclic direction
}

void get_cartesian_dist_cyclix(double xi, double yi, double zi, double xj, double yj, double zj, double twist, double *dx, double *dy, double *dz){
    *dx = xj*cos(yj+twist*zj) - xi*cos(yi+twist*zi);
    *dy = xj*sin(yj+twist*zj) - xi*sin(yi+twist*zi);
    *dz = zj - zi;
}

