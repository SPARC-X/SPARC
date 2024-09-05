#ifndef CYCLIX_MLFF_TOOLS_H
#define CYCLIX_MLFF_TOOLS_H

#include "mlff_types.h"

void get_img_cyclix(double L2, double temp_tol, int *img_ny, int *img_py);

void get_cartesian_dist_cyclix(double xi, double yi, double zi, double xj, double yj, double zj, double twist, double *dx, double *dy, double *dz);
#endif