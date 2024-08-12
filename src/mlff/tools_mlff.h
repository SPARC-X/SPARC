#ifndef TOOLS_MLFF_H
#define TOOLS_MLFF_H

// typedef int value_type;

#include "mlff_types.h"



// void tridiag_gen(double *A, double *B, double *C, double *D, int len);
// void getYD_gen(double *X, double *Y, double *YD, int len);
// void SplineInterp(double *X1,double *Y1,int len1,double *X2,double *Y2,int len2,double *YD);
// void init_dyarray(dyArray *a);
// void realloc_dyarray(dyArray *a, size_t new_capacity);
// void dyarray_double_capacity(dyArray *a);
// void dyarray_half_capacity(dyArray *a);
// void append_dyarray(dyArray *a, value_type element);
// value_type pop_dyarray(dyArray *a);
// void clear_dyarray(dyArray *a);
// void delete_dyarray(dyArray *a);
// void print_dyarray(const dyArray *a);
// void show_dyarray(const dyArray *a);
double dotProduct(double* vect_A, double* vect_B, int size_vector);
double get_mean(double a[], int n);
double get_variance(double a[], int n);
double largest(double* arr, int n);
double smallest(double* arr, int n);
double lin_search_double(double *arr, int n, double x);
double lin_search_INT(int *arr, int n, int x);


#endif
