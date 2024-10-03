/**
 * @file    tools.h
 * @brief   This file declares the tool functions.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#ifndef TOOLS_H
#define TOOLS_H

#ifdef USE_MKL
    #define MKL_Complex16 double _Complex
    #include <mkl.h>
#endif
#ifdef USE_FFTW
    #include <fftw3.h>
#endif

#include "isddft.h"

typedef struct _Sortstruct {
    double value;
    int index;
} Sortstruct;



/**
 * @brief   Compare function for qsort().
 */
int cmp(const void *a,const void *b);


/**
 * @brief   Convert string to lower case.
 */
char *strlwr(char *str);


/**
 * @brief   Convert string to upper case.
 */
char *strupr(char *str);


/**
 * @brief   Compare two strings ignoring case.
 *
 * @ref  https://code.woboq.org/userspace/glibc/string/strcmp.c.html
 *       https://searchcode.com/codesearch/view/10193301/
 */
int strcmpi(const char *p1, const char *p2);


/**
 * @brief   Simplify a linux style path (works for both absolute paths 
 *          and relative paths).
 *
 * @ref  LeetCode problem https://leetcode.com/problems/simplify-path/.
 */
void simplifyPath(char* path, char *newpath, size_t newlen);



/**
 * @brief Exact file path from a file name.
 * 
 * @param filename File name.
 * @param path (OUTPUT) Path (relative or absolute) where the file is located.
 * @param maxlen Maximum length of the path.
 */
void extract_path_from_file(const char *filename, char *path, int maxlen);


/**
 * @brief Combine a path and a file name and simplify the path.
 *        
 * If the file name is already provided in the absolute path form, then
 * path is ignored. Otherwise, the new file name would be a simplified path
 * equivalent to "path/fname".
 * 
 * @param path The path relative to which fname is given.
 * @param fname The name provided relative to path.
 * @param newfname (OUTPUT) The new file name.
 */
void combine_path_filename(const char *path, const char *fname, char *newfname, int maxlen);


/**
 * @brief   The following code for factorizing an integer is copied from Rosetta Code
 *          https://rosettacode.org/wiki/Factors_of_an_integer#C . It is only used when
 *          SPARC_Dims_create_2D_BLCYC() fails to find any reasonable decomposition of
 *          npband.
 */
typedef struct {
    int *list;
    short count; 
} Factors;

void xferFactors( Factors *fctrs, int *flist, int flix );
Factors *factor( int num, Factors *fctrs);
void sorted_factor( int num, Factors *fctrs);

// Equivalent to ceil(x / (double)y), where x, y are positive integers
int ceil_div(const unsigned int x, const unsigned int y);

/**
 * @brief   Calculates derivatives of a tabulated function required for spline interpolation.
 *
 *          Given arrays X[1..len] and Y[1..len] containing a tabulated function Y[i] = f(X[i]),
 *          with X[1] < X[2] < ... < X[len]. getYD_gen() finds the derivatives of f(x) at X[1..len].
 *          The output YD[] will be used for applying the cubic spline interpolation.
 *
 *          Reference: Cubic Spline Interpolation: A Review, George Wolberg, Technical Report, 
 *          Department of Computer Science, Columbia University
 */
void getYD_gen(double *X, double *Y, double *YD, int len);


/**
 * @brief   Cubic spline interpolation from precalculated data.
 *
 *          This function does cubic spline interpolation. Given arrays X1[1..len1] and Y1[1..len1] 
 *          containing a tabulated function Y1[i] = f(X1[i]), with X1[1] < X1[2] < ... < X1[len1], 
 *          and given the arrays YD[1..len1] (the output from getYD_gen()), which contains the first 
 *          derivative of the tabulated function f(x) at X1[1..len1], and given X2[1..len2], this 
 *          routine returns a cubic-spline interpolated array Y2[1..len2]. 
 *          
 *          Reference: Numerical recipes in C.
 */
void SplineInterp(double *X1, double *Y1, int len1, double *X2, double *Y2, int len2, double *YD);


/**
 * @brief   Sort an array in ascending order and return original index.
 */
void Sort(double *X, int len, double *Xsorted, int *Ind);


/**
 * @brief   Sort input values and then apply spline interpolation.
 *
 *          This function is the same as SplineInterp except it doesn't assume input array X2[1...len2]
 *          is ascendingly ordered. It first apply a qsort() to sort X2 and then call SplineInterp to
 *          do the interpolation. Note: qsort() is VERY EXPENSIVE, therefore try to avoid calling 
 *          this function too ofen if possible. For example, if the same X2 array is used multiple times,
 *          one should sort X2 once and then call SplineInterp multiple times.
 */
void SortSplineInterp(double *X1, double *Y1, int len1, double *X2, double *Y2, int len2, double *YD);



/**
 * @brief   Cubic spline evaluation from precalculated data. This function
 *          assumes X1 is a uniformly increasing grid.
 *
 *          This function runs much faster than SortSplineInterp above. This 
 *          function assumes that X1 is uniform, and therefore one can 
 *          directly find the interval any point in X2 is located.
 */
void SplineInterpUniform(
    double *X1,double *Y1,int len1,double *X2,double *Y2,int len2,double *YD
);



/**
 * @brief   Cubic spline evaluation from precalculated data. This function
 *          assumes X1 is a monotically increasing grid (but not necessarily
 *          uniform).
 *
 *          This function runs faster than SortSplineInterp above in general.
 *          This function assumes that X1 is monotically increasing, and 
 *          therefore one can use a binary search approach to find the 
 *          interval any point in X2 is located.
 */
void SplineInterpNonuniform(
    double *X1,double *Y1,int len1,double *X2,double *Y2,int len2,double *YD
);



/**
 * @brief   Binary search to find which interval a given number 
 *          is located. list[...] is assumed to be monotonically
 *          increasing.
 */
int binary_interval_search(const double *list, const int len, const double x);


/**
 * @brief   The main funciton for Cubic spline evaluation from precalculated
 *          data. This function calls the appropriate uniform or non-uniform
 *          routines.         
 */
void SplineInterpMain(double *X1,double *Y1,int len1,
    double *X2,double *Y2,int len2,double *YD,int isUniform);


/**
 * @brief   Cubic spline coefficients.
 *
 *          This function finds the coefficients of the cubic polynomials within each data interval
 *          (X[k],X[k+1]). Given arrays X, Y, and YD with len entries, the coefficients are stored 
 *          in arrays A3, A2, A1 and A0. Note that the input array X is ascendingly ordered.
 */
//void SplineCoeff(double *X,double *Y,double *YD,int len,double *A3,double *A2,double *A1,double *A0);


/**
 * @brief   Cubic spline coefficients, but without copying Y and YD into A0 and A1, respectively.
 */
void SplineCoeff(double *X,double *Y,double *YD,int len,double *A3,double *A2);



/**
 * @brief   Solves a tridiagonal system using Gauss Elimination.
 *
 *          Reference: Cubic Spline Interpolation: A Review, George Wolberg, Technical Report, 
 *          Department of Computer Science, Columbia University.
 */
void tridiag_gen(double *A, double *B, double *C, double *D, int len);


/**
 * @brief   Calculates the terms involving factorials in finite difference 
 *          weights calculation.
 *
 *          This function determines the expression fract(n,k) = (n!)^2/((n-k)!(n+k)!).
 *
 * @param n
 * @param k
 */
double fract(int n,int k);




/**
 * @brief   Calculate global 2-norm of a vector among the given communicator. 
 *          
 */
void Vector2Norm(const double *Vec, const int len, double *vec_2norm, MPI_Comm comm);

/**
 * @brief   Calculate global 2-norm of a vector among the given communicator. 
 *          
 */
void Vector2Norm_complex(const double _Complex *Vec, const int len, double *vec_2norm, MPI_Comm comm);


/**
 * @brief   Calculate global 2-norm of a vector among the given communicator. 
 *          
 */
void VectorDotProduct(const double *Vec1, const double *Vec2, const int len, double *vec_dot, MPI_Comm comm);

/**
 * @brief   Calculate global 2-norm of a vector with integration weight among the given communicator. 
 *          
 */
void VectorDotProduct_wt(const double *Vec1, const double *Vec2, const double *Intgwt, const int len, double *vec_dot, MPI_Comm comm); 


/**
 * @brief   Calculate global 2-norm of a vector among the given communicator. 
 *          
 */
void VectorDotProduct_complex(const double _Complex *Vec1, const double _Complex *Vec2, const int len, double *vec_dot, MPI_Comm comm);


/**
 * @brief   Calculate global sum of a vector among the given communicator. 
 *          
 */
void VectorSum(const double *Vec, const int len, double *vec_sum, MPI_Comm comm);

/**
 * @brief   Calculate global weighted sum of a vector among the given communicator. 
 */
void VectorSum_wt(const double *Vec, const double *weight, const int len, double *vec_sum, MPI_Comm comm);


/**
 * @brief   Calculate shift of a vector, x = x + c.
 *          
 */
void VectorShift(double *Vec, const int len, const double c, MPI_Comm comm);


/**
 * @brief   Scale a vector, x = x * c.
 */
void VectorScale(double *Vec, const int len, const double c, MPI_Comm comm);


/**
 * @brief   Create a perturbed unit constant vector distributed among the given communicator. 
 *      
 */
void SetUnitOnesVector(double *Vec, int n_loc, int n_global, double pert_fac, MPI_Comm comm);


/**
 * @brief Create a random matrix with each entry a random number within given range.
 *        In order to increase the flexibility of the routine, this routine also
 *        accepts a shift to the seed, so that the same process can call this routine
 *        multiple times and get different random arrays.
 * 
 * @param Mat Local part of the matrix.
 * @param m Number of rows of the local copy of the matrix.
 * @param n Number of columns of the local part of the matrix.
 * @param rand_min Minimum value of the random entries.
 * @param rand_max Maximum value of the random entries.
 * @param seed Random number generator seed, seed >= 1.
 */
void SetRandMat_seed(double *Mat, int m, int n, double rand_min, double rand_max, int seed);


/**
 * @brief   Create a random matrix with each entry a random number within given range. 
 * 
 *          Note that each process within comm will have different random entries.
 *
 * @param Mat   Local part of the matrix.
 * @param m     Number of rows of the local copy of the matrix.
 * @param n     Number of columns of the local part of the matrix.
 */
void SetRandMat(double *Mat, int m, int n, double rand_min, double rand_max, MPI_Comm comm);


/**
 * @brief   Create a random matrix with each entry a random number within given range. 
 * 
 *          Note that each process within comm will have different random entries.
 *
 * @param Mat   Local part of the matrix.
 * @param m     Number of rows of the local copy of the matrix.
 * @param n     Number of columns of the local part of the matrix.
 */
void SetRandMat_complex(double _Complex *Mat, int m, int n, double rand_min, double rand_max, MPI_Comm comm);


/**
 * @brief   Create a random vector with fixed seeds corresponding to it's global index.
 * 
 *			This function is to fix the random number seeds so that all random numbers 
 *			generated in parallel under MPI are the same as those generated in sequential 
 *          execution.
 */
void SeededRandVec (
	double *Vec, const int DMVert[6], const int gridsizes[3],
	const double rand_min, const double rand_max,
	const int seed_offset
);


/**
 * @brief   Create a random vector with fixed seeds corresponding to it's global index.
 * 
 *			This function is to fix the random number seeds so that all random numbers 
 *			generated in parallel under MPI are the same as those generated in sequential 
 *          execution.
 */
void SeededRandVec_complex (
	double _Complex *Vec, const int DMVert[6], const int gridsizes[3],
	const double rand_min, const double rand_max,
	const int seed_offset
);


/**
 * @brief Check if a grid lies outside the global domain.
 *
 * @param ip Grid local coordinate index in the 1st dir.
 * @param jp Grid local coordinate index in the 2nd dir.
 * @param kp Grid local coordinate index in the 3rd dir.
 * @param origin_shift_i Shift of the origin of the above local index in the 1st dir.
 * @param origin_shift_j Shift of the origin of the above local index in the 2nd dir.
 * @param origin_shift_k Shift of the origin of the above local index in the 3rd dir.
 * @param DMVerts Local domain vertices.
 * @param gridsizes Global grid sizes.
 * @return int
 */
int is_grid_outside(const int ip, const int jp, const int kp,
    const int origin_shift_i, const int origin_shift_j, const int origin_shift_k,
    const int DMVerts[6], const int gridsizes[3]);


/**
 * @brief Check if a grid lies outside the global domain and returns which region it
 * lies in (indexed from 0 to 25, center region is indexed -1).
 *
 * @param ip Grid local coordinate index in the 1st dir.
 * @param jp Grid local coordinate index in the 2nd dir.
 * @param kp Grid local coordinate index in the 3rd dir.
 * @param origin_shift_i Shift of the origin of the above local index in the 1st dir.
 * @param origin_shift_j Shift of the origin of the above local index in the 2nd dir.
 * @param origin_shift_k Shift of the origin of the above local index in the 3rd dir.
 * @param DMVerts Local domain vertices.
 * @param gridsizes Global grid sizes.
 * @return int The index of region the grid lies in. The index goes from bottom left
 * corner (0-based) to upper right corner (25). We skip the center region, which is
 * inside the domain, and continues after that. The index goes from 0 to 25. If grid
 * is at the center region (i.e., within the domain), -1 is returned.
 */
int grid_outside_region(const int ip, const int jp, const int kp,
    const int origin_shift_i, const int origin_shift_j, const int origin_shift_k,
    const int DMVerts[6], const int gridsizes[3]);


/**
 * @brief   Create a rectangular grid in n-d space. 
 *
 *          This function tries to achieve the same objective as the
 *          MATLAB(TM) function ndgrid. The result X is a (ncopy*nnodes)-by-ndims 
 *          matrix that contains all the coordinates in the n-d space.
 *          Note that X is column major.
 *
 * @param ndims     Number of dimensions of the space.
 * @param xs        An array of length ndims containing start positions 
 *                  in each dimension.
 * @param xe        An array of length ndims containing end positions 
 *                  in each dimension.
 * @param X         Output matrix containing ncopy of all the coordinates in 
 *                  the n-d space.
 */
void c_ndgrid(int ndims, int *xs, int *xe, int *X);



/**
 * @brief   Create a rectangular grid in n-d space using recursion. 
 *
 *          This function tries to achieve the same objective as the
 *          MATLAB(TM) function ndgrid. The result X is a (ncopy*nnodes)-by-ndims 
 *          matrix that contains all the coordinates in the n-d space.
 *          Note that X is column major.
 *
 * @param ndims     Number of dimensions of the space.
 * @param ncopy     Copies of coordinates wanted, usually this is set to 1.                   
 * @param nnodes    Total number of nodes in the n-d space. Note that 
 *                  nnodes = (xe[0]-xs[0]+1)*...*(xe[ndims-1]-xs[ndims-1]+1).
 * @param xs        An array of length ndims containing start positions 
 *                  in each dimension.
 * @param xe        An array of length ndims containing end positions 
 *                  in each dimension.
 * @param X         Output matrix containing ncopy of all the coordinates in 
 *                  the n-d space.
 */
void c_ndgrid_recursion(int ndims, int ncopy, int nnodes, int *xs, int *xe, int *X);



/**
 * @brief   Convert memory in Bytes to appropriate units.
 *
 * @param bytes       Memory in bytes.
 * @param n           Maximum string length for storing the formated bytes.
 * @param size_format Formated bytes in new units.
 */
void formatBytes(double bytes, int n, char *size_format);


/**
 * @brief   Find unique elements in an int array.
 *          Warning: this is a brute force implementation (O(n^2)).
 *
 * @param array  Original array, will be overwritten by the results.
 * @param n      Original length.
 */
int unique(int *array, int n);


/**
 * @brief   Calculate real spherical harmonics for given positions and given l and m. 
 *
 *          Only for l = 0, 1, ..., 6.
 */
void RealSphericalHarmonic(const int len, double *x, double *y,double *z, double *r, 
                           const int l, const int m, double *Ylm);

/**
 * @brief   Calculate Complex spherical harmonics for given positions and given l and m. 
 *
 *          Only for l = 0, 1, ..., 6.
 */
void ComplexSphericalHarmonic(const int len, double *x, double *y,double *z, double *r, 
                            const int l, const int m, double _Complex *Ylm);


void Calc_dist(SPARC_OBJ *pSPARC, int nxp, int nyp, int nzp, double x0_i_shift, double y0_i_shift, double z0_i_shift, double *R, double rchrg, int *count_interp);



/**
 * @brief   Print a 3D-vector distributed in comm to file.
 *
 *          This routine collects the vector distributed in comm
 *          to rank 0 (of comm) and writes to a file.
 *
 * @param x          Local part of the vector (input).
 * @param gridsizes  Array of length 3, total number of nodes in each direction. 
 * @param DMVertices Array of length 6, domain vertices of the local pieces of x.
 * @param fname      The name of the file to which the vector will be written.
 * @param comm       MPI communicator.
 */
void print_vec(
    void *x, int *gridsizes, int *DMVertices, 
    int unit_size, char *fname, MPI_Comm comm
);

/**
 * @brief   Read a 3D-vector from file and distributed in comm.
 *
 *          This routine reads a vector from a file by rank 0 and distributed 
 *          in comm by domain decomposition.
 *
 * @param x          Local part of the vector (output).
 * @param gridsizes  Array of length 3, total number of nodes in each direction. 
 * @param DMVertices Array of length 6, domain vertices of the local pieces of x.
 * @param option     the format of data, 0 - a vertical vector, 1 - cube
 * @param fname      The name of the file to which the vector will be read.
 * @param comm       MPI communicator.
 */
void read_vec(
    double *x, int *gridsizes, int *DMVertices, int option,
    char *fname, MPI_Comm comm
);


void read_cube(int Nx_, int Ny_, int Nz_, double *rho, char *fname);


/**
 * @brief   Function to check the below-tag format
 *          Note: used in readfiles.c for readion function
 */
void check_below_entries(FILE *ion_fp, char *tag);

/**
 * @brief   Check the input options in ion file
 */
int check_num_input(FILE *fp, void *array, char TYPE);

/*
 @ brief: function to calculate a 3x3 matrix times a vector
*/
void matrixTimesVec_3d(double *A, double *b, double *c);

/*
 @ brief: function to calculate a 3x3 matrix times a vector
*/
void matrixTimesVec_3d_complex(double *A, double _Complex *b, double _Complex *c);

#if defined(USE_MKL)
/**
 * @brief   MKL multi-dimension FFT interface, real to complex, following conjugate even distribution. 
 */
void MKL_MDFFT_real(double *r2c_3dinput, MKL_LONG *dim_sizes, MKL_LONG *strides_out, double _Complex *r2c_3doutput);

/**
 * @brief   MKL multi-dimension FFT interface, complex to complex
 */
void MKL_MDFFT(double _Complex *c2c_3dinput, MKL_LONG *dim_sizes, MKL_LONG *strides_out, double _Complex *c2c_3doutput);

/**
 * @brief   MKL multi-dimension iFFT interface, complex to real, following conjugate even distribution. 
 */
void MKL_MDiFFT_real(double _Complex *c2r_3dinput, MKL_LONG *dim_sizes, MKL_LONG *strides_in, double *c2r_3doutput);

/**
 * @brief   MKL multi-dimension iFFT interface, complex to complex. 
 */
void MKL_MDiFFT(double _Complex *c2c_3dinput, MKL_LONG *dim_sizes, MKL_LONG *strides_out, double _Complex *c2c_3doutput);

void MKL_MDFFT_batch_real(double *r2c_3dinput, int ncol, MKL_LONG *dim_sizes, int in_dist, double _Complex *r2c_3doutput, MKL_LONG *strides_out, int out_dist);

void MKL_MDiFFT_batch_real(double _Complex *c2r_3dinput, int ncol, MKL_LONG *dim_sizes, int in_dist, MKL_LONG *strides_in, double *c2r_3doutput, int out_dist);

void MKL_MDFFT_batch(double _Complex *c2c_3dinput, int ncol, MKL_LONG *dim_sizes, int in_dist, double _Complex *c2c_3doutput, MKL_LONG *strides_out, int out_dist);

void MKL_MDiFFT_batch(double _Complex *c2c_3dinput, int ncol, MKL_LONG *dim_sizes, int in_dist, double _Complex *c2c_3doutput, MKL_LONG *strides_out, int out_dist);
#endif

#if defined(USE_FFTW)
/**
 * @brief   FFTW multi-dimension FFT interface, complex to complex. 
 */
void FFTW_MDFFT(int *dim_sizes, double _Complex *c2c_3dinput, double _Complex *c2c_3doutput);

/**
 * @brief   FFTW multi-dimension iFFT interface, complex to complex. 
 */
void FFTW_MDiFFT(int *dim_sizes, double _Complex *c2c_3dinput, double _Complex *c2c_3doutput);

/**
 * @brief   FFTW multi-dimension FFT interface, real to complex. 
 */
void FFTW_MDFFT_real(int *dim_sizes, double *r2c_3dinput, double _Complex *r2c_3doutput);

/**
 * @brief   FFTW multi-dimension FFT interface, complex to real. 
 */
void FFTW_MDiFFT_real(int *dim_sizes, double _Complex *c2r_3dinput, double *c2r_3doutput);

void FFTW_MDFFT_batch(int *dim_sizes, int ncol, double _Complex *c2c_3dinput, int idist, double _Complex *c2c_3doutput, int odist);

void FFTW_MDiFFT_batch(int *dim_sizes, int ncol, double _Complex *c2c_3dinput, int idist, double _Complex *c2c_3doutput, int odist);

void FFTW_MDFFT_batch_real(int *dim_sizes, int ncol, double *r2c_3dinput, int *inembed, int idist, double _Complex *r2c_3doutput, int *onembed, int odist);

void FFTW_MDiFFT_batch_real(int *dim_sizes, int ncol, double _Complex *c2r_3dinput, int *inembed, int idist, double *c2r_3doutput, int *onembed, int odist);
#endif

/**
 * @brief   Function to compute exponential integral E_n(x)
 *          From Numerical Recipes
 */
double expint(const int n, const double x);

/**
 * @brief Restrict any function defined on a FD grid to a sub-grid by extracting
 *        the values that fall in the sub-grid.
 *
 *        Note that all the input indices for v_i are relative to the grid owned
 *        by the current process, while the indices for v_o are relative to the
 *        sub-grid.
 *
 * @param v_i              : Input data on the original grid
 * @param v_o (OUT)        : Output data on the sub-grid
 * @param stride_y_o       : Distance between v_o(i, j, k) and v_o(i, j+1, k)
 * @param stride_y_i       : Distance between v_i(i, j, k) and v_i(i, j+1, k)
 * @param stride_z_o       : Distance between v_o(i, j, k) and v_o(i, j, k+1)
 * @param stride_z_i       : Distance between v_i(i, j, k) and v_i(i, j, k+1)
 * @param [x_spos, x_epos] : X index range of v_o that will be computed
 * @param [y_spos, y_epos] : Y index range of v_o that will be computed
 * @param [z_spos, z_epos] : Z index range of v_o that will be computed
 * @param x_i_spos         : X start index in v_i that will be restricted
 * @param y_i_spos         : Y start index in v_i that will be restricted
 * @param z_i_spos         : Z start index in v_i that will be restricted
 *
 */
void restrict_to_subgrid(
    const void *v_i,      void *v_o,
    const int stride_y_o, const int stride_y_i,
    const int stride_z_o, const int stride_z_i,
    const int x_o_spos,   const int x_o_epos,
    const int y_o_spos,   const int y_o_epos,
    const int z_o_spos,   const int z_o_epos,
    const int x_i_spos,   const int y_i_spos,
    const int z_i_spos,   const int unit_size
);

/**
 * @brief change a = [b c] to a = [b; c] in-place as in Matlab 
 *        b and c have the same size nrow x ncol
 *
 * @param a (OUT)    : Input array
 * @param nrow       : number of rows of b or c
 * @param ncol       : number of columns of b or c
 *
 */
void Row2Col(void *a, const int nrow, const int ncol, const size_t unit_size);

/**
 * @brief change a = [b; c] to a = [b c] in-place as in Matlab 
 *        b and c have the same size nrow x ncol
 *
 * @param a (OUT)    : Input array
 * @param nrow       : number of rows of b or c
 * @param ncol       : number of columns of b or c
 *
 */
void Col2Row(void *a, const int nrow, const int ncol, const size_t unit_size);


/**
 * @brief   Printing matrix
 */
void print_matrix(double *A, int nrow, int ncol, char ACC);

#ifdef __cplusplus
extern "C" {
#endif
/** @ brief   Copy column-major matrix block
 *
 *  @param unit_size  Size of data element in bytes (double == 8, double _Complex == 16)
 *  @param src_       Pointer to the top-left element of the source matrix 
 *  @param lds        Leading dimension of the source matrix
 *  @param nrow       Number of rows to copy
 *  @param ncol       Number of columns to copy
 *  @param dst_       Pointer to the top-left element of the destination matrix
 *  @param ldd        Leading dimension of the destination matrix
 */
void copy_mat_blk(
    const size_t unit_size, const void *src_, const int lds, 
    const int nrow, const int ncol, void *dst_, const int ldd
);

#ifdef __cplusplus
}
#endif

/**
 * @brief    Reshape into block separation from Cartesian order of a vector in domain parallelization
 *          
 *          The block separation of a vector means that the vector is in the order
 *          [core0, core1, core2,...], corei is the part in i-th core of domain communicator
 *          Cartesian order means that the vector is in the order (x,y,z)
 */
void cart_to_block_dp(void *vec_cart, int ncol, int (*DMVertices)[6], int size_comm, 
                    int Nx, int Ny, int Nd, void *vec_bdp, int unit_size);

/**
 * @brief   Reshape into Cartesian order from the block separation of a vector in domain parallelization
 *          
 *          The block separation of a vector means that the vector is in the order
 *          [core0, core1, core2,...], corei is the part in i-th core of domain communicator
 *          Cartesian order means that the vector is in the order (x,y,z)
 */
void block_dp_to_cart(void *vec_bdp, int ncol, int (*DMVertices)[6], int *displs, int size_comm, 
                    int Nx, int Ny, int Nd, void *vec_cart, int unit_size);

/**
 * @brief   Transfer vectors from dmcomm to kptcomm_topo
 */
void Transfer_dmcomm_to_kptcomm_topo(SPARC_OBJ *pSPARC, int Nspinor, int ncols, void *vec_dmcomm, void *vec_kptcomm_topo, int unit_size);

#endif // TOOL_H
