/**
 * @file    tools.c
 * @brief   This file contains the tool functions.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */
 
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <float.h>
#include <assert.h>

#include "tools.h"
#include "parallelization.h"
#include "isddft.h"

#define min(x,y) ((x)<(y)?(x):(y))
#define max(x,y) ((x)>(y)?(x):(y))



/**
 * @brief   Compare function for qsort().
 */
int cmp(const void *a,const void *b)
{
    Sortstruct *a1 = (Sortstruct *)a;
    Sortstruct *a2 = (Sortstruct *)b;
    return a1->value < a2->value ? -1 : a1->value > a2->value;
}



/**
 * @brief   Convert string to lower case.
 */
char *strlwr(char *str)
{
  unsigned char *p = (unsigned char *)str;

  while (*p) {
     *p = tolower((unsigned char)*p);
      p++;
  }

  return str;
}



/**
 * @brief   Convert string to upper case.
 */
char *strupr(char *str)
{
  unsigned char *p = (unsigned char *)str;

  while (*p) {
     *p = toupper((unsigned char)*p);
      p++;
  }

  return str;
}



/**
 * @brief   Compare two strings ignoring case.
 *
 * @ref  https://code.woboq.org/userspace/glibc/string/strcmp.c.html
 *       https://searchcode.com/codesearch/view/10193301/
 */
int strcmpi(const char *p1, const char *p2)
{
    const unsigned char *s1 = (const unsigned char *) p1;
    const unsigned char *s2 = (const unsigned char *) p2;
    unsigned char c1, c2;

    // in case p1 and p2 are pointed to the same string
    if (s1 == s2) return 0;

    do {
        // convert both strs to lower case first
        c1 = tolower(*s1++);
        c2 = tolower(*s2++);
        if (c1 == '\0') return c1 - c2;
    } while (c1 == c2);

    return c1 - c2;
}


/**
 * @brief   Simplify a linux style path. 
 *
 *          Simplify a path (absolute or relative) to the simplest format. 
 *          E.g., 
 *            "././foo/../foo/a.txt" => "foo/a.txt",
 *            "/home/././foo/../foo/a.txt" => "/home/foo/a.txt".
 *
 * @param path    Input path.
 * @param newpath Simplified path. (OUTPUT)
 * @param newlen  Maximum size of newpath.
 *
 * @ref  LeetCode problem 71 (https://leetcode.com/problems/simplify-path/).
 */
void simplifyPath(char* path, char *newpath, size_t newlen)
{
    char* getNextLevel(char** level, char* path, size_t level_size);
    char* goLower(char* tail, char* level);
    char* goUpper(char* head, char* tail, char *prefix);

    if (strlen(path) == 0) {
        printf("path is empty!\n");
        return;
    }

    int len = strlen(path) + 1;
     // assign at least 8 byte, e.g., "./" needs 3 bytes (+'\0')
    len = len < 8 ? 8 : len;

    char* newcpy = calloc(len, sizeof(char));
    char* prefix = calloc(len, sizeof(char)); // stores final prefix such as "../"

    char* head = newcpy;
    char* tail = head;
    char* level;

    // remove leading spaces
    while (*path != '\0' && *path == ' ') path++;

    // take care of the first level
    *tail = *path; 
    if (*path != '/') {
        while (*path != '\0' && *path != '/') {
            *tail = *path;
            tail++; path++;
        }
    }
    *tail = '\0';

    // init prefix
    if (strcmp(head,"/") == 0 || strlen(head) == 0) { // abs path
        *prefix = '/';
    } else if (strcmp(head,"..") == 0) {
        snprintf(prefix, len, "%s", "../");
        head += 2;
    } else if (strcmp(head,".") == 0) {
        snprintf(prefix, len, "%s", "./");
        head++;
    } else {
        snprintf(prefix, len, "%s", "./");
    }

    while (*path != '\0') {
        path = getNextLevel(&level, path, len);
        if (strcmp(level, "/.") == 0 || strcmp(level, "/") == 0) {
            free(level);
            continue;
        } else if (strcmp(level, "/..") == 0)
            tail = goUpper(head, tail, prefix);
        else
            tail = goLower(tail, level);
        free(level);
    }

    // Ignore leading multiple "/"
    if (*head == '/')
        while (*(head+1) == '/') head++;
    if (strlen(prefix) > 0 && prefix[strlen(prefix)-1] == '/' && *head == '/') head++;
    char *prefix_head = prefix;
    
    // remove the preceeding './' 
    if (strlen(prefix) >= 2 && prefix[0] == '.' && prefix[1] == '/') {
        prefix_head += 2;
    }

    if (strlen(prefix_head) == 0 && strlen(head) == 0) {
        head[0] = '.';
        head[1] = '\0';
    }

    snprintf(newpath, newlen, "%s%s", prefix_head, head);
    free(newcpy);
    free(prefix);
}

char* getNextLevel(char** level, char* path, size_t level_size)
{
    *level = malloc(sizeof(char) * level_size);
    int l = 0;

    // Always get first character
    (*level)[l++] = *path++;

    // Ignore multiple "/"
    for (; *path == '/'; path++);

    // copy current level
    for (; *path != '/' && *path != '\0'; path++)
        (*level)[l++] = *path;

    (*level)[l] = '\0';

    return path;
}

char* goLower(char* tail, char* level)
{
    do {
        *tail = *level;
        level++; tail++;
    } while (*level != '\0');
    *tail = '\0';
    return tail;
}


char* goUpper(char* head, char* tail, char* prefix)
{
    // the second condition is to prevent going up the root /
    if (tail == head && strcmp(prefix, "/") != 0) {
        strcat(prefix,"../");
    }

    for (; *tail != '/' && tail != head; tail--);

    *tail = '\0';
    return tail;
}



/**
 * @brief   The following code for factorizing an integer is copied from Rosetta Code
 *          (https://rosettacode.org/wiki/Factors_of_an_integer#C).
 */
// The defined type is moved to the header file: tools.h
// typedef struct {
// 	int *list;
// 	short count; 
// } Factors;
void xferFactors( Factors *fctrs, int *flist, int flix ) 
{
    int ix, ij;
    int newSize = fctrs->count + flix;
    if (newSize > flix)  {
        fctrs->list = realloc( fctrs->list, newSize * sizeof(int));
    }
    else {
        fctrs->list = malloc(  newSize * sizeof(int));
    }
    for (ij=0,ix=fctrs->count; ix<newSize; ij++,ix++) {
        fctrs->list[ix] = flist[ij];
    }
    fctrs->count = newSize;
}

Factors *factor( int num, Factors *fctrs)
{
    int flist[301], flix;
    int dvsr;
    flix = 0;
    fctrs->count = 0;
    free(fctrs->list);
    fctrs->list = NULL;
    for (dvsr=1; dvsr*dvsr < num; dvsr++) {
        if (num % dvsr != 0) continue;
        if ( flix == 300) {
            xferFactors( fctrs, flist, flix );
            flix = 0;
        }
        flist[flix++] = dvsr;
        flist[flix++] = num/dvsr;
    }
    if (dvsr*dvsr == num) 
        flist[flix++] = dvsr;
    if (flix > 0)
        xferFactors( fctrs, flist, flix );
 
    return fctrs;
}



void sorted_factor( int num, Factors *fctrs) {
    // call factors to do the calculation
    factor(num, fctrs);

    short len = fctrs->count;
    int *new_list = (int *)malloc(len * sizeof(int));

    // the returned list comes in pairs, f1xf2, f3xf4, ...
    // sort the list in ascending order
    // copy the 1st half
    for (int i = 0; i < (len+1)/2; i++) {
        new_list[i] = fctrs->list[2*i];
    }

    short ind = len-1;
    // copy the 2nd half
    for (int i = 1; i < len; i+=2) {
        new_list[ind--] = fctrs->list[i];
    }   

    free(fctrs->list);
    fctrs->list = new_list;

}


// Equivalent to ceil(x / (double)y), where x, y are positive integers
int ceil_div(const unsigned int x, const unsigned int y)
{
    return (x + y - 1) / y;
}


/**
 * @brief   Calculates derivatives of a tabulated function required for spline interpolation.
 */
void getYD_gen(double *X, double *Y, double *YD, int len) {
    int i;
    double h0,h1,r0,r1,*A,*B,*C;
          
    A = (double *)malloc(sizeof(double)*len);
    B = (double *)malloc(sizeof(double)*len);
    C = (double *)malloc(sizeof(double)*len);
    if (A == NULL || B == NULL || C == NULL) {
        printf("Memory allocation failed!\n");
        exit(EXIT_FAILURE);
    }

    h0 =  X[1]-X[0]; h1 = X[2]-X[1];
    r0 = (Y[1]-Y[0])/h0; r1=(Y[2]-Y[1])/h1;
    B[0] = h1*(h0+h1);
    C[0] = (h0+h1)*(h0+h1);
    YD[0] = r0*(3*h0*h1 + 2*h1*h1) + r1*h0*h0;
               
    for(i=1;i<len-1;i++) {
        h0 = X[i]-X[i-1]; h1=X[i+1]-X[i];
        r0 = (Y[i]-Y[i-1])/h0;  r1=(Y[i+1]-Y[i])/h1;
        A[i] = h1;
        B[i] = 2*(h0+h1);
        C[i] = h0;
        YD[i] = 3*(r0*h1 + r1*h0);
    }
           
    A[i] = (h0+h1)*(h0+h1);
    B[i] = h0*(h0+h1);
    YD[i] = r0*h1*h1 + r1*(3*h0*h1 + 2*h0*h0);
     
    tridiag_gen(A,B,C,YD,len);
    
    free(A); free(B); free(C);                                     
}



/**
 * @brief   Solves a tridiagonal system using Gauss Elimination.
 */
void tridiag_gen(double *A, double *B, double *C, double *D, int len) {
    int i;
    double b, *F;
    F = (double *)malloc(sizeof(double)*len);
    if (F == NULL) {
        printf("Memory allocation failed!\n");
        exit(EXIT_FAILURE);
    }
    
    // Gauss elimination; forward substitution
    b = B[0];
    if (fabs(b) < 1e-14) {
        printf("Divide by zero in tridiag_gen\n"); 
        exit(EXIT_FAILURE);
    }
    D[0] = D[0]/b;
    for (i = 1; i<len; i++) {
        F[i] = C[i-1] / b;
        b= B[i] - A[i] * F[i];
        if (fabs(b) < 1e-14) {
            printf("Divide by zero in tridiag_gen\n"); 
            exit(EXIT_FAILURE);
        }
        D[i] = (D[i] - D[i-1] * A[i])/b;
    }
    // backsubstitution 
    for (i = len-2; i >= 0; i--)
        D[i] -= (D[i+1]*F[i+1]);
        
    free(F);
}



/**
 * @brief   Cubic spline evaluation from precalculated data.
 */
void SplineInterp(double *X1,double *Y1,int len1,double *X2,double *Y2,int len2,double *YD) {
    if (len2 <= 0) return;
    int i,j;
    double A0,A1,A2,A3,x,dx,dy,p1,p2,p3;    
    if(X2[0]<X1[0] || X2[len2-1]>X1[len1-1]) {
        printf("First input X in table=%lf, last input X in table=%lf, "
               "interpolate at x[first]=%lf, x[last]=%lf\n",X1[0],X1[len1-1],X2[0],X2[len2-1]);
        printf("Out of range in spline interpolation!\n");
        exit(EXIT_FAILURE);
    }
    // p1 is left endpoint of the interval
    // p2 is resampling position
    // p3 is right endpoint of interval
    // j is input index of current interval  
    A0 = A1 = A2 = A3 = 0.0;
    p1 = p3 = X2[0]-1;  // force coefficient initialization  
    for (i = j = 0; i < len2; i++) {
        // check if in new interval
        p2 = X2[i];
        if (p2 > p3) {
            //find interval which contains p2 
            for (; j<len1 && p2>X1[j]; j++);
            if (p2 < X1[j]) j--;
            p1 = X1[j]; 
            p3 = X1[j+1]; 
            // coefficients
            dx = 1.0 / (X1[j+1] - X1[j]);
            dy = (Y1[j+1] - Y1[j]) * dx;
            A0 = Y1[j];
            A1 = YD[j];
            A2 = dx * (3.0 * dy - 2.0 * YD[j] - YD[j+1]);
            A3 = dx * dx * (-2.0*dy + YD[j] + YD[j+1]);  
        }
        // use Horner's rule to calculate cubic polynomial
        x = p2-p1;
        Y2[i] = ((A3*x + A2) * x + A1) * x + A0;     
    } 
}


/**
 * @brief   Sort an array in ascending order and return original index.
 */
void Sort(double *X, int len, double *Xsorted, int *Ind) {
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
 * @brief   Sort input values and then apply spline interpolation.
 */
void SortSplineInterp(double *X1, double *Y1, int len1, double *X2, double *Y2, int len2, double *YD) {
    int i, *Ind_sort, len_interp;
    double *Y2_sort;
    
    // first sort X2 in ascending order
    // NOTE: we're using the memory of Y2 to store sorted X2 temporarily to save memory!
    Ind_sort = (int *)malloc( len2 * sizeof(int) );
    Sort(X2, len2, Y2, Ind_sort);
    
    Y2_sort = (double *)malloc( len2 * sizeof(double) );
    if (Y2_sort == NULL) {
        printf("\nMemory allocation failed!\n");
        exit(EXIT_FAILURE);
    }
    
    len_interp = len2;
    while (len_interp >= 1 && Y2[len_interp-1] > X1[len1-1]) len_interp--;
    
    // apply cubic spline interpolation to find Y2_sort, note here Y2 stores sorted values of X2
    SplineInterp(X1, Y1, len1, Y2, Y2_sort, len_interp, YD);
    
    // rearrange Y2_sort to original order
    for (i = 0; i < len_interp; i++) {
        Y2[Ind_sort[i]] = Y2_sort[i];
    }
    
    free(Y2_sort); free(Ind_sort);
}



/**
 * @brief   Cubic spline evaluation from precalculated data. This assumes
 *          X1 is a uniformly increasing grid.
 */
void SplineInterpUniform(
	double *X1,double *Y1,int len1,double *X2,double *Y2,int len2,double *YD
)
{
	if (len2 <= 0 || len1 < 2) return;
	int i,j;
	double A0,A1,A2,A3,x,dx,dy,p1,p2,p3;
	// p1 is left endpoint of the interval
	// p2 is resampling position
	// p3 is right endpoint of interval
	// j is index of current interval  
	double X1_max = X1[len1-1];
	double delta_x1;
	delta_x1 = (X1[1] - X1[0]); // assuming len1 >= 2
	p3 = X2[0]-1;  // force coefficient initialization  
	for (i = 0; i < len2; i++) {
		// check if in new interval
		p2 = X2[i];
		// assume X1 is a uniform grid with len1>=2
		if (p2 > X1_max) {
			j = len1 - 2;
			continue; // comment this to enable interpolation out of range
		} else {
			j = floor((p2 - X1[0]) / delta_x1);
		}
		// interval j = (X1[j], X1[j+1])
		p1 = X1[j]; 
		p3 = X1[j+1]; 
		// coefficients
		dx = 1.0 / (p3 - p1);
		dy = (Y1[j+1] - Y1[j]) * dx;
		A0 = Y1[j];
		A1 = YD[j];
		A2 = dx * (3.0 * dy - 2.0 * YD[j] - YD[j+1]);
		A3 = dx * dx * (-2.0*dy + YD[j] + YD[j+1]);  
		// use Horner's rule to calculate cubic polynomial
		x = p2-p1;
		Y2[i] = ((A3*x + A2) * x + A1) * x + A0;     
	}
}



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
)
{

SortSplineInterp(X1, Y1, len1, X2, Y2, len2, YD);
return;

    if (len2 <= 0) return;
    int i,j;
    double A0,A1,A2,A3,x,dx,dy,p1,p2,p3;
    // p1 is left endpoint of the interval
    // p2 is resampling position
    // p3 is right endpoint of interval
    // j is input index of current interval  
    double X1_max = X1[len1-1];
    p3 = X2[0]-1;  // force coefficient initialization  
    for (i = 0; i < len2; i++) {
        // check if in new interval
        p2 = X2[i];
        if (p2 > X1_max) {
        	j = len1 - 2;
        	continue; // comment this to enable interpolation out of range
        } else {
        	j = binary_interval_search(X1, len1, p2); 
        }
        p1 = X1[j]; 
        p3 = X1[j+1]; 
        // coefficients
        dx = 1.0 / (p3 - p1);
        dy = (Y1[j+1] - Y1[j]) * dx;
        A0 = Y1[j];
        A1 = YD[j];
        A2 = dx * (3.0 * dy - 2.0 * YD[j] - YD[j+1]);
        A3 = dx * dx * (-2.0*dy + YD[j] + YD[j+1]);  
        // use Horner's rule to calculate cubic polynomial
        x = p2-p1;
        Y2[i] = ((A3*x + A2) * x + A1) * x + A0;     
    } 
}



/**
 * @brief   Binary search to find which interval a given number 
 *          is located. list[...] is assumed to be monotonically
 *          increasing.
 */
int binary_interval_search(const double *list, const int len, const double x)
{
	int first = 0;
	int last = len - 1;
	int middle = (first+last)/2;
	while (first+1 < last) {
		if (list[middle]<= x && x < list[middle+1]) {
			return middle;
		}
		if (list[middle] < x) {
			first = middle;
		} else {
			last = middle;
		}
		middle = (first + last)/2;
	}
	return first;
}


/**
 * @brief   The main funciton for Cubic spline evaluation from precalculated
 *          data. This function calls the appropriate uniform or non-uniform
 *          routines.         
 */
void SplineInterpMain(double *X1,double *Y1,int len1,
    double *X2,double *Y2,int len2,double *YD,int isUniform) 
{
    if (isUniform) SplineInterpUniform(X1,Y1,len1,X2,Y2,len2,YD);
    else SplineInterpNonuniform(X1,Y1,len1,X2,Y2,len2,YD);
}


/**
 * @brief   Cubic spline coefficients.
 */
// void SplineCoeff(double *X,double *Y,double *YD,int len,double *A3,double *A2,double *A1,double *A0) {
// 	int j;
// 	double x,dx,dy;
// 	for (j = 0; j < len-1; j++) {
// 		dx = 1.0 / (X[j+1] - X[j]);
// 		dy = (Y[j+1] - Y[j]) * dx;
// 		A0[j] = Y[j]; 
// 		A1[j] = YD[j];
// 		A2[j] = dx * (3.0 * dy - 2.0 * YD[j] - YD[j+1]);
// 		A3[j] = dx * dx * (-2.0*dy + YD[j] + YD[j+1]);
// 	}
// }


/**
 * @brief   Cubic spline coefficients, but without copying Y and YD into A0 and A1, respectively.
 */
void SplineCoeff(double *X,double *Y,double *YD,int len,double *A3,double *A2) {
    int j;
    double dx,dy;
    for (j = 0; j < len-1; j++) {
        dx = 1.0 / (X[j+1] - X[j]);
        dy = (Y[j+1] - Y[j]) * dx;
        // note: there's no need to copy Y or YD into another array
        // A0[j] = Y[j]; 
        // A1[j] = YD[j];
        A2[j] = dx * (3.0 * dy - 2.0 * YD[j] - YD[j+1]);
        A3[j] = dx * dx * (-2.0*dy + YD[j] + YD[j+1]);
    }
}


/**
 * @brief   Calculates the terms involving factorials in finite difference 
 *          weights calculation.
 */
double fract(int n,int k) {
    int i;
    double Nr=1, Dr=1, val;
    for(i=n-k+1; i<=n; i++)
        Nr*=i;
    for(i=n+1; i<=n+k; i++)
        Dr*=i;
    val = Nr/Dr;
    return (val);
}



/**
 * @brief   Calculate global 2-norm of a vector among the given communicator. 
 *          
 */
void Vector2Norm(const double *Vec, const int len, double *vec_2norm, MPI_Comm comm)
{
    int k;
    double sqsum = 0.0;
    for(k = 0; k < len; k++)
        sqsum += Vec[k]*Vec[k];
    MPI_Allreduce(&sqsum, vec_2norm, 1, MPI_DOUBLE, MPI_SUM, comm);
    *vec_2norm = sqrt(*vec_2norm);
}



/**
 * @brief   Calculate global 2-norm of a vector among the given communicator.      
 */
void Vector2Norm_complex(const double complex *Vec, const int len, double *vec_2norm, MPI_Comm comm)
{
    int k;
    double sqsum = 0.0;
    for(k = 0; k < len; k++)
        sqsum += conj(Vec[k]) * Vec[k];
    MPI_Allreduce(&sqsum, vec_2norm, 1, MPI_DOUBLE, MPI_SUM, comm);
    *vec_2norm = sqrt(*vec_2norm);
}



/**
 * @brief   Calculate global 2-norm of a vector among the given communicator. 
 */
void VectorDotProduct(const double *Vec1, const double *Vec2, const int len, double *vec_dot, MPI_Comm comm) 
{
    int i;
    double vec_dot_loc = 0.0;
    for (i = 0; i < len; i++)
        vec_dot_loc += Vec1[i] * Vec2[i];
    MPI_Allreduce(&vec_dot_loc, vec_dot, 1, MPI_DOUBLE, MPI_SUM, comm);
}


/**
 * @brief   Calculate global 2-norm of a vector with integration weight among the given communicator. 
 */
void VectorDotProduct_wt(const double *Vec1, const double *Vec2, const double *Intgwt, const int len, double *vec_dot, MPI_Comm comm) 
{
    int i;
    double vec_dot_loc = 0.0;
    for (i = 0; i < len; i++)
        vec_dot_loc += Vec1[i] * Vec2[i] * Intgwt[i];
    MPI_Allreduce(&vec_dot_loc, vec_dot, 1, MPI_DOUBLE, MPI_SUM, comm);
}



/**
 * @brief   Calculate global 2-norm of a vector among the given communicator. 
 */
void VectorDotProduct_complex(const double complex *Vec1, const double complex *Vec2, const int len, double *vec_dot, MPI_Comm comm) 
{
    int i;
    double vec_dot_loc = 0.0;
    for (i = 0; i < len; i++)
        vec_dot_loc += conj(Vec1[i]) * Vec2[i];
    MPI_Allreduce(&vec_dot_loc, vec_dot, 1, MPI_DOUBLE, MPI_SUM, comm);
}



/**
 * @brief   Calculate global sum of a vector among the given communicator. 
 */
void VectorSum(const double *Vec, const int len, double *vec_sum, MPI_Comm comm)
{
    if (comm == MPI_COMM_NULL) return;
    int k;
    double sum = 0.0;
    for (k = 0; k < len; k++)
        sum += Vec[k];
    MPI_Allreduce(&sum, vec_sum, 1, MPI_DOUBLE, MPI_SUM, comm);
}



/**
 * @brief   Calculate shift of a vector, x = x + c.
 */
void VectorShift(double *Vec, const int len, const double c, MPI_Comm comm)
{
    if (comm == MPI_COMM_NULL) return;
    int k;
    for (k = 0; k < len; k++)
        Vec[k] += c;
}



/**
 * @brief   Scale a vector, x = x * c.
 */
void VectorScale(double *Vec, const int len, const double c, MPI_Comm comm)
{
    if (comm == MPI_COMM_NULL) return;
    for (int k = 0; k < len; k++)
        Vec[k] *= c;
}



/**
 * @brief   Create a random matrix with each entry a random number within given range. 
 * 
 *          Note that each process within comm will have different random entries.
 *
 * @param Mat   Local part of the matrix.
 * @param m     Number of rows of the local copy of the matrix.
 * @param n     Number of columns of the local part of the matrix.
 */
void SetRandMat(double *Mat, int m, int n, double rand_min, double rand_max, MPI_Comm comm)
{
    int rank, i, len_tot;
    MPI_Comm_rank(comm, &rank);

    int seed_shift = 1;
    int seed_temp = rank * 100 + seed_shift;
    srand(seed_temp);

    len_tot = m * n;
    for (i = 0; i < len_tot; i++) {
        Mat[i] = rand_min + (rand_max - rand_min) * ((double) rand() / RAND_MAX);
    }
}


/**
 * @brief   Create a random matrix with each entry a random number within given range. 
 * 
 *          Note that each process within comm will have different random entries.
 *
 * @param Mat   Local part of the matrix.
 * @param m     Number of rows of the local copy of the matrix.
 * @param n     Number of columns of the local part of the matrix.
 */
void SetRandMat_complex(double complex *Mat, int m, int n, double rand_min, double rand_max, MPI_Comm comm)
{
    int rank, i, len_tot;
    MPI_Comm_rank(comm, &rank);

    int seed_shift = 1;
    int seed_temp = rank * 100 + seed_shift;
    srand(seed_temp);

    len_tot = m * n;
    for (i = 0; i < len_tot; i++) {
        Mat[i] = rand_min + (rand_max - rand_min) * ((double) rand() / RAND_MAX);
    }
}



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
)
{
	int shift_gz = gridsizes[0] * gridsizes[1];
	int shift_gy = gridsizes[0];
	int DMnx = DMVert[1] - DMVert[0] + 1;
	int DMny = DMVert[3] - DMVert[2] + 1;
	int DMnz = DMVert[5] - DMVert[4] + 1;
	
    int shift_z = DMnx * DMny;
    int shift_y = DMnx;

    int i, j, k, ig, jg, kg;
    for (k = 0, kg = DMVert[4]; k < DMnz; k++, kg++) {
    	for (j = 0, jg = DMVert[2]; j < DMny; j++, jg++) { 
    		for (i = 0, ig = DMVert[0]; i < DMnx; i++, ig++) {
    			int index_g = kg * shift_gz + jg * shift_gy + ig; // global index
    			int index   = k  * shift_z  + j  * shift_y  + i ; // local index
    			srand(index_g + 1 + seed_offset);
    			Vec[index] = rand_min + (rand_max - rand_min) * ((double) rand() / RAND_MAX);
    		}
    	}
    }
}



/**
 * @brief   Create a random vector with fixed seeds corresponding to it's global index.
 * 
 *			This function is to fix the random number seeds so that all random numbers 
 *			generated in parallel under MPI are the same as those generated in sequential 
 *          execution.
 */
void SeededRandVec_complex (
	double complex *Vec, const int DMVert[6], const int gridsizes[3],
	const double rand_min, const double rand_max,
	const int seed_offset
)
{
	int shift_gz = gridsizes[0] * gridsizes[1];
	int shift_gy = gridsizes[0];
	int DMnx = DMVert[1] - DMVert[0] + 1;
	int DMny = DMVert[3] - DMVert[2] + 1;
	int DMnz = DMVert[5] - DMVert[4] + 1;
	
    int shift_z = DMnx * DMny;
    int shift_y = DMnx;

    int i, j, k, ig, jg, kg;
    for (k = 0, kg = DMVert[4]; k < DMnz; k++, kg++) {
    	for (j = 0, jg = DMVert[2]; j < DMny; j++, jg++) { 
    		for (i = 0, ig = DMVert[0]; i < DMnx; i++, ig++) {
    			int index_g = kg * shift_gz + jg * shift_gy + ig; // global index
    			int index   = k  * shift_z  + j  * shift_y  + i ; // local index
    			srand(index_g + 1 + seed_offset);
    			Vec[index] = rand_min + (rand_max - rand_min) * ((double) rand() / RAND_MAX);
    		}
    	}
    }
}


/**
 * @brief   Create a perturbed unit constant vector distributed among the given communicator. 
 */
void SetUnitOnesVector(double *Vec, int n_loc, int n_global, double pert_fac, MPI_Comm comm)
{
    int i;
    double vscal;
    if (pert_fac == 0.0) {
        vscal = 1.0 / sqrt(n_global);
        for (i = 0; i < n_loc; i++)
            Vec[i] = vscal;
    } else {
        int rank;
        MPI_Comm_rank(comm, &rank);

        srand(rank+1);
        //srand(rank+1+(int)MPI_Wtime());

        for (i = 0; i < n_loc; i++)
            Vec[i] = 1.0 - pert_fac + 2.0 * pert_fac * (double)rand() / RAND_MAX;
        
        Vector2Norm(Vec, n_loc, &vscal, comm); // find 2-norm
        vscal = 1.0 / vscal;

        for (i = 0; i < n_loc; i++)
            Vec[i] *= vscal;
    }
}



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
    const int DMVerts[6], const int gridsizes[3])
{
    int i_global = ip + origin_shift_i + DMVerts[0];
    int j_global = jp + origin_shift_j + DMVerts[2];
    int k_global = kp + origin_shift_k + DMVerts[4];
    int is_out = (i_global < 0) || (i_global >= gridsizes[0]) ||
                 (j_global < 0) || (j_global >= gridsizes[1]) ||
                 (k_global < 0) || (k_global >= gridsizes[2]);
    return is_out;
}



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
    const int DMVerts[6], const int gridsizes[3])
{
    int i_global = ip + origin_shift_i + DMVerts[0];
    int j_global = jp + origin_shift_j + DMVerts[2];
    int k_global = kp + origin_shift_k + DMVerts[4];
    int i_ind = (0 <= i_global) - (i_global < gridsizes[0]); // -1 - left, 0 - center, 1 - right
    int j_ind = (0 <= j_global) - (j_global < gridsizes[1]); // -1 - left, 0 - center, 1 - right
    int k_ind = (0 <= k_global) - (k_global < gridsizes[2]); // -1 - left, 0 - center, 1 - right
    int region_ind = (k_ind+1) * 9 + (j_ind+1) * 3 + (i_ind+1);
    if (region_ind == 13) return -1; // grid is not outside the domain
    else if (region_ind < 13) return region_ind; // region before the center (original domain)
    else return region_ind - 1; // region after the center, we skip the center index 13
}



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
void c_ndgrid(int ndims, int *xs, int *xe, int *X) 
{
    if (ndims < 1) return;
    int nnodes = 1;
    unsigned i;
    // find total number of nodes in the n-d space
    for (i = 0; i < ndims; i++) {
        nnodes *= xe[i] - xs[i] + 1;
    }
    // start the recursion 
    c_ndgrid_recursion(ndims, 1, nnodes, xs, xe, X);
}



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
void c_ndgrid_recursion(int ndims, int ncopy, int nnodes, int *xs, int *xe, int *X) 
{
    if (ndims < 1) return;
    int i, j, n, nnd, nnodes_ndm1, start_pos, count;
    nnd = xe[ndims-1] - xs[ndims-1] + 1; // # of nodes in ndims-th dimension
    nnodes_ndm1 = nnodes / nnd; // # of nodes in each (ndims-1)-d space
    start_pos = nnodes * ncopy * (ndims - 1);
    count = 0;
    for (n = 0; n < ncopy; n++) {
        for (i = xs[ndims-1]; i <= xe[ndims-1]; i++) {
            for (j = 0; j < nnodes_ndm1; j++, count++) {
                X[start_pos + count] = i;
            }
        }
    }
    ncopy = ncopy * nnodes / nnodes_ndm1;
    c_ndgrid_recursion(ndims-1, ncopy, nnodes_ndm1, xs, xe, X);
}



/**
 * @brief   Convert memory in Bytes to appropriate units.
 *
 * @param bytes       Memory in bytes.
 * @param n           Maximum string length for storing the formated bytes.
 * @param size_format Formated bytes in new units.
 */
void formatBytes(double bytes, int n, char *size_format) {
#define NUM_UNITS 9
    char *Suffix[9] = { "B", "kB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB" };
    double formatedbytes = bytes;
    int i;
    for (i = 0; i < NUM_UNITS && formatedbytes >= 1024.0; i++) {
        formatedbytes /= 1024.0;
    }
    
    i = min(i, NUM_UNITS-1);
    snprintf(size_format, n, "%.2f %s", formatedbytes, Suffix[i]);

    if (i == NUM_UNITS && formatedbytes > 1024.0) {
        printf("WARNING: memory size is too large: %.2e Bytes >= 1024 %s\n", bytes, Suffix[NUM_UNITS-1]);
        exit(EXIT_FAILURE);
    }
#undef NUM_UNITS
}



/**
 * @brief   Calculate real spherical harmonics for given positions and given l and m. 
 *
 *          Only for l = 0, 1, ..., 6.
 */
void RealSphericalHarmonic(const int len, double *x, double *y,double *z, double *r, const int l, const int m, double *Ylm)
{
    // only l=0,1,2,3,4,5,6 implemented for now

    //double pi=M_PI;
    double p;                   	  
    int i; 
    
    /* l = 0 */
    double C00 = 0.282094791773878; // 0.5*sqrt(1/pi)
    /* l = 1 */
    double C1m1 = 0.488602511902920; // sqrt(3/(4*pi))
    double C10 = 0.488602511902920; // sqrt(3/(4*pi))
    double C1p1 = 0.488602511902920; // sqrt(3/(4*pi))
    /* l = 2 */
    double C2m2 = 1.092548430592079; // 0.5*sqrt(15/pi)
    double C2m1 = 1.092548430592079; // 0.5*sqrt(15/pi)  
    double C20 =  0.315391565252520; // 0.25*sqrt(5/pi)
    double C2p1 = 1.092548430592079; // 0.5*sqrt(15/pi)  
    double C2p2 = 0.546274215296040; // 0.25*sqrt(15/pi)
    /* l = 3 */
    double C3m3 = 0.590043589926644; // 0.25*sqrt(35/(2*pi))   
    double C3m2 = 2.890611442640554; // 0.5*sqrt(105/(pi))
    double C3m1 = 0.457045799464466; // 0.25*sqrt(21/(2*pi))
    double C30 =  0.373176332590115; // 0.25*sqrt(7/pi)
    double C3p1 = 0.457045799464466; // 0.25*sqrt(21/(2*pi))
    double C3p2 = 1.445305721320277; //  0.25*sqrt(105/(pi))
    double C3p3 = 0.590043589926644; //  0.25*sqrt(35/(2*pi))
    /* l = 4 */
    double C4m4 = 2.503342941796705; // (3.0/4.0)*sqrt(35.0/pi)
    double C4m3 = 1.770130769779930; // (3.0/4.0)*sqrt(35.0/(2.0*pi))   
    double C4m2 = 0.946174695757560; // (3.0/4.0)*sqrt(5.0/pi)
    double C4m1 = 0.669046543557289; // (3.0/4.0)*sqrt(5.0/(2.0*pi))
    double C40 =  0.105785546915204; // (3.0/16.0)*sqrt(1.0/pi)
    double C4p1 = 0.669046543557289; // (3.0/4.0)*sqrt(5.0/(2.0*pi))
    double C4p2 = 0.473087347878780; // (3.0/8.0)*sqrt(5.0/(pi))
    double C4p3 = 1.770130769779930; // (3.0/4.0)*sqrt(35.0/(2.0*pi))
    double C4p4 = 0.625835735449176; // (3.0/16.0)*sqrt(35.0/pi) 
    /* l = 5 */
    double C5m5 = 0.656382056840170; // (3.0*sqrt(2.0*77.0/pi)/32.0)
    double C5m4 = 2.075662314881042; // (3.0/16.0)*sqrt(385.0/pi)
    double C5m3 = 0.489238299435250; // (sqrt(2.0*385.0/pi)/32.0)
    double C5m2 = 4.793536784973324; // (1.0/8.0)*sqrt(1155.0/pi)*2.0
    double C5m1 = 0.452946651195697; // (1.0/16.0)*sqrt(165.0/pi)
    double C50 =  0.116950322453424; // (1.0/16.0)*sqrt(11.0/pi)
    double C5p1 = 0.452946651195697; // (1.0/16.0)*sqrt(165.0/pi)
    double C5p2 = 2.396768392486662; // (1.0/8.0)*sqrt(1155.0/pi)
    double C5p3 = 0.489238299435250; // (sqrt(2.0*385.0/pi)/32.0)
    double C5p4 = 2.075662314881042; // (3.0/16.0)*sqrt(385.0/pi)
    double C5p5 = 0.656382056840170; // (3.0*sqrt(2.0)/32.0)*sqrt(77.0/pi)
    /* l = 6 */
    double C6m6 = 0.683184105191914; // (sqrt(2.0*3003.0/pi)/64.0)
    double C6m5 = 2.366619162231752; // (3.0/32.0)*sqrt(2.0*1001.0/pi)
    double C6m4 = 0.504564900728724; // (3.0/32.0)*sqrt(91.0/pi)
    double C6m3 = 0.921205259514923; // (sqrt(2.0*1365.0/pi)/32.0)
    double C6m2 = 0.460602629757462; // (sqrt(2.0*1365/pi)/64.0)
    double C6m1 = 0.582621362518731; // (sqrt(273.0/pi)/16.0)
    double C60 =  0.0635692022676284;// (sqrt(13.0/pi)/32.0)
    double C6p1 = 0.582621362518731; // (sqrt(273.0/pi)/16.0)
    double C6p2 = 0.460602629757462; // (sqrt(2.0*1365.0/pi)/64.0)
    double C6p3 = 0.921205259514923; // (sqrt(2.0*1365.0/pi)/32.0)
    double C6p4 = 0.504564900728724; // (3.0/32.0)*sqrt(91.0/pi)
    double C6p5 = 2.366619162231752; // (3.0/32.0)*sqrt(2.0*1001.0/pi)
    double C6p6 = 0.683184105191914; // (sqrt(2.0*3003.0/pi)/64.0)
    
    switch (l)
    {
        /* l = 0 */
        case 0:
            for (i = 0; i < len; i++) Ylm[i] = C00;
            break;
        
        /* l = 1 */
        case 1: 
            switch (m) 
            {
                case -1: /* m = -1 */
                    for (i = 0; i < len; i++) 
                        Ylm[i] = C1m1 * (y[i] / r[i]);
                    break;
                
                case 0: /* m = 0 */
                    for (i = 0; i < len; i++)
                        Ylm[i] = C10 * (z[i] / r[i]);
                    break;
                
                case 1: /* m = 1 */
                    for (i = 0; i < len; i++)
                        Ylm[i] = C1p1 * (x[i] / r[i]);
                    break;
                
                /* incorrect m */
                default: printf("<m> must be an integer between %d and %d!\n", -l, l); break;
            }
            break;
        
        /* l = 2 */
        case 2: 
            switch (m) 
            {      
                case -2: /* m = -2 */
                    for (i = 0; i < len; i++)
                        Ylm[i] = C2m2 * (x[i]*y[i])/(r[i]*r[i]);
                    break;

                case -1: /* m = -1 */
                    for (i = 0; i < len; i++)
                        Ylm[i] = C2m1*(y[i]*z[i])/(r[i]*r[i]);
                    break;

                case 0: /* m = 0 */
                    for (i = 0; i < len; i++)
                        Ylm[i] = C20*(-x[i]*x[i] - y[i]*y[i] + 2.0*z[i]*z[i])/(r[i]*r[i]);
                    break;
                
                case 1: /* m = 1 */
                    for (i = 0; i < len; i++)
                        Ylm[i] = C2p1*(z[i]*x[i])/(r[i]*r[i]);
                    break;
                
                case 2: /* m = 2 */
                    for (i = 0; i < len; i++)
                        Ylm[i] = C2p2*(x[i]*x[i] - y[i]*y[i])/(r[i]*r[i]);
                    break;
                
                /* incorrect m */
                default: printf("<m> must be an integer between %d and %d!\n", -l, l); 
                         break;
            }
            break;

        /* l = 3 */
        case 3: 
            switch (m) 
            {   
                case -3: /* m = -3 */
                    for (i = 0; i < len; i++)
                        Ylm[i] = C3m3*(3*x[i]*x[i] - y[i]*y[i])*y[i]/(r[i]*r[i]*r[i]);
                    break;
               
                case -2: /* m = -2 */
                    for (i = 0; i < len; i++)
                        Ylm[i] = C3m2*(x[i]*y[i]*z[i])/(r[i]*r[i]*r[i]);
                    break;

                case -1: /* m = -1 */
                    for (i = 0; i < len; i++)
                        Ylm[i] = C3m1*y[i]*(4*z[i]*z[i] - x[i]*x[i] - y[i]*y[i])/(r[i]*r[i]*r[i]);
                    break;

                case 0: /* m = 0 */
                    for (i = 0; i < len; i++)
                        Ylm[i] = C30*z[i]*(2*z[i]*z[i]-3*x[i]*x[i]-3*y[i]*y[i])/(r[i]*r[i]*r[i]);
                    break;
                
                case 1: /* m = 1 */
                    for (i = 0; i < len; i++)
                        Ylm[i] = C3p1*x[i]*(4*z[i]*z[i] - x[i]*x[i] - y[i]*y[i])/(r[i]*r[i]*r[i]);
                    break;
                
                case 2: /* m = 2 */
                    for (i = 0; i < len; i++)
                        Ylm[i] = C3p2*z[i]*(x[i]*x[i] - y[i]*y[i])/(r[i]*r[i]*r[i]);
                    break;
                
                case 3: /* m = 3 */
                    for (i = 0; i < len; i++)
                        Ylm[i] = C3p3*x[i]*(x[i]*x[i]-3*y[i]*y[i])/(r[i]*r[i]*r[i]);
                    break;
                
                /* incorrect m */
                default: printf("<m> must be an integer between %d and %d!\n", -l, l); 
                         break;
            }
            break;
        
        /* l = 4 */
        case 4: 
            switch (m) 
            {     
                case -4: /* m = -4 */
                    for (i = 0; i < len; i++)
                        Ylm[i]=C4m4*(x[i]*y[i]*(x[i]*x[i]-y[i]*y[i]))/(r[i]*r[i]*r[i]*r[i]);
                    break;
                
                case -3: /* m = -3 */
                    for (i = 0; i < len; i++)
                        Ylm[i]=C4m3*(3.0*x[i]*x[i]-y[i]*y[i])*y[i]*z[i]/(r[i]*r[i]*r[i]*r[i]);
                    break;
               
                case -2: /* m = -2 */
                    for (i = 0; i < len; i++)
                        Ylm[i]=C4m2*x[i]*y[i]*(7.0*z[i]*z[i]-r[i]*r[i])/(r[i]*r[i]*r[i]*r[i]);
                    break;

                case -1: /* m = -1 */
                    for (i = 0; i < len; i++)
                        Ylm[i]=C4m1*y[i]*z[i]*(7.0*z[i]*z[i]-3.0*r[i]*r[i])/(r[i]*r[i]*r[i]*r[i]);
                    break;

                case 0: /* m = 0 */
                    for (i = 0; i < len; i++)
                        Ylm[i]=C40*(35.0*z[i]*z[i]*z[i]*z[i]-30.0*z[i]*z[i]*r[i]*r[i]+3.0*r[i]*r[i]*r[i]*r[i])/(r[i]*r[i]*r[i]*r[i]);
                    break;
                
                case 1: /* m = 1 */
                    for (i = 0; i < len; i++)
                        Ylm[i]=C4p1*x[i]*z[i]*(7.0*z[i]*z[i]-3.0*r[i]*r[i])/(r[i]*r[i]*r[i]*r[i]);
                    break;
                
                case 2: /* m = 2 */
                    for (i = 0; i < len; i++)
                        Ylm[i]=C4p2*(x[i]*x[i]-y[i]*y[i])*(7.0*z[i]*z[i]-r[i]*r[i])/(r[i]*r[i]*r[i]*r[i]);
                    break;
                
                case 3: /* m = 3 */
                    for (i = 0; i < len; i++)
                        Ylm[i]=C4p3*(x[i]*x[i]-3.0*y[i]*y[i])*x[i]*z[i]/(r[i]*r[i]*r[i]*r[i]);
                    break;
                
                case 4: /* m = 4 */
                    for (i = 0; i < len; i++)
                        Ylm[i]=C4p4*(x[i]*x[i]*(x[i]*x[i]-3.0*y[i]*y[i]) - y[i]*y[i]*(3.0*x[i]*x[i]-y[i]*y[i]))/(r[i]*r[i]*r[i]*r[i]);
                    break;
                    
                /* incorrect m */
                default: printf("<m> must be an integer between %d and %d!\n", -l, l); 
                         break;
            }
            break;
      
        /* l = 5 */
        case 5: 
            //p = sqrt(x[i]*x[i]+y[i]*y[i]);
            switch (m) 
            {   
                case -5: /* m = -5 */
                    for (i = 0; i < len; i++) {
                        p = sqrt(x[i]*x[i]+y[i]*y[i]);
                        Ylm[i] = C5m5*(8.0*x[i]*x[i]*x[i]*x[i]*y[i]-4.0*x[i]*x[i]*y[i]*y[i]*y[i] + 4.0*pow(y[i],5)-3.0*y[i]*p*p*p*p)/(r[i]*r[i]*r[i]*r[i]*r[i]);
                    }
                    break;
                
                case -4: /* m = -4 */
                    for (i = 0; i < len; i++) {
                        //p = sqrt(x[i]*x[i]+y[i]*y[i]);
                        Ylm[i] = C5m4*(4.0*x[i]*x[i]*x[i]*y[i] - 4.0*x[i]*y[i]*y[i]*y[i])*z[i]/(r[i]*r[i]*r[i]*r[i]*r[i]);
                    }
                    break;
                
                case -3: /* m = -3 */
                    for (i = 0; i < len; i++) {
                        p = sqrt(x[i]*x[i]+y[i]*y[i]);
                        Ylm[i] = C5m3*(3.0*y[i]*p*p - 4.0*y[i]*y[i]*y[i])*(9.0*z[i]*z[i]-r[i]*r[i])/(r[i]*r[i]*r[i]*r[i]*r[i]);
                    }
                    break;
               
                case -2: /* m = -2 */
                    for (i = 0; i < len; i++) {
                        //p = sqrt(x[i]*x[i]+y[i]*y[i]);
                        Ylm[i] = C5m2*x[i]*y[i]*(3.0*z[i]*z[i]*z[i]-z[i]*r[i]*r[i])/(r[i]*r[i]*r[i]*r[i]*r[i]);
                    }
                    break;

                case -1: /* m = -1 */
                    for (i = 0; i < len; i++) {
                        //p = sqrt(x[i]*x[i]+y[i]*y[i]);
                        Ylm[i] = C5m1*y[i]*(21.0*z[i]*z[i]*z[i]*z[i] - 14.0*r[i]*r[i]*z[i]*z[i]+r[i]*r[i]*r[i]*r[i])/(r[i]*r[i]*r[i]*r[i]*r[i]);
                    }
                    break;

                case 0: /* m = 0 */
                    for (i = 0; i < len; i++) {
                        //p = sqrt(x[i]*x[i]+y[i]*y[i]);
                        Ylm[i] = C50*(63.0*z[i]*z[i]*z[i]*z[i]*z[i] -70.0*z[i]*z[i]*z[i]*r[i]*r[i] + 15.0*z[i]*r[i]*r[i]*r[i]*r[i])/(r[i]*r[i]*r[i]*r[i]*r[i]);
                    }
                    break;
                
                case 1: /* m = 1 */
                    for (i = 0; i < len; i++) {
                        //p = sqrt(x[i]*x[i]+y[i]*y[i]);
                        Ylm[i] = C5p1*x[i]*(21.0*z[i]*z[i]*z[i]*z[i] - 14.0*r[i]*r[i]*z[i]*z[i]+r[i]*r[i]*r[i]*r[i])/(r[i]*r[i]*r[i]*r[i]*r[i]);
                    }
                    break;
                
                case 2: /* m = 2 */
                    for (i = 0; i < len; i++) {
                        //p = sqrt(x[i]*x[i]+y[i]*y[i]);
                        Ylm[i] = C5p2*(x[i]*x[i]-y[i]*y[i])*(3.0*z[i]*z[i]*z[i] - r[i]*r[i]*z[i])/(r[i]*r[i]*r[i]*r[i]*r[i]);
                    }
                    break;
                
                case 3: /* m = 3 */
                    for (i = 0; i < len; i++) {
                        p = sqrt(x[i]*x[i]+y[i]*y[i]);
                        Ylm[i] = C5p3*(4.0*x[i]*x[i]*x[i]-3.0*p*p*x[i])*(9.0*z[i]*z[i]-r[i]*r[i])/(r[i]*r[i]*r[i]*r[i]*r[i]);
                    }
                    break;
                
                case 4: /* m = 4 */
                    for (i = 0; i < len; i++) {
                        p = sqrt(x[i]*x[i]+y[i]*y[i]);
                        Ylm[i] = C5p4*(4.0*(x[i]*x[i]*x[i]*x[i]+y[i]*y[i]*y[i]*y[i])-3.0*p*p*p*p)*z[i]/(r[i]*r[i]*r[i]*r[i]*r[i]);
                    }
                    break;
                
                case 5: /* m = 5 */
                    for (i = 0; i < len; i++) {
                        p = sqrt(x[i]*x[i]+y[i]*y[i]);
                        Ylm[i] = C5p5*(4.0*x[i]*x[i]*x[i]*x[i]*x[i] + 8.0*x[i]*y[i]*y[i]*y[i]*y[i] -4.0*x[i]*x[i]*x[i]*y[i]*y[i] -3.0*x[i]*p*p*p*p)/(r[i]*r[i]*r[i]*r[i]*r[i]);
                    }
                    break;
                    
                /* incorrect m */
                default: printf("<m> must be an integer between %d and %d!\n", -l, l); 
                         break;
            }
            break;

        /* l = 6 */
        case 6: 
            //p = sqrt(x[i]*x[i]+y[i]*y[i]);
            switch (m) 
            {   
                case -6: /* m = -6 */
                    for (i = 0; i < len; i++) {
                        p = sqrt(x[i]*x[i]+y[i]*y[i]);
                        Ylm[i] = C6m6*(12.0*pow(x[i],5)*y[i]+12.0*x[i]*pow(y[i],5) - 8.0*x[i]*x[i]*x[i]*y[i]*y[i]*y[i]-6.0*x[i]*y[i]*pow(p,4))/(r[i]*r[i]*r[i]*r[i]*r[i]*r[i]);
                    }
                    break;
                
                case -5: /* m = -5 */
                    for (i = 0; i < len; i++) {
                        p = sqrt(x[i]*x[i]+y[i]*y[i]);
                        Ylm[i] = C6m5*(8.0*pow(x[i],4)*y[i] - 4.0*x[i]*x[i]*y[i]*y[i]*y[i] + 4.0*pow(y[i],5) -3.0*y[i]*pow(p,4))*z[i]/(r[i]*r[i]*r[i]*r[i]*r[i]*r[i]);
                    }
                    break;
                
                case -4: /* m = -4 */
                    for (i = 0; i < len; i++)
                        Ylm[i] = C6m4*(4.0*x[i]*x[i]*x[i]*y[i] -4.0*x[i]*y[i]*y[i]*y[i])*(11.0*z[i]*z[i]-r[i]*r[i])/(r[i]*r[i]*r[i]*r[i]*r[i]*r[i]);
                    break;
                
                case -3: /* m = -3 */
                    for (i = 0; i < len; i++) {
                        p = sqrt(x[i]*x[i]+y[i]*y[i]);
                        Ylm[i] = C6m3*(-4.0*y[i]*y[i]*y[i] + 3.0*y[i]*p*p)*(11.0*z[i]*z[i]*z[i] - 3.0*z[i]*r[i]*r[i])/(r[i]*r[i]*r[i]*r[i]*r[i]*r[i]);
                    }
                    break;
               
                case -2: /* m = -2 */
                    for (i = 0; i < len; i++) {
                        //p = sqrt(x[i]*x[i]+y[i]*y[i]);
                        Ylm[i] = C6m2*(2.0*x[i]*y[i])*(33.0*pow(z[i],4)-18.0*z[i]*z[i]*r[i]*r[i] + pow(r[i],4))/(r[i]*r[i]*r[i]*r[i]*r[i]*r[i]);
                    }
                    break;

                case -1: /* m = -1 */
                    for (i = 0; i < len; i++)
                        Ylm[i] = C6m1*y[i]*(33.0*pow(z[i],5)-30.0*z[i]*z[i]*z[i]*r[i]*r[i] +5.0*z[i]*pow(r[i],4))/(r[i]*r[i]*r[i]*r[i]*r[i]*r[i]);
                    break;

                case 0: /* m = 0 */
                    for (i = 0; i < len; i++) {
                        //p = sqrt(x[i]*x[i]+y[i]*y[i]);
                        Ylm[i] = C60*(231.0*pow(z[i],6)-315*pow(z[i],4)*r[i]*r[i] + 105.0*z[i]*z[i]*pow(r[i],4) -5.0*pow(r[i],6))/(r[i]*r[i]*r[i]*r[i]*r[i]*r[i]);
                    }
                    break;
                
                case 1: /* m = 1 */
                    for (i = 0; i < len; i++) {
                        //p = sqrt(x[i]*x[i]+y[i]*y[i]);
                        Ylm[i] = C6p1*x[i]*(33.0*pow(z[i],5)-30.0*z[i]*z[i]*z[i]*r[i]*r[i] +5.0*z[i]*pow(r[i],4))/(r[i]*r[i]*r[i]*r[i]*r[i]*r[i]);
                    }
                    break;
                
                case 2: /* m = 2 */
                    for (i = 0; i < len; i++) {
                        //p = sqrt(x[i]*x[i]+y[i]*y[i]);
                        Ylm[i] = C6p2*(x[i]*x[i]-y[i]*y[i])*(33.0*pow(z[i],4) - 18.0*z[i]*z[i]*r[i]*r[i] + pow(r[i],4))/(r[i]*r[i]*r[i]*r[i]*r[i]*r[i]);
                    }
                    break;
                
                case 3: /* m = 3 */
                    for (i = 0; i < len; i++) {
                        p = sqrt(x[i]*x[i]+y[i]*y[i]);
                        Ylm[i] = C6p3*(4.0*x[i]*x[i]*x[i] -3.0*x[i]*p*p)*(11.0*z[i]*z[i]*z[i] - 3.0*z[i]*r[i]*r[i])/(r[i]*r[i]*r[i]*r[i]*r[i]*r[i]);
                    }
                    break;
                
                case 4: /* m = 4 */
                    for (i = 0; i < len; i++) {
                        p = sqrt(x[i]*x[i]+y[i]*y[i]);
                        Ylm[i] = C6p4*(4.0*pow(x[i],4)+4.0*pow(y[i],4) -3.0*pow(p,4))*(11.0*z[i]*z[i] -r[i]*r[i])/(r[i]*r[i]*r[i]*r[i]*r[i]*r[i]);
                    }
                    break;
                
                case 5: /* m = 5 */
                    for (i = 0; i < len; i++) {
                        p = sqrt(x[i]*x[i]+y[i]*y[i]);
                        Ylm[i] = C6p5*(4.0*pow(x[i],5) + 8.0*x[i]*pow(y[i],4)-4.0*x[i]*x[i]*x[i]*y[i]*y[i]-3.0*x[i]*pow(p,4))*z[i]/(r[i]*r[i]*r[i]*r[i]*r[i]*r[i]);
                    }
                    break;
                
                case 6: /* m = 6 */
                    for (i = 0; i < len; i++) {
                        p = sqrt(x[i]*x[i]+y[i]*y[i]);
                        Ylm[i] = C6p6*(4.0*pow(x[i],6)-4.0*pow(y[i],6) +12.0*x[i]*x[i]*pow(y[i],4)-12.0*pow(x[i],4)*y[i]*y[i] + 3.0*y[i]*y[i]*pow(p,4)-3.0*x[i]*x[i]*pow(p,4))/(r[i]*r[i]*r[i]*r[i]*r[i]*r[i]);
                    }
                    break;
                    
                /* incorrect m */
                default: printf("<m> must be an integer between %d and %d!\n", -l, l); 
                         break;
            }
            break;
        
        default: printf("<l> must be an integer between 0 and 6!\n"); break;
    }
    
    if (l > 0) {
        for (i = 0; i < len; i++) {
            if (r[i] < 1e-10) Ylm[i] = 0.0;
        }
    }
}


/**
 * @brief   Calculate Complex spherical harmonics for given positions and given l and m. 
 *
 *          Only for l = 0, 1, ..., 6.
 */
void ComplexSphericalHarmonic(const int len, double *x, double *y,double *z, double *r, const int l, const int m, double complex *Ylm)
{
    // only l=0,1,2,3,4,5,6 implemented for now

    int i; 
    
    /* l = 0 */
    double C00 = 0.282094791773878;
    /* l = 1 */
    double C11 = 0.345494149471335;
    double C10 = 0.488602511902920;
    /* l = 2 */
    double C22 = 0.386274202023190;
    double C21 = 0.772548404046379;
    double C20 = 0.315391565252520;
    /* l = 3 */
    double C33 = 0.417223823632784;
    double C32 = 1.021985476433282;
    double C31 = 0.323180184114151;
    double C30 = 0.373176332590115;
    /* l = 4 */
    double C44 = 0.442532692444983;
    double C43 = 1.251671470898352;
    double C42 = 0.334523271778645;
    double C41 = 0.473087347878780;
    double C40 = 0.105785546915204;
    /* l = 5 */
    double C55 = 0.464132203440858;
    double C54 = 1.467714898305751; 
    double C53 = 0.345943719146840;
    double C52 = 1.694771183260899;
    double C51 = 0.320281648576215;
    double C50 = 0.116950322453424;
    /* l = 6 */
    double C66 = 0.483084113580066; 
    double C65 = 1.673452458100098;
    double C64 = 0.356781262853998; 
    double C63 = 0.651390485867716; 
    double C62 = 0.325695242933858; 
    double C61 = 0.411975516301141;
    double C60 = 0.063569202267628;
    
    switch (l)
    {
        /* l = 0 */
        case 0:
            for (i = 0; i < len; i++) Ylm[i] = C00;
            break;
        
        /* l = 1 */
        case 1: 
            switch (m) 
            {
                case -1: /* m = -1 */
                    for (i = 0; i < len; i++) 
                        Ylm[i] = C11 * ((x[i]-I*y[i])/r[i]);
                    break;
                
                case 0: /* m = 0 */
                    for (i = 0; i < len; i++)
                        Ylm[i] = C10 * (z[i] / r[i]);
                    break;
                
                case 1: /* m = 1 */
                    for (i = 0; i < len; i++)
                        Ylm[i] = -C11 * ((x[i]+I*y[i])/r[i]);
                    break;
                
                /* incorrect m */
                default: printf("<m> must be an integer between %d and %d!\n", -l, l); break;
            }
            break;
        
        /* l = 2 */
        case 2: 
            switch (m) 
            {      
                case -2: /* m = -2 */
                    for (i = 0; i < len; i++)
                        Ylm[i] = C22 * ((x[i]-I*y[i])*(x[i]-I*y[i]))/(r[i]*r[i]);
                    break;

                case -1: /* m = -1 */
                    for (i = 0; i < len; i++)
                        Ylm[i] = C21 * ((x[i]-I*y[i])*z[i])/(r[i]*r[i]);
                    break;

                case 0: /* m = 0 */
                    for (i = 0; i < len; i++)
                        Ylm[i] = C20 * (2*z[i]*z[i] - x[i]*x[i] - y[i]*y[i])/(r[i]*r[i]);
                    break;
                
                case 1: /* m = 1 */
                    for (i = 0; i < len; i++)
                        Ylm[i] = -C21 * ((x[i]+I*y[i])*z[i])/(r[i]*r[i]);
                    break;
                
                case 2: /* m = 2 */
                    for (i = 0; i < len; i++)
                        Ylm[i] = C22 * ((x[i]+I*y[i])*(x[i]+I*y[i]))/(r[i]*r[i]);
                    break;
                
                /* incorrect m */
                default: printf("<m> must be an integer between %d and %d!\n", -l, l); 
                         break;
            }
            break;

        /* l = 3 */
        case 3: 
            switch (m) 
            {   
                case -3: /* m = -3 */
                    for (i = 0; i < len; i++)
                        Ylm[i] = C33 * ((x[i]-I*y[i])*(x[i]-I*y[i])*(x[i]-I*y[i]))/(r[i]*r[i]*r[i]);
                    break;
               
                case -2: /* m = -2 */
                    for (i = 0; i < len; i++)
                        Ylm[i] = C32 * ((x[i]-I*y[i])*((x[i]-I*y[i]))*z[i])/(r[i]*r[i]*r[i]);
                    break;

                case -1: /* m = -1 */
                    for (i = 0; i < len; i++)
                        Ylm[i] = C31 * (x[i]-I*y[i])*(4*z[i]*z[i] - x[i]*x[i] - y[i]*y[i])/(r[i]*r[i]*r[i]);
                    break;

                case 0: /* m = 0 */
                    for (i = 0; i < len; i++)
                        Ylm[i] = C30 * z[i]*(2*z[i]*z[i] - 3*x[i]*x[i] - 3*y[i]*y[i])/(r[i]*r[i]*r[i]);
                    break;
                
                case 1: /* m = 1 */
                    for (i = 0; i < len; i++)
                        Ylm[i] = -C31 * (x[i]+I*y[i])*(4*z[i]*z[i] - x[i]*x[i] - y[i]*y[i])/(r[i]*r[i]*r[i]);
                    break;
                
                case 2: /* m = 2 */
                    for (i = 0; i < len; i++)
                        Ylm[i] = C32 * (((x[i]+I*y[i])*(x[i]+I*y[i]))*z[i])/(r[i]*r[i]*r[i]);
                    break;
                
                case 3: /* m = 3 */
                    for (i = 0; i < len; i++)
                        Ylm[i] = -C33 * ((x[i]+I*y[i])*(x[i]+I*y[i])*(x[i]+I*y[i]))/(r[i]*r[i]*r[i]);
                    break;
                
                /* incorrect m */
                default: printf("<m> must be an integer between %d and %d!\n", -l, l); 
                         break;
            }
            break;
        
        /* l = 4 */
        case 4: 
            switch (m) 
            {     
                case -4: /* m = -4 */
                    for (i = 0; i < len; i++)
                        Ylm[i] = C44 * ((x[i]-I*y[i])*(x[i]-I*y[i])*(x[i]-I*y[i])*(x[i]-I*y[i]))/pow(r[i],4);
                    break;
                
                case -3: /* m = -3 */
                    for (i = 0; i < len; i++)
                        Ylm[i] = C43 * (((x[i]-I*y[i])*(x[i]-I*y[i])*(x[i]-I*y[i]))*z[i])/pow(r[i],4);
                    break;
               
                case -2: /* m = -2 */
                    for (i = 0; i < len; i++)
                        Ylm[i] = C42 * (((x[i]-I*y[i])*(x[i]-I*y[i]))*(7*z[i]*z[i] - r[i]*r[i]))/pow(r[i],4);
                    break;

                case -1: /* m = -1 */
                    for (i = 0; i < len; i++)
                        Ylm[i] = C41 * ((x[i]-I*y[i])*z[i]*(7*z[i]*z[i] - 3*r[i]*r[i]))/pow(r[i],4);
                    break;

                case 0: /* m = 0 */
                    for (i = 0; i < len; i++)
                        Ylm[i] = C40 * (35*z[i]*z[i]*z[i]*z[i] - 30*(z[i]*z[i])*(r[i]*r[i]) + 3*r[i]*r[i]*r[i]*r[i])/pow(r[i],4);
                    break;
                
                case 1: /* m = 1 */
                    for (i = 0; i < len; i++)
                        Ylm[i] = -C41 * ((x[i]+I*y[i])*z[i]*(7*z[i]*z[i] - 3*r[i]*r[i]))/pow(r[i],4);
                    break;
                
                case 2: /* m = 2 */
                    for (i = 0; i < len; i++)
                        Ylm[i] = C42 * (((x[i]+I*y[i])*(x[i]+I*y[i]))*(7*z[i]*z[i] - r[i]*r[i]))/pow(r[i],4);
                    break;
                
                case 3: /* m = 3 */
                    for (i = 0; i < len; i++)
                        Ylm[i] = -C43 * (((x[i]+I*y[i])*(x[i]+I*y[i])*(x[i]+I*y[i]))*z[i])/pow(r[i],4);
                    break;
                
                case 4: /* m = 4 */
                    for (i = 0; i < len; i++)
                        Ylm[i] = C44 * ((x[i]+I*y[i])*(x[i]+I*y[i])*(x[i]+I*y[i])*(x[i]+I*y[i]))/pow(r[i],4);
                    break;
                    
                /* incorrect m */
                default: printf("<m> must be an integer between %d and %d!\n", -l, l); 
                         break;
            }
            break;
      
        /* l = 5 */
        case 5: 
            switch (m) 
            {   
                case -5: /* m = -5 */
                    for (i = 0; i < len; i++) {
                        Ylm[i] = C55 * (cpow(x[i]-I*y[i],5))/pow(r[i],5);
                    }
                    break;
                
                case -4: /* m = -4 */
                    for (i = 0; i < len; i++) {
                        Ylm[i] = C54 * (cpow(x[i]-I*y[i],4)*z[i])/pow(r[i],5);
                    }
                    break;
                
                case -3: /* m = -3 */
                    for (i = 0; i < len; i++) {
                        Ylm[i] = C53 * (((x[i]-I*y[i])*(x[i]-I*y[i])*(x[i]-I*y[i]))*(9*z[i]*z[i] - r[i]*r[i]))/pow(r[i],5);
                    }
                    break;
               
                case -2: /* m = -2 */
                    for (i = 0; i < len; i++) {
                        Ylm[i] = C52 * (((x[i]-I*y[i])*(x[i]-I*y[i]))*(3*z[i]*z[i]*z[i] - z[i]*r[i]*r[i]))/pow(r[i],5);
                    }
                    break;

                case -1: /* m = -1 */
                    for (i = 0; i < len; i++) {
                        Ylm[i] = C51 * ((x[i]-I*y[i])*(21*pow(z[i],4) - 14*z[i]*z[i]*r[i]*r[i] + pow(r[i],4)))/pow(r[i],5);
                    }
                    break;

                case 0: /* m = 0 */
                    for (i = 0; i < len; i++) {
                        Ylm[i] = C50 * (63*pow(z[i],5) - 70*(z[i]*z[i]*z[i])*(r[i]*r[i]) + 15*z[i]*pow(r[i],4))/pow(r[i],5);
                    }
                    break;
                
                case 1: /* m = 1 */
                    for (i = 0; i < len; i++) {
                        Ylm[i] = -C51 * ((x[i]+I*y[i])*(21*pow(z[i],4) - 14*z[i]*z[i]*r[i]*r[i] + pow(r[i],4)))/pow(r[i],5);
                    }
                    break;
                
                case 2: /* m = 2 */
                    for (i = 0; i < len; i++) {
                        Ylm[i] = C52 * (((x[i]+I*y[i])*(x[i]+I*y[i]))*(3*z[i]*z[i]*z[i] - z[i]*r[i]*r[i]))/pow(r[i],5);
                    }
                    break;
                
                case 3: /* m = 3 */
                    for (i = 0; i < len; i++) {
                        Ylm[i] = -C53 * (((x[i]+I*y[i])*(x[i]+I*y[i])*(x[i]+I*y[i]))*(9*z[i]*z[i] - r[i]*r[i]))/pow(r[i],5);
                    }
                    break;
                
                case 4: /* m = 4 */
                    for (i = 0; i < len; i++) {
                        Ylm[i] = C54 * (cpow(x[i]+I*y[i],4)*z[i])/pow(r[i],5);
                    }
                    break;
                
                case 5: /* m = 5 */
                    for (i = 0; i < len; i++) {
                        Ylm[i] = -C55 * cpow(x[i]+I*y[i],5)/pow(r[i],5);
                    }
                    break;
                    
                /* incorrect m */
                default: printf("<m> must be an integer between %d and %d!\n", -l, l); 
                         break;
            }
            break;

        /* l = 6 */
        case 6: 
            switch (m) 
            {   
                case -6: /* m = -6 */
                    for (i = 0; i < len; i++) {
                        Ylm[i] = C66 * cpow(x[i]-I*y[i],6)/pow(r[i],6);
                    }
                    break;
                
                case -5: /* m = -5 */
                    for (i = 0; i < len; i++) {
                        Ylm[i] = C65 * (cpow(x[i]-I*y[i],5)*z[i])/pow(r[i],6);
                    }
                    break;
                
                case -4: /* m = -4 */
                    for (i = 0; i < len; i++)
                        Ylm[i] = C64 * (cpow(x[i]-I*y[i],4)*(11*z[i]*z[i] - r[i]*r[i]))/pow(r[i],6);
                    break;
                
                case -3: /* m = -3 */
                    for (i = 0; i < len; i++) {
                        Ylm[i] = C63 * (((x[i]-I*y[i])*(x[i]-I*y[i])*(x[i]-I*y[i]))*(11*z[i]*z[i]*z[i] - 3*z[i]*r[i]*r[i]))/pow(r[i],6);
                    }
                    break;
               
                case -2: /* m = -2 */
                    for (i = 0; i < len; i++) {
                        Ylm[i] = C62 * (((x[i]-I*y[i])*(x[i]-I*y[i]))*(33*pow(z[i],4) - 18*(z[i]*z[i])*(r[i]*r[i]) + pow(r[i],4)))/pow(r[i],6);
                    }
                    break;

                case -1: /* m = -1 */
                    for (i = 0; i < len; i++)
                        Ylm[i] = C61 * ((x[i]-I*y[i])*(33*pow(z[i],5) - 30*(z[i]*z[i]*z[i])*(r[i]*r[i]) + 5*z[i]*pow(r[i],4)))/pow(r[i],6);
                    break;

                case 0: /* m = 0 */
                    for (i = 0; i < len; i++) {                        
                        Ylm[i] = C60 * (231*pow(z[i],6) - 315*pow(z[i],4)*(r[i]*r[i]) + 105*(z[i]*z[i])*pow(r[i],4) - 5*pow(r[i],6))/pow(r[i],6);
                    }
                    break;
                
                case 1: /* m = 1 */
                    for (i = 0; i < len; i++) {
                        Ylm[i] = -C61 * ((x[i]+I*y[i])*(33*pow(z[i],5) - 30*(z[i]*z[i]*z[i])*(r[i]*r[i]) + 5*z[i]*pow(r[i],4)))/pow(r[i],6);
                    }
                    break;
                
                case 2: /* m = 2 */
                    for (i = 0; i < len; i++) {
                        Ylm[i] = C62 * (((x[i]+I*y[i])*(x[i]+I*y[i]))*(33*pow(z[i],4) - 18*(z[i]*z[i])*(r[i]*r[i]) + pow(r[i],4)))/pow(r[i],6);
                    }
                    break;
                
                case 3: /* m = 3 */
                    for (i = 0; i < len; i++) {
                        Ylm[i] = -C63 * (((x[i]+I*y[i])*(x[i]+I*y[i])*(x[i]+I*y[i]))*(11*z[i]*z[i]*z[i] - 3*z[i]*(r[i]*r[i])))/pow(r[i],6);
                    }
                    break;
                
                case 4: /* m = 4 */
                    for (i = 0; i < len; i++) {
                        Ylm[i] = C64 * (cpow(x[i]+I*y[i],4)*(11*z[i]*z[i] - r[i]*r[i]))/pow(r[i],6);
                    }
                    break;
                
                case 5: /* m = 5 */
                    for (i = 0; i < len; i++) {
                        Ylm[i] = -C65 * (cpow(x[i]+I*y[i],5)*z[i])/pow(r[i],6);
                    }
                    break;
                
                case 6: /* m = 6 */
                    for (i = 0; i < len; i++) {
                        Ylm[i] = C66 * cpow(x[i]+I*y[i],6)/pow(r[i],6);
                    }
                    break;
                    
                /* incorrect m */
                default: printf("<m> must be an integer between %d and %d!\n", -l, l); 
                         break;
            }
            break;
        
        default: printf("<l> must be an integer between 0 and 6!\n"); break;
    }
    
    if (l > 0) {
        for (i = 0; i < len; i++) {
            if (r[i] < 1e-10) Ylm[i] = 0.0;
        }
    }
}


/*
 @ brief: function to calculate distance between two points
*/
void Calc_dist(SPARC_OBJ *pSPARC, int nxp, int nyp, int nzp, double x0_i_shift, double y0_i_shift, double z0_i_shift, double *R, double rchrg, int *count_interp)
{
    
    int i, j, k;
    double x, y, z;
    
    int count = 0;
    *count_interp = 0;
    
    
    if(pSPARC->cell_typ == 0){
        for (k = 0; k < nzp; k++) {
            z = k * pSPARC->delta_z - z0_i_shift; 
            for (j = 0; j < nyp; j++) {
                y = j * pSPARC->delta_y - y0_i_shift;
                for (i = 0; i < nxp; i++) {
                    x = i * pSPARC->delta_x - x0_i_shift;
                    R[count] = sqrt(x*x + y*y + z*z);
                    if (R[count] <= rchrg) (*count_interp)++;
                    (count)++;
                }
            }
        }
    } else {
        for (k = 0; k < nzp; k++) {
            z = k * pSPARC->delta_z - z0_i_shift; 
            for (j = 0; j < nyp; j++) {
                y = j * pSPARC->delta_y - y0_i_shift;
                for (i = 0; i < nxp; i++) {
                    x = i * pSPARC->delta_x - x0_i_shift;
                    R[count] = sqrt(pSPARC->metricT[0] * (x*x) + pSPARC->metricT[1] * (x*y) + pSPARC->metricT[2] * (x*z) 
                                   + pSPARC->metricT[4] * (y*y) + pSPARC->metricT[5] * (y*z) + pSPARC->metricT[8] * (z*z) );     
                    if (R[count] <= rchrg) (*count_interp)++;
                    (count)++;
                }
            }
        }
    }    
	// return count_interp;
}



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
    double *x, int *gridsizes, int *DMVertices, 
    char *fname, MPI_Comm comm
) 
{
    if (comm == MPI_COMM_NULL) return;
    int nproc_comm, rank_comm, i, j, k, index;
    MPI_Comm_size(comm, &nproc_comm);
    MPI_Comm_rank(comm, &rank_comm);

    // global size of the vector
    int Nx = gridsizes[0];
    int Ny = gridsizes[1];
    int Nz = gridsizes[2];
    int Nd = Nx * Ny * Nz;

    double *x_global = NULL;

    if (nproc_comm > 1) { // if there's more than one process, need to collect x first
        int sdims[3], periods[3], my_coords[3];
        MPI_Cart_get(comm, 3, sdims, periods, my_coords);
        /* use DD2DD to collect distributed data */
        // create a cartesian topology on one process (rank 0)
        int rdims[3] = {1,1,1}, rDMVert[6];
        MPI_Comm recv_comm;
        if (rank_comm) {
            recv_comm = MPI_COMM_NULL;
        } else {
            int rperiods[3] = {1,1,1};
            // create a cartesian topology on one process (rank 0)
            MPI_Cart_create(MPI_COMM_SELF, 3, rdims, rperiods, 0, &recv_comm);
        }
        
        D2D_OBJ d2d_sender, d2d_recvr;
        rDMVert[0] = 0; rDMVert[1] = Nx-1;
        rDMVert[2] = 0; rDMVert[3] = Ny-1;
        rDMVert[4] = 0; rDMVert[5] = Nz-1;
        
        // set up D2D targets, note that this is time consuming if 
        // number of processes is large (> 1000), in that case, do
        // this step only once and keep the d2d target objects
        Set_D2D_Target(
            &d2d_sender, &d2d_recvr, gridsizes, DMVertices, 
            rDMVert, comm, sdims, recv_comm, rdims, comm
        );
        if (rank_comm == 0) {
            x_global = (double*)malloc(Nd * sizeof(double));
        }
        
        // collect vector to one process   
        D2D(&d2d_sender, &d2d_recvr, gridsizes, DMVertices, x, rDMVert, 
            x_global, comm, sdims, recv_comm, rdims, comm);
        
        // free D2D targets
        Free_D2D_Target(&d2d_sender, &d2d_recvr, comm, recv_comm);
        
    } else {
        x_global = x;
    }
    
    if (rank_comm == 0) {
        FILE *output_fp = fopen(fname,"w");
        if (output_fp == NULL) {
            printf("\nCannot open file \"%s\"\n",fname);
            exit(EXIT_FAILURE);
        }
        for (k = 0; k < Nz; k++) {
            for (j = 0; j < Ny; j++) {
                for (i = 0; i < Nx; i++) {
                    index = k*Nx*Ny + j*Nx + i;
                    fprintf(output_fp,"%22.15E\n",x_global[index]);
                }
            }
        }
        fclose(output_fp);
    }
    
    // free the collected data after printing to file
    if (nproc_comm > 1) {
        if (rank_comm == 0) {
            free(x_global);
        }
    }
}



/**
 * @brief   Read a 3D-vector from file and distributed in comm.
 *
 *          This routine reads a vector from a file by rank 0 and distributed 
 *          in comm by domain decomposition.
 *
 * @param x          Local part of the vector (output).
 * @param gridsizes  Array of length 3, total number of nodes in each direction. 
 * @param DMVertices Array of length 6, domain vertices of the local pieces of x.
 * @param fname      The name of the file to which the vector will be read.
 * @param comm       MPI communicator.
 */
void read_vec(
    double *x, int *gridsizes, int *DMVertices, 
    char *fname, MPI_Comm comm
) 
{
    if (comm == MPI_COMM_NULL) return;
    int nproc_comm, rank_comm;
    MPI_Comm_size(comm, &nproc_comm);
    MPI_Comm_rank(comm, &rank_comm);

    // global size of the vector
    int Nx = gridsizes[0];
    int Ny = gridsizes[1];
    int Nz = gridsizes[2];
    int Nd = Nx * Ny * Nz;

    double *x_global = NULL;
    if (rank_comm == 0) {
        if (nproc_comm > 1) {
            x_global = (double*)malloc(Nd * sizeof(double));
        } else {
            x_global = x;
        }
        // read from file
        FILE *input_fp = fopen(fname,"r");
        if (input_fp == NULL) {
            printf("\nCannot open file \"%s\"\n",fname);
            exit(EXIT_FAILURE);
        }
        // read x_global
        int i;
        double vtemp;
        for (i = 0; i < Nd; i++) {
            fscanf(input_fp, "%lf", &vtemp);
            x_global[i] = vtemp;
        }
        fclose(input_fp);
    }

    if (nproc_comm > 1) { // if there's more than one process, need to distribute x 
        int rdims[3], rperiods[3], my_coords[3];
        MPI_Cart_get(comm, 3, rdims, rperiods, my_coords);
        /* use DD2DD to collect distributed data */
        // create a cartesian topology on one process (rank 0)
        int sdims[3] = {1,1,1}, sDMVert[6];
        MPI_Comm send_comm;
        if (rank_comm) {
            send_comm = MPI_COMM_NULL;
        } else {
            int speriods[3] = {1,1,1};
            // create a cartesian topology on one process (rank 0)
            MPI_Cart_create(MPI_COMM_SELF, 3, sdims, speriods, 0, &send_comm);
        }
        
        D2D_OBJ d2d_sender, d2d_recvr;
        sDMVert[0] = 0; sDMVert[1] = Nx-1;
        sDMVert[2] = 0; sDMVert[3] = Ny-1;
        sDMVert[4] = 0; sDMVert[5] = Nz-1;
        
        // set up D2D targets, note that this is time consuming if 
        // number of processes is large (> 1000), in that case, do
        // this step only once and keep the d2d target objects
        Set_D2D_Target(
            &d2d_sender, &d2d_recvr, gridsizes, sDMVert, DMVertices, 
            send_comm, sdims, comm, rdims, comm
        );
        
        // collect vector to one process   
        D2D(&d2d_sender, &d2d_recvr, gridsizes, sDMVert, x_global, 
            DMVertices, x, send_comm, sdims, comm, rdims, comm);
        
        // free D2D targets
        Free_D2D_Target(&d2d_sender, &d2d_recvr, send_comm, comm);
    }
    
    // free the collected data after printing to file
    if (nproc_comm > 1) {
        if (rank_comm == 0) {
            free(x_global);
        }
    }
}


/**
 * @brief   Function to check the below-tag format
 *          Note: used in readfiles.c for readion function
 */
void check_below_entries(FILE *ion_fp, char *tag) 
{
    int i;
    char *str = malloc(L_STRING * sizeof(char));

    str[0] = '\0';
    fscanf(ion_fp, "%[^\n]%*c", str);
    for (i = 0; i < strlen(str); i++) {
        if (isdigit(str[i])) {
            printf(RED"ERROR: Please remove the data in the same line as the %s tag in the ION file. All entries should be strictly below the %s tag.\n"RESET, tag, tag);
            exit(EXIT_FAILURE);
        }
    }
    free(str);
}


/**
 * @brief   Check the input options in ion file
 */
int check_num_input(FILE *fp, void *array, char TYPE)
{
    int nums_now, bytes_now;
    int bytes_consumed = 0, nums_read = 0;
    char *str = malloc(L_STRING * sizeof(char));
    str[0] = '\0';

    if (TYPE != 'I' && TYPE != 'D') {
        printf("ERROR: Unknown type\n");
        exit(EXIT_FAILURE);
    }

    fscanf(fp,"%s",str);
    fseek ( fp , -strlen(str) , SEEK_CUR );
    
    if (str[0] == '#') {
        fscanf(fp, "%*[^\n]\n"); // skip current line
        free(str);
        return -1;
    }
    
    fscanf(fp, "%[^\n]%*c", str);

    if (TYPE == 'I') {
        while ( ( nums_now = 
                sscanf( str + bytes_consumed, "%d%n", (int *)array + nums_read, & bytes_now )
                ) > 0 && nums_read < 10) {
            bytes_consumed += bytes_now;
            nums_read += nums_now;
        }
    } else if (TYPE == 'D') {
        while ( ( nums_now = 
                sscanf( str + bytes_consumed, "%lf%n", (double *)array + nums_read, & bytes_now )
                ) > 0 && nums_read < 10) {
            bytes_consumed += bytes_now;
            nums_read += nums_now;
        }
    }
    
    free(str);
    return nums_read;
}

/*
 @ brief: function to calculate a 3x3 matrix times a vector
*/
void matrixTimesVec_3d(double *A, double *b, double *c) {
#define A(i,j) A[i+j*3]
    int i, j;
    for (i = 0;  i < 3; i++) {
        for (j = 0; j < 3; j++) {
            c[i] += A(i,j) * b[j];
        }
    }
#undef A
}

/*
 @ brief: function to calculate a 3x3 matrix times a vector
*/
void matrixTimesVec_3d_complex(double *A, double _Complex *b, double _Complex *c) {
#define A(i,j) A[i+j*3]
    int i, j;
    for (i = 0;  i < 3; i++) {
        for (j = 0; j < 3; j++) {
            c[i] += A(i,j) * b[j];
        }
    }
#undef A
}


#if defined(USE_MKL)
/**
 * @brief   MKL multi-dimension FFT interface, real to complex, following conjugate even distribution. 
 */
void MKL_MDFFT_real(double *r2c_3dinput, MKL_LONG *dim_sizes, MKL_LONG *strides_out, double _Complex *r2c_3doutput) {
    DFTI_DESCRIPTOR_HANDLE my_desc_handle = NULL;
    MKL_LONG status;
    /********************************************************************/

    status = DftiCreateDescriptor(&my_desc_handle,
                                  DFTI_DOUBLE, DFTI_REAL, 3, dim_sizes);
    status = DftiSetValue(my_desc_handle,
                          DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    status = DftiSetValue(my_desc_handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    status = DftiSetValue(my_desc_handle, DFTI_OUTPUT_STRIDES, strides_out);

    status = DftiCommitDescriptor(my_desc_handle);
    status = DftiComputeForward(my_desc_handle, r2c_3dinput, r2c_3doutput);
    status = DftiFreeDescriptor(&my_desc_handle);
    
    if (status && !DftiErrorClass(status, DFTI_NO_ERROR)) {
        printf("Error: %s\n", DftiErrorMessage(status));
    }
}

/**
 * @brief   MKL multi-dimension FFT interface, complex to complex
 */
void MKL_MDFFT(double _Complex *c2c_3dinput, MKL_LONG *dim_sizes, MKL_LONG *strides_out, double _Complex *c2c_3doutput)
{
    DFTI_DESCRIPTOR_HANDLE my_desc_handle = NULL;
    MKL_LONG status;
    /********************************************************************/

    status = DftiCreateDescriptor(&my_desc_handle,
                                  DFTI_DOUBLE, DFTI_COMPLEX, 3, dim_sizes);
    status = DftiSetValue(my_desc_handle,
                          DFTI_COMPLEX_COMPLEX, DFTI_COMPLEX_COMPLEX);
    status = DftiSetValue(my_desc_handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    status = DftiSetValue(my_desc_handle, DFTI_OUTPUT_STRIDES, strides_out);

    status = DftiCommitDescriptor(my_desc_handle);
    status = DftiComputeForward(my_desc_handle, c2c_3dinput, c2c_3doutput);
    status = DftiFreeDescriptor(&my_desc_handle);
    
    if (status && !DftiErrorClass(status, DFTI_NO_ERROR)) {
        printf("Error: %s\n", DftiErrorMessage(status));
    }
}


/**
 * @brief   MKL multi-dimension iFFT interface, complex to real, following conjugate even distribution. 
 */
void MKL_MDiFFT_real(double _Complex *c2r_3dinput, MKL_LONG *dim_sizes, MKL_LONG *strides_in, double *c2r_3doutput) {
    DFTI_DESCRIPTOR_HANDLE my_desc_handle = NULL;
    MKL_LONG status;
    /********************************************************************/

    status = DftiCreateDescriptor(&my_desc_handle,
                                  DFTI_DOUBLE, DFTI_REAL, 3, dim_sizes);
    status = DftiSetValue(my_desc_handle,
                          DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    status = DftiSetValue(my_desc_handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    status = DftiSetValue(my_desc_handle, DFTI_INPUT_STRIDES, strides_in);
    status = DftiCommitDescriptor(my_desc_handle);
    status = DftiComputeBackward(my_desc_handle, c2r_3dinput, c2r_3doutput);
    status = DftiFreeDescriptor(&my_desc_handle);

    if (status && !DftiErrorClass(status, DFTI_NO_ERROR)) {
        printf("Error: %s\n", DftiErrorMessage(status));
    }

    // scale the result to make it the same as definition of IFFT
    int N = dim_sizes[2]*dim_sizes[1]*dim_sizes[0];
    for (int i = 0; i < N; i++) {
        c2r_3doutput[i] /= N;
    }
}

/**
 * @brief   MKL multi-dimension iFFT interface, complex to complex. 
 */
void MKL_MDiFFT(double _Complex *c2c_3dinput, MKL_LONG *dim_sizes, MKL_LONG *strides_out, double _Complex *c2c_3doutput)
{
    DFTI_DESCRIPTOR_HANDLE my_desc_handle = NULL;
    MKL_LONG status;
    /********************************************************************/
    
    status = DftiCreateDescriptor(&my_desc_handle,
                                  DFTI_DOUBLE, DFTI_COMPLEX, 3, dim_sizes);
    status = DftiSetValue(my_desc_handle,
                          DFTI_COMPLEX_COMPLEX, DFTI_COMPLEX_COMPLEX);
    status = DftiSetValue(my_desc_handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    status = DftiSetValue(my_desc_handle, DFTI_OUTPUT_STRIDES, strides_out);

    status = DftiCommitDescriptor(my_desc_handle);
    status = DftiComputeBackward(my_desc_handle, c2c_3dinput, c2c_3doutput);
    status = DftiFreeDescriptor(&my_desc_handle);

    if (status && !DftiErrorClass(status, DFTI_NO_ERROR)) {
        printf("Error: %s\n", DftiErrorMessage(status));
    }

    // scale the result to make it the same as definition of IFFT
    int N = dim_sizes[2]*dim_sizes[1]*dim_sizes[0];
    for (int i = 0; i < N; i++) {
        c2c_3doutput[i] /= N;
    }
}
#endif


#if defined(USE_FFTW)
/**
 * @brief   FFTW multi-dimension FFT interface, complex to complex. 
 */
void FFTW_MDFFT(int *dim_sizes, double _Complex *c2c_3dinput, double _Complex *c2c_3doutput) {
    fftw_complex *in, *out;
    fftw_plan p;
    int N = dim_sizes[0] * dim_sizes[1] * dim_sizes[2];
    p = fftw_plan_dft(3, dim_sizes, c2c_3dinput, c2c_3doutput, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);
}

/**
 * @brief   FFTW multi-dimension iFFT interface, complex to complex. 
 */
void FFTW_MDiFFT(int *dim_sizes, double _Complex *c2c_3dinput, double _Complex *c2c_3doutput) {
    fftw_complex *in, *out;
    fftw_plan p;
    int N = dim_sizes[0] * dim_sizes[1] * dim_sizes[2], i;
    p = fftw_plan_dft(3, dim_sizes, c2c_3dinput, c2c_3doutput, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);
    for (i = 0; i < N; i++)
        c2c_3doutput[i] /= N;
}

/**
 * @brief   FFTW multi-dimension FFT interface, real to complex. 
 */
void FFTW_MDFFT_real(int *dim_sizes, double *r2c_3dinput, double _Complex *r2c_3doutput) {
    fftw_complex *in, *out;
    fftw_plan p;
    int N = dim_sizes[0] * dim_sizes[1] * dim_sizes[2];
    p = fftw_plan_dft_r2c(3, dim_sizes, r2c_3dinput, r2c_3doutput, FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);
}

/**
 * @brief   FFTW multi-dimension FFT interface, complex to real. 
 */
void FFTW_MDiFFT_real(int *dim_sizes, double _Complex *c2r_3dinput, double *c2r_3doutput) {
    fftw_complex *in, *out;
    fftw_plan p;
    int N = dim_sizes[0] * dim_sizes[1] * dim_sizes[2], i;
    p = fftw_plan_dft_c2r(3, dim_sizes, c2r_3dinput, c2r_3doutput, FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);
    for (i = 0; i < N; i++)
        c2r_3doutput[i] /= N;
}
#endif


/**
 * @brief   Function to compute exponential integral E_n(x)
 *          From Numerical Recipes
 */
double expint(const int n, const double x)
{
    static const int MAXIT = 200;
    static const double EULER = 0.577215664901533;
    double EPS, BIG;
    EPS = DBL_EPSILON;
    BIG = DBL_MAX * EPS;
    
    int i, ii, nm1;
    double a, b, c, d, del, fact, h, psi, ans;

    nm1 = n - 1;
    if (n < 0 || x < 0.0 || (x==0.0 && (n==0 || n==1))) {
        printf("ERROR: bad arguments in expint\n");
        exit(-1);
    }
    
    if (n == 0) 
        ans = exp(-x)/x;
    else if (x == 0.0)
        ans = 1./nm1;
    else {
        if (x > 1.0) {
            b = x+n;
            c = BIG;
            d = 1.0/b;
            h = d;
            for (i = 1; i <= MAXIT; i++) {
                a = -i*(nm1+i);
                b += 2.0;
                d = 1.0/(a*d+b);
                c = b+a/c;
                del = c*d;
                h *= del;
                if (fabs(del-1.0) <= EPS) {
                    ans=h*exp(-x);
                    return ans; 
                }
            }
            printf("ERROR: Continued fraction failed in expint\n");
            exit(-1);
        } else {
            ans = (nm1!=0 ? 1.0/nm1 : -log(x)-EULER);
            fact=1.0;
            for (i=1; i <= MAXIT; i++) {
                fact *= -x/i;
                if (i != nm1) 
                    del = -fact/(i-nm1);
                else {
                    psi = -EULER;
                    for (ii=1;ii<=nm1;ii++) 
                        psi += 1.0/ii;
                    del = fact*(-log(x)+psi);
                }
                ans += del;
                if (fabs(del) < fabs(ans)*EPS) return ans;
            }
            printf("ERROR: Series failed in expint\n");
            exit(-1);
        }
    }
    return ans;
}
