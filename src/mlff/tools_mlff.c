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
#include <assert.h>
#include <math.h>

#include "mlff_types.h"

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))



/*
dotProduct function returns the dot product of two arrays.

[Input]
1. vect_A: pointer to first array
2. vect_B: pointer to second array
3. size_vector: size of the vector
[Output]
1. product: dot product
*/

double dotProduct(double* vect_A, double* vect_B, int size_vector)
{
    double product = 0;
 
    // Loop for calculate cot product
    for (int i = 0; i < size_vector; i++)
        product = product + vect_A[i] * vect_B[i];

    return product;
}

/*
get_mean function returns the mean of values in an array.

[Input]
1. a: pointer to  array
2. n: size of array
[Output]
1. mean: mean
*/


double get_mean(double a[], int n)
{
    // Compute mean (average of elements)
    double sum = 0;
    for (int i = 0; i < n; i++)
        sum += a[i];
    double mean = (double)sum /
                  (double)n;
 
    return mean;
}

/*
get_variance function returns the variance of values in an array.

[Input]
1. a: pointer to  array
2. n: size of array
[Output]
1. sqDiff / (n-1): variance
*/

double get_variance(double a[], int n)
{
    // Compute mean (average of elements)
    double mean = get_mean(a,n);
    // Compute sum squared
    // differences with mean.
    double sqDiff = 0.0;
    for (int i = 0; i < n; i++)
        sqDiff += (a[i] - mean) *
                  (a[i] - mean);
    return sqDiff / (n-1);
}

double largest(double* arr, int n)
{
    int i;
    
    // Initialize maximum element
    double max = arr[0];
 
    // Traverse array elements from second and
    // compare every element with current max 
    for (i = 1; i < n; i++)
        if (arr[i] > max)
            max = arr[i];
 
    return max;
}

double smallest(double* arr, int n)
{
    int i;
    
    // Initialize maximum element
    double min = arr[0];
 
    // Traverse array elements from second and
    // compare every element with current max 
    for (i = 1; i < n; i++)
        if (arr[i] < min)
            min = arr[i];
 
    return min;
}



/*
lin_search function searches for the first occurence of an element in the array 

[Input]
1. arr: pointer to array
2. n: length of array
3. x: element to search

[Output]
1. i: ID of the element in the array
*/

double lin_search_double(double *arr, int n, double x)
{
    int i;
    for (i = 0; i < n; i++)
        if (arr[i] == x)
            return i;
    return -1;
}

double lin_search_INT(int *arr, int n, int x)
{
    int i;
    for (i = 0; i < n; i++)
        if (arr[i] == x)
            return i;
    return -1;
}