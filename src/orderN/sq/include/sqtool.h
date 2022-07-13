/**
 * @file    sqtool.h
 * @brief   This file contains the function declarations for SQ method.
 *
 * @authors Xin Jing <xjing30@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#ifndef SQTOOL_H
#define SQTOOL_H 


void Vec_copy(int *a, int *b, int n);
void Get_xyz_rs_counts(int *counts, int *layers, int *psize, int *x_counts, int *y_counts, int *z_counts);
void Get_reverse(int *start, int shift1, int n, int *array, int shift2);
void Get_in_order(int *start, int shift1, int n, int *array, int shift2);
void Find_size_dir(int rem, int coords, int psize, int *small, int *large);

void VectorDotProduct_local(double ***v1, double ***v2, int Nd[3], double *val);
void Vector2Norm_local(double ***vec, int Nd[3], double * val);
void Vec3Dto1D(double ***vec_in, double *vec_out, int *dims);
void symmetrize_stress_SQ(double *stress_out, double *stress_in);
#endif // SQ_H