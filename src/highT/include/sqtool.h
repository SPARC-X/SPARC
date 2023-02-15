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


/**
 * @brief tools in SQ-type communication 
 */
void Get_xyz_rs_counts(int *counts, int *layers, int *psize, int *x_counts, int *y_counts, int *z_counts);

/**
 * @brief tools in SQ-type communication 
 */
void Get_reverse(int *start, int shift1, int n, int *array, int shift2);

/**
 * @brief tools in SQ-type communication 
 */
void Get_in_order(int *start, int shift1, int n, int *array, int shift2);

/**
 * @brief tools in SQ-type communication 
 */
void Find_size_dir(int rem, int coords, int psize, int *small, int *large);

/**
 * @brief symmetrize SQ stress
 */
void symmetrize_stress_SQ(double *stress_out, double *stress_in);

#endif