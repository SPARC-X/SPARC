/**
 * @file    blacs.h
 * @brief   This file contains the delaration of blacs functions.
 *
 * @author  Qimen Xu <qimenxu@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#ifndef BLACS_H
#define BLACS_H 

extern void   Cblacs_pinfo( int* mypnum, int* nprocs);
extern void   Cblacs_get( int context, int request, int* value);
extern int    Cblacs_gridinit( int* context, char * order, int np_row, int np_col);
extern void   Cblacs_gridinfo( int context, int*  np_row, int* np_col, int*  my_row, int*  my_col);
extern void   Cblacs_gridexit( int context);
extern void   Cblacs_exit( int error_code);
extern void   Cblacs_gridmap (int *ConTxt, int *usermap, int ldup, int nprow0, int npcol0);
extern int    Csys2blacs_handle(MPI_Comm comm);
extern void   Cfree_blacs_system_handle(int handle);

#endif //BLACS_H
