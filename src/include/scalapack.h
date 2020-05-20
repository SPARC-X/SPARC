/**
 * @file    scalapack.h
 * @brief   Declares the scalapack functions that are not defined in MKL.
 *
 * @author  Qimen Xu <qimenxu@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */


#ifndef SCALAPACK_H
#define SCALAPACK_H

// #include <complex.h>
// #include <mkl_scalapack.h> // for numroc_, descinit_, pdgemr2d_, pzgemr2d_

extern void   pdlawrite_();
extern void   pdelset_();
extern double pdlamch_();
extern int    indxg2p_();
extern int    indxg2l_();
extern int    numroc_();
extern void   descinit_();
extern void   pdlaset_();
extern double pdlange_();
extern void   pdlacpy_();
extern int    indxg2p_();

extern void   pdgemr2d_();
extern void   pdgemm_();
extern void   pdsygvx_();
extern void   pdgesv_();
extern void   pdgesvd_();

extern void   pzgemr2d_();
extern void   pzgemm_();
extern void   pzhegvx_();

#endif // SCALAPACK_H
