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

#ifdef IBM
#define numroc_ numroc
#define descinit_ descinit
#define pdlamch_ pdlamch
#define pdgemr2d_ pdgemr2d
#define pdgemm_ pdgemm
#define pdsygvx_ pdsygvx
#define pdsyevx_ pdsyevx
#define pzgemm_ pzgemm
#define pzgemr2d_ pzgemr2d
#define pzhegvx_ pzhegvx
#define pdsyrk_ pdsyrk
#define pdsyr2k_ pdsyr2k
#define pdaxpy_ pdaxpy
#define pdsyevd_ pdsyevd
#define pdsyev_ pdsyev
#define pdpotrf_ pdpotrf
#define pddot_ pddot
#define pdcopy_ pdcopy
#define pdscal_ pdscal
#define pdnrm2_ pdnrm2
#define pdgemv_ pdgemv
#define pdtrsm_ pdtrsm
#define pdlatra_ pdlatra
#define pztrsm_ pztrsm
#endif

extern void   pdlawrite_();
extern void   pdelset_();
extern double pdlamch_();
extern int    indxg2p_();
extern int    indxg2l_();
extern int    numroc_();
extern void   descinit_();
extern int    Csys2blacs_handle();
extern void   pdlaset_();
extern double pdlange_();
extern void   pdlacpy_();
extern int    indxg2p_();

extern void   pdgemr2d_();
extern void   pdgemm_();
extern void   pdsygvx_();
extern void   pdsyevx_();
extern void   pdgesv_();
extern void   pdgesvd_();

extern void   pzgemr2d_();
extern void   pzgemm_();
extern void   pzhegvx_();

extern void   pdpotrf_();
extern void   pdtrmr2d_();
extern void   pdtrtri_();
extern void   pdscal_();

extern void   pdgeadd_();
extern void   pdgemv_();
extern void   pdtrmm_();
extern void   pdnrm2_();
extern void   pddot_();
extern void   pdaxpy_();
extern void   pdcopy_();
extern double pdlatra_();
extern void   pdtrsm_();
extern void   pztrsm_();
extern void   pdsyrk_();
extern void   pdsyev_();
extern void   pdsyevd_();
#endif // SCALAPACK_H
