/**
 * @file    EVA_Lap_MV_common.h
 * @brief   External Vectorization and Acceleration (EVA) module
 *          Functions used in both EVA_Lap_MV_orth.c and EVA_Lap_MV_nonorth.c
 *
 * @author  Hua Huang <huangh223@gatech.edu>
 *          Edmond Chow <echow@cc.gatech.edu>
 * 
 * Copyright (c) 2018-2019 Edmond Group at Georgia Tech.
 */
 
#ifndef __EVA_LAP_MV_COMMON_H__
#define __EVA_LAP_MV_COMMON_H__

#include "EVA_buff.h"

void EVA_Lap_MV_copy_inner_x(EVA_buff_t EVA_buff, const int ncol, const double *x);

void EVA_Lap_MV_copy_periodic_halo(
    EVA_buff_t EVA_buff, const double *x, const int ncol, 
    const int k_spos, const int j_spos, const int i_spos,
    const int k_epos, const int j_epos, const int i_epos,
    const int kp_spos, const int jp_spos, const int ip_spos
);

void EVA_Lap_MV_pack_halo(
    EVA_buff_t EVA_buff, const double *x, const int ncol, 
    const int k_spos, const int j_spos, const int i_spos,
    const int k_epos, const int j_epos, const int i_epos,
    int *_count
);

void EVA_Lap_MV_unpack_halo(
    EVA_buff_t EVA_buff, const int ncol, 
    const int k_spos, const int j_spos, const int i_spos,
    const int k_epos, const int j_epos, const int i_epos,
    int *_count
);

void EVA_Lap_MV_stencil_omp_dd(
    const int nthreads, const int DMnx, const int DMny, const int DMnz,
    int *nblk_x, int *nblk_y, int *nblk_z, 
    int *x_blk_size, int *y_blk_size, int *z_blk_size
);

static inline void EVA_Lap_MV_unpack_dd_idx(
    const int iblk, const int nblk_x, const int nblk_xy, 
    const int DMnx, const int DMny, const int DMnz,
    const int x_blk_size, const int y_blk_size, const int z_blk_size,
    int *x_spos, int *y_spos, int *z_spos,
    int *x_epos, int *y_epos, int *z_epos
)
{
    int iblk_z = iblk / nblk_xy;
    int iblk_y = (iblk % nblk_xy) / nblk_x;
    int iblk_x = iblk % nblk_x;
    *z_spos = iblk_z * z_blk_size;
    *y_spos = iblk_y * y_blk_size;
    *x_spos = iblk_x * x_blk_size;
    *z_epos = ((*z_spos) + z_blk_size > DMnz) ? DMnz : (*z_spos) + z_blk_size;
    *y_epos = ((*y_spos) + y_blk_size > DMny) ? DMny : (*y_spos) + y_blk_size;
    *x_epos = ((*x_spos) + x_blk_size > DMnx) ? DMnx : (*x_spos) + x_blk_size;
}

#endif
