#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <mpi.h>
#include <omp.h>
#include <math.h>

#include "EVA_buff.h"
#include "EVA_Lap_MV_common.h"

void EVA_Lap_MV_copy_inner_x(EVA_buff_t EVA_buff, const int ncol, const double *x)
{
    int FDn  = EVA_buff->FDn;
    int DMnx = EVA_buff->DMnx;
    int DMny = EVA_buff->DMny;
    int DMnz = EVA_buff->DMnz;
    int DMnd = EVA_buff->DMnd;
    int DMnx_ex = EVA_buff->DMnx_ex;
    int DMny_ex = EVA_buff->DMny_ex;
    int DMnd_ex = EVA_buff->DMnd_ex;
    
    double *x_ex  = EVA_buff->x_ex;

    size_t line_size = sizeof(double) * DMnx;
    #pragma omp parallel for collapse(3)
    for (int n = 0; n < ncol; n++) 
    {
        for (int kp = FDn; kp < DMnz+FDn; kp++) 
        {
            for (int jp = FDn; jp < DMny+FDn; jp++) 
            {
                int ex_idx_base = n * DMnd_ex + kp * DMnx_ex * DMny_ex;
                int  x_idx_base = n * DMnd + (kp-FDn) * DMnx * DMny;
                double *x_ex_ptr = x_ex + ex_idx_base + jp * DMnx_ex + FDn;
                const double *x_ptr = x +  x_idx_base + (jp-FDn) * DMnx;
                memcpy(x_ex_ptr, x_ptr, line_size);
            }
        }
    }
}

void EVA_Lap_MV_copy_periodic_halo(
    EVA_buff_t EVA_buff, const double *x, const int ncol, 
    const int k_spos, const int j_spos, const int i_spos,
    const int k_epos, const int j_epos, const int i_epos,
    const int kp_spos, const int jp_spos, const int ip_spos
)
{
    int DMnx = EVA_buff->DMnx;
    int DMnd = EVA_buff->DMnd;
    int DMnxny = DMnx * EVA_buff->DMny;
    int DMnx_ex = EVA_buff->DMnx_ex;
    int DMnd_ex = EVA_buff->DMnd_ex;
    int DMnxny_ex = DMnx_ex * EVA_buff->DMny_ex;
    
    double *x_ex = EVA_buff->x_ex;
    const int nelem = i_epos - i_spos;
    #pragma omp parallel for collapse(3)
    for (int n = 0; n < ncol; n++) 
    {
        for (int k = k_spos; k < k_epos; k++) 
        {
            for (int j = j_spos; j < j_epos; j++) 
            {
                int kp = kp_spos + (k - k_spos);
                int jp = jp_spos + (j - j_spos);
                double *x_ex_ptr = x_ex + n * DMnd_ex + kp * DMnxny_ex + jp * DMnx_ex + ip_spos;
                const double *x_ptr = x + n * DMnd + k * DMnxny + j * DMnx + i_spos;
                memcpy(x_ex_ptr, x_ptr, sizeof(double) * nelem);
            }
        }
    }
}

void EVA_Lap_MV_pack_halo(
    EVA_buff_t EVA_buff, const double *x, const int ncol, 
    const int k_spos, const int j_spos, const int i_spos,
    const int k_epos, const int j_epos, const int i_epos,
    int *_count
)
{
    int DMnx = EVA_buff->DMnx;
    int DMnd = EVA_buff->DMnd;
    int DMnxny = DMnx * EVA_buff->DMny;
    double *x_out = EVA_buff->x_out;
    
    int count = *_count;
    int ni = i_epos - i_spos;
    int nk = k_epos - k_spos;
    int nj = j_epos - j_spos;
    #pragma omp parallel for collapse(3)
    for (int n = 0; n < ncol; n++) 
    {
        for (int k = k_spos; k < k_epos; k++) 
        {
            for (int j = j_spos; j < j_epos; j++) 
            {
                int x_offset = n * DMnd + k * DMnxny + j * DMnx + i_spos;
                int x_out_offset = (n * nk * nj + (k - k_spos) * nj + (j - j_spos)) * ni + count;
                const double *x_ptr = x + x_offset;
                double *x_out_ptr = x_out + x_out_offset;
                memcpy(x_out_ptr, x_ptr, sizeof(double) * ni);
            }
        }
    }
    count += ni * nk * nj * ncol;
    *_count = count;
}

void EVA_Lap_MV_unpack_halo(
    EVA_buff_t EVA_buff, const int ncol, 
    const int k_spos, const int j_spos, const int i_spos,
    const int k_epos, const int j_epos, const int i_epos,
    int *_count
)
{
    int DMnx_ex = EVA_buff->DMnx_ex;
    int DMnd_ex = EVA_buff->DMnd_ex;
    int DMnxny_ex = DMnx_ex * EVA_buff->DMny_ex;
    double *x_ex = EVA_buff->x_ex;
    double *x_in = EVA_buff->x_in;
    
    int count = *_count;
    int ni = i_epos - i_spos;
    int nk = k_epos - k_spos;
    int nj = j_epos - j_spos;
    #pragma omp parallel for collapse(3)
    for (int n = 0; n < ncol; n++) 
    {
        for (int k = k_spos; k < k_epos; k++) 
        {
            for (int j = j_spos; j < j_epos; j++) 
            {
                int x_in_offset = (n * nk * nj + (k - k_spos) * nj + (j - j_spos)) * ni + count;
                int x_ex_offset = n * DMnd_ex + k * DMnxny_ex + j * DMnx_ex + i_spos;
                const double *x_in_ptr = x_in + x_in_offset;
                double *x_ex_ptr = x_ex + x_ex_offset;
                memcpy(x_ex_ptr, x_in_ptr, sizeof(double) * ni);
            }
        }
    }
    count += ni * nk * nj * ncol;
    *_count = count;
}

#define X_BLK_SIZE 32
#define Y_BLK_SIZE 32
#define Z_BLK_SIZE 16

void EVA_Lap_MV_stencil_omp_dd(
    const int nthreads, const int DMnx, const int DMny, const int DMnz,
    int *nblk_x, int *nblk_y, int *nblk_z, 
    int *x_blk_size, int *y_blk_size, int *z_blk_size
)
{
    if (nthreads <= 8)
    {
        switch (nthreads)
        {
            case 1: { *nblk_x = 1; *nblk_y = 1; *nblk_z = 1; break; }
            case 2: { *nblk_x = 1; *nblk_y = 2; *nblk_z = 1; break; }
            case 3: { *nblk_x = 1; *nblk_y = 3; *nblk_z = 1; break; }
            case 4: { *nblk_x = 1; *nblk_y = 2; *nblk_z = 2; break; }
            case 5: { *nblk_x = 1; *nblk_y = 5; *nblk_z = 1; break; }
            case 6: { *nblk_x = 1; *nblk_y = 3; *nblk_z = 2; break; }
            case 7: { *nblk_x = 1; *nblk_y = 7; *nblk_z = 1; break; }
            case 8: { *nblk_x = 2; *nblk_y = 2; *nblk_z = 2; break; }
        }
        *x_blk_size = (DMnx + (*nblk_x) - 1) / (*nblk_x);
        *y_blk_size = (DMny + (*nblk_y) - 1) / (*nblk_y);
        *z_blk_size = (DMnz + (*nblk_z) - 1) / (*nblk_z);
    } else {
        *nblk_x = (DMnx + X_BLK_SIZE - 1) / X_BLK_SIZE;
        *nblk_y = (DMny + Y_BLK_SIZE - 1) / Y_BLK_SIZE;
        *nblk_z = (DMnz + Z_BLK_SIZE - 1) / Z_BLK_SIZE;
        *x_blk_size = X_BLK_SIZE;
        *y_blk_size = Y_BLK_SIZE;
        *z_blk_size = Z_BLK_SIZE;
    }
}
