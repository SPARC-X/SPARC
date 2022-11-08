/**
 * @file    nlocVecRoutines.c
 * @brief   This file contains functions for nonlocal components.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 *          Hua Huang <huangh223@gatech.edu>
 *          Edmond Chow <echow@cc.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */
 
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
/* BLAS routines */
#ifdef USE_MKL
    #include <mkl.h> // for cblas_* functions
#else
    #include <cblas.h>
#endif

#include "nlocVecRoutines.h"
#include "tools.h"
#include "isddft.h"
#include "initialization.h"

#define TEMP_TOL 1e-12

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)>(b)?(b):(a))



/**
 * @brief   Find the list of all atoms that influence the processor 
 *          domain in psi-domain.
 */
void GetInfluencingAtoms_nloc(SPARC_OBJ *pSPARC, ATOM_NLOC_INFLUENCE_OBJ **Atom_Influence_nloc, int *DMVertices, MPI_Comm comm) 
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#ifdef DEBUG
    if (rank == 0) printf("Finding atoms that has nonlocal influence on the local process domain ... \n");
#endif
    // processors that are not in the dmcomm will remain idle
    // on input make comm of processes with bandcomm_index or kptcomm_index MPI_COMM_NULL!
    if (comm == MPI_COMM_NULL) {
        return; 
    }

#ifdef DEBUG
    double t1, t2;
    t1 = MPI_Wtime();
#endif

    int nproc_comm, rank_comm;
    MPI_Comm_size(comm, &nproc_comm);
    MPI_Comm_rank(comm, &rank_comm);
    
    double DMxs, DMxe, DMys, DMye, DMzs, DMze;
    double Lx, Ly, Lz, rc, x0, y0, z0, x0_i, y0_i, z0_i, x2, y2, z2, r2, rc2, rcbox_x, rcbox_y, rcbox_z, x, y, z;
    int count_overlap_nloc, count_overlap_nloc_sphere, ityp, i, j, k, count, i_DM, j_DM, k_DM, 
        iat, atmcount, atmcount2, DMnx, DMny;
    int pp, qq, rr, ppmin, ppmax, qqmin, qqmax, rrmin, rrmax;
    int rc_xl, rc_xr, rc_yl, rc_yr, rc_zl, rc_zr, ndc;
    
    Lx = pSPARC->range_x;
    Ly = pSPARC->range_y;
    Lz = pSPARC->range_z;
    
    // TODO: notice the difference here from rb-domain, since nonlocal projectors decay drastically, using the fd-nodes as edges are more appropriate
    DMxs = pSPARC->xin + DMVertices[0] * pSPARC->delta_x;
    DMxe = pSPARC->xin + (DMVertices[1]) * pSPARC->delta_x; // note that this is not the actual edge, add BCx to get actual domain edge
    DMys = DMVertices[2] * pSPARC->delta_y;
    DMye = (DMVertices[3]) * pSPARC->delta_y; // note that this is not the actual edge, add BCx to get actual domain edge
    DMzs = DMVertices[4] * pSPARC->delta_z;
    DMze = (DMVertices[5]) * pSPARC->delta_z; // note that this is not the actual edge, add BCx to get actual domain edge
    
    // number of nodes in the local distributed domain
    DMnx = DMVertices[1] - DMVertices[0] + 1;
    DMny = DMVertices[3] - DMVertices[2] + 1;

    // TODO: in the future, it's better to save only the atom types that have influence on the local domain
    *Atom_Influence_nloc = (ATOM_NLOC_INFLUENCE_OBJ *)malloc(sizeof(ATOM_NLOC_INFLUENCE_OBJ) * pSPARC->Ntypes);
    
    ATOM_NLOC_INFLUENCE_OBJ Atom_Influence_nloc_temp;

    // find which atoms have nonlocal influence on the distributed domain owned by current process
    atmcount = 0; atmcount2 = 0;
    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        rc = 0.0;
        // find max rc 
        for (i = 0; i <= pSPARC->psd[ityp].lmax; i++) {
            rc = max(rc, pSPARC->psd[ityp].rc[i]);
        }
        rc2 = rc * rc;
        
        if(pSPARC->cell_typ == 0) {
            rcbox_x = rcbox_y = rcbox_z = rc;            
        } else {
            // TODO: make an appropriate box in initialization 
            rcbox_x = pSPARC->CUTOFF_x[ityp];
            rcbox_y = pSPARC->CUTOFF_y[ityp];
            rcbox_z = pSPARC->CUTOFF_z[ityp];
        }

        // first loop over all atoms of each type to find number of influencing atoms
        count_overlap_nloc = 0;
        for (i = 0; i < pSPARC->nAtomv[ityp]; i++) {
            // get atom positions
            x0 = pSPARC->atom_pos[3*atmcount];
            y0 = pSPARC->atom_pos[3*atmcount+1];
            z0 = pSPARC->atom_pos[3*atmcount+2];
            atmcount++;
            ppmin = ppmax = qqmin = qqmax = rrmin = rrmax = 0;
            if (pSPARC->BCx == 0) {
                if (pSPARC->SQFlag == 1) {
                    // rcut_x is the real ruct in x direction. 
                    double rcut_x = pSPARC->pSQ->nloc[0] * pSPARC->delta_x;
                    ppmax = floor((rcbox_x + Lx - x0 + rcut_x) / Lx + TEMP_TOL);
                    ppmin = -floor((rcbox_x + x0 + rcut_x) / Lx + TEMP_TOL);    
                } else {
                    ppmax = floor((rcbox_x + Lx - x0) / Lx + TEMP_TOL);
                    ppmin = -floor((rcbox_x + x0) / Lx + TEMP_TOL);
                }
            }
            if (pSPARC->BCy == 0) {
                if (pSPARC->SQFlag == 1) {
                    double rcut_y = pSPARC->pSQ->nloc[1] * pSPARC->delta_y;
                    qqmax = floor((rcbox_y + Ly - y0 + rcut_y) / Ly + TEMP_TOL);
                    qqmin = -floor((rcbox_y + y0 + rcut_y) / Ly + TEMP_TOL);
                } else {
                    qqmax = floor((rcbox_y + Ly - y0) / Ly + TEMP_TOL);
                    qqmin = -floor((rcbox_y + y0) / Ly + TEMP_TOL);
                }
            }
            if (pSPARC->BCz == 0) {
                if (pSPARC->SQFlag == 1) {
                    double rcut_z = pSPARC->pSQ->nloc[2] * pSPARC->delta_z;
                    rrmax = floor((rcbox_z + Lz - z0 + rcut_z) / Lz + TEMP_TOL);
                    rrmin = -floor((rcbox_z + z0 + rcut_z) / Lz + TEMP_TOL);
                } else {
                    rrmax = floor((rcbox_z + Lz - z0) / Lz + TEMP_TOL);
                    rrmin = -floor((rcbox_z + z0) / Lz + TEMP_TOL);
                }
            }

            // check how many of it's images interacts with the local distributed domain
            for (rr = rrmin; rr <= rrmax; rr++) {
                z0_i = z0 + Lz * rr; // z coord of image atom
                if ((z0_i < DMzs - rcbox_z) || (z0_i >= DMze + rcbox_z)) continue;
                for (qq = qqmin; qq <= qqmax; qq++) {
                    y0_i = y0 + Ly * qq; // y coord of image atom
                    if ((y0_i < DMys - rcbox_y) || (y0_i >= DMye + rcbox_y)) continue;
                    for (pp = ppmin; pp <= ppmax; pp++) {
                        x0_i = x0 + Lx * pp; // x coord of image atom
                        if ((x0_i < DMxs - rcbox_x) || (x0_i >= DMxe + rcbox_x)) continue;
                        count_overlap_nloc++;
                    }
                }
            }
        } // end for loop over atoms of each type, for the first time
        
        Atom_Influence_nloc_temp.n_atom = count_overlap_nloc;
        Atom_Influence_nloc_temp.coords = (double *)malloc(sizeof(double) * count_overlap_nloc * 3);
        Atom_Influence_nloc_temp.atom_index = (int *)malloc(sizeof(int) * count_overlap_nloc);
        Atom_Influence_nloc_temp.xs = (int *)malloc(sizeof(int) * count_overlap_nloc);
        Atom_Influence_nloc_temp.ys = (int *)malloc(sizeof(int) * count_overlap_nloc);
        Atom_Influence_nloc_temp.zs = (int *)malloc(sizeof(int) * count_overlap_nloc);
        Atom_Influence_nloc_temp.xe = (int *)malloc(sizeof(int) * count_overlap_nloc);
        Atom_Influence_nloc_temp.ye = (int *)malloc(sizeof(int) * count_overlap_nloc);
        Atom_Influence_nloc_temp.ze = (int *)malloc(sizeof(int) * count_overlap_nloc);
        Atom_Influence_nloc_temp.ndc = (int *)malloc(sizeof(int) * count_overlap_nloc);
        Atom_Influence_nloc_temp.grid_pos = (int **)malloc(sizeof(int*) * count_overlap_nloc);

        // when there's no atom of this type that have influence, go to next type of atom
        if (Atom_Influence_nloc_temp.n_atom == 0) {
            (*Atom_Influence_nloc)[ityp].n_atom = 0;
            atmcount2 = atmcount;
            free(Atom_Influence_nloc_temp.coords);
            free(Atom_Influence_nloc_temp.atom_index);
            free(Atom_Influence_nloc_temp.xs);
            free(Atom_Influence_nloc_temp.ys);
            free(Atom_Influence_nloc_temp.zs);
            free(Atom_Influence_nloc_temp.xe);
            free(Atom_Influence_nloc_temp.ye);
            free(Atom_Influence_nloc_temp.ze);
            free(Atom_Influence_nloc_temp.ndc);
            free(Atom_Influence_nloc_temp.grid_pos);
            continue;
        }
        
        // loop over atoms of this type again to find overlapping region and atom info
        count_overlap_nloc = 0;
        count_overlap_nloc_sphere = 0;
        for (iat = 0; iat < pSPARC->nAtomv[ityp]; iat++) {
            // get atom positions
            x0 = pSPARC->atom_pos[3*atmcount2];
            y0 = pSPARC->atom_pos[3*atmcount2+1];
            z0 = pSPARC->atom_pos[3*atmcount2+2];
            atmcount2++;
            ppmin = ppmax = qqmin = qqmax = rrmin = rrmax = 0;
            if (pSPARC->BCx == 0) {
                if (pSPARC->SQFlag == 1) {
                    double rcut_x = pSPARC->pSQ->nloc[0] * pSPARC->delta_x;
                    ppmax = floor((rcbox_x + Lx - x0 + rcut_x) / Lx + TEMP_TOL);
                    ppmin = -floor((rcbox_x + x0 + rcut_x) / Lx + TEMP_TOL);    
                } else {
                    ppmax = floor((rcbox_x + Lx - x0) / Lx + TEMP_TOL);
                    ppmin = -floor((rcbox_x + x0) / Lx + TEMP_TOL);
                }
            }
            if (pSPARC->BCy == 0) {
                if (pSPARC->SQFlag == 1) {
                    double rcut_y = pSPARC->pSQ->nloc[1] * pSPARC->delta_y;
                    qqmax = floor((rcbox_y + Ly - y0 + rcut_y) / Ly + TEMP_TOL);
                    qqmin = -floor((rcbox_y + y0 + rcut_y) / Ly + TEMP_TOL);
                } else {
                    qqmax = floor((rcbox_y + Ly - y0) / Ly + TEMP_TOL);
                    qqmin = -floor((rcbox_y + y0) / Ly + TEMP_TOL);
                }
            }
            if (pSPARC->BCz == 0) {
                if (pSPARC->SQFlag == 1) {
                    double rcut_z = pSPARC->pSQ->nloc[2] * pSPARC->delta_z;
                    rrmax = floor((rcbox_z + Lz - z0 + rcut_z) / Lz + TEMP_TOL);
                    rrmin = -floor((rcbox_z + z0 + rcut_z) / Lz + TEMP_TOL);
                } else {
                    rrmax = floor((rcbox_z + Lz - z0) / Lz + TEMP_TOL);
                    rrmin = -floor((rcbox_z + z0) / Lz + TEMP_TOL);
                }
            }
            
            // check if this image interacts with the local distributed domain
            for (rr = rrmin; rr <= rrmax; rr++) {
                z0_i = z0 + Lz * rr; // z coord of image atom
                if ((z0_i < DMzs - rcbox_z) || (z0_i >= DMze + rcbox_z)) continue;
                for (qq = qqmin; qq <= qqmax; qq++) {
                    y0_i = y0 + Ly * qq; // y coord of image atom
                    if ((y0_i < DMys - rcbox_y) || (y0_i >= DMye + rcbox_y)) continue;
                    for (pp = ppmin; pp <= ppmax; pp++) {
                        x0_i = x0 + Lx * pp; // x coord of image atom
                        if ((x0_i < DMxs - rcbox_x) || (x0_i >= DMxe + rcbox_x)) continue;
                        
                        // store coordinates of the overlapping atom
                        Atom_Influence_nloc_temp.coords[count_overlap_nloc*3  ] = x0_i;
                        Atom_Influence_nloc_temp.coords[count_overlap_nloc*3+1] = y0_i;
                        Atom_Influence_nloc_temp.coords[count_overlap_nloc*3+2] = z0_i;
                        
                        // record the original atom index this image atom corresponds to
                        Atom_Influence_nloc_temp.atom_index[count_overlap_nloc] = atmcount2-1;
                        
                        // find start & end nodes of the rc-region of the image atom
                        // This way, we try to make sure all points inside rc-region
                        // is strictly less that rc distance away from the image atom
                        rc_xl = ceil( (x0_i - pSPARC->xin - rcbox_x)/pSPARC->delta_x);
                        rc_xr = floor((x0_i - pSPARC->xin + rcbox_x)/pSPARC->delta_x);
                        rc_yl = ceil( (y0_i - rcbox_y)/pSPARC->delta_y);
                        rc_yr = floor((y0_i + rcbox_y)/pSPARC->delta_y);
                        rc_zl = ceil( (z0_i - rcbox_z)/pSPARC->delta_z);
                        rc_zr = floor((z0_i + rcbox_z)/pSPARC->delta_z);
                        
                        // TODO: check if rc-region is out of fundamental domain for BC == 1!
                        // find overlap of rc-region of the image and the local dist. domain
                        Atom_Influence_nloc_temp.xs[count_overlap_nloc] = max(DMVertices[0], rc_xl);
                        Atom_Influence_nloc_temp.xe[count_overlap_nloc] = min(DMVertices[1], rc_xr);
                        Atom_Influence_nloc_temp.ys[count_overlap_nloc] = max(DMVertices[2], rc_yl);
                        Atom_Influence_nloc_temp.ye[count_overlap_nloc] = min(DMVertices[3], rc_yr);
                        Atom_Influence_nloc_temp.zs[count_overlap_nloc] = max(DMVertices[4], rc_zl);
                        Atom_Influence_nloc_temp.ze[count_overlap_nloc] = min(DMVertices[5], rc_zr);

                        // find the spherical rc-region
                        ndc = (Atom_Influence_nloc_temp.xe[count_overlap_nloc] - Atom_Influence_nloc_temp.xs[count_overlap_nloc] + 1)
                            * (Atom_Influence_nloc_temp.ye[count_overlap_nloc] - Atom_Influence_nloc_temp.ys[count_overlap_nloc] + 1)
                            * (Atom_Influence_nloc_temp.ze[count_overlap_nloc] - Atom_Influence_nloc_temp.zs[count_overlap_nloc] + 1);
                        
                        // first allocate memory for the rectangular rc-region, resize later to the spherical rc-region
                        Atom_Influence_nloc_temp.grid_pos[count_overlap_nloc] = (int *)malloc(sizeof(int) * ndc);
                        if(pSPARC->cell_typ == 0) {
                            count = 0;
                            for (k = Atom_Influence_nloc_temp.zs[count_overlap_nloc]; k <= Atom_Influence_nloc_temp.ze[count_overlap_nloc]; k++) {
                                k_DM = k - DMVertices[4];
                                z2 = k * pSPARC->delta_z - z0_i;
                                z2 *= z2;
                                for (j = Atom_Influence_nloc_temp.ys[count_overlap_nloc]; j <= Atom_Influence_nloc_temp.ye[count_overlap_nloc]; j++) {
                                    j_DM = j - DMVertices[2];
                                    y2 = j * pSPARC->delta_y - y0_i;
                                    y2 *= y2;
                                    for (i = Atom_Influence_nloc_temp.xs[count_overlap_nloc]; i <= Atom_Influence_nloc_temp.xe[count_overlap_nloc]; i++) {
                                        i_DM = i - DMVertices[0];
                                        x2 = i * pSPARC->delta_x - x0_i;
                                        x2 *= x2;
                                        r2 = x2 + y2 + z2;
                                        if (r2 <= rc2) {
                                            Atom_Influence_nloc_temp.grid_pos[count_overlap_nloc][count] = k_DM * (DMnx * DMny) + j_DM * DMnx + i_DM;
                                            count++;
                                        }
                                    }
                                }
                            }
                        } else if(pSPARC->cell_typ > 10 && pSPARC->cell_typ < 20) {
                           count = 0;
                            for (k = Atom_Influence_nloc_temp.zs[count_overlap_nloc]; k <= Atom_Influence_nloc_temp.ze[count_overlap_nloc]; k++) {
                                k_DM = k - DMVertices[4];
                                z = k * pSPARC->delta_z - z0_i;
                                for (j = Atom_Influence_nloc_temp.ys[count_overlap_nloc]; j <= Atom_Influence_nloc_temp.ye[count_overlap_nloc]; j++) {
                                    j_DM = j - DMVertices[2];
                                    y = j * pSPARC->delta_y - y0_i;
                                    for (i = Atom_Influence_nloc_temp.xs[count_overlap_nloc]; i <= Atom_Influence_nloc_temp.xe[count_overlap_nloc]; i++) {
                                        i_DM = i - DMVertices[0];
                                        x = i * pSPARC->delta_x - x0_i;
                                        r2 = pSPARC->metricT[0] * (x*x) + pSPARC->metricT[1] * (x*y) + pSPARC->metricT[2] * (x*z) 
                                           + pSPARC->metricT[4] * (y*y) + pSPARC->metricT[5] * (y*z) + pSPARC->metricT[8] * (z*z);
                                        if (r2 <= rc2) {
                                            Atom_Influence_nloc_temp.grid_pos[count_overlap_nloc][count] = k_DM * (DMnx * DMny) + j_DM * DMnx + i_DM;
                                            count++;
                                        }
                                    }
                                }
                            } 
                        } else {
                            count = 0;
                        }   
                        // TODO: in some cases count is 0! check if ndc is 0 and remove those!
                        Atom_Influence_nloc_temp.ndc[count_overlap_nloc] = count;
                        count_overlap_nloc++;
                        
                        if (count > 0) {
                            count_overlap_nloc_sphere++;
                        }
                    }
                }
            }
        }
        
        if (count_overlap_nloc_sphere == 0) {
            (*Atom_Influence_nloc)[ityp].n_atom = 0;
            atmcount2 = atmcount;
            free(Atom_Influence_nloc_temp.coords);
            free(Atom_Influence_nloc_temp.atom_index);
            free(Atom_Influence_nloc_temp.xs);
            free(Atom_Influence_nloc_temp.ys);
            free(Atom_Influence_nloc_temp.zs);
            free(Atom_Influence_nloc_temp.xe);
            free(Atom_Influence_nloc_temp.ye);
            free(Atom_Influence_nloc_temp.ze);
            free(Atom_Influence_nloc_temp.ndc);
            for (i = 0; i < count_overlap_nloc; i++) {
                free(Atom_Influence_nloc_temp.grid_pos[i]);
            }
            free(Atom_Influence_nloc_temp.grid_pos);
            continue;
        }

        (*Atom_Influence_nloc)[ityp].n_atom = count_overlap_nloc_sphere;
        (*Atom_Influence_nloc)[ityp].coords = (double *)malloc(sizeof(double) * count_overlap_nloc_sphere * 3);
        (*Atom_Influence_nloc)[ityp].atom_index = (int *)malloc(sizeof(int) * count_overlap_nloc_sphere);
        (*Atom_Influence_nloc)[ityp].xs = (int *)malloc(sizeof(int) * count_overlap_nloc_sphere);
        (*Atom_Influence_nloc)[ityp].ys = (int *)malloc(sizeof(int) * count_overlap_nloc_sphere);
        (*Atom_Influence_nloc)[ityp].zs = (int *)malloc(sizeof(int) * count_overlap_nloc_sphere);
        (*Atom_Influence_nloc)[ityp].xe = (int *)malloc(sizeof(int) * count_overlap_nloc_sphere);
        (*Atom_Influence_nloc)[ityp].ye = (int *)malloc(sizeof(int) * count_overlap_nloc_sphere);
        (*Atom_Influence_nloc)[ityp].ze = (int *)malloc(sizeof(int) * count_overlap_nloc_sphere);
        (*Atom_Influence_nloc)[ityp].ndc = (int *)malloc(sizeof(int) * count_overlap_nloc_sphere);
        (*Atom_Influence_nloc)[ityp].grid_pos = (int **)malloc(sizeof(int*) * count_overlap_nloc_sphere);
        
        count = 0;
        for (i = 0; i < count_overlap_nloc; i++) {
            if ( Atom_Influence_nloc_temp.ndc[i] > 0 ) {
                ndc = Atom_Influence_nloc_temp.ndc[i];
                (*Atom_Influence_nloc)[ityp].coords[count*3] = Atom_Influence_nloc_temp.coords[i*3];
                (*Atom_Influence_nloc)[ityp].coords[count*3+1] = Atom_Influence_nloc_temp.coords[i*3+1];
                (*Atom_Influence_nloc)[ityp].coords[count*3+2] = Atom_Influence_nloc_temp.coords[i*3+2];
                (*Atom_Influence_nloc)[ityp].atom_index[count] = Atom_Influence_nloc_temp.atom_index[i];
                (*Atom_Influence_nloc)[ityp].xs[count] = Atom_Influence_nloc_temp.xs[i];
                (*Atom_Influence_nloc)[ityp].ys[count] = Atom_Influence_nloc_temp.ys[i];
                (*Atom_Influence_nloc)[ityp].zs[count] = Atom_Influence_nloc_temp.zs[i];
                (*Atom_Influence_nloc)[ityp].xe[count] = Atom_Influence_nloc_temp.xe[i];
                (*Atom_Influence_nloc)[ityp].ye[count] = Atom_Influence_nloc_temp.ye[i];
                (*Atom_Influence_nloc)[ityp].ze[count] = Atom_Influence_nloc_temp.ze[i];
                (*Atom_Influence_nloc)[ityp].ndc[count] = Atom_Influence_nloc_temp.ndc[i];
                (*Atom_Influence_nloc)[ityp].grid_pos[count] = (int *)malloc(sizeof(int) * ndc);
                for (j = 0; j < ndc; j++) {
                    (*Atom_Influence_nloc)[ityp].grid_pos[count][j] = Atom_Influence_nloc_temp.grid_pos[i][j];
                }
                count++;
            }
            free(Atom_Influence_nloc_temp.grid_pos[i]);
        }
        
        free(Atom_Influence_nloc_temp.coords);
        free(Atom_Influence_nloc_temp.atom_index);
        free(Atom_Influence_nloc_temp.xs);
        free(Atom_Influence_nloc_temp.ys);
        free(Atom_Influence_nloc_temp.zs);
        free(Atom_Influence_nloc_temp.xe);
        free(Atom_Influence_nloc_temp.ye);
        free(Atom_Influence_nloc_temp.ze);
        free(Atom_Influence_nloc_temp.ndc);
        free(Atom_Influence_nloc_temp.grid_pos);
    }
    
#ifdef DEBUG
    t2 = MPI_Wtime();
    if(!rank) printf(GRN"rank = %d, time for nonlocal influencing atoms: %.3f ms\n"RESET, rank, (t2-t1)*1e3);
#endif
}




/**
 * @brief   Calculate nonlocal projectors. 
 */
void CalculateNonlocalProjectors(SPARC_OBJ *pSPARC, NLOC_PROJ_OBJ **nlocProj, 
        ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc, int *DMVertices, MPI_Comm comm)
{
    // processors that are not in the dmcomm will continue
    if (comm == MPI_COMM_NULL) {
        return; // upon input, make sure process with bandcomm_index < 0 provides MPI_COMM_NULL
    }
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double t1, t2, t_tot, t_old;

    t_tot = t_old = 0.0;
#ifdef DEBUG
    if (rank == 0) printf("Calculate nonlocal projectors ... \n");
#endif    
    int l, np, lcount, lcount2, m, psd_len, col_count, indx, ityp, iat, ipos, ndc, lloc, lmax, DMnx, DMny;
    int i_DM, j_DM, k_DM;
    double x0_i, y0_i, z0_i, *rc_pos_x, *rc_pos_y, *rc_pos_z, *rc_pos_r, *Ylm, *UdV_sort, x2, y2, z2, x, y, z;
    
    // number of nodes in the local distributed domain
    DMnx = DMVertices[1] - DMVertices[0] + 1;
    DMny = DMVertices[3] - DMVertices[2] + 1;

    (*nlocProj) = (NLOC_PROJ_OBJ *)malloc( sizeof(NLOC_PROJ_OBJ) * pSPARC->Ntypes ); // TODO: deallocate!!
    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) { 
        // allocate memory for projectors
        (*nlocProj)[ityp].Chi = (double **)malloc( sizeof(double *) * Atom_Influence_nloc[ityp].n_atom);
        lloc = pSPARC->localPsd[ityp]; // local projector index
        lmax = pSPARC->psd[ityp].lmax;
        psd_len = pSPARC->psd[ityp].size;
        // number of projectors per atom
        (*nlocProj)[ityp].nproj = 0;
        for (l = 0; l <= lmax; l++) {
            if (l == lloc) continue;
            (*nlocProj)[ityp].nproj += pSPARC->psd[ityp].ppl[l] * (2 * l + 1);
        }
        if (! (*nlocProj)[ityp].nproj) continue;
        for (iat = 0; iat < Atom_Influence_nloc[ityp].n_atom; iat++) {
            // store coordinates of the overlapping atom
            x0_i = Atom_Influence_nloc[ityp].coords[iat*3  ];
            y0_i = Atom_Influence_nloc[ityp].coords[iat*3+1];
            z0_i = Atom_Influence_nloc[ityp].coords[iat*3+2];
            // grid nodes in (spherical) rc-domain
            ndc = Atom_Influence_nloc[ityp].ndc[iat]; 
            (*nlocProj)[ityp].Chi[iat] = (double *)malloc( sizeof(double) * ndc * (*nlocProj)[ityp].nproj); 
            rc_pos_x = (double *)malloc( sizeof(double) * ndc );
            rc_pos_y = (double *)malloc( sizeof(double) * ndc );
            rc_pos_z = (double *)malloc( sizeof(double) * ndc );
            rc_pos_r = (double *)malloc( sizeof(double) * ndc );
            Ylm = (double *)malloc( sizeof(double) * ndc );
            UdV_sort = (double *)malloc( sizeof(double) * ndc );
            // use spline to fit UdV
            if(pSPARC->cell_typ == 0){
                for (ipos = 0; ipos < ndc; ipos++) {
                    indx = Atom_Influence_nloc[ityp].grid_pos[iat][ipos];
                    k_DM = indx / (DMnx * DMny);
                    j_DM = (indx - k_DM * (DMnx * DMny)) / DMnx;
                    i_DM = indx % DMnx;
                    x2 = (i_DM + DMVertices[0]) * pSPARC->delta_x - x0_i;
                    y2 = (j_DM + DMVertices[2]) * pSPARC->delta_y - y0_i;
                    z2 = (k_DM + DMVertices[4]) * pSPARC->delta_z - z0_i;
                    rc_pos_x[ipos] = x2;
                    rc_pos_y[ipos] = y2;
                    rc_pos_z[ipos] = z2;
                    x2 *= x2; y2 *= y2; z2 *= z2;
                    rc_pos_r[ipos] = sqrt(x2+y2+z2);
                }
            } else if(pSPARC->cell_typ > 10 && pSPARC->cell_typ < 20) {
                for (ipos = 0; ipos < ndc; ipos++) {
                    indx = Atom_Influence_nloc[ityp].grid_pos[iat][ipos];
                    k_DM = indx / (DMnx * DMny);
                    j_DM = (indx - k_DM * (DMnx * DMny)) / DMnx;
                    i_DM = indx % DMnx;
                    x = (i_DM + DMVertices[0]) * pSPARC->delta_x - x0_i;
                    y = (j_DM + DMVertices[2]) * pSPARC->delta_y - y0_i;
                    z = (k_DM + DMVertices[4]) * pSPARC->delta_z - z0_i;
                    nonCart2Cart_coord(pSPARC, &x, &y, &z);
                    rc_pos_x[ipos] = x;
                    rc_pos_y[ipos] = y;
                    rc_pos_z[ipos] = z;
                    x2 = x * x; y2 = y * y; z2 = z*z;
                    rc_pos_r[ipos] = sqrt(x2+y2+z2);
                }
            }    

            lcount = lcount2 = col_count = 0;
            // multiply spherical harmonics and UdV
            for (l = 0; l <= lmax; l++) {
                // skip the local projector
                if  (l == lloc) { lcount2 += pSPARC->psd[ityp].ppl[l]; continue;}
                for (np = 0; np < pSPARC->psd[ityp].ppl[l]; np++) {
                    // find UdV using spline interpolation
					if (pSPARC->psd[ityp].is_r_uniform == 1) {
						SplineInterpUniform(pSPARC->psd[ityp].RadialGrid, pSPARC->psd[ityp].UdV+lcount2*psd_len, psd_len, 
						                    rc_pos_r, UdV_sort, ndc, pSPARC->psd[ityp].SplineFitUdV+lcount*psd_len);
					} else {
						SplineInterpNonuniform(pSPARC->psd[ityp].RadialGrid, pSPARC->psd[ityp].UdV+lcount2*psd_len, psd_len, 
						                       rc_pos_r, UdV_sort, ndc, pSPARC->psd[ityp].SplineFitUdV+lcount*psd_len); 
					}
                    for (m = -l; m <= l; m++) {
                        t1 = MPI_Wtime();
                        // find spherical harmonics, Ylm
                        RealSphericalHarmonic(ndc, rc_pos_x, rc_pos_y, rc_pos_z, rc_pos_r, l, m, Ylm);
                        t2 = MPI_Wtime();
                        t_tot += t2 - t1;
                        
                        // calculate Chi = UdV * Ylm
                        for (ipos = 0; ipos < ndc; ipos++) {
                            (*nlocProj)[ityp].Chi[iat][col_count*ndc+ipos] = Ylm[ipos] * UdV_sort[ipos];
                        }
                        col_count++;
                    }
                    lcount++; lcount2++;
                }
            }
            free(rc_pos_x);
            free(rc_pos_y);
            free(rc_pos_z);
            free(rc_pos_r);
            free(Ylm);
            free(UdV_sort);
        }
    }
#ifdef DEBUG    
    if(!rank) printf(BLU "rank = %d, Time for spherical harmonics: %.3f ms\n" RESET, rank, t_tot*1e3);
#endif    
}



/**
 * @brief   Calculate nonlocal projectors. 
 */
void CalculateNonlocalProjectors_kpt(SPARC_OBJ *pSPARC, NLOC_PROJ_OBJ **nlocProj, 
        ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc, int *DMVertices, MPI_Comm comm)
{
    // processors that are not in the dmcomm will continue
    if (comm == MPI_COMM_NULL) {
        return; // upon input, make sure process with bandcomm_index < 0 provides MPI_COMM_NULL
    }
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    double t1, t2, t_tot, t_old;

    t_tot = t_old = 0.0;
#ifdef DEBUG
    if (rank == 0) printf("Calculate nonlocal projectors ... \n");
#endif    
    int l, np, lcount, lcount2, m, psd_len, col_count, indx, ityp, iat, ipos, ndc, lloc, lmax, DMnx, DMny;
    int i_DM, j_DM, k_DM;
    double x0_i, y0_i, z0_i, *rc_pos_x, *rc_pos_y, *rc_pos_z, *rc_pos_r, *Ylm, *UdV_sort, x2, y2, z2, x, y, z;
    
    // number of nodes in the local distributed domain
    DMnx = DMVertices[1] - DMVertices[0] + 1;
    DMny = DMVertices[3] - DMVertices[2] + 1;
    // DMnz = DMVertices[5] - DMVertices[4] + 1;  

    (*nlocProj) = (NLOC_PROJ_OBJ *)malloc( sizeof(NLOC_PROJ_OBJ) * pSPARC->Ntypes ); // TODO: deallocate!!
    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) { 
        // allocate memory for projectors
        (*nlocProj)[ityp].Chi_c = (double complex **)malloc( sizeof(double complex *) * Atom_Influence_nloc[ityp].n_atom);
        lloc = pSPARC->localPsd[ityp]; // local projector index
        lmax = pSPARC->psd[ityp].lmax;
        psd_len = pSPARC->psd[ityp].size;
        // number of projectors per atom
        (*nlocProj)[ityp].nproj = 0;
        for (l = 0; l <= lmax; l++) {
            if (l == lloc) continue;
            (*nlocProj)[ityp].nproj += pSPARC->psd[ityp].ppl[l] * (2 * l + 1);
        }
        if (! (*nlocProj)[ityp].nproj) continue;
        for (iat = 0; iat < Atom_Influence_nloc[ityp].n_atom; iat++) {
            // store coordinates of the overlapping atom
            x0_i = Atom_Influence_nloc[ityp].coords[iat*3  ];
            y0_i = Atom_Influence_nloc[ityp].coords[iat*3+1];
            z0_i = Atom_Influence_nloc[ityp].coords[iat*3+2];
            // grid nodes in (spherical) rc-domain
            ndc = Atom_Influence_nloc[ityp].ndc[iat]; 
            (*nlocProj)[ityp].Chi_c[iat] = (double complex *)malloc( sizeof(double complex) * ndc * (*nlocProj)[ityp].nproj);
            rc_pos_x = (double *)malloc( sizeof(double) * ndc );
            rc_pos_y = (double *)malloc( sizeof(double) * ndc );
            rc_pos_z = (double *)malloc( sizeof(double) * ndc );
            rc_pos_r = (double *)malloc( sizeof(double) * ndc );
            Ylm = (double *)malloc( sizeof(double) * ndc );
            UdV_sort = (double *)malloc( sizeof(double) * ndc );
            // use spline to fit UdV
            if(pSPARC->cell_typ == 0){
                for (ipos = 0; ipos < ndc; ipos++) {
                    indx = Atom_Influence_nloc[ityp].grid_pos[iat][ipos];
                    k_DM = indx / (DMnx * DMny);
                    j_DM = (indx - k_DM * (DMnx * DMny)) / DMnx;
                    i_DM = indx % DMnx;
                    x2 = (i_DM + DMVertices[0]) * pSPARC->delta_x - x0_i;
                    y2 = (j_DM + DMVertices[2]) * pSPARC->delta_y - y0_i;
                    z2 = (k_DM + DMVertices[4]) * pSPARC->delta_z - z0_i;
                    rc_pos_x[ipos] = x2;
                    rc_pos_y[ipos] = y2;
                    rc_pos_z[ipos] = z2;
                    x2 *= x2; y2 *= y2; z2 *= z2;
                    rc_pos_r[ipos] = sqrt(x2+y2+z2);
                }
            } else if(pSPARC->cell_typ > 10 && pSPARC->cell_typ < 20) {
                for (ipos = 0; ipos < ndc; ipos++) {
                    indx = Atom_Influence_nloc[ityp].grid_pos[iat][ipos];
                    k_DM = indx / (DMnx * DMny);
                    j_DM = (indx - k_DM * (DMnx * DMny)) / DMnx;
                    i_DM = indx % DMnx;
                    x = (i_DM + DMVertices[0]) * pSPARC->delta_x - x0_i;
                    y = (j_DM + DMVertices[2]) * pSPARC->delta_y - y0_i;
                    z = (k_DM + DMVertices[4]) * pSPARC->delta_z - z0_i;
                    nonCart2Cart_coord(pSPARC, &x, &y, &z);
                    rc_pos_x[ipos] = x;
                    rc_pos_y[ipos] = y;
                    rc_pos_z[ipos] = z;
                    x2 = x * x; y2 = y * y; z2 = z*z;
                    rc_pos_r[ipos] = sqrt(x2+y2+z2);
                }
            }    

            lcount = lcount2 = col_count = 0;
            // multiply spherical harmonics and UdV
            for (l = 0; l <= lmax; l++) {
                // skip the local projector
                if  (l == lloc) { lcount2 += pSPARC->psd[ityp].ppl[l]; continue;}
                for (np = 0; np < pSPARC->psd[ityp].ppl[l]; np++) {
                    // find UdV using spline interpolation
                    if (pSPARC->psd[ityp].is_r_uniform == 1) {
						SplineInterpUniform(pSPARC->psd[ityp].RadialGrid, pSPARC->psd[ityp].UdV+lcount2*psd_len, psd_len, 
						                    rc_pos_r, UdV_sort, ndc, pSPARC->psd[ityp].SplineFitUdV+lcount*psd_len);
					} else {
						SplineInterpNonuniform(pSPARC->psd[ityp].RadialGrid, pSPARC->psd[ityp].UdV+lcount2*psd_len, psd_len, 
						                       rc_pos_r, UdV_sort, ndc, pSPARC->psd[ityp].SplineFitUdV+lcount*psd_len); 
					}
                    for (m = -l; m <= l; m++) {
                        t1 = MPI_Wtime();
                        // find spherical harmonics, Ylm
                        RealSphericalHarmonic(ndc, rc_pos_x, rc_pos_y, rc_pos_z, rc_pos_r, l, m, Ylm);
                        t2 = MPI_Wtime();
                        t_tot += t2 - t1;
                        // calculate Chi = UdV * Ylm
                        for (ipos = 0; ipos < ndc; ipos++) {
                            (*nlocProj)[ityp].Chi_c[iat][col_count*ndc+ipos] = Ylm[ipos] * UdV_sort[ipos];
                        }
                        col_count++;
                    }
                    lcount++; lcount2++;
                }
            }

            free(rc_pos_x);
            free(rc_pos_y);
            free(rc_pos_z);
            free(rc_pos_r);
            free(Ylm);
            free(UdV_sort);
        }
    }
    
#ifdef DEBUG    
    if(!rank) printf(BLU"rank = %d, Time for spherical harmonics: %.3f ms\n"RESET, rank, t_tot*1e3);
#endif    
}



/**
 * @brief   Calculate indices for storing nonlocal inner product in an array. 
 *
 *          We will store the inner product < Chi_Jlm, x_n > in a continuous array "alpha",
 *          the dimensions are in the order: <lm>, n, J. Here we find out the sizes of the 
 *          inner product corresponding to atom J, and the total number of inner products
 *          corresponding to each vector x_n.
 */
void CalculateNonlocalInnerProductIndex(SPARC_OBJ *pSPARC)
{
    int ityp, iat, l, lmax, lloc, atom_index, nproj;

    pSPARC->IP_displ = (int *)malloc( sizeof(int) * (pSPARC->n_atom+1));
    
    atom_index = 0;
    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        lmax = pSPARC->psd[ityp].lmax;
        lloc = pSPARC->localPsd[ityp];
        // number of projectors per atom (of this type)
        nproj = 0;
        for (l = 0; l <= lmax; l++) {
            if (l == lloc) continue;
            nproj += pSPARC->psd[ityp].ppl[l] * (2 * l + 1);
        }
        pSPARC->IP_displ[0] = 0;
        for (iat = 0; iat < pSPARC->nAtomv[ityp]; iat++) {
            pSPARC->IP_displ[atom_index+1] = pSPARC->IP_displ[atom_index] + nproj;
            atom_index++;
        }
    }
}



/**
 * @brief   Calculate Vnl times vectors in a matrix-free way.
 */
void Vnl_vec_mult(const SPARC_OBJ *pSPARC, int DMnd, ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc, 
                  NLOC_PROJ_OBJ *nlocProj, int ncol, double *x, double *Hx, MPI_Comm comm)
{
    int i, n, np, count;
    /* compute nonlocal operator times vector(s) */
    int ityp, iat, l, m, ldispl, lmax, ndc, atom_index;
    double *alpha, *x_rc, *Vnlx;
    alpha = (double *)calloc( pSPARC->IP_displ[pSPARC->n_atom] * ncol, sizeof(double));
    //first find inner product
    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        //int lloc = pSPARC->localPsd[ityp];
        //lmax = pSPARC->psd[ityp].lmax;
        if (! nlocProj[ityp].nproj) continue; // this is typical for hydrogen
        for (iat = 0; iat < Atom_Influence_nloc[ityp].n_atom; iat++) {
            ndc = Atom_Influence_nloc[ityp].ndc[iat]; 
            x_rc = (double *)malloc( ndc * ncol * sizeof(double));
            atom_index = Atom_Influence_nloc[ityp].atom_index[iat];
            for (n = 0; n < ncol; n++) {
                for (i = 0; i < ndc; i++) {
                    x_rc[n*ndc+i] = x[n*DMnd + Atom_Influence_nloc[ityp].grid_pos[iat][i]];
                }
            }
            cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, nlocProj[ityp].nproj, ncol, ndc, 
                pSPARC->dV, nlocProj[ityp].Chi[iat], ndc, x_rc, ndc, 1.0, 
                alpha+pSPARC->IP_displ[atom_index]*ncol, nlocProj[ityp].nproj);          
            free(x_rc);
        }
    }

    // if there are domain parallelization over each band, we need to sum over all processes over domain comm
    int commsize;
    MPI_Comm_size(comm, &commsize);
    if (commsize > 1) {
        MPI_Allreduce(MPI_IN_PLACE, alpha, pSPARC->IP_displ[pSPARC->n_atom] * ncol, MPI_DOUBLE, MPI_SUM, comm);
    }
    
    // go over all atoms and multiply gamma_Jl to the inner product
    count = 0;
    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        int lloc = pSPARC->localPsd[ityp];
        lmax = pSPARC->psd[ityp].lmax;
        for (iat = 0; iat < pSPARC->nAtomv[ityp]; iat++) {
            for (n = 0; n < ncol; n++) {
                ldispl = 0;
                for (l = 0; l <= lmax; l++) {
                    // skip the local l
                    if (l == lloc) {
                        ldispl += pSPARC->psd[ityp].ppl[l];
                        continue;
                    }
                    for (np = 0; np < pSPARC->psd[ityp].ppl[l]; np++) {
                        for (m = -l; m <= l; m++) {
                            alpha[count++] *= pSPARC->psd[ityp].Gamma[ldispl+np];
                        }
                    }
                    ldispl += pSPARC->psd[ityp].ppl[l];
                }
            }
        }
    }

    // multiply the inner product and the nonlocal projector
    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        if (! nlocProj[ityp].nproj) continue; // this is typical for hydrogen
        for (iat = 0; iat < Atom_Influence_nloc[ityp].n_atom; iat++) {
            ndc = Atom_Influence_nloc[ityp].ndc[iat]; 
            atom_index = Atom_Influence_nloc[ityp].atom_index[iat];
            Vnlx = (double *)malloc( ndc * ncol * sizeof(double));
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, ndc, ncol, nlocProj[ityp].nproj, 1.0, nlocProj[ityp].Chi[iat], ndc, 
                        alpha+pSPARC->IP_displ[atom_index]*ncol, nlocProj[ityp].nproj, 0.0, Vnlx, ndc); 
            for (n = 0; n < ncol; n++) {
                for (i = 0; i < ndc; i++) {
                    Hx[n*DMnd + Atom_Influence_nloc[ityp].grid_pos[iat][i]] += Vnlx[n*ndc+i];
                }
            }     
            free(Vnlx);
        }
    }
    free(alpha);
}


/**
 * @brief   Calculate Vnl times vectors in a matrix-free way with Bloch factor
 */
void Vnl_vec_mult_kpt(const SPARC_OBJ *pSPARC, int DMnd, ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc, 
                      NLOC_PROJ_OBJ *nlocProj, int ncol, double complex *x, double complex *Hx, int kpt, MPI_Comm comm)
{
    int i, n, np, count;
    /* compute nonlocal operator times vector(s) */
    int ityp, iat, l, m, ldispl, lmax, ndc, atom_index;
    double x0_i, y0_i, z0_i;
    double complex *alpha, *x_rc, *Vnlx;
    alpha = (double complex *) calloc( pSPARC->IP_displ[pSPARC->n_atom] * ncol, sizeof(double complex));
    double Lx = pSPARC->range_x;
    double Ly = pSPARC->range_y;
    double Lz = pSPARC->range_z;
    double k1 = pSPARC->k1_loc[kpt];
    double k2 = pSPARC->k2_loc[kpt];
    double k3 = pSPARC->k3_loc[kpt];
    double theta;
    double complex bloch_fac, a, b;
    
    //first find inner product
    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        if (! nlocProj[ityp].nproj) continue; // this is typical for hydrogen
        for (iat = 0; iat < Atom_Influence_nloc[ityp].n_atom; iat++) {
            x0_i = Atom_Influence_nloc[ityp].coords[iat*3  ];
            y0_i = Atom_Influence_nloc[ityp].coords[iat*3+1];
            z0_i = Atom_Influence_nloc[ityp].coords[iat*3+2];
            theta = -k1 * (floor(x0_i/Lx) * Lx) - k2 * (floor(y0_i/Ly) * Ly) - k3 * (floor(z0_i/Lz) * Lz);
            bloch_fac = cos(theta) + sin(theta) * I;
            a = bloch_fac * pSPARC->dV;
            b = 1.0;
            ndc = Atom_Influence_nloc[ityp].ndc[iat]; 
            x_rc = (double complex *)malloc( ndc * ncol * sizeof(double complex));
            atom_index = Atom_Influence_nloc[ityp].atom_index[iat];
            for (n = 0; n < ncol; n++) {
                for (i = 0; i < ndc; i++) {
                    x_rc[n*ndc+i] = x[n*DMnd + Atom_Influence_nloc[ityp].grid_pos[iat][i]];
                }
            }
            cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, nlocProj[ityp].nproj, ncol, ndc, 
                &a, nlocProj[ityp].Chi_c[iat], ndc, x_rc, ndc, &b, 
                alpha+pSPARC->IP_displ[atom_index]*ncol, nlocProj[ityp].nproj);
            free(x_rc);
        }
    }

    // if there are domain parallelization over each band, we need to sum over all processes over domain comm
    int commsize;
    MPI_Comm_size(comm, &commsize);
    if (commsize > 1) {
        MPI_Allreduce(MPI_IN_PLACE, alpha, pSPARC->IP_displ[pSPARC->n_atom] * ncol, MPI_DOUBLE_COMPLEX, MPI_SUM, comm);
    }
    
    // go over all atoms and multiply gamma_Jl to the inner product
    count = 0;
    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        int lloc = pSPARC->localPsd[ityp];
        lmax = pSPARC->psd[ityp].lmax;
        for (iat = 0; iat < pSPARC->nAtomv[ityp]; iat++) {
            for (n = 0; n < ncol; n++) {
                ldispl = 0;
                for (l = 0; l <= lmax; l++) {
                    // skip the local l
                    if (l == lloc) {
                        ldispl += pSPARC->psd[ityp].ppl[l];
                        continue;
                    }
                    for (np = 0; np < pSPARC->psd[ityp].ppl[l]; np++) {
                        for (m = -l; m <= l; m++) {
                            alpha[count++] *= pSPARC->psd[ityp].Gamma[ldispl+np];
                        }
                    }
                    ldispl += pSPARC->psd[ityp].ppl[l];
                }
            }
        }
    }
        
    // multiply the inner product and the nonlocal projector
    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        if (! nlocProj[ityp].nproj) continue; // this is typical for hydrogen
        for (iat = 0; iat < Atom_Influence_nloc[ityp].n_atom; iat++) {
            x0_i = Atom_Influence_nloc[ityp].coords[iat*3  ];
            y0_i = Atom_Influence_nloc[ityp].coords[iat*3+1];
            z0_i = Atom_Influence_nloc[ityp].coords[iat*3+2];
            theta = -k1 * (floor(x0_i/Lx) * Lx) - k2 * (floor(y0_i/Ly) * Ly) - k3 * (floor(z0_i/Lz) * Lz);
            bloch_fac = cos(theta) - sin(theta) * I;
            b = 0.0;
            ndc = Atom_Influence_nloc[ityp].ndc[iat]; 
            atom_index = Atom_Influence_nloc[ityp].atom_index[iat];
            Vnlx = (double complex *)malloc( ndc * ncol * sizeof(double complex));
            cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, ndc, ncol, nlocProj[ityp].nproj, &bloch_fac, nlocProj[ityp].Chi_c[iat], ndc, 
                          alpha+pSPARC->IP_displ[atom_index]*ncol, nlocProj[ityp].nproj, &b, Vnlx, ndc); 
            for (n = 0; n < ncol; n++) {
                for (i = 0; i < ndc; i++) {
                    Hx[n*DMnd + Atom_Influence_nloc[ityp].grid_pos[iat][i]] += Vnlx[n*ndc+i];
                }
            }
            free(Vnlx);
        }
    }
    free(alpha);
}
