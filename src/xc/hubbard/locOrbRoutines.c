/**
 * @file locOrbRoutines.c 
 * @brief This file contains the routines pertaining with local orbitals in the context of DFT+U
 * @author  Sayan Bhowmik <sbhowmik9@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Reference:   Dudarev, S. L., et al. "Electron-energy-loss spectra and the structural stability of nickel oxide:  An LSDA+U study"
 *              Phys. Rev. B 57, 1505
 */

#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <string.h>
/* BLAS routines */
#ifdef USE_MKL
#include <mkl.h> // for cblas_* functions
#else
#include <cblas.h>
#endif

#include "locOrbRoutines.h"
#include "tools.h"
#include "isddft.h"
#include "initialization.h"

#define TEMP_TOL 1e-12

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) > (b) ? (b) : (a))

/**
 * @brief   Find the list of all atoms that influence the processor
 *          domain in psi-domain.
 */
void GetInfluencingAtoms_loc(SPARC_OBJ *pSPARC, ATOM_LOC_INFLUENCE_OBJ **Atom_Influence_loc, int *DMVertices, MPI_Comm comm)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#ifdef DEBUG
    if (rank == 0)
        printf("Finding atoms that has local orbitals influence on the local process domain ... \n");
#endif
    // processors that are not in the dmcomm will remain idle
    // on input make comm of processes with bandcomm_index or kptcomm_index MPI_COMM_NULL!
    if (comm == MPI_COMM_NULL)
    {
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
    int count_overlap_loc, count_overlap_loc_sphere, ityp, i, j, k, count, i_DM, j_DM, k_DM, 
        iat, atmcount, atmcount2, DMnx, DMny;
    int pp, qq, rr, ppmin, ppmax, qqmin, qqmax, rrmin, rrmax;
    int rc_xl, rc_xr, rc_yl, rc_yr, rc_zl, rc_zr, ndc;

    Lx = pSPARC->range_x;
    Ly = pSPARC->range_y;
    Lz = pSPARC->range_z;
    
    // TODO: notice the difference here from rb-domain, since local projectors decay drastically, using the fd-nodes as edges are more appropriate
    DMxs = pSPARC->xin + DMVertices[0] * pSPARC->delta_x;
    DMxe = pSPARC->xin + (DMVertices[1]) * pSPARC->delta_x; // note that this is not the actual edge, add BCx to get actual domain edge
    DMys = DMVertices[2] * pSPARC->delta_y;
    DMye = (DMVertices[3]) * pSPARC->delta_y; // note that this is not the actual edge, add BCx to get actual domain edge
    DMzs = DMVertices[4] * pSPARC->delta_z;
    DMze = (DMVertices[5]) * pSPARC->delta_z; // note that this is not the actual edge, add BCx to get actual domain edge
    
    // number of nodes in the local distributed domain
    DMnx = DMVertices[1] - DMVertices[0] + 1;
    DMny = DMVertices[3] - DMVertices[2] + 1;


    *Atom_Influence_loc = (ATOM_LOC_INFLUENCE_OBJ *)malloc(sizeof(ATOM_LOC_INFLUENCE_OBJ) * pSPARC->Ntypes);

    ATOM_LOC_INFLUENCE_OBJ Atom_Influence_loc_temp;

    // find which atoms have local influence on the distributed domain owned by current process
    atmcount = 0; atmcount2 = 0;
    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++)
    {
        if (!pSPARC->atom_solve_flag[ityp]){ // condition to check if U corrections are desired for atomtype
            atmcount += pSPARC->nAtomv[ityp];
            atmcount2 += pSPARC->nAtomv[ityp];
            continue;
        }

        // find rc for each U type atom - done during U initialization
        rcbox_x = pSPARC->AtmU[ityp].rc[0];
        rcbox_y = pSPARC->AtmU[ityp].rc[1];
        rcbox_z = pSPARC->AtmU[ityp].rc[2];

        // rc = max(max(rcbox_x,rcbox_y),rcbox_z);
        rc2 = pSPARC->AtmU[ityp].rc2;

        // first loop over all atoms of each type to find number of influencing atoms
        count_overlap_loc = 0;
        for (i = 0; i < pSPARC->nAtomv[ityp]; i++) {
            // get atom positions
            x0 = pSPARC->atom_pos[3*atmcount];
            y0 = pSPARC->atom_pos[3*atmcount+1];
            z0 = pSPARC->atom_pos[3*atmcount+2];
            atmcount++;
            ppmin = ppmax = qqmin = qqmax = rrmin = rrmax = 0;
            if (pSPARC->BCx == 0) {
                if (pSPARC->sqAmbientFlag == 1 || pSPARC->sqHighTFlag == 1) {
  
                } else {
                    ppmax = floor((rcbox_x + Lx - x0) / Lx + TEMP_TOL);
                    ppmin = -floor((rcbox_x + x0) / Lx + TEMP_TOL);
                }
            }
            if (pSPARC->BCy == 0) {
                if (pSPARC->sqAmbientFlag == 1 || pSPARC->sqHighTFlag == 1) {

                } else {
                    qqmax = floor((rcbox_y + Ly - y0) / Ly + TEMP_TOL);
                    qqmin = -floor((rcbox_y + y0) / Ly + TEMP_TOL);
                }
            }
            if (pSPARC->BCz == 0) {
                if (pSPARC->sqAmbientFlag == 1 || pSPARC->sqHighTFlag == 1) {

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
                        count_overlap_loc++;
                    }
                }
            }
        } // end for loop over atoms of each type, for the first time

        Atom_Influence_loc_temp.n_atom = count_overlap_loc;
        Atom_Influence_loc_temp.coords = (double *)malloc(sizeof(double) * count_overlap_loc * 3);
        Atom_Influence_loc_temp.atom_index = (int *)malloc(sizeof(int) * count_overlap_loc);
        Atom_Influence_loc_temp.xs = (int *)malloc(sizeof(int) * count_overlap_loc);
        Atom_Influence_loc_temp.ys = (int *)malloc(sizeof(int) * count_overlap_loc);
        Atom_Influence_loc_temp.zs = (int *)malloc(sizeof(int) * count_overlap_loc);
        Atom_Influence_loc_temp.xe = (int *)malloc(sizeof(int) * count_overlap_loc);
        Atom_Influence_loc_temp.ye = (int *)malloc(sizeof(int) * count_overlap_loc);
        Atom_Influence_loc_temp.ze = (int *)malloc(sizeof(int) * count_overlap_loc);
        Atom_Influence_loc_temp.ndc = (int *)malloc(sizeof(int) * count_overlap_loc);
        Atom_Influence_loc_temp.grid_pos = (int **)malloc(sizeof(int*) * count_overlap_loc);

        // when there's no atom of this type that have influence, go to next type of atom
        if (Atom_Influence_loc_temp.n_atom == 0) {
            (*Atom_Influence_loc)[ityp].n_atom = 0;
            atmcount2 = atmcount;
            free(Atom_Influence_loc_temp.coords);
            free(Atom_Influence_loc_temp.atom_index);
            free(Atom_Influence_loc_temp.xs);
            free(Atom_Influence_loc_temp.ys);
            free(Atom_Influence_loc_temp.zs);
            free(Atom_Influence_loc_temp.xe);
            free(Atom_Influence_loc_temp.ye);
            free(Atom_Influence_loc_temp.ze);
            free(Atom_Influence_loc_temp.ndc);
            free(Atom_Influence_loc_temp.grid_pos);
            continue;
        }

        // loop over atoms of this type again to find overlapping region and atom info
        count_overlap_loc = 0;
        count_overlap_loc_sphere = 0;
        for (iat = 0; iat < pSPARC->nAtomv[ityp]; iat++) {
            // get atom positions
            x0 = pSPARC->atom_pos[3*atmcount2];
            y0 = pSPARC->atom_pos[3*atmcount2+1];
            z0 = pSPARC->atom_pos[3*atmcount2+2];
            atmcount2++;
            ppmin = ppmax = qqmin = qqmax = rrmin = rrmax = 0;
            if (pSPARC->BCx == 0) {
                if (pSPARC->sqAmbientFlag == 1 || pSPARC->sqHighTFlag == 1) {
  
                } else {
                    ppmax = floor((rcbox_x + Lx - x0) / Lx + TEMP_TOL);
                    ppmin = -floor((rcbox_x + x0) / Lx + TEMP_TOL);
                }
            }
            if (pSPARC->BCy == 0) {
                if (pSPARC->sqAmbientFlag == 1 || pSPARC->sqHighTFlag == 1) {

                } else {
                    qqmax = floor((rcbox_y + Ly - y0) / Ly + TEMP_TOL);
                    qqmin = -floor((rcbox_y + y0) / Ly + TEMP_TOL);
                }
            }
            if (pSPARC->BCz == 0) {
                if (pSPARC->sqAmbientFlag == 1 || pSPARC->sqHighTFlag == 1) {

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
                        Atom_Influence_loc_temp.coords[count_overlap_loc*3  ] = x0_i;
                        Atom_Influence_loc_temp.coords[count_overlap_loc*3+1] = y0_i;
                        Atom_Influence_loc_temp.coords[count_overlap_loc*3+2] = z0_i;
                        
                        // record the original atom index this image atom corresponds to
                        Atom_Influence_loc_temp.atom_index[count_overlap_loc] = atmcount2-1;
                        
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
                        Atom_Influence_loc_temp.xs[count_overlap_loc] = max(DMVertices[0], rc_xl);
                        Atom_Influence_loc_temp.xe[count_overlap_loc] = min(DMVertices[1], rc_xr);
                        Atom_Influence_loc_temp.ys[count_overlap_loc] = max(DMVertices[2], rc_yl);
                        Atom_Influence_loc_temp.ye[count_overlap_loc] = min(DMVertices[3], rc_yr);
                        Atom_Influence_loc_temp.zs[count_overlap_loc] = max(DMVertices[4], rc_zl);
                        Atom_Influence_loc_temp.ze[count_overlap_loc] = min(DMVertices[5], rc_zr);

                        // find the spherical rc-region
                        ndc = (Atom_Influence_loc_temp.xe[count_overlap_loc] - Atom_Influence_loc_temp.xs[count_overlap_loc] + 1)
                            * (Atom_Influence_loc_temp.ye[count_overlap_loc] - Atom_Influence_loc_temp.ys[count_overlap_loc] + 1)
                            * (Atom_Influence_loc_temp.ze[count_overlap_loc] - Atom_Influence_loc_temp.zs[count_overlap_loc] + 1);
                        
                        // first allocate memory for the rectangular rc-region, resize later to the spherical rc-region
                        Atom_Influence_loc_temp.grid_pos[count_overlap_loc] = (int *)malloc(sizeof(int) * ndc);
                        count = 0;
                        for (k = Atom_Influence_loc_temp.zs[count_overlap_loc]; k <= Atom_Influence_loc_temp.ze[count_overlap_loc]; k++) {
                            k_DM = k - DMVertices[4];
                            z = k * pSPARC->delta_z;
                            for (j = Atom_Influence_loc_temp.ys[count_overlap_loc]; j <= Atom_Influence_loc_temp.ye[count_overlap_loc]; j++) {
                                j_DM = j - DMVertices[2];
                                y = j * pSPARC->delta_y;
                                for (i = Atom_Influence_loc_temp.xs[count_overlap_loc]; i <= Atom_Influence_loc_temp.xe[count_overlap_loc]; i++) {
                                    i_DM = i - DMVertices[0];
                                    x = pSPARC->xin + i * pSPARC->delta_x;
                                    CalculateDistance(pSPARC, x, y, z, x0_i, y0_i, z0_i, &r2);
                                    r2 *= r2;
                                    if (r2 <= rc2) {
                                        Atom_Influence_loc_temp.grid_pos[count_overlap_loc][count] = k_DM * (DMnx * DMny) + j_DM * DMnx + i_DM;
                                        count++;
                                    }
                                }
                            }
                        }
                        // TODO: in some cases count is 0! check if ndc is 0 and remove those!
                        Atom_Influence_loc_temp.ndc[count_overlap_loc] = count;
                        count_overlap_loc++;
                        // printf("Img: %d, Count: %d\n",count_overlap_loc-1, count);
                        
                        if (count > 0) {
                            count_overlap_loc_sphere++;
                        }
                    }
                }
            }
        }

        if (count_overlap_loc_sphere == 0)
        {
            (*Atom_Influence_loc)[ityp].n_atom = 0;
            atmcount2 = atmcount;
            free(Atom_Influence_loc_temp.coords);
            free(Atom_Influence_loc_temp.atom_index);
            free(Atom_Influence_loc_temp.xs);
            free(Atom_Influence_loc_temp.ys);
            free(Atom_Influence_loc_temp.zs);
            free(Atom_Influence_loc_temp.xe);
            free(Atom_Influence_loc_temp.ye);
            free(Atom_Influence_loc_temp.ze);
            free(Atom_Influence_loc_temp.ndc);
            for (i = 0; i < count_overlap_loc; i++)
            {
                free(Atom_Influence_loc_temp.grid_pos[i]);
            }
            free(Atom_Influence_loc_temp.grid_pos);
            continue;
        }

        (*Atom_Influence_loc)[ityp].n_atom = count_overlap_loc_sphere;
        (*Atom_Influence_loc)[ityp].coords = (double *)malloc(sizeof(double) * count_overlap_loc_sphere * 3);
        (*Atom_Influence_loc)[ityp].atom_index = (int *)malloc(sizeof(int) * count_overlap_loc_sphere);
        (*Atom_Influence_loc)[ityp].xs = (int *)malloc(sizeof(int) * count_overlap_loc_sphere);
        (*Atom_Influence_loc)[ityp].ys = (int *)malloc(sizeof(int) * count_overlap_loc_sphere);
        (*Atom_Influence_loc)[ityp].zs = (int *)malloc(sizeof(int) * count_overlap_loc_sphere);
        (*Atom_Influence_loc)[ityp].xe = (int *)malloc(sizeof(int) * count_overlap_loc_sphere);
        (*Atom_Influence_loc)[ityp].ye = (int *)malloc(sizeof(int) * count_overlap_loc_sphere);
        (*Atom_Influence_loc)[ityp].ze = (int *)malloc(sizeof(int) * count_overlap_loc_sphere);
        (*Atom_Influence_loc)[ityp].ndc = (int *)malloc(sizeof(int) * count_overlap_loc_sphere);
        (*Atom_Influence_loc)[ityp].grid_pos = (int **)malloc(sizeof(int*) * count_overlap_loc_sphere);


        count = 0;
        for (i = 0; i < count_overlap_loc; i++)
        {
            if (Atom_Influence_loc_temp.ndc[i] > 0)
            {
                ndc = Atom_Influence_loc_temp.ndc[i];
                (*Atom_Influence_loc)[ityp].coords[count * 3] = Atom_Influence_loc_temp.coords[i * 3];
                (*Atom_Influence_loc)[ityp].coords[count * 3 + 1] = Atom_Influence_loc_temp.coords[i * 3 + 1];
                (*Atom_Influence_loc)[ityp].coords[count * 3 + 2] = Atom_Influence_loc_temp.coords[i * 3 + 2];
                (*Atom_Influence_loc)[ityp].atom_index[count] = Atom_Influence_loc_temp.atom_index[i];
                (*Atom_Influence_loc)[ityp].xs[count] = Atom_Influence_loc_temp.xs[i];
                (*Atom_Influence_loc)[ityp].ys[count] = Atom_Influence_loc_temp.ys[i];
                (*Atom_Influence_loc)[ityp].zs[count] = Atom_Influence_loc_temp.zs[i];
                (*Atom_Influence_loc)[ityp].xe[count] = Atom_Influence_loc_temp.xe[i];
                (*Atom_Influence_loc)[ityp].ye[count] = Atom_Influence_loc_temp.ye[i];
                (*Atom_Influence_loc)[ityp].ze[count] = Atom_Influence_loc_temp.ze[i];
                (*Atom_Influence_loc)[ityp].ndc[count] = Atom_Influence_loc_temp.ndc[i];
                (*Atom_Influence_loc)[ityp].grid_pos[count] = (int *)malloc(sizeof(int) * ndc);
                for (j = 0; j < ndc; j++)
                {
                    (*Atom_Influence_loc)[ityp].grid_pos[count][j] = Atom_Influence_loc_temp.grid_pos[i][j];
                }

                count++;
            }
            free(Atom_Influence_loc_temp.grid_pos[i]);
        }

        free(Atom_Influence_loc_temp.coords);
        free(Atom_Influence_loc_temp.atom_index);
        free(Atom_Influence_loc_temp.xs);
        free(Atom_Influence_loc_temp.ys);
        free(Atom_Influence_loc_temp.zs);
        free(Atom_Influence_loc_temp.xe);
        free(Atom_Influence_loc_temp.ye);
        free(Atom_Influence_loc_temp.ze);
        free(Atom_Influence_loc_temp.ndc);
        free(Atom_Influence_loc_temp.grid_pos);
        
    }

#ifdef DEBUG
    t2 = MPI_Wtime();
    if (!rank)
        printf(GRN "rank = %d, time for local influencing atoms: %.3f ms\n" RESET, rank, (t2 - t1) * 1e3);
#endif
}

/**
 * @brief   Calculate local projectors.
 */
void CalculateLocalProjectors(SPARC_OBJ *pSPARC, LOC_PROJ_OBJ **locProj,
                              ATOM_LOC_INFLUENCE_OBJ *Atom_Influence_loc, int *DMVertices, MPI_Comm comm)
{
    // processors that are not in the dmcomm will continue
    if (comm == MPI_COMM_NULL)
    {
        return; // upon input, make sure process with bandcomm_index < 0 provides MPI_COMM_NULL
    }

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double t1, t2, t_tot, t_old;

    t_tot = t_old = 0.0;
#ifdef DEBUG
    if (rank == 0)
        printf("Calculate local orbital projectors ... \n");
#endif
    int l, np, lcount, m, orb_len, col_count, indx, ityp, iat, ipos, ndc, lmax, DMnx, DMny;
    int i_DM, j_DM, k_DM;
    double x0_i, y0_i, z0_i, *rc_pos_x, *rc_pos_y, *rc_pos_z, *rc_pos_r, *Ylm, *UdV_sort, x2, y2, z2, x, y, z;

    // number of nodes in the local distributed domain
    DMnx = DMVertices[1] - DMVertices[0] + 1;
    DMny = DMVertices[3] - DMVertices[2] + 1;

    (*locProj) = (LOC_PROJ_OBJ *)malloc(sizeof(LOC_PROJ_OBJ) * pSPARC->Ntypes);
    double y0, z0, xi, yi, zi, ty, tz;
    double xin = pSPARC->xin + DMVertices[0] * pSPARC->delta_x;

    // Has no cyclix part
    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++)
    {
        if (!pSPARC->atom_solve_flag[ityp]){ // condition to check if U corrections are desired for atomtype
            continue;
        }

        // allocate memory for projectors
        (*locProj)[ityp].Orb = (double **)malloc(sizeof(double *) * Atom_Influence_loc[ityp].n_atom);
        // No cyclix
        lmax = pSPARC->AtmU[ityp].max_l;
        orb_len = pSPARC->AtmU[ityp].size; // vector length of each orbital from atom solve

        // number of local projectors per atom
        (*locProj)[ityp].nproj = 0;
        for (l = 0; l <= lmax; l++)
        {
            if (pSPARC->AtmU[ityp].U[l] == 0)
                continue;                                                      // Skip for zero U values for a particular l orbital
            (*locProj)[ityp].nproj += pSPARC->AtmU[ityp].ppl[l] * (2 * l + 1); // pSPARC->AtmU[ityp].ppl[l] is >1 only when core states are involved
        }
        if (!(*locProj)[ityp].nproj) continue;

        for (iat = 0; iat < Atom_Influence_loc[ityp].n_atom; iat++)
        {
            // store coordinates of the overlapping atom
            x0_i = Atom_Influence_loc[ityp].coords[iat * 3];
            y0_i = Atom_Influence_loc[ityp].coords[iat * 3 + 1];
            z0_i = Atom_Influence_loc[ityp].coords[iat * 3 + 2];
            // grid nodes in (spherical) cutoff domain
            ndc = Atom_Influence_loc[ityp].ndc[iat];
            (*locProj)[ityp].Orb[iat] = (double *)malloc(sizeof(double) * ndc * (*locProj)[ityp].nproj);

            // No cyclix
            rc_pos_x = (double *)malloc( sizeof(double) * ndc );
            rc_pos_y = (double *)malloc( sizeof(double) * ndc );
            rc_pos_z = (double *)malloc( sizeof(double) * ndc );
            rc_pos_r = (double *)malloc( sizeof(double) * ndc );
            Ylm = (double *)malloc( sizeof(double) * ndc );
            UdV_sort = (double *)malloc( sizeof(double) * ndc );

            // use spline to fit UdV
            if (pSPARC->cell_typ == 0)
            {
                for (ipos = 0; ipos < ndc; ipos++)
                {
                    indx = Atom_Influence_loc[ityp].grid_pos[iat][ipos];
                    k_DM = indx / (DMnx * DMny);
                    j_DM = (indx - k_DM * (DMnx * DMny)) / DMnx;
                    i_DM = indx % DMnx;
                    x2 = (i_DM + DMVertices[0]) * pSPARC->delta_x - x0_i;
                    y2 = (j_DM + DMVertices[2]) * pSPARC->delta_y - y0_i;
                    z2 = (k_DM + DMVertices[4]) * pSPARC->delta_z - z0_i;
                    rc_pos_x[ipos] = x2;
                    rc_pos_y[ipos] = y2;
                    rc_pos_z[ipos] = z2;
                    x2 *= x2;
                    y2 *= y2;
                    z2 *= z2;
                    rc_pos_r[ipos] = sqrt(x2 + y2 + z2);
                }
            }
            else if (pSPARC->cell_typ > 10 && pSPARC->cell_typ < 20)
            {
                for (ipos = 0; ipos < ndc; ipos++)
                {
                    indx = Atom_Influence_loc[ityp].grid_pos[iat][ipos];
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
                    x2 = x * x;
                    y2 = y * y;
                    z2 = z * z;
                    rc_pos_r[ipos] = sqrt(x2 + y2 + z2);
                }
            } // No cyclix

            lcount = col_count = 0; 
            // multiply spherical harmonics and UdV
            for (l = 0; l <= lmax; l++)
            {
                if (pSPARC->AtmU[ityp].U[l] == 0){
                    lcount += pSPARC->AtmU[ityp].ppl[l];
                    continue; // Skip for zero U values for a particular l orbital
                }

                for (np = 0; np < pSPARC->AtmU[ityp].ppl[l]; np++) {
                    // find UdV using spline interpolation, radial grid is non uniform
                    SplineInterpNonuniform(pSPARC->AtmU[ityp].RadialGrid, pSPARC->AtmU[ityp].orbitals + lcount * orb_len, orb_len,
                        rc_pos_r, UdV_sort, ndc, pSPARC->AtmU[ityp].SplineFitOrb + lcount * orb_len);

                    for (m = -l; m <= l; m++) {
                        t1 = MPI_Wtime();
                        // find spherical harmonics, Ylm
                        RealSphericalHarmonic(ndc, rc_pos_x, rc_pos_y, rc_pos_z, rc_pos_r, l, m, Ylm);
                        t2 = MPI_Wtime();
                        t_tot += t2 - t1;
                        
                        // calculate Chi = UdV * Ylm
                        for (ipos = 0; ipos < ndc; ipos++) {
                            (*locProj)[ityp].Orb[iat][col_count*ndc+ipos] = Ylm[ipos] * UdV_sort[ipos];
                        }
                        col_count++;
                    }
                    lcount++;
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
    if (!rank)
        printf(BLU "rank = %d, Time for spherical harmonics: %.3f ms\n" RESET, rank, t_tot * 1e3);
#endif
}

/**
 * @brief   Calculate local projectors.
 */
void CalculateLocalProjectors_kpt(SPARC_OBJ *pSPARC, LOC_PROJ_OBJ **locProj,
                              ATOM_LOC_INFLUENCE_OBJ *Atom_Influence_loc, int *DMVertices, MPI_Comm comm)
{
    // processors that are not in the dmcomm will continue
    if (comm == MPI_COMM_NULL)
    {
        return; // upon input, make sure process with bandcomm_index < 0 provides MPI_COMM_NULL
    }

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double t1, t2, t_tot, t_old;

    t_tot = t_old = 0.0;
#ifdef DEBUG
    if (rank == 0)
        printf("Calculate local orbital projectors ... \n");
#endif
    int l, np, lcount, m, orb_len, col_count, indx, ityp, iat, ipos, ndc, lmax, DMnx, DMny;
    int i_DM, j_DM, k_DM;
    double x0_i, y0_i, z0_i, *rc_pos_x, *rc_pos_y, *rc_pos_z, *rc_pos_r, *Ylm, *UdV_sort, x2, y2, z2, x, y, z;

    // number of nodes in the local distributed domain
    DMnx = DMVertices[1] - DMVertices[0] + 1;
    DMny = DMVertices[3] - DMVertices[2] + 1;

    (*locProj) = (LOC_PROJ_OBJ *)malloc(sizeof(LOC_PROJ_OBJ) * pSPARC->Ntypes);
    double y0, z0, xi, yi, zi, ty, tz;
    double xin = pSPARC->xin + DMVertices[0] * pSPARC->delta_x;

    // Has no cyclix part
    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++)
    {
        if (!pSPARC->atom_solve_flag[ityp]){ // condition to check if U corrections are desired for atomtype
            continue;
        }

        // allocate memory for projectors
        (*locProj)[ityp].Orb_c = (double _Complex **)malloc(sizeof(double _Complex *) * Atom_Influence_loc[ityp].n_atom);
        // No cyclix
        lmax = pSPARC->AtmU[ityp].max_l;
        orb_len = pSPARC->AtmU[ityp].size; // vector length of each orbital from atom solve

        // number of local projectors per atom
        (*locProj)[ityp].nproj = 0;
        for (l = 0; l <= lmax; l++)
        {
            if (pSPARC->AtmU[ityp].U[l] == 0)
                continue;                                                      // Skip for zero U values for a particular l orbital
            (*locProj)[ityp].nproj += pSPARC->AtmU[ityp].ppl[l] * (2 * l + 1); // pSPARC->AtmU[ityp].ppl[l] is >1 only when core states are involved
        }
        if (!(*locProj)[ityp].nproj) continue;

        for (iat = 0; iat < Atom_Influence_loc[ityp].n_atom; iat++)
        {
            // store coordinates of the overlapping atom
            x0_i = Atom_Influence_loc[ityp].coords[iat * 3];
            y0_i = Atom_Influence_loc[ityp].coords[iat * 3 + 1];
            z0_i = Atom_Influence_loc[ityp].coords[iat * 3 + 2];
            // grid nodes in (spherical) cutoff domain
            ndc = Atom_Influence_loc[ityp].ndc[iat];
            (*locProj)[ityp].Orb_c[iat] = (double _Complex *)malloc(sizeof(double _Complex) * ndc * (*locProj)[ityp].nproj);

            // No cyclix
            rc_pos_x = (double *)malloc( sizeof(double) * ndc );
            rc_pos_y = (double *)malloc( sizeof(double) * ndc );
            rc_pos_z = (double *)malloc( sizeof(double) * ndc );
            rc_pos_r = (double *)malloc( sizeof(double) * ndc );
            Ylm = (double *)malloc( sizeof(double) * ndc );
            UdV_sort = (double *)malloc( sizeof(double) * ndc );

            // use spline to fit UdV
            if (pSPARC->cell_typ == 0)
            {
                for (ipos = 0; ipos < ndc; ipos++)
                {
                    indx = Atom_Influence_loc[ityp].grid_pos[iat][ipos];
                    k_DM = indx / (DMnx * DMny);
                    j_DM = (indx - k_DM * (DMnx * DMny)) / DMnx;
                    i_DM = indx % DMnx;
                    x2 = (i_DM + DMVertices[0]) * pSPARC->delta_x - x0_i;
                    y2 = (j_DM + DMVertices[2]) * pSPARC->delta_y - y0_i;
                    z2 = (k_DM + DMVertices[4]) * pSPARC->delta_z - z0_i;
                    rc_pos_x[ipos] = x2;
                    rc_pos_y[ipos] = y2;
                    rc_pos_z[ipos] = z2;
                    x2 *= x2;
                    y2 *= y2;
                    z2 *= z2;
                    rc_pos_r[ipos] = sqrt(x2 + y2 + z2);
                }
            }
            else if (pSPARC->cell_typ > 10 && pSPARC->cell_typ < 20)
            {
                for (ipos = 0; ipos < ndc; ipos++)
                {
                    indx = Atom_Influence_loc[ityp].grid_pos[iat][ipos];
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
                    x2 = x * x;
                    y2 = y * y;
                    z2 = z * z;
                    rc_pos_r[ipos] = sqrt(x2 + y2 + z2);
                }
            } // No cyclix

            lcount = col_count = 0; 
            // multiply spherical harmonics and UdV
            for (l = 0; l <= lmax; l++)
            {
                if (pSPARC->AtmU[ityp].U[l] == 0){
                    lcount += pSPARC->AtmU[ityp].ppl[l];
                    continue; // Skip for zero U values for a particular l orbital
                }

                for (np = 0; np < pSPARC->AtmU[ityp].ppl[l]; np++) {
                    // find UdV using spline interpolation, radial grid is non uniform
                    SplineInterpNonuniform(pSPARC->AtmU[ityp].RadialGrid, pSPARC->AtmU[ityp].orbitals + lcount * orb_len, orb_len,
                        rc_pos_r, UdV_sort, ndc, pSPARC->AtmU[ityp].SplineFitOrb + lcount * orb_len);

                    for (m = -l; m <= l; m++) {
                        t1 = MPI_Wtime();
                        // find spherical harmonics, Ylm
                        RealSphericalHarmonic(ndc, rc_pos_x, rc_pos_y, rc_pos_z, rc_pos_r, l, m, Ylm);
                        t2 = MPI_Wtime();
                        t_tot += t2 - t1;
                        
                        // calculate Chi = UdV * Ylm
                        for (ipos = 0; ipos < ndc; ipos++) {
                            (*locProj)[ityp].Orb_c[iat][col_count*ndc+ipos] = Ylm[ipos] * UdV_sort[ipos];
                        }
                        col_count++;
                    }
                    lcount++;
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
    if (!rank)
        printf(BLU "rank = %d, Time for spherical harmonics: %.3f ms\n" RESET, rank, t_tot * 1e3);
#endif
}

/**
 * @brief   Calculate indices for storing nonlocal inner product in an array.
 *
 *          We will store the inner product < Orb_Jlm, x_n > in a continuous array "alpha",
 *          the dimensions are in the order: <lm>, n, J. Here we find out the sizes of the
 *          inner product corresponding to atom J, and the total number of inner products
 *          corresponding to each vector x_n.
 */
void CalculateLocalInnerProductIndex(SPARC_OBJ *pSPARC)
{
    int ityp, iat, l, lmax, atom_index, atom_index2, nproj, id;

    pSPARC->IP_displ_U = (int *)malloc(sizeof(int) * (pSPARC->n_atom + 1));
    memset(pSPARC->IP_displ_U, -1, sizeof(int) * (pSPARC->n_atom + 1));

    atom_index = atom_index2 = id = 0;
    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++)
    {
        if (!pSPARC->atom_solve_flag[ityp]) {
            atom_index += pSPARC->nAtomv[ityp];
            if (id == 0) atom_index2 = atom_index;
            continue;
        }

        lmax = pSPARC->AtmU[ityp].max_l;

        if (id == 0) {
            pSPARC->IP_displ_U[atom_index] = 0;
            id++;
        }

        // number of projectors per atom (of this type)
        nproj = 0;
        for (l = 0; l <= lmax; l++)
        {
            if (pSPARC->AtmU[ityp].U[l] == 0)
                continue;                                     // Skip for zero U values for a particular l orbital
            nproj += pSPARC->AtmU[ityp].ppl[l] * (2 * l + 1); // pSPARC->AtmU[ityp].ppl[l] > 1 if core states are included
        }

        for (iat = 0; iat < pSPARC->nAtomv[ityp]; iat++)
        {
            if (iat == 0 && id !=0) {
                pSPARC->IP_displ_U[atom_index] = pSPARC->IP_displ_U[atom_index2];
            } 
            pSPARC->IP_displ_U[atom_index + 1] = pSPARC->IP_displ_U[atom_index] + nproj;
            atom_index++;
        }
        atom_index2 = atom_index;
    }
}

/**
 * @brief   Calculate Vhub times vectors in a matrix-free way.
 */
void Vhub_vec_mult(const SPARC_OBJ *pSPARC, int DMnd, ATOM_LOC_INFLUENCE_OBJ *Atom_Influence_loc,
                   LOC_PROJ_OBJ *locProj, int ncol, double *x, int ldi, double *Hx, int ldo, int spin, MPI_Comm comm)
{

    int i, n, np, count, atm_idx;
    /* compute hubbard operator times vector(s) */
    int ityp, iat, l, m, ldispl, lmax, ndc, atom_index, nproj;
    double *alpha, *pre_fac_alpha, *x_rc, *Vhub;

    int n_atom = pSPARC->n_atom;
    atm_idx = -1;
    for (int JJ = n_atom; JJ >= 0; JJ--) {
        if (pSPARC->IP_displ_U[JJ] >= 0) {
            atm_idx = JJ; // last entry of IP_displ_U array corresponding to the last atom with U correction
            break;
        }
    }

    int rank;
    MPI_Comm_rank(comm, &rank);

    alpha = (double *)calloc(pSPARC->IP_displ_U[atm_idx] * ncol, sizeof(double));

    pre_fac_alpha = (double *)calloc(pSPARC->IP_displ_U[atm_idx] * ncol, sizeof(double));

    // first find inner product
    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++)
    {
        if (!pSPARC->atom_solve_flag[ityp]) {
            continue;
        }

        if (!locProj[ityp].nproj)
            continue;
        for (iat = 0; iat < Atom_Influence_loc[ityp].n_atom; iat++)
        {
            ndc = Atom_Influence_loc[ityp].ndc[iat];
            x_rc = (double *)malloc(ndc * ncol * sizeof(double));
            atom_index = Atom_Influence_loc[ityp].atom_index[iat];

            for (n = 0; n < ncol; n++)
            {
                for (i = 0; i < ndc; i++)
                {
                    x_rc[n * ndc + i] = x[n * ldi + Atom_Influence_loc[ityp].grid_pos[iat][i]];
                }
            }

            // No cyclix
            cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, locProj[ityp].nproj, ncol, ndc,
                        pSPARC->dV, locProj[ityp].Orb[iat], ndc, x_rc, ndc, 1.0,
                        alpha + pSPARC->IP_displ_U[atom_index] * ncol, locProj[ityp].nproj);
            
            free(x_rc);
        }
    }

    // if there are domain parallelization over each band, we need to sum over all processes over domain comm
    int commsize;
    MPI_Comm_size(comm, &commsize);
    if (commsize > 1)
    {
        MPI_Allreduce(MPI_IN_PLACE, alpha, pSPARC->IP_displ_U[atm_idx] * ncol, MPI_DOUBLE, MPI_SUM, comm);
    }

    // Multiply pre-factor
    double *pre_fac;
    int angnum;
    int atmcount = 0;
    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++)
    {
        if (!pSPARC->atom_solve_flag[ityp]) {
            atmcount += pSPARC->nAtomv[ityp];
            continue;
        }

        if (!locProj[ityp].nproj){
            atmcount += pSPARC->nAtomv[ityp];
            continue;
        }
        // angnum = pSPARC->AtmU[ityp].angnum;
        angnum = locProj[ityp].nproj;
        for (iat = 0; iat < pSPARC->nAtomv[ityp]; iat++)
        {
            // Calculate prefactor U(\delta_{mn} - rho_{mn})
            pre_fac = (double *)calloc(angnum*angnum, sizeof(double));
            for (int col = 0; col < angnum; col++) {
                for (int row = 0; row < angnum; row++) {
                    if (row == col) {
                        pre_fac[angnum*col + row] += 0.5;
                    }
                    pre_fac[angnum*col + row] -= pSPARC->rho_mn[atmcount][spin][angnum*col + row];
                }
            }

            // Scale the rows by Uval[row]
            for (int row = 0; row < angnum; row++) {
                for (int col = 0; col < angnum; col++) {
                    pre_fac[angnum*col + row] *= pSPARC->AtmU[ityp].Uval[row];
                }
            }

            // multiply pre factor to alpha
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, angnum, ncol, angnum,
                        1.0, pre_fac, angnum, alpha + pSPARC->IP_displ_U[atmcount] * ncol, angnum, 0.0,
                        pre_fac_alpha + pSPARC->IP_displ_U[atmcount] * ncol, angnum);

            free(pre_fac);
            atmcount++;
        }
    }
    free(alpha);

    // multiply the inner product and the local projector
    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++)
    {
        if (!pSPARC->atom_solve_flag[ityp]) {
            continue;
        }

        if (!locProj[ityp].nproj){
            continue;
        }

        for (iat = 0; iat < Atom_Influence_loc[ityp].n_atom; iat++)
        {
            ndc = Atom_Influence_loc[ityp].ndc[iat];
            atom_index = Atom_Influence_loc[ityp].atom_index[iat];
            Vhub = (double *)calloc(ndc * ncol, sizeof(double));
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, ndc, ncol, locProj[ityp].nproj, 1.0, locProj[ityp].Orb[iat], ndc,
                        pre_fac_alpha + pSPARC->IP_displ_U[atom_index] * ncol, locProj[ityp].nproj, 0.0, Vhub, ndc);
            for (n = 0; n < ncol; n++)
            {
                for (i = 0; i < ndc; i++)
                {
                    Hx[n * ldo + Atom_Influence_loc[ityp].grid_pos[iat][i]] += Vhub[n * ndc + i];
                }
            }

            free(Vhub);
        }
    }
    free(pre_fac_alpha);

}


/**
 * @brief   Calculate Vhub times vectors in a matrix-free way with Bloch factor.
 */
 void Vhub_vec_mult_kpt(const SPARC_OBJ *pSPARC, int DMnd, ATOM_LOC_INFLUENCE_OBJ *Atom_Influence_loc,
    LOC_PROJ_OBJ *locProj, int ncol, double _Complex *x, int ldi, double _Complex *Hx, int ldo, int spin, int kpt, MPI_Comm comm)
{
    int i, n, np, count, atm_idx;
    /* compute hubbard operator times vector(s) */
    int ityp, iat, l, m, ldispl, lmax, ndc, atom_index, nproj;
    double x0_i, y0_i, z0_i;

    double _Complex *alpha, *pre_fac, *pre_fac_alpha, *x_rc, *Vhub;

    int n_atom = pSPARC->n_atom;
    atm_idx = -1;
    for (int JJ = n_atom; JJ >= 0; JJ--) {
        if (pSPARC->IP_displ_U[JJ] >= 0) {
            atm_idx = JJ; // last entry of IP_displ_U array corresponding to the last atom with U correction
            break;
        }
    }

    alpha = (double _Complex *)calloc(pSPARC->IP_displ_U[atm_idx] * ncol, sizeof(double _Complex));

    pre_fac_alpha = (double _Complex *)calloc(pSPARC->IP_displ_U[atm_idx] * ncol, sizeof(double _Complex));

    double Lx = pSPARC->range_x;
    double Ly = pSPARC->range_y;
    double Lz = pSPARC->range_z;
    double k1 = pSPARC->k1_loc[kpt];
    double k2 = pSPARC->k2_loc[kpt];
    double k3 = pSPARC->k3_loc[kpt];
    double theta;
    double _Complex bloch_fac, a, b;

    // first find inner product
    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++)
    {
        if (!pSPARC->atom_solve_flag[ityp]) {
            continue;
        }

        if (!locProj[ityp].nproj)
            continue;
        for (iat = 0; iat < Atom_Influence_loc[ityp].n_atom; iat++)
        {
            x0_i = Atom_Influence_loc[ityp].coords[iat*3  ];
            y0_i = Atom_Influence_loc[ityp].coords[iat*3+1];
            z0_i = Atom_Influence_loc[ityp].coords[iat*3+2];
            theta = -k1 * (floor(x0_i/Lx) * Lx) - k2 * (floor(y0_i/Ly) * Ly) - k3 * (floor(z0_i/Lz) * Lz);
            bloch_fac = cos(theta) + sin(theta) * I;
            a = bloch_fac * pSPARC->dV; // no cyclix
            b = 1.0;

            ndc = Atom_Influence_loc[ityp].ndc[iat];
            x_rc = (double _Complex *)malloc(ndc * ncol * sizeof(double _Complex));
            atom_index = Atom_Influence_loc[ityp].atom_index[iat];
            for (n = 0; n < ncol; n++)
            {
                for (i = 0; i < ndc; i++)
                {
                    x_rc[n * ndc + i] = x[n * ldi + Atom_Influence_loc[ityp].grid_pos[iat][i]];
                }
            }
            // No cyclix
            cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, locProj[ityp].nproj, ncol, ndc,
                        &a, locProj[ityp].Orb_c[iat], ndc, x_rc, ndc, &b,
                        alpha + pSPARC->IP_displ_U[atom_index] * ncol, locProj[ityp].nproj);
            
            free(x_rc);
        }
    }

    // if there are domain parallelization over each band, we need to sum over all processes over domain comm
    int commsize;
    MPI_Comm_size(comm, &commsize);
    if (commsize > 1)
    {
        MPI_Allreduce(MPI_IN_PLACE, alpha, pSPARC->IP_displ_U[atm_idx] * ncol, MPI_DOUBLE_COMPLEX, MPI_SUM, comm);
    }

    // Multiply pre-factor
    int angnum;
    int atmcount = 0;
    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++)
    {
        if (!pSPARC->atom_solve_flag[ityp]) {
            atmcount += pSPARC->nAtomv[ityp];
            continue;
        }

        if (!locProj[ityp].nproj){
            atmcount += pSPARC->nAtomv[ityp];
            continue;
        }
        // angnum = pSPARC->AtmU[ityp].angnum;
        angnum = locProj[ityp].nproj;
        for (iat = 0; iat < pSPARC->nAtomv[ityp]; iat++)
        {
            // Calculate prefactor U(\delta_{mn} - rho_{mn})
            pre_fac = (double _Complex *)calloc(angnum*angnum, sizeof(double _Complex));
            for (int col = 0; col < angnum; col++) {
                for (int row = 0; row < angnum; row++) {
                    if (row == col) {
                        pre_fac[angnum*col + row] += 0.5;
                    }
                    // pre_fac[angnum*col + row] -= pSPARC->rho_mn_c[atmcount][spin][angnum*col + row];
                    pre_fac[angnum*col + row] -= pSPARC->rho_mn[atmcount][spin][angnum*col + row];
                }
            }

            // Scale the rows by Uval[row]
            for (int row = 0; row < angnum; row++) {
                for (int col = 0; col < angnum; col++) {
                    pre_fac[angnum*col + row] *= pSPARC->AtmU[ityp].Uval[row];
                }
            }

            // multiply pre factor to alpha
            a = 1.0; b = 0.0;
            cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, angnum, ncol, angnum,
                        &a, pre_fac, angnum, alpha + pSPARC->IP_displ_U[atmcount] * ncol, angnum, &b,
                        pre_fac_alpha + pSPARC->IP_displ_U[atmcount] * ncol, angnum);

            free(pre_fac);
            atmcount++;
        }
    }
    free(alpha);

    // multiply the inner product and the local projector
    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++)
    {
        if (!pSPARC->atom_solve_flag[ityp]) {
            continue;
        }

        if (!locProj[ityp].nproj){
            continue;
        }

        for (iat = 0; iat < Atom_Influence_loc[ityp].n_atom; iat++)
        {
            x0_i = Atom_Influence_loc[ityp].coords[iat*3  ];
            y0_i = Atom_Influence_loc[ityp].coords[iat*3+1];
            z0_i = Atom_Influence_loc[ityp].coords[iat*3+2];
            theta = -k1 * (floor(x0_i/Lx) * Lx) - k2 * (floor(y0_i/Ly) * Ly) - k3 * (floor(z0_i/Lz) * Lz);
            bloch_fac = cos(theta) - sin(theta) * I;
            b = 0.0;

            ndc = Atom_Influence_loc[ityp].ndc[iat];
            atom_index = Atom_Influence_loc[ityp].atom_index[iat];
            Vhub = (double _Complex *)calloc(ndc * ncol, sizeof(double _Complex));
            cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, ndc, ncol, locProj[ityp].nproj, &bloch_fac, locProj[ityp].Orb_c[iat], ndc,
                        pre_fac_alpha + pSPARC->IP_displ_U[atom_index] * ncol, locProj[ityp].nproj, &b, Vhub, ndc);
            for (n = 0; n < ncol; n++)
            {
                for (i = 0; i < ndc; i++)
                {
                    Hx[n * ldo + Atom_Influence_loc[ityp].grid_pos[iat][i]] += Vhub[n * ndc + i];
                }
            }

            free(Vhub);
        }
    }
    free(pre_fac_alpha);
}
