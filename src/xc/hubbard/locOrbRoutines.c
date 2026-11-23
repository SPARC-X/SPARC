/**
 * @file locOrbRoutines.c 
 * @brief This file contains the routines pertaining with local orbitals in the context of DFT+U
 * @author  Sayan Bhowmik <sbhowmik9@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Reference:   Dudarev, S. L., et al. "Electron-energy-loss spectra and the structural stability of nickel oxide:  An LSDA+U study"
 *              Phys. Rev. B 57, 1505
 *
 * The Hubbard product applies the periodic images of each U atom folded into one orbital, brick by brick (see Vhub_vec_mult
 * below). The per-image formulation used before is in SPARC commit a44cc61.
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
    double Lx, Ly, Lz, x0, y0, z0, x0_i, y0_i, z0_i, r2, rc2, rcbox_x, rcbox_y, rcbox_z, x, y, z;
    //double rc, x2, y2, z2;
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
    //double y0, z0, xi, yi, zi, ty, tz;
    //double xin = pSPARC->xin + DMVertices[0] * pSPARC->delta_x;

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
    //double y0, z0, xi, yi, zi, ty, tz;
    //double xin = pSPARC->xin + DMVertices[0] * pSPARC->delta_x;

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

/*
 * Hubbard product V_U X = sum_a Phi_a U (1/2 I - rho_a) (dV Phi_a^T X), in two steps (exact, since X is (Bloch-)periodic):
 * 1. Fold: sum the periodic images of each U atom once, Phi~_a = sum_R w_R Phi_{a,R}, with w_R = 1 at the Gamma point and
 *    w_R = e^{-i theta_R} at a k-point (one set per k-point).
 * 2. Bricks: cut the local grid into HUB_BRICK_NX x HUB_BRICK_NY x HUB_BRICK_NZ bricks; per brick, the folded orbitals of
 *    all atoms touching it form one dense matrix Phi_b. Then alpha = dV sum_b Phi_b^H X_b, beta = U (1/2 I - rho) alpha,
 *    HX_b += Phi_b beta.
 * Real (Gamma) and complex (k-point) values share one code path with w = 1 or 2 doubles per value. The same projection
 * gives the occupation matrix and the forces; the stress and pressure use orbitals folded with an extra weight
 * e . (x - R_image) per image. The folded orbitals of a geometry are kept as "plans", one per (influence, projectors,
 * k-point). The per-image formulation used before is in SPARC commit a44cc61 (Vhub_vec_mult, Vhub_vec_mult_kpt).
 */

#define HUB_BRICK_NX 8       // brick extent in x (contiguous in memory)
#define HUB_BRICK_NY 8       // brick extent in y
#define HUB_BRICK_NZ 4       // brick extent in z
#define HUB_PLAN_MB 1024.0   // memory for the plans of one process (MB); least recently used plans are freed beyond it

typedef struct {
    int atom_index, nproj;   // global atom index, orbitals per atom
    long nu;                 // set-up only: grid points of the folded support, sorted,
    int *idx;                //   their local grid indices
    double _Complex *Phiu;   //   and the folded orbitals on them (nu x nproj, column-major)
} HUB_ATOM;

typedef struct {
    int x0, y0, z0, nx, ny, nz;   // corner and size in the local grid
    int na, S;                    // atoms touching the brick, orbital columns
    int *slot, *coff;             // atom j: plan->at[slot[j]], columns coff[j] .. coff[j] + nproj - 1
    double *Phi;                  // R x S values, R = nx * ny * nz; w doubles per value
} HUB_BRICK;

typedef struct {
    const void *infl, *proj;      // the Atom_Influence / locProj pair this plan was built for
    int kpt;                      // local k-point index, -1 at the Gamma point
    int weighted; double e[3];    // stress/pressure: each image weighted by e . (x - R_image)
    int w;                        // doubles per value: 1 (Gamma, real) or 2 (k-point, complex)
    int Nx, Ny, Nz;               // local grid
    int gx0, gy0, gz0;            // its first point in the global grid
    int nat; HUB_ATOM *at;        // U atoms touching the local grid
    int nbrick; HUB_BRICK *brk;   // non-empty bricks
    int maxR, maxS;               // largest brick (points, columns)
    double mb;                    // memory of the brick orbitals
    long last_use;                // for freeing the least recently used plan
} HUB_PLAN;

static HUB_PLAN *hub_plans = NULL;
static int hub_nplans = 0, hub_cap_plans = 0;
static long hub_use_count = 0;
static double *hub_buf_x = NULL, *hub_buf_c = NULL, *hub_alpha = NULL, *hub_beta = NULL;   // brick of X, brick coefficients
static long hub_nbuf_x = 0, hub_nbuf_c = 0, hub_nalpha = 0;                                // their sizes in doubles

// make room for n plans in hub_plans
static void hub_plans_reserve(int n)
{
    if (n <= hub_cap_plans) return;
    while (hub_cap_plans < n) hub_cap_plans = hub_cap_plans ? 2 * hub_cap_plans : 4;
    hub_plans = realloc(hub_plans, hub_cap_plans * sizeof(HUB_PLAN));
}

static void free_plan(HUB_PLAN *pl)
{
    for (int q = 0; q < pl->nat; q++) { free(pl->at[q].idx); free(pl->at[q].Phiu); }
    for (int b = 0; b < pl->nbrick; b++) { free(pl->brk[b].slot); free(pl->brk[b].coff); free(pl->brk[b].Phi); }
    free(pl->at); free(pl->brk);
    memset(pl, 0, sizeof(HUB_PLAN));
}

/**
 * @brief   Free the folded orbitals and work space of the Hubbard product (on every change of geometry).
 */
void Vhub_free_plans(void)
{
    for (int i = 0; i < hub_nplans; i++) free_plan(&hub_plans[i]);
    free(hub_plans); hub_plans = NULL; hub_nplans = hub_cap_plans = 0;
    free(hub_buf_x); free(hub_buf_c); free(hub_alpha); free(hub_beta);
    hub_buf_x = hub_buf_c = hub_alpha = hub_beta = NULL; hub_nbuf_x = hub_nbuf_c = hub_nalpha = 0;
}

static int cmp_int(const void *a, const void *b) { return (*(const int *)a > *(const int *)b) - (*(const int *)a < *(const int *)b); }

/**
 * @brief   Weight of image iat in the fold: 1 at the Gamma point, e^{-i theta} at a k-point, theta = -k . (lattice shift).
 */
static double _Complex image_phase(const SPARC_OBJ *pSPARC, const ATOM_LOC_INFLUENCE_OBJ *infl, int iat, int kpt)
{
    if (kpt < 0) return 1.0;
    double Lx = pSPARC->range_x, Ly = pSPARC->range_y, Lz = pSPARC->range_z;
    double x0 = infl->coords[iat * 3], y0 = infl->coords[iat * 3 + 1], z0 = infl->coords[iat * 3 + 2];
    double theta = -pSPARC->k1_loc[kpt] * (floor(x0 / Lx) * Lx) - pSPARC->k2_loc[kpt] * (floor(y0 / Ly) * Ly)
                   - pSPARC->k3_loc[kpt] * (floor(z0 / Lz) * Lz);
    return cos(theta) - sin(theta) * I;
}

/**
 * @brief   Stress and pressure weight of image iat at local grid point g: e . (x_g - R_image). 1 for an unweighted plan.
 */
static double image_weight(const SPARC_OBJ *pSPARC, const HUB_PLAN *pl, const ATOM_LOC_INFLUENCE_OBJ *infl, int iat, long g)
{
    if (!pl->weighted) return 1.0;
    long NxNy = (long)pl->Nx * pl->Ny;
    double x1 = ((g % pl->Nx) + pl->gx0) * pSPARC->delta_x - infl->coords[iat * 3];
    double x2 = (((g % NxNy) / pl->Nx) + pl->gy0) * pSPARC->delta_y - infl->coords[iat * 3 + 1];
    double x3 = ((g / NxNy) + pl->gz0) * pSPARC->delta_z - infl->coords[iat * 3 + 2];
    return pl->e[0] * x1 + pl->e[1] * x2 + pl->e[2] * x3;
}

/**
 * @brief   Step 1: Phi~_a = sum_R w_R Phi_{a,R} on the local grid, for every U atom with an image touching it, stored per
 *          atom on its support (the grid points any image touches), sorted by grid index.
 */
static void Vhub_fold_orbitals(const SPARC_OBJ *pSPARC, ATOM_LOC_INFLUENCE_OBJ *Atom_Influence_loc, LOC_PROJ_OBJ *locProj, HUB_PLAN *pl)
{
    long Nd = (long)pl->Nx * pl->Ny * pl->Nz;
    int npmax = 1;
    for (int t = 0; t < pSPARC->Ntypes; t++) if (pSPARC->atom_solve_flag[t] && locProj[t].nproj > npmax) npmax = locProj[t].nproj;
    double _Complex *acc = calloc((size_t)Nd * npmax, sizeof(double _Complex));   // folded orbitals of the current atom
    unsigned char *mark = calloc(Nd > 0 ? Nd : 1, 1);                             // points already in its support
    int *pts = malloc((Nd > 0 ? Nd : 1) * sizeof(int));                           // its support
    pl->at = calloc(pSPARC->n_atom, sizeof(HUB_ATOM)); pl->nat = 0;
    for (int t = 0; t < pSPARC->Ntypes; t++) {
        if (!pSPARC->atom_solve_flag[t] || !locProj[t].nproj) continue;
        int np = locProj[t].nproj;
        for (int i0 = 0; i0 < Atom_Influence_loc[t].n_atom; i0++) {
            int a = Atom_Influence_loc[t].atom_index[i0], seen = 0;
            for (int q = 0; q < pl->nat; q++) if (pl->at[q].atom_index == a) { seen = 1; break; }
            if (seen) continue;
            // sum the images of atom a, each with its weight
            long nu = 0;
            for (int iat = 0; iat < Atom_Influence_loc[t].n_atom; iat++) {
                if (Atom_Influence_loc[t].atom_index[iat] != a) continue;
                int ndc = Atom_Influence_loc[t].ndc[iat]; const int *gp = Atom_Influence_loc[t].grid_pos[iat];
                double _Complex wR = image_phase(pSPARC, &Atom_Influence_loc[t], iat, pl->kpt);
                for (int i = 0; i < ndc; i++) {
                    long g = gp[i]; if (!mark[g]) { mark[g] = 1; pts[nu++] = (int)g; }
                    for (int p = 0; p < np; p++) {
                        size_t k = (size_t)p * ndc + i;   // Gamma: real projectors (Orb); k-point: complex ones (Orb_c)
                        acc[(size_t)p * Nd + g] += (pl->kpt < 0) ? locProj[t].Orb[iat][k] : wR * locProj[t].Orb_c[iat][k];
                    }
                }
            }
            qsort(pts, nu, sizeof(int), cmp_int);
            HUB_ATOM *A = &pl->at[pl->nat++]; A->atom_index = a; A->nproj = np; A->nu = nu;
            A->idx = malloc((nu > 0 ? nu : 1) * sizeof(int));
            A->Phiu = malloc((size_t)(nu > 0 ? nu : 1) * np * sizeof(double _Complex));
            for (long i = 0; i < nu; i++) A->idx[i] = pts[i];
            for (int p = 0; p < np; p++) for (long i = 0; i < nu; i++) A->Phiu[(size_t)p * nu + i] = acc[(size_t)p * Nd + A->idx[i]];
            for (long i = 0; i < nu; i++) { long g = pts[i]; mark[g] = 0; for (int p = 0; p < np; p++) acc[(size_t)p * Nd + g] = 0.0; }
        }
    }
    free(acc); free(mark); free(pts);
}

/**
 * @brief   Step 2: cut the local grid into bricks and store, per non-empty brick, the folded orbitals of the atoms touching
 *          it as one dense matrix. Frees the per-atom copies afterwards.
 */
static void Vhub_build_bricks(HUB_PLAN *pl)
{
    int Nx = pl->Nx, Ny = pl->Ny, Nz = pl->Nz, w = pl->w; long NxNy = (long)Nx * Ny;
    int nbx = (Nx + HUB_BRICK_NX - 1) / HUB_BRICK_NX, nby = (Ny + HUB_BRICK_NY - 1) / HUB_BRICK_NY, nbz = (Nz + HUB_BRICK_NZ - 1) / HUB_BRICK_NZ;
    long nb = (long)nbx * nby * nbz;
    #define BRICK_OF(g) ((((g) / NxNy) / HUB_BRICK_NZ) * (long)nby * nbx + ((((g) % NxNy) / Nx) / HUB_BRICK_NY) * (long)nbx + (((g) % Nx) / HUB_BRICK_NX))
    int *cnt = calloc(nb, sizeof(int)), *S = calloc(nb, sizeof(int)), *cur = malloc(nb * sizeof(int)), *curoff = malloc(nb * sizeof(int));
    long *map = malloc(nb * sizeof(long));
    for (long b = 0; b < nb; b++) cur[b] = -1;
    // pass 1: atoms and orbital columns per brick
    for (int q = 0; q < pl->nat; q++) {
        const HUB_ATOM *A = &pl->at[q];
        for (long i = 0; i < A->nu; i++) { long b = BRICK_OF((long)A->idx[i]); if (cur[b] != q) { cur[b] = q; cnt[b]++; S[b] += A->nproj; } }
    }
    int nbk = 0;
    for (long b = 0; b < nb; b++) map[b] = cnt[b] ? nbk++ : -1;
    pl->brk = calloc(nbk > 0 ? nbk : 1, sizeof(HUB_BRICK)); pl->nbrick = nbk; pl->maxR = 0; pl->maxS = 0;
    for (long b = 0; b < nb; b++) {
        if (map[b] < 0) continue;
        HUB_BRICK *K = &pl->brk[map[b]];
        int bz = (int)(b / ((long)nbx * nby)), by = (int)((b / nbx) % nby), bx = (int)(b % nbx);
        K->x0 = bx * HUB_BRICK_NX; K->y0 = by * HUB_BRICK_NY; K->z0 = bz * HUB_BRICK_NZ;
        K->nx = (K->x0 + HUB_BRICK_NX <= Nx) ? HUB_BRICK_NX : Nx - K->x0;
        K->ny = (K->y0 + HUB_BRICK_NY <= Ny) ? HUB_BRICK_NY : Ny - K->y0;
        K->nz = (K->z0 + HUB_BRICK_NZ <= Nz) ? HUB_BRICK_NZ : Nz - K->z0;
        K->S = S[b]; K->na = 0;
        K->slot = malloc(cnt[b] * sizeof(int)); K->coff = malloc(cnt[b] * sizeof(int));
        int R = K->nx * K->ny * K->nz;
        K->Phi = calloc((size_t)R * K->S * w, sizeof(double));
        if (R > pl->maxR) pl->maxR = R;
        if (K->S > pl->maxS) pl->maxS = K->S;
        cur[b] = -1; curoff[b] = 0;
    }
    // pass 2: copy each atom's folded orbitals into the bricks its support meets
    double stored = 0;
    for (int q = 0; q < pl->nat; q++) {
        const HUB_ATOM *A = &pl->at[q];
        for (long i = 0; i < A->nu; i++) {
            long g = A->idx[i], b = BRICK_OF(g); HUB_BRICK *K = &pl->brk[map[b]];
            if (cur[b] != q) { cur[b] = q; K->slot[K->na] = q; K->coff[K->na] = curoff[b]; K->na++; curoff[b] += A->nproj; }
            int off = K->coff[K->na - 1];
            int lx = (int)(g % Nx) - K->x0, ly = (int)((g % NxNy) / Nx) - K->y0, lz = (int)(g / NxNy) - K->z0;
            long r = ((long)lz * K->ny + ly) * K->nx + lx, R = (long)K->nx * K->ny * K->nz;
            for (int p = 0; p < A->nproj; p++) {
                double _Complex v = A->Phiu[(size_t)p * A->nu + i]; double *dst = K->Phi + ((size_t)(off + p) * R + r) * w;
                dst[0] = creal(v); if (w == 2) dst[1] = cimag(v);
            }
        }
    }
    #undef BRICK_OF
    for (int k = 0; k < nbk; k++) stored += (double)pl->brk[k].nx * pl->brk[k].ny * pl->brk[k].nz * pl->brk[k].S;
    pl->mb = stored * 8 * w / 1e6;
    free(cnt); free(S); free(cur); free(curoff); free(map);
    for (int q = 0; q < pl->nat; q++) { free(pl->at[q].idx); free(pl->at[q].Phiu); pl->at[q].idx = NULL; pl->at[q].Phiu = NULL; }
}

static int same_weight(const HUB_PLAN *pl, const double *e)
{
    if (!e) return !pl->weighted;
    return pl->weighted && pl->e[0] == e[0] && pl->e[1] == e[1] && pl->e[2] == e[2];
}

/**
 * @brief   Free the weighted (stress/pressure) folded orbitals; call after the stress or pressure.
 */
void Vhub_free_weighted(void)
{
    for (int i = 0; i < hub_nplans; ) {
        if (hub_plans[i].weighted) { free_plan(&hub_plans[i]); hub_plans[i] = hub_plans[--hub_nplans]; } else i++;
    }
}

/**
 * @brief   A weighted plan (stress, pressure) has the bricks, atoms and columns of the unweighted plan of the same k-point;
 *          only the values differ. So copy that layout and add each image's weighted orbital straight into the bricks.
 */
static void Vhub_build_weighted_bricks(const SPARC_OBJ *pSPARC, ATOM_LOC_INFLUENCE_OBJ *Atom_Influence_loc, LOC_PROJ_OBJ *locProj,
                                       const HUB_PLAN *base, HUB_PLAN *pl)
{
    int Nx = base->Nx, Ny = base->Ny, Nz = base->Nz, w = base->w; long NxNy = (long)Nx * Ny;
    pl->nat = base->nat; pl->nbrick = base->nbrick; pl->maxR = base->maxR; pl->maxS = base->maxS; pl->mb = base->mb;
    pl->at = calloc(pl->nat > 0 ? pl->nat : 1, sizeof(HUB_ATOM));
    for (int q = 0; q < pl->nat; q++) { pl->at[q].atom_index = base->at[q].atom_index; pl->at[q].nproj = base->at[q].nproj; }
    pl->brk = calloc(pl->nbrick > 0 ? pl->nbrick : 1, sizeof(HUB_BRICK));
    int nbx = (Nx + HUB_BRICK_NX - 1) / HUB_BRICK_NX, nby = (Ny + HUB_BRICK_NY - 1) / HUB_BRICK_NY, nbz = (Nz + HUB_BRICK_NZ - 1) / HUB_BRICK_NZ;
    long nb = (long)nbx * nby * nbz, *map = malloc(nb * sizeof(long));   // brick of the grid -> brick of the plan
    for (long b = 0; b < nb; b++) map[b] = -1;
    for (int k = 0; k < pl->nbrick; k++) {
        const HUB_BRICK *B = &base->brk[k]; HUB_BRICK *K = &pl->brk[k];
        *K = *B;
        K->slot = malloc(B->na * sizeof(int)); K->coff = malloc(B->na * sizeof(int));
        memcpy(K->slot, B->slot, B->na * sizeof(int)); memcpy(K->coff, B->coff, B->na * sizeof(int));
        K->Phi = calloc((size_t)B->nx * B->ny * B->nz * B->S * w, sizeof(double));
        map[((long)(B->z0 / HUB_BRICK_NZ) * nby + B->y0 / HUB_BRICK_NY) * nbx + B->x0 / HUB_BRICK_NX] = k;
    }
    int *qof = malloc(pSPARC->n_atom * sizeof(int));   // global atom index -> atom of the plan
    for (int a = 0; a < pSPARC->n_atom; a++) qof[a] = -1;
    for (int q = 0; q < pl->nat; q++) qof[pl->at[q].atom_index] = q;
    double **dst = NULL, *wx = NULL; long *step = NULL; int ndc_max = 0;   // per point of an image: where column 0 goes,
    for (int t = 0; t < pSPARC->Ntypes; t++) {                             // the column stride, the weight
        if (!pSPARC->atom_solve_flag[t] || !locProj[t].nproj) continue;
        int np = locProj[t].nproj;
        for (int iat = 0; iat < Atom_Influence_loc[t].n_atom; iat++) {
            int q = qof[Atom_Influence_loc[t].atom_index[iat]], ndc = Atom_Influence_loc[t].ndc[iat], off = 0; long R = 0;
            const int *gp = Atom_Influence_loc[t].grid_pos[iat];
            double _Complex wR = image_phase(pSPARC, &Atom_Influence_loc[t], iat, pl->kpt);
            if (ndc > ndc_max) {
                ndc_max = ndc; free(dst); free(wx); free(step);
                dst = malloc(ndc * sizeof(double *)); wx = malloc(ndc * sizeof(double)); step = malloc(ndc * sizeof(long));
            }
            HUB_BRICK *K = NULL; long lastb = -1;
            for (int i = 0; i < ndc; i++) {
                long g = gp[i], b = map[(((g / NxNy) / HUB_BRICK_NZ) * nby + ((g % NxNy) / Nx) / HUB_BRICK_NY) * nbx + (g % Nx) / HUB_BRICK_NX];
                if (b != lastb) {   // the atom's columns in this brick
                    K = &pl->brk[b]; lastb = b; R = (long)K->nx * K->ny * K->nz;
                    for (int j = 0; j < K->na; j++) if (K->slot[j] == q) { off = K->coff[j]; break; }
                }
                long r = ((long)((g / NxNy) - K->z0) * K->ny + ((g % NxNy) / Nx - K->y0)) * K->nx + (g % Nx - K->x0);
                dst[i] = K->Phi + ((size_t)off * R + r) * w; step[i] = R * w;
                wx[i] = image_weight(pSPARC, pl, &Atom_Influence_loc[t], iat, g);
            }
            for (int p = 0; p < np; p++) {   // column by column: Orb is read in order
                if (w == 1) {
                    const double *o = locProj[t].Orb[iat] + (size_t)p * ndc;
                    for (int i = 0; i < ndc; i++) dst[i][p * step[i]] += wx[i] * o[i];
                } else {
                    const double _Complex *o = locProj[t].Orb_c[iat] + (size_t)p * ndc;
                    for (int i = 0; i < ndc; i++) {
                        double _Complex v = wx[i] * wR * o[i]; double *d = dst[i] + p * step[i];
                        d[0] += creal(v); d[1] += cimag(v);
                    }
                }
            }
        }
    }
    free(map); free(qof); free(dst); free(wx); free(step);
}

/**
 * @brief   The plan for (Atom_Influence_loc, locProj, kpt, weight e; e = NULL: unweighted), built on first use. Only one
 *          weighted plan is kept at a time; it is built from the unweighted plan and not counted in HUB_PLAN_MB.
 */
static HUB_PLAN *Vhub_get_plan(const SPARC_OBJ *pSPARC, ATOM_LOC_INFLUENCE_OBJ *Atom_Influence_loc, LOC_PROJ_OBJ *locProj,
                               int kpt, const double *e, MPI_Comm comm)
{
    hub_use_count++;
    for (int i = 0; i < hub_nplans; i++)
        if (hub_plans[i].infl == Atom_Influence_loc && hub_plans[i].proj == locProj && hub_plans[i].kpt == kpt && same_weight(&hub_plans[i], e)) {
            hub_plans[i].last_use = hub_use_count; return &hub_plans[i];
        }
    if (e) {
        Vhub_free_weighted();
        hub_plans_reserve(hub_nplans + 2);   // room for the unweighted plan and this one: no realloc while base is held
        HUB_PLAN *base = Vhub_get_plan(pSPARC, Atom_Influence_loc, locProj, kpt, NULL, comm);
        HUB_PLAN *pl = &hub_plans[hub_nplans++]; memset(pl, 0, sizeof(HUB_PLAN));
        pl->infl = base->infl; pl->proj = base->proj; pl->kpt = kpt; pl->w = base->w; pl->last_use = hub_use_count;
        pl->Nx = base->Nx; pl->Ny = base->Ny; pl->Nz = base->Nz; pl->gx0 = base->gx0; pl->gy0 = base->gy0; pl->gz0 = base->gz0;
        pl->weighted = 1; pl->e[0] = e[0]; pl->e[1] = e[1]; pl->e[2] = e[2];
        Vhub_build_weighted_bricks(pSPARC, Atom_Influence_loc, locProj, base, pl);
        return pl;
    }
    for (;;) {   // free the least recently used plans beyond HUB_PLAN_MB
        double total = 0; int lru = -1;
        for (int i = 0; i < hub_nplans; i++) { total += hub_plans[i].mb; if (lru < 0 || hub_plans[i].last_use < hub_plans[lru].last_use) lru = i; }
        if (hub_nplans == 0 || total <= HUB_PLAN_MB) break;
        free_plan(&hub_plans[lru]); hub_plans[lru] = hub_plans[--hub_nplans];
    }
    hub_plans_reserve(hub_nplans + 1);
    const int *DMV = (comm == pSPARC->kptcomm_topo) ? pSPARC->DMVertices_kptcomm : pSPARC->DMVertices_dmcomm;
    HUB_PLAN *pl = &hub_plans[hub_nplans++]; memset(pl, 0, sizeof(HUB_PLAN));
    pl->infl = Atom_Influence_loc; pl->proj = locProj; pl->kpt = kpt; pl->w = (kpt < 0) ? 1 : 2; pl->last_use = hub_use_count;
    pl->Nx = DMV[1] - DMV[0] + 1; pl->Ny = DMV[3] - DMV[2] + 1; pl->Nz = DMV[5] - DMV[4] + 1;
    pl->gx0 = DMV[0]; pl->gy0 = DMV[2]; pl->gz0 = DMV[4];
    Vhub_fold_orbitals(pSPARC, Atom_Influence_loc, locProj, pl);
    Vhub_build_bricks(pl);
    return pl;
}

static void grow(double **p, long *n, long need)
{
    if (need > *n) { free(*p); *n = need; *p = malloc((size_t)need * sizeof(double)); }
}

// brick <-> grid copies by x-runs (each run is contiguous in X and HX); w doubles per value
static void brick_gather(const HUB_PLAN *pl, const HUB_BRICK *K, int ncol, const double *x, int ldi, double *buf)
{
    long NxNy = (long)pl->Nx * pl->Ny; int R = K->nx * K->ny * K->nz, w = pl->w, run = K->nx * w;
    for (int n = 0; n < ncol; n++) {
        const double *xn = x + (size_t)n * ldi * w; double *d = buf + (size_t)n * R * w;
        for (int lz = 0; lz < K->nz; lz++) for (int ly = 0; ly < K->ny; ly++) {
            double *dst = d + ((size_t)lz * K->ny + ly) * run;
            const double *src = xn + ((K->z0 + lz) * NxNy + (long)(K->y0 + ly) * pl->Nx + K->x0) * w;
            for (int i = 0; i < run; i++) dst[i] = src[i];
        }
    }
}
static void brick_scatter_add(const HUB_PLAN *pl, const HUB_BRICK *K, int ncol, const double *buf, double *y, int ldo)
{
    long NxNy = (long)pl->Nx * pl->Ny; int R = K->nx * K->ny * K->nz, w = pl->w, run = K->nx * w;
    for (int n = 0; n < ncol; n++) {
        double *yn = y + (size_t)n * ldo * w; const double *s = buf + (size_t)n * R * w;
        for (int lz = 0; lz < K->nz; lz++) for (int ly = 0; ly < K->ny; ly++) {
            double *dst = yn + ((K->z0 + lz) * NxNy + (long)(K->y0 + ly) * pl->Nx + K->x0) * w;
            const double *src = s + ((size_t)lz * K->ny + ly) * run;
            for (int i = 0; i < run; i++) dst[i] += src[i];
        }
    }
}

/**
 * @brief   out_a += scale * sum_b Phi_b^H (R_b X) for every atom a (out at IP_displ_U[a] * ncol, nproj x ncol per atom).
 */
static void Vhub_project(const SPARC_OBJ *pSPARC, const HUB_PLAN *pl, int ncol, const double *x, int ldi, double scale, double *out)
{
    int w = pl->w;
    grow(&hub_buf_x, &hub_nbuf_x, (long)pl->maxR * ncol * w); grow(&hub_buf_c, &hub_nbuf_c, (long)pl->maxS * ncol * w);
    for (int b = 0; b < pl->nbrick; b++) {
        const HUB_BRICK *K = &pl->brk[b]; int R = K->nx * K->ny * K->nz;
        brick_gather(pl, K, ncol, x, ldi, hub_buf_x);
        if (w == 1) {
            cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, K->S, ncol, R, scale, K->Phi, R, hub_buf_x, R, 0.0, hub_buf_c, K->S);
        } else {
            double _Complex a = scale, z = 0.0;
            cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, K->S, ncol, R, &a, K->Phi, R, hub_buf_x, R, &z, hub_buf_c, K->S);
        }
        for (int j = 0; j < K->na; j++) {
            const HUB_ATOM *A = &pl->at[K->slot[j]]; double *al = out + (size_t)pSPARC->IP_displ_U[A->atom_index] * ncol * w;
            for (int n = 0; n < ncol; n++) for (int p = 0; p < A->nproj * w; p++)
                al[p + (size_t)n * A->nproj * w] += hub_buf_c[K->coff[j] * w + p + (size_t)n * K->S * w];
        }
    }
}

/**
 * @brief   R_b HX += Phi_b beta_b for every brick (beta laid out as out in Vhub_project).
 */
static void Vhub_add_back(const SPARC_OBJ *pSPARC, const HUB_PLAN *pl, int ncol, const double *beta, double *Hx, int ldo)
{
    int w = pl->w;
    grow(&hub_buf_x, &hub_nbuf_x, (long)pl->maxR * ncol * w); grow(&hub_buf_c, &hub_nbuf_c, (long)pl->maxS * ncol * w);
    for (int b = 0; b < pl->nbrick; b++) {
        const HUB_BRICK *K = &pl->brk[b]; int R = K->nx * K->ny * K->nz;
        for (int j = 0; j < K->na; j++) {
            const HUB_ATOM *A = &pl->at[K->slot[j]]; const double *pa = beta + (size_t)pSPARC->IP_displ_U[A->atom_index] * ncol * w;
            for (int n = 0; n < ncol; n++) for (int p = 0; p < A->nproj * w; p++)
                hub_buf_c[K->coff[j] * w + p + (size_t)n * K->S * w] = pa[p + (size_t)n * A->nproj * w];
        }
        if (w == 1) {
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, R, ncol, K->S, 1.0, K->Phi, R, hub_buf_c, K->S, 0.0, hub_buf_x, R);
        } else {
            double _Complex one = 1.0, z = 0.0;
            cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, R, ncol, K->S, &one, K->Phi, R, hub_buf_c, K->S, &z, hub_buf_x, R);
        }
        brick_scatter_add(pl, K, ncol, hub_buf_x, Hx, ldo);
    }
}

// number of coefficient rows: IP_displ_U of the last atom with U correction
static int n_coeff(const SPARC_OBJ *pSPARC)
{
    for (int J = pSPARC->n_atom; J >= 0; J--) if (pSPARC->IP_displ_U[J] >= 0) return pSPARC->IP_displ_U[J];
    return 0;
}

/**
 * @brief   pre_fac_alpha = U (1/2 I - rho) alpha for every U atom (alpha after the sum over the domain).
 */
static void Vhub_pre_factor_alpha(const SPARC_OBJ *pSPARC, LOC_PROJ_OBJ *locProj, int ncol, int spin,
                                  double *alpha, double *pre_fac_alpha)
{
    int ityp, iat;
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
}

/**
 * @brief   pre_fac_alpha = U (1/2 I - rho) alpha for every U atom (alpha complex, after the sum over the domain).
 */
static void Vhub_pre_factor_alpha_kpt(const SPARC_OBJ *pSPARC, LOC_PROJ_OBJ *locProj, int ncol, int spin,
                                      double _Complex *alpha, double _Complex *pre_fac_alpha)
{
    int ityp, iat;
    double _Complex *pre_fac, a, b;
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
}

// Hx += V_U x with the plan of k-point kpt (-1: Gamma point; x, Hx: w doubles per value)
static void Vhub_apply(const SPARC_OBJ *pSPARC, ATOM_LOC_INFLUENCE_OBJ *Atom_Influence_loc, LOC_PROJ_OBJ *locProj, int ncol,
                       double *x, int ldi, double *Hx, int ldo, int spin, int kpt, MPI_Comm comm)
{
    HUB_PLAN *pl = Vhub_get_plan(pSPARC, Atom_Influence_loc, locProj, kpt, NULL, comm);
    long nal = (long)n_coeff(pSPARC) * ncol * pl->w;
    if (nal > hub_nalpha) {
        free(hub_alpha); free(hub_beta); hub_nalpha = nal;
        hub_alpha = malloc(nal * sizeof(double)); hub_beta = malloc(nal * sizeof(double));
    }
    memset(hub_alpha, 0, nal * sizeof(double));
    Vhub_project(pSPARC, pl, ncol, x, ldi, pSPARC->dV, hub_alpha);

    // if there are domain parallelization over each band, we need to sum over all processes over domain comm
    int commsize;
    MPI_Comm_size(comm, &commsize);
    if (commsize > 1) MPI_Allreduce(MPI_IN_PLACE, hub_alpha, nal, MPI_DOUBLE, MPI_SUM, comm);

    if (pl->w == 1) Vhub_pre_factor_alpha(pSPARC, locProj, ncol, spin, hub_alpha, hub_beta);
    else Vhub_pre_factor_alpha_kpt(pSPARC, locProj, ncol, spin, (double _Complex *)hub_alpha, (double _Complex *)hub_beta);
    Vhub_add_back(pSPARC, pl, ncol, hub_beta, Hx, ldo);
}

/**
 * @brief   Calculate Vhub times vectors in a matrix-free way.
 */
void Vhub_vec_mult(const SPARC_OBJ *pSPARC, ATOM_LOC_INFLUENCE_OBJ *Atom_Influence_loc,
                   LOC_PROJ_OBJ *locProj, int ncol, double *x, int ldi, double *Hx, int ldo, int spin, MPI_Comm comm)
{
    Vhub_apply(pSPARC, Atom_Influence_loc, locProj, ncol, x, ldi, Hx, ldo, spin, -1, comm);
}

/**
 * @brief   Calculate Vhub times vectors in a matrix-free way with Bloch factor.
 */
void Vhub_vec_mult_kpt(const SPARC_OBJ *pSPARC, ATOM_LOC_INFLUENCE_OBJ *Atom_Influence_loc,
    LOC_PROJ_OBJ *locProj, int ncol, double _Complex *x, int ldi, double _Complex *Hx, int ldo, int spin, int kpt, MPI_Comm comm)
{
    Vhub_apply(pSPARC, Atom_Influence_loc, locProj, ncol, (double *)x, ldi, (double *)Hx, ldo, spin, kpt, comm);
}

/**
 * @brief   out += scale * Phi~^H x for every U atom touching the local domain, without the sum over the domain
 *          (occupation matrix). kpt = -1: Gamma point (real x, out); otherwise complex, as pairs of doubles.
 */
void Vhub_inner_product(const SPARC_OBJ *pSPARC, ATOM_LOC_INFLUENCE_OBJ *Atom_Influence_loc, LOC_PROJ_OBJ *locProj,
                        int ncol, const double *x, int ldi, double scale, double *out, int kpt, MPI_Comm comm)
{
    Vhub_project(pSPARC, Vhub_get_plan(pSPARC, Atom_Influence_loc, locProj, kpt, NULL, comm), ncol, x, ldi, scale, out);
}

/**
 * @brief   Projections of the bands of this process for the forces, stress and pressure (x: DMnd x Nspinor x Nband_bandcomm):
 *              out += scale * sum_R w_R Phi_R^T x,
 *          w_R the Bloch factor at a k-point (kpt >= 0), times e . (x - R) if e is given (stress, pressure). With conj_out
 *          the complex conjugate is added. No sum over the domain.
 */
void Vhub_inner_product_bands(const SPARC_OBJ *pSPARC, const double *x, int kpt, const double *e, double scale, int conj_out,
                              double *out)
{
    int ncol = pSPARC->Nband_bandcomm, DMnd = pSPARC->Nd_d_dmcomm, Nspinor = pSPARC->Nspinor_spincomm, w = (kpt < 0) ? 1 : 2;
    long len = (long)n_coeff(pSPARC) * ncol * w;   // one spinor
    double *tmp = conj_out ? calloc(len > 0 ? len : 1, sizeof(double)) : NULL;
    for (int s = 0; s < Nspinor; s++) {
        HUB_PLAN *pl = Vhub_get_plan(pSPARC, pSPARC->Atom_Influence_loc_orb, pSPARC->locProj, kpt, e, pSPARC->dmcomm);
        double *o = out + len * s;
        if (!conj_out) { Vhub_project(pSPARC, pl, ncol, x + (size_t)s * DMnd * w, DMnd * Nspinor, scale, o); continue; }
        memset(tmp, 0, len * sizeof(double));
        Vhub_project(pSPARC, pl, ncol, x + (size_t)s * DMnd * w, DMnd * Nspinor, scale, tmp);
        for (long i = 0; i < len; i += 2) { o[i] += tmp[i]; o[i + 1] -= tmp[i + 1]; }
    }
    free(tmp);
}
