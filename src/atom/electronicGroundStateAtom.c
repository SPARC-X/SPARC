#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <mpi.h>
#ifdef USE_MKL
    #define MKL_Complex16 double _Complex
    #include <mkl.h>
#else
    #include <cblas.h>
    #include <lapacke.h>
#endif

#include "electronicGroundStateAtom.h"
#include "isddftAtom.h"
#include "tools.h"
#include "exchangeCorrelation.h"
#include "mGGAscan.h"
#include "mGGArscan.h"
#include "mGGAr2scan.h"
#include "mGGAtoolsAtom.h"
#include "mGGApotentialAtom.h"
#include "exxPotentialEnergyAtom.h"
#include "exxToolsAtom.h"

#define max(x,y) ((x)>(y)?(x):(y))

/**
 * @brief Calculate the electronic ground state
 *        Refer to Eq. 7 of Bhowmik, S. et al. Comput. Phys. Commun. 308 (2025) 109448
 * 
 */
void electronicGroundState_atom(SPARC_ATOM_OBJ *pSPARC_ATOM){
    double t1, t2;
    // Calculate non local potential
    t1 = MPI_Wtime();
    compute_nonlocal_potential(pSPARC_ATOM);
    t2 = MPI_Wtime();
#ifdef DEBUG
    printf("\nSetting non local potential took %.3f ms\n",(t2-t1)*1000);
#endif

    // Calculate local potential
    t1 = MPI_Wtime();
    compute_local_potential(pSPARC_ATOM);
    t2 = MPI_Wtime();
#ifdef DEBUG
    printf("\nSetting local potential took %.3f ms\n",(t2-t1)*1000);
#endif
    
    int Nd = pSPARC_ATOM->Nd;
    int col = pSPARC_ATOM->nspin;
    int val_len = pSPARC_ATOM->val_len;
    pSPARC_ATOM->orbitals = (double *)malloc(col*(Nd - 1)*val_len*sizeof(double));
    pSPARC_ATOM->eigenVal = (double *)malloc(col*val_len*sizeof(double));
    pSPARC_ATOM->orbital_l = (int *)malloc(col*val_len*sizeof(int));
    pSPARC_ATOM->occ = (int *)malloc(col*val_len*sizeof(int));

    // Store rho-previous and initialise mixing variables
    pSPARC_ATOM->mixing_hist_xk = (double *)malloc(col*(Nd-1)*sizeof(double));
    pSPARC_ATOM->mixing_hist_fk = (double *)calloc(col*(Nd-1), sizeof(double));

    pSPARC_ATOM->mixing_hist_xkm1 = (double *)malloc(col*(Nd-1)*sizeof(double));
    pSPARC_ATOM->mixing_hist_fkm1 = (double *)calloc(col*(Nd-1), sizeof(double));
    if (pSPARC_ATOM->spinFlag == 0) {
        memcpy(pSPARC_ATOM->mixing_hist_xkm1, pSPARC_ATOM->electronDens, (Nd - 1)*sizeof(double));
    } else {
        memcpy(pSPARC_ATOM->mixing_hist_xkm1, pSPARC_ATOM->electronDens+Nd-1, col*(Nd - 1)*sizeof(double));
    }

    memcpy(pSPARC_ATOM->mixing_hist_xk, pSPARC_ATOM->mixing_hist_xkm1, col*(Nd - 1)*sizeof(double));
    pSPARC_ATOM->X = (double *)calloc(col*(Nd-1)*pSPARC_ATOM->MixingHistory, sizeof(double));
    pSPARC_ATOM->F = (double *)calloc(col*(Nd-1)*pSPARC_ATOM->MixingHistory, sizeof(double));

    /* SCF loop for local and semi-local part */
    scfLoopAtom(pSPARC_ATOM);
    while (!pSPARC_ATOM->scf_success) {
#ifdef DEBUG
        printf("#######################################################################\n");
        printf("                         WARNING! BE CAREFUL.                          \n");
        printf("#######################################################################\n");
        printf("\nAdjusting SCF tolerance for convergence....\n");
#endif
        pSPARC_ATOM->SCF_tol *= 10;
#ifdef DEBUG
        printf("Adjusted SCF tolerance now is %0.6e\n",pSPARC_ATOM->SCF_tol);
#endif
        scfLoopAtom(pSPARC_ATOM);
    }
    if ((pSPARC_ATOM->scf_success) && (pSPARC_ATOM->usefock == 0)) {
#ifdef DEBUG
        printf("\nAtom ground state calculation successful!\nDensity converged to %0.6e\n", pSPARC_ATOM->SCF_tol);
#endif
    }

    /* Hybrid Functional */
    if (pSPARC_ATOM->usefock) pSPARC_ATOM->usefock++;

    if (pSPARC_ATOM->usefock > 0) {
        int countExx = 0; double exx_error;
        double *denMatPrev = (double *)malloc((Nd-1)*(Nd-1)*sizeof(double));
        double *denMat = (double *)malloc((Nd-1)*(Nd-1)*sizeof(double));

        densityMatrix(pSPARC_ATOM, denMatPrev);

        while (countExx < pSPARC_ATOM->MAXIT_FOCK) {
#ifdef DEBUG
            printf("*************************************************************\n");
            printf("No.%d Exx outer loop.\n",countExx+1);
#endif

            exxOperatorAtom(pSPARC_ATOM);

            scfLoopAtom(pSPARC_ATOM);
            densityMatrix(pSPARC_ATOM, denMat);
            normedRelativeError(denMatPrev, denMat, (Nd-1)*(Nd-1), &exx_error);
#ifdef DEBUG
            printf("\nExx outer loop error: %.14e \n",exx_error);
#endif

            if ((exx_error <= pSPARC_ATOM->TOL_FOCK) && (countExx+1 >= pSPARC_ATOM->MINIT_FOCK)) {
// #ifdef DEBUG
                printf("Desired outer loop accuracy of %0.6e reached!\n",pSPARC_ATOM->TOL_FOCK);
                printf("Finished outer loop in %d steps.\n", countExx+1);
// #endif
                exxOperatorAtom(pSPARC_ATOM);
                break;
            } else {
                memcpy(denMatPrev, denMat, sizeof(double)*(Nd-1)*(Nd-1));
                countExx++;
            }
        }

        if ((countExx = pSPARC_ATOM->MAXIT_FOCK) && (exx_error > pSPARC_ATOM->TOL_FOCK)){
// #ifdef DEBUG
            printf("Warning: Exact Exchange outer loop did not converge. Maximum outer loop iterations reached!\n");
// #endif
            exxOperatorAtom(pSPARC_ATOM);
        }
        
        free(denMat); free(denMatPrev);
    }
}

/////////////////////////// Potentials - Non local, Local, Hartree, XC /////////////////////////////////////////

/**
 * @brief Computes the nonlocal potential matrices from pseudopotential
 *        Refer to Eq. 5b, 24b of Bhowmik, S. et al. Comput. Phys. Commun. 308 (2025) 109448
 */
void compute_nonlocal_potential(SPARC_ATOM_OBJ *pSPARC_ATOM) {
    if (pSPARC_ATOM->psd == NULL) {
#ifdef DEBUG
        printf("Error: pSPARC_ATOM->psd is NULL\n");
#endif
        exit(EXIT_FAILURE);
    }
#ifdef DEBUG
    printf("\nComputing non local potential.\n");
#endif

    int lmax = pSPARC_ATOM->psd->lmax;
    double *rc = pSPARC_ATOM->psd->rc;
    double *r_grid_vloc = pSPARC_ATOM->psd->RadialGrid;
    double *w = pSPARC_ATOM->w;
    double *r = pSPARC_ATOM->r;
    double *int_scale = pSPARC_ATOM->int_scale;
    int Nd = pSPARC_ATOM->Nd;
    int psd_len = pSPARC_ATOM->psd->size;

    int *lpos = (int *)calloc((lmax+2), sizeof(int));
    lpos[0] = 0;
    for (int l = 0; l <= lmax; l++) {
        lpos[l+1] = lpos[l] + pSPARC_ATOM->psd->ppl[l];
    }

    //  Allocate memory for the array of VNLOC_OBJ_ATOM (Vnl)
    pSPARC_ATOM->Vnl = (VNLOC_OBJ_ATOM *)malloc((lmax + 1) * sizeof(VNLOC_OBJ_ATOM));
    if (!pSPARC_ATOM->Vnl) {
#ifdef DEBUG
        printf("Error: Memory allocation failed for Vnl array\n");
#endif
        exit(EXIT_FAILURE);
    }


    for (int l = 0; l <= lmax; l++) {
        if (l == *pSPARC_ATOM->localPsd) {
            continue; // Skip the local potential channel
        }

        double rc_max = rc[l];

        // Identify indices where r <= rc_max
        int *index = (int *)malloc((Nd + 1) * sizeof(int));
        int count = 0;
        for (int j = 0; j <= Nd; j++) {
            if (r[j] <= rc_max) {
                index[j] = 1;
                count++;
            } else {
                index[j] = 0;
            }
        }

        double *r_proj_in = (double *)malloc(count * sizeof(double));
        // int idx = 0;
        int idx = count - 1;
        for (int j = 0; j <= Nd; j++) {
            if (index[j]) {
                // r_proj_in[idx++] = r[j];
                r_proj_in[idx--] = r[j];
            }
        }

        double *gamma_Jl = &(pSPARC_ATOM->psd->Gamma[lpos[l]]);
        double *proj = &(pSPARC_ATOM->psd->UdV[lpos[l] * pSPARC_ATOM->psd->size]);
        double *Chi_l = (double *)calloc(count, sizeof(double));

        // Initialize matrix V
        double *V = (double *)calloc((Nd + 1) * (Nd + 1), sizeof(double));

        // Process each projector
        for (int pp = 0; pp < pSPARC_ATOM->psd->ppl[l]; pp++) {
            // Interpolate Chi_l
            
            double *proj_col = &proj[pp * pSPARC_ATOM->psd->size];
            SplineInterp(r_grid_vloc, proj_col, pSPARC_ATOM->psd->size, r_proj_in, 
                        Chi_l, count, 
                        pSPARC_ATOM->psd->SplineFitUdV+(lpos[l] + pp)*psd_len);
            
            // Compute rChi_l and weighted rChi_l
            double *rChi_l = (double *)calloc(Nd + 1, sizeof(double));
            double *wt_rChi_l = (double *)calloc(Nd + 1, sizeof(double));
            idx = count - 1;
            for (int j = 0; j <= Nd; j++) {
                if (index[j]) {
                    rChi_l[j] = Chi_l[idx--] * r[j];
                    wt_rChi_l[j] = (w[j] * rChi_l[j]) / int_scale[j];
                }
            }

            // Update V
            for (int row = 0; row <= Nd; row++) {
                for (int col = 0; col <= Nd; col++) {
                    V[col * (Nd + 1) + row] += gamma_Jl[pp] * rChi_l[row] * wt_rChi_l[col];
                }
            }
            free(rChi_l);
            free(wt_rChi_l);
        }
        free(Chi_l);

        // Store Vnl
        pSPARC_ATOM->Vnl[l].V = (double *)malloc((Nd - 1) * (Nd - 1) * sizeof(double));
        idx = 0;
        for (int col = 1; col <= Nd-1; col++){
            for (int row = 1; row <= Nd-1; row++){
                pSPARC_ATOM->Vnl[l].V[idx++] = V[col * (Nd + 1) + row];
            }
        }

        free(V);
        free(r_proj_in);
        free(index);
    }

    free(lpos);
}

/**
 * @brief Local part of pseudopotential V_{loc}
 *        Refer to Eq. 5c, 24c of Bhowmik, S. et al. Comput. Phys. Commun. 308 (2025) 109448
 * 
 */
void compute_local_potential(SPARC_ATOM_OBJ *pSPARC_ATOM) {
#ifdef DEBUG
    printf("\nComputing local potential.\n");
#endif

    double *r_grid_vloc = pSPARC_ATOM->psd->RadialGrid;
    double *rVloc = pSPARC_ATOM->psd->rVloc;
    int psd_len = pSPARC_ATOM->psd->size;
    double rc = r_grid_vloc[psd_len - 1];
    int Nd = pSPARC_ATOM->Nd;
    double *r = pSPARC_ATOM->r;

    // Identify indices where r <= rc_max
    int *index = (int *)malloc((Nd + 1) * sizeof(int));
    int count = 0;
    for (int j = 0; j <= Nd; j++) {
        if (r[j] <= rc) {
            index[j] = 1;
            count++;
        } else {
            index[j] = 0;
        }
    }

    double *r_proj_in = (double *)malloc(count * sizeof(double));
    int idx = count - 1;
    for (int j = 0; j <= Nd; j++) {
        if (index[j]) {
            r_proj_in[idx--] = r[j];
        }
    }

    double *VJ_interp = (double *)malloc(count * sizeof(double));
    SplineInterp(r_grid_vloc, rVloc, psd_len, r_proj_in, VJ_interp, count, 
                pSPARC_ATOM->psd->SplinerVlocD);
    
    double *VJ = (double *)malloc((Nd + 1) * sizeof(double));
    idx = count - 1;
    for (int j = 0; j <= Nd; j++) {
        if (index[j]) {
            VJ[j] = VJ_interp[idx--];
        } else {
            VJ[j] = -(*pSPARC_ATOM->Znucl);
        }
    }

    free(index); free(r_proj_in); free(VJ_interp); 

    pSPARC_ATOM->VJ = (double *)malloc((pSPARC_ATOM->Nd - 1) * sizeof(double));
    for (int i = 1; i < Nd; i++) {
        pSPARC_ATOM->VJ[(i-1)] = VJ[i]/r[i];
    }

    free(VJ);
}

/**
 * @brief Poisson solve to get the Hartree Potential
 *        Refer to Eq. 8b, 17b, and 24c of Bhowmik, S. et al. Comput. Phys. Commun. 308 (2025) 109448
 * 
 */
void poissonSolve_atom(SPARC_ATOM_OBJ *pSPARC_ATOM) {
    int Nd = pSPARC_ATOM->Nd;
    double *r = pSPARC_ATOM->r;
    double *laplacian = pSPARC_ATOM->laplacian;

    double *L = (double *)calloc(Nd * Nd, sizeof(double));
    for (int col = 0; col < Nd; col++) {
        for (int row = 0; row < Nd; row++) {
            L[col*Nd + row] = laplacian[col*(Nd + 1) + row];
        }
    }

    for (int col = 0; col < Nd; col++) {
        L[col * Nd] = 0.0;
    }

    L[0] = 1.0;

    double *rho = pSPARC_ATOM->electronDens;
    double *modified_rho = (double *)malloc(Nd * sizeof(double));
    modified_rho[0] = 0.0;
    for (int j = 0; j < Nd - 1; j++){
        modified_rho[j+1] = rho[j];
    }

    double *RHS = (double *)malloc(Nd * sizeof(double));
    for (int j = 0; j < Nd; j++) {
        RHS[j] = -4 * M_PI * r[j] *modified_rho[j];
    }
    RHS[0] = *pSPARC_ATOM->Znucl;

    free(modified_rho);

    int info;
    int *ipiv = (int *)malloc(Nd * sizeof(int));
    double *rphi = (double *)malloc(Nd * sizeof(double));

    for (int i = 0; i < Nd; i++){
        rphi[i] = RHS[i];
    }

    info = LAPACKE_dgesv(LAPACK_COL_MAJOR, Nd, 1, L, Nd, ipiv, rphi, Nd);
    if (info != 0) {
#ifdef DEBUG
        fprintf(stderr, "Error: LAPACK dgesv failed with info = %d.\n", info);
#endif
        free(ipiv);
        free(rphi);
        exit(EXIT_FAILURE);
    }

    for (int i = 1; i < Nd; i++) {
        pSPARC_ATOM->phi[i-1] = rphi[i] / r[i];
    }

    // Free allocated memory
    free(ipiv);
    free(rphi);
    free(L); free(RHS);
}

/**
 * @brief Calculate XC Potential for a given electron sensity (and set of wavefunctions)
 *        Refer to Eq. 11a-c of Bhowmik, S. et al. Comput. Phys. Commun. 308 (2025) 109448
 * 
 */
void Calculate_Vxc_atom(SPARC_ATOM_OBJ *pSPARC_ATOM){
    int ncol = pSPARC_ATOM->nspden;
    int Nd = pSPARC_ATOM->Nd;
    int sz = ncol * (Nd - 1); // first and last points are not needed
    double *rho = (double *)malloc(sz * sizeof(double));
    double *r_rho, *sigma, *Drho, *tau;
    r_rho = sigma = Drho = NULL;
    double *D = NULL;

    // add core electron density if needed
    add_rho_core_atom(pSPARC_ATOM, pSPARC_ATOM->electronDens, rho, ncol);

    // calculate sigma
    if (pSPARC_ATOM->isGradient){
        r_rho = (double *)malloc(sz * sizeof(double));
        sigma = (double *)malloc(sz * sizeof(double));
        Drho = (double *)malloc(sz * sizeof(double));

        // calculate r_rho
        for (int n = 0; n < ncol; n++){
            for (int i = 0; i < Nd-1; i++){
                r_rho[i + n*(Nd - 1)] = pSPARC_ATOM->r[i+1]*rho[i + n*(Nd - 1)];
            }
        }

        // Extract gradient matrix
        D = (double *)malloc((Nd - 1) * (Nd - 1) * sizeof(double));
        for (int j = 1; j < Nd; j++) {  // Iterate column-first
            for (int i = 1; i < Nd; i++) {
                D[(i - 1) + (j - 1) * (Nd - 1)] = pSPARC_ATOM->grad[i + j * (Nd+1)];  // Proper column-major indexing
            }
        }

        // matrix vector product
        for (int n = 0; n < ncol; n++){
            cblas_dgemv(CblasColMajor, CblasNoTrans, Nd - 1, Nd - 1, 1.0, D, Nd - 1, r_rho + n * (Nd - 1), 1, 0.0, Drho + n * (Nd - 1), 1);
        }

        // Drho = (D*r_rho - rho)./r
        for (int n = 0; n < ncol; n++){
            for (int i = 0; i < Nd - 1; i++){
                Drho[i + n*(Nd - 1)] = (Drho[i + n*(Nd - 1)] - rho[i + n*(Nd - 1)])/pSPARC_ATOM->r[i+1];
            }
        }

        // get sigma and truncate to tolerance if needed
        for (int i = 0; i < sz; i++){
            sigma[i] = Drho[i]*Drho[i];
            if (sigma[i] < pSPARC_ATOM->xc_sigmatol){
                sigma[i] = pSPARC_ATOM->xc_sigmatol;
            }
        }

        free(r_rho);
    }

    // metaGGA
    if (pSPARC_ATOM->ixc[2]) { // 1st SCF, calculate PBE
        if (pSPARC_ATOM->countPotentialCalculate == 0) {
            pSPARC_ATOM->ixc[0] = 2; pSPARC_ATOM->ixc[1] = 3;
            pSPARC_ATOM->ixc[2] = 1; pSPARC_ATOM->ixc[3] = 0;
            pSPARC_ATOM->xcoption[0] = 1; pSPARC_ATOM->xcoption[1] = 1;
        } else {
            tau = pSPARC_ATOM->tau;
        }
    }

    if (pSPARC_ATOM->spinFlag == 0){
        double *ex = (double *)malloc((Nd - 1) * sizeof(double));
        double *ec = (double *)malloc((Nd - 1) * sizeof(double));
        double *vx = (double *)malloc((Nd - 1) * sizeof(double));
        double *vc = (double *)malloc((Nd - 1) * sizeof(double));
        double *v2x, *v2c, *v3x, *v3c;
        v2x = v2c = NULL;
        if (pSPARC_ATOM->isGradient) {
            v2x = (double *)malloc((Nd - 1) * sizeof(double));
            v2c = (double *)malloc((Nd - 1) * sizeof(double));
        }
        v3x = v3c = NULL;
        if (pSPARC_ATOM->ixc[2]) // metaGGA, d(n\epsilon)/d\tau
        {
            v3x = (double *)malloc((Nd - 1) * sizeof(double));
            v3c = (double *)malloc((Nd - 1) * sizeof(double));
        }

        // iexch
        switch (pSPARC_ATOM->ixc[0])
        {
            case 1:
                slater(Nd - 1, rho, ex, vx);
                break;
            case 2:
                pbex(Nd - 1, rho, sigma, pSPARC_ATOM->xcoption[0], ex, vx, v2x);
                break;
            case 3:
                rPW86x(Nd - 1, rho, sigma, ex, vx, v2x);
                break;
            case 4:
                scanx(Nd - 1, rho, sigma, tau, ex, vx, v2x, v3x);
                break;
            case 5:
                rscanx(Nd - 1, rho, sigma, tau, ex, vx, v2x, v3x);
                break;
            case 6:
                r2scanx(Nd - 1, rho, sigma, tau, ex, vx, v2x, v3x);
                break;
            default:
                memset(ex, 0, sizeof(double) * (Nd - 1));
                memset(vx, 0, sizeof(double) * (Nd - 1));
                if (pSPARC_ATOM->isGradient){
                    memset(v2x, 0, sizeof(double) * (Nd - 1));
                }
                break;
        }

        // icorr
        switch (pSPARC_ATOM->ixc[1])
        {
            case 1:
                pz(Nd - 1, rho, ec, vc);
                break;
            case 2:
                pw(Nd - 1, rho, ec, vc);
                break;
            case 3:
                pbec(Nd - 1, rho, sigma, pSPARC_ATOM->xcoption[1], ec, vc, v2c);
                break;
            case 4:
                scanc(Nd - 1, rho, sigma, tau, ec, vc, v2c, v3c);
                break;
            case 5:
                rscanc(Nd - 1, rho, sigma, tau, ec, vc, v2c, v3c);
                break;
            case 6:
                r2scanc(Nd - 1, rho, sigma, tau, ec, vc, v2c, v3c);
                break;
            default:
                memset(ec, 0, sizeof(double) * (Nd - 1));
                memset(vc, 0, sizeof(double) * (Nd - 1));
                if (pSPARC_ATOM->isGradient){
                    memset(v2c, 0, sizeof(double) * (Nd - 1));
                }
                break;
        }

        if (pSPARC_ATOM->usefock > 1) {
            for (int i = 0; i < Nd - 1; i++) {
                ex[i] *= (1-pSPARC_ATOM->exx_frac);
                vx[i] *= (1-pSPARC_ATOM->exx_frac);
                v2x[i] *= (1-pSPARC_ATOM->exx_frac);
            }
        }

        for (int i = 0; i < Nd - 1; i++){
            pSPARC_ATOM->e_xc[i] = ex[i] + ec[i];
            pSPARC_ATOM->XCPotential[i] = vx[i] + vc[i];
            if (pSPARC_ATOM->isGradient){
                pSPARC_ATOM->Dxcdgrho[i] = v2x[i] + v2c[i];
            }
            if (pSPARC_ATOM->ixc[2] && (pSPARC_ATOM->countPotentialCalculate > 0)) {
                pSPARC_ATOM->vxcMGGA3[i] = v3x[i] + v3c[i];
            }
        }

        if (pSPARC_ATOM->isGradient) {
            // v2xc .* drho 
            double *v2xc_drho = (double *)malloc((Nd - 1) * sizeof(double));
            for (int i = 0; i < Nd - 1; i++){
                v2xc_drho[i] = pSPARC_ATOM->Dxcdgrho[i] * Drho[i];
            }

            // D * (v2xc .* drho)
            double *temp_result = (double *)malloc((Nd - 1) * sizeof(double));
            cblas_dgemv(CblasColMajor, CblasNoTrans, Nd - 1, Nd - 1, 1.0, D, Nd - 1, v2xc_drho, 1, 0.0, temp_result, 1);

            // (2 ./ r) .* v2xc .* drho
            double *scaled_v2xc_drho = (double *)malloc((Nd - 1) * sizeof(double));
            for (int i = 0; i < Nd - 1; i++) {
                scaled_v2xc_drho[i] = (2.0 / pSPARC_ATOM->r[i + 1]) * v2xc_drho[i];
            }

            for (int i = 0; i < Nd - 1; i++) {
                pSPARC_ATOM->XCPotential[i] = pSPARC_ATOM->XCPotential[i] - temp_result[i] - scaled_v2xc_drho[i];
            }

            free(v2xc_drho);
            free(temp_result);
            free(scaled_v2xc_drho);
        }

        free(ex);
        free(ec);
        free(vx);
        free(vc);
        if (pSPARC_ATOM->isGradient){
            free(v2x);
            free(v2c);
        }
        if (pSPARC_ATOM->ixc[2]){
            free(v3x);
            free(v3c);
        }
    } else {
        double *ex = (double *)malloc((Nd - 1) * sizeof(double));
        double *ec = (double *)malloc((Nd - 1) * sizeof(double));
        double *vx = (double *)malloc(2*(Nd - 1) * sizeof(double));
        double *vc = (double *)malloc(2*(Nd - 1) * sizeof(double));
        double *v2x, *v2c, *v3x, *v3c;
        v2x = v2c = v3x = v3c = NULL;
        if (pSPARC_ATOM->isGradient) {
            v2x = (double *)malloc(2*(Nd - 1) * sizeof(double));
            v2c = (double *)malloc((Nd - 1) * sizeof(double));
        }
        if (pSPARC_ATOM->ixc[2]) // metaGGA, d(n\epsilon)/d\tau
        {
            v3x = (double *)malloc(2*(Nd - 1) * sizeof(double));
            v3c = (double *)malloc((Nd - 1) * sizeof(double));
        }

        // iexch
        switch (pSPARC_ATOM->ixc[0])
        {
            case 1:
                slater_spin((Nd-1), rho, ex, vx);
                break;
            case 2:
                pbex_spin((Nd - 1), rho, sigma, pSPARC_ATOM->xcoption[0], ex, vx, v2x);
                break;
            case 3:
                rPW86x_spin((Nd - 1), rho, sigma + (Nd - 1), ex, vx, v2x);
                break;
            case 4:
                scanx_spin((Nd - 1), rho, sigma, tau, ex, vx, v2x, v3x);
                break;
            case 5:
                rscanx_spin((Nd - 1), rho, sigma, tau, ex, vx, v2x, v3x);
                break;
            case 6:
                r2scanx_spin((Nd - 1), rho, sigma, tau, ex, vx, v2x, v3x);
                break;
            default:
                memset(ex, 0, sizeof(double) * (Nd - 1));
                memset(vx, 0, sizeof(double) * (Nd - 1)*2);
                if (pSPARC_ATOM->isGradient){
                    memset(v2x, 0, sizeof(double) * (Nd - 1)*2);
                }
                break;
        }

        // icorr
        switch(pSPARC_ATOM->ixc[1])
        {
            case 1:
                pz_spin((Nd - 1), rho, ec, vc);
                break;
            case 2:
                pw_spin((Nd - 1), rho, ec, vc);
                break;
            case 3:
                pbec_spin((Nd - 1), rho, sigma, pSPARC_ATOM->xcoption[1], ec, vc, v2c);
                break;
            case 4:
                scanc_spin((Nd - 1), rho, sigma, tau, ec, vc, v2c, v3c);
                break;
            case 5:
                rscanc_spin((Nd - 1), rho, sigma, tau, ec, vc, v2c, v3c);
                break;
            case 6:
                r2scanc_spin((Nd - 1), rho, sigma, tau, ec, vc, v2c, v3c);
                break;
            default:
                memset(ec, 0, sizeof(double) * (Nd - 1));
                memset(vc, 0, sizeof(double) * (Nd - 1)*2);
                if (pSPARC_ATOM->isGradient) {
                    memset(v2c, 0, sizeof(double) * (Nd - 1));
                }
                break;
        }

        if (pSPARC_ATOM->usefock > 1) {
            for (int i = 0; i < Nd-1; i++) {
                ex[i] *= (1-pSPARC_ATOM->exx_frac);
                for(int spn_i = 0; spn_i < 2; spn_i++) {
                    vx[i + spn_i*(Nd-1)] *= (1-pSPARC_ATOM->exx_frac);
                    v2x[i + spn_i*(Nd-1)] *= (1-pSPARC_ATOM->exx_frac);
                }
            }
        }

        for (int i = 0; i < Nd - 1; i++){
            pSPARC_ATOM->e_xc[i] = ex[i] + ec[i];
            pSPARC_ATOM->XCPotential[i] = vx[i] + vc[i];
            pSPARC_ATOM->XCPotential[i+Nd-1] = vx[i+Nd-1] + vc[i+Nd-1];
            if (pSPARC_ATOM->isGradient) {
                pSPARC_ATOM->Dxcdgrho[i] = v2c[i];
                pSPARC_ATOM->Dxcdgrho[i+Nd-1] = v2x[i];
                pSPARC_ATOM->Dxcdgrho[i+2*(Nd-1)] = v2x[i+Nd-1];
            }
            if (pSPARC_ATOM->ixc[2] && (pSPARC_ATOM->countPotentialCalculate > 0)) {
                pSPARC_ATOM->vxcMGGA3[i] = v3x[i] + v3c[i];
                pSPARC_ATOM->vxcMGGA3[i+Nd-1] = v3x[i+Nd-1] + v3c[i];
            }
        }

        if (pSPARC_ATOM->isGradient) {
            double *v2xc_drho = (double *)malloc(3 * (Nd - 1) * sizeof(double));
            double *Vxc_temp = (double *)malloc(3 * (Nd - 1) * sizeof(double));
            double *D_result = (double *)malloc(3 * (Nd - 1) * sizeof(double));

            // v2xc .* drho (element-wise multiplication for each column)
            for (int n = 0; n < 3; n++) {
                for (int i = 0; i < Nd - 1; i++){
                    v2xc_drho[n * (Nd - 1) + i] = pSPARC_ATOM->Dxcdgrho[n * (Nd - 1) + i] * Drho[n * (Nd - 1) + i];
                }
            }

            // D * (v2xc .* drho)
            for (int n = 0; n < ncol; n++){
                cblas_dgemv(CblasColMajor, CblasNoTrans, Nd - 1, Nd - 1, 1.0,
                            D, Nd - 1, v2xc_drho + n * (Nd - 1), 1, 0.0, D_result + n * (Nd - 1), 1);
            }

            // (2./r) .* v2xc .* drho
            for (int n = 0; n < 3; n++) {
                for (int i = 0; i < Nd - 1; i++){
                    Vxc_temp[n * (Nd - 1) + i] = D_result[n * (Nd - 1) + i] +
                                           (2.0 / pSPARC_ATOM->r[i + 1]) * v2xc_drho[n * (Nd - 1) + i];
                }
            }

            for (int i = 0; i < Nd - 1; i++) {
                pSPARC_ATOM->XCPotential[i] -= (Vxc_temp[i + (Nd - 1)] + Vxc_temp[i]);
                pSPARC_ATOM->XCPotential[i + (Nd - 1)] -= (Vxc_temp[i + 2*(Nd - 1)] + Vxc_temp[i]);
            }

            free(v2xc_drho);
            free(Vxc_temp);
            free(D_result);

        }

        free(ex);
        free(ec);
        free(vx);
        free(vc);
        if (pSPARC_ATOM->isGradient){
            free(v2x);
            free(v2c);
        }
        if (pSPARC_ATOM->ixc[2]){
            free(v3x);
            free(v3c);
        }
    }

    // restore metaGGA labels after 1st SCF
    if (pSPARC_ATOM->ixc[2]) {
        if (pSPARC_ATOM->countPotentialCalculate == 0){
            if (strcmpi(pSPARC_ATOM->XC, "SCAN") == 0) {
                pSPARC_ATOM->ixc[0] = 4; pSPARC_ATOM->ixc[1] = 4; 
                pSPARC_ATOM->ixc[2] = 1; pSPARC_ATOM->ixc[3] = 0;
                pSPARC_ATOM->xcoption[0] = 0; pSPARC_ATOM->xcoption[1] = 0;
            } else if (strcmpi(pSPARC_ATOM->XC, "RSCAN") == 0) {
                pSPARC_ATOM->ixc[0] = 5; pSPARC_ATOM->ixc[1] = 5; 
                pSPARC_ATOM->ixc[2] = 1; pSPARC_ATOM->ixc[3] = 0;
                pSPARC_ATOM->xcoption[0] = 0; pSPARC_ATOM->xcoption[1] = 0;
            } else if (strcmpi(pSPARC_ATOM->XC, "R2SCAN") == 0) {
                pSPARC_ATOM->ixc[0] = 6; pSPARC_ATOM->ixc[1] = 6; 
                pSPARC_ATOM->ixc[2] = 1; pSPARC_ATOM->ixc[3] = 0;
                pSPARC_ATOM->xcoption[0] = 0; pSPARC_ATOM->xcoption[1] = 0;
            }
        }
    }

    free(rho);
    if (pSPARC_ATOM->isGradient){
        free(sigma);
        free(Drho);
        free(D);
    }
}

/**
* @brief add core electron density if needed
*/
void add_rho_core_atom(SPARC_ATOM_OBJ *pSPARC_ATOM, double *rho_in, double *rho_out, int ncol){
    assert(ncol == 1 || ncol == 3);
    int Nd = pSPARC_ATOM->Nd;
    for (int n = 0; n < ncol; n++){
        for (int i = 0; i < Nd - 1; i++) // i < Nd - 1 as first and last points are removed from Nd + 1 points in total (in MATLAB loop runs from 2:Nd)
        {
            rho_out[i + n*(Nd - 1)] = rho_in[i + n*(Nd - 1)];
            // for NLCC we use rho+rho_core to evaluate Vxc[rho+rho_core]
            if (pSPARC_ATOM->NLCC_flag){
                if (n == 0)
                    rho_out[i + n*(Nd - 1)] += pSPARC_ATOM->electronDens_core[i]; // i+1 as the first point is removed in storing the single atom electronDens
                else
                    rho_out[i + n*(Nd - 1)] += 0.5 * pSPARC_ATOM->electronDens_core[i]; // i+1 as the first point is removed in storing the single atom electronDens
            }
            if(rho_out[i + n*(Nd - 1)] < pSPARC_ATOM->xc_rhotol)
                rho_out[i + n*(Nd - 1)] = pSPARC_ATOM->xc_rhotol;
        }
    }

    if (ncol == 3){
        for (int i = 0; i < Nd - 1; i++){
            rho_out[i] = rho_out[(Nd - 1) + i] + rho_out[2*(Nd - 1) + i];
        }
    }
 }

///////////////////////////////////// SCF Functions /////////////////////////////////////////

/**
 * @brief SCF Loop 
 *        1. Poisson Solve to calculate Hartree potential
 *        2. Calculate XC potential
 *        3. Perform Eigen solve
 *        4. Update wavefunctions
 *        5. Calculate electron density (Refer to Eq. 2, 18, and 23 of 
 *           Bhowmik, S. et al. Comput. Phys. Commun. 308 (2025) 109448)
 */
void scfLoopAtom(SPARC_ATOM_OBJ *pSPARC_ATOM) {
    int Nd = pSPARC_ATOM->Nd;
    int col = pSPARC_ATOM->nspin;
    double *w = pSPARC_ATOM->w;
    double *r = pSPARC_ATOM->r;
    double *int_scale = pSPARC_ATOM->int_scale;

    double scf_tol = pSPARC_ATOM->SCF_tol;
    if (pSPARC_ATOM->usefock > 0) {
        if (pSPARC_ATOM->usefock % 2 != 0) scf_tol = pSPARC_ATOM->TOL_SCF_INIT;
    }

    int iter = 0; // Storing the iteration count
    int orb_count; // To store the total number of orbitals

    // States related variables
    int val_len = pSPARC_ATOM->val_len;
    int min_l = pSPARC_ATOM->min_l;
    int max_l = pSPARC_ATOM->max_l;
    int no_orb_per_l;

    // Allocate memory for eigenvectors/eigenvalues
    double *eigvecs = (double *)malloc(col*(Nd - 1)*(Nd - 1)*sizeof(double));
    double *eigvals = (double *)malloc(col*(Nd - 1)*sizeof(double));

    // Initialise pointers to rho_up and rho_dw
    double *rhoUp = (double *)malloc((Nd - 1)*sizeof(double)); 
    double *rhoDw = NULL;
    if (pSPARC_ATOM->spinFlag) rhoDw = (double *)malloc((Nd - 1)*sizeof(double));
    int *o_up = NULL, *o_dw = NULL;

    // Allocate memory for normalizing eigen-vector
    double *eigvec = (double *)malloc((Nd-1)*sizeof(double));
    double integral;

    double scf_error;
    double *mag = NULL;

    // Store timings
    double t1, t2, t1_SCF, t2_SCF;
#ifdef DEBUG
    printf("*************************************************************\n");
    printf("\nStarting SCF loop with guess-density.\n");
#endif
    t1_SCF = MPI_Wtime();
    while (iter+1 <= pSPARC_ATOM->MAXIT_SCF) {
#ifdef DEBUG
        printf("========================SCF Iteration #%d=======================\n",iter+1);
#endif

        // Poisson solve
        t1 = MPI_Wtime();
        poissonSolve_atom(pSPARC_ATOM);
        t2 = MPI_Wtime();
#ifdef DEBUG
        printf("\nThe Poisson solve took %.3f ms\n",(t2-t1)*1000);
#endif

        // XC
        if ((pSPARC_ATOM->ixc[2]) && pSPARC_ATOM->countPotentialCalculate==0){
#ifdef DEBUG
            printf("This step is PBE.\n");
#endif
        }
        t1 = MPI_Wtime();
        Calculate_Vxc_atom(pSPARC_ATOM);
        t2 = MPI_Wtime();
#ifdef DEBUG
        printf("The Vxc calculation took %.3f ms\n",(t2-t1)*1000);
#endif
        pSPARC_ATOM->countPotentialCalculate++; // increase potential calculation counter for MGGA, etc.

        // Reset electron density memory first
        memset(rhoUp,0.0,sizeof(double)*(Nd-1)); // reset memory
        if (pSPARC_ATOM->spinFlag) {
            memset(rhoDw,0.0,sizeof(double)*(Nd-1)); // reset memory
        }

        // Eigen solve per azimuthal quantum number
        orb_count = 0;
        t1 = MPI_Wtime();
        for (int l = min_l; l <= max_l; l++) {
            // printf("The min_l is %d\n",min_l); // debug
            if (l == 0) {
                no_orb_per_l = pSPARC_ATOM->lcount0;
                o_up = pSPARC_ATOM->f0_up;
                o_dw = pSPARC_ATOM->f0_dw;
            } else if (l == 1) {
                no_orb_per_l = pSPARC_ATOM->lcount1;
                o_up = pSPARC_ATOM->f1_up;
                o_dw = pSPARC_ATOM->f1_dw;
            } else if (l == 2) {
                no_orb_per_l = pSPARC_ATOM->lcount2;
                o_up = pSPARC_ATOM->f2_up;
                o_dw = pSPARC_ATOM->f2_dw;
            } else if (l == 3) {
                no_orb_per_l = pSPARC_ATOM->lcount3;
                o_up = pSPARC_ATOM->f3_up;
                o_dw = pSPARC_ATOM->f3_dw;
            }

            eigenSolve(pSPARC_ATOM, l, eigvecs, eigvals);

            // Normalize and store the orbitals
            if (pSPARC_ATOM->spinFlag == 0) {
                for (int jj = 0; jj < no_orb_per_l; jj++) {
                    memcpy(eigvec, eigvecs+jj*(Nd-1), (Nd-1)*sizeof(double));
                    integralOrbitalSquared(pSPARC_ATOM, eigvec, &integral);
                    for (int ii = 0; ii < Nd - 1; ii++) {
                        eigvecs[ii + jj*(Nd - 1)] /= sqrt(integral);
                        rhoUp[ii] += (0.25/M_PI)*(o_up[jj] + o_dw[jj])
                                    *pow(eigvecs[ii + jj*(Nd - 1)],2)/(r[ii+1]*r[ii+1]);
                    }
                    memcpy(pSPARC_ATOM->orbitals+orb_count*(Nd-1), 
                            eigvecs+jj*(Nd-1), (Nd-1)*sizeof(double));
                    pSPARC_ATOM->eigenVal[orb_count] = eigvals[jj];
                    pSPARC_ATOM->orbital_l[orb_count] = l;
                    pSPARC_ATOM->occ[orb_count] = (o_up[jj] + o_dw[jj]);
                    orb_count++;
                }
            } else {

                for (int jj = 0; jj < no_orb_per_l; jj++) {
                    // Up spin
                    memcpy(eigvec, eigvecs+jj*(Nd-1), (Nd-1)*sizeof(double));
                    integralOrbitalSquared(pSPARC_ATOM, eigvec, &integral);
                    for (int ii = 0; ii < Nd - 1; ii++) {
                        eigvecs[ii + jj*(Nd - 1)] /= sqrt(integral);
                        rhoUp[ii] += (0.25/M_PI)*o_up[jj]
                                    *pow(eigvecs[ii + jj*(Nd - 1)],2)/(r[ii+1]*r[ii+1]);
                    }
                    memcpy(pSPARC_ATOM->orbitals+orb_count*(Nd-1), 
                            eigvecs+jj*(Nd-1), (Nd-1)*sizeof(double));
                    pSPARC_ATOM->eigenVal[orb_count] = eigvals[jj];
                    pSPARC_ATOM->orbital_l[orb_count] = l;

                    // Dw spin
                    memcpy(eigvec, eigvecs+jj*(Nd-1)+(Nd-1)*(Nd-1), (Nd-1)*sizeof(double));
                    integralOrbitalSquared(pSPARC_ATOM, eigvec, &integral);
                    for (int ii = 0; ii < Nd - 1; ii++) {
                        eigvecs[ii + jj*(Nd - 1) + (Nd - 1)*(Nd - 1)] /= sqrt(integral);
                        rhoDw[ii] += (0.25/M_PI)*o_dw[jj]
                                    *pow(eigvecs[ii + jj*(Nd - 1) + (Nd - 1)*(Nd - 1)],2)/(r[ii+1]*r[ii+1]);
                    }
                    memcpy(pSPARC_ATOM->orbitals+orb_count*(Nd-1)+val_len*(Nd-1), 
                            eigvecs+jj*(Nd-1)+(Nd-1)*(Nd-1), (Nd-1)*sizeof(double));
                    pSPARC_ATOM->eigenVal[orb_count+val_len] = eigvals[jj+Nd-1];
                    pSPARC_ATOM->orbital_l[orb_count+val_len] = l;

                    pSPARC_ATOM->occ[orb_count] = o_up[jj];
                    pSPARC_ATOM->occ[orb_count+val_len] = o_dw[jj];
                    orb_count++;
                }
            }
        }
        t2 = MPI_Wtime();
#ifdef DEBUG
        printf("The Eigen solve took %.3f ms\n",(t2-t1)*1000);
#endif

        // Copy the densities
        if (pSPARC_ATOM->spinFlag == 0) {
            memcpy(pSPARC_ATOM->electronDens, rhoUp, sizeof(double)*(Nd-1));
        } else {
            memcpy(pSPARC_ATOM->electronDens+Nd-1, rhoUp, sizeof(double)*(Nd-1));
            memcpy(pSPARC_ATOM->electronDens+2*(Nd-1), rhoDw, sizeof(double)*(Nd-1));
            for (int jj = 0; jj < Nd-1; jj++) {
                pSPARC_ATOM->electronDens[jj] = pSPARC_ATOM->electronDens[jj+Nd-1]
                                                + pSPARC_ATOM->electronDens[jj+2*(Nd-1)];
            }
        }

        // Calculate Kinetic Energy Density if metaGGA
        if ((pSPARC_ATOM->ixc[2]) && (pSPARC_ATOM->countPotentialCalculate > 0))
            kineticEnergyDensityAtom(pSPARC_ATOM);
        
        SCFerrorAtom(pSPARC_ATOM, &scf_error);
#ifdef DEBUG
        printf("\nRelative error in density:  %0.10e\n",scf_error);
#endif

        if (pSPARC_ATOM->spinFlag) {
            integral = 0.0;
            mag = pSPARC_ATOM->mag;
            for (int i = 0; i < Nd - 1; i++){
                integral += 4*M_PI*(w[i+1]/int_scale[i+1])*(r[i+1]*r[i+1]*mag[i]);
            }
            pSPARC_ATOM->netM = integral;
#ifdef DEBUG
            printf("Net magnetization:  %.6f\n",pSPARC_ATOM->netM);
#endif
        }

        // Check scf error and break
        if (scf_error < scf_tol) {
#ifdef DEBUG
            printf("\nDensity has already converged to %0.4e.\n", scf_tol);
            printf("Total number of SCF iterations taken:  %d\n",iter+1);
            printf("*************************************************************\n");
#endif
            pSPARC_ATOM->scf_success = 1;
            break;
        }

        periodicPulayMixAtom(pSPARC_ATOM, iter);
        if (pSPARC_ATOM->spinFlag) {
            for (int jj = 0; jj < Nd - 1; jj++){
                pSPARC_ATOM->mag[jj] = pSPARC_ATOM->electronDens[jj + (Nd - 1)] - pSPARC_ATOM->electronDens[jj + 2*(Nd - 1)];
            }
        }

        // Update counter
        iter++;

    }
    t2_SCF = MPI_Wtime();
#ifdef DEBUG
    printf("The SCF cycle took %.3f s\n\n",(t2_SCF-t1_SCF));
#endif
    pSPARC_ATOM->tSCF = t2_SCF-t1_SCF;

    if (iter == pSPARC_ATOM->MAXIT_SCF) {
#ifdef DEBUG
        printf("The SCF cycle did not converge! \n\n");
#endif
    }

    free(eigvecs); free(eigvals); free(eigvec);
    free(rhoUp);
    if (pSPARC_ATOM->spinFlag) free(rhoDw);
}

/**
 * @brief Form the electronic Hamiltonian and perform the Eigen Solve
 * 
 *        H_{el} \tilde{R}_{nl} = E_{el} \tilde{R}_{nl}
 *        Refer to Eq. 8a (and Eq. 17a) of Bhowmik, S. et al. Comput. Phys. Commun. 308 (2025) 109448
 * 
 */
void eigenSolve(SPARC_ATOM_OBJ *pSPARC_ATOM, int l, double *eigvecs, double *eigvals) {
    int Nd = pSPARC_ATOM->Nd;
    double *r = (double *)malloc((Nd - 1) * sizeof(double));
    for (int i = 1; i < Nd; i++) {
        r[i-1] = pSPARC_ATOM->r[i];
    }

    // Store -0.5*laplacian
    double *KEOp = (double *)calloc((Nd - 1) * (Nd - 1), sizeof(double));
    int idx = 0;
    for (int col = 1; col <= Nd-1; col++){
        for (int row = 1; row <= Nd-1; row++){
            KEOp[idx++] = -0.5*pSPARC_ATOM->laplacian[col * (Nd + 1) + row];
        }
    }

    // Form the hamiltonian H_up
    double *Hup = (double *)calloc((Nd - 1) * (Nd - 1), sizeof(double));
    double *Hdw = NULL;
    for (int col = 0; col < Nd - 1; col++) {
        for (int row = 0; row < Nd - 1; row++) {
            Hup[col*(Nd-1) + row] = KEOp[col*(Nd-1) + row] + pSPARC_ATOM->Vnl[l].V[col*(Nd-1) + row];
            if (col == row) {
                Hup[col*(Nd-1) + row] += pSPARC_ATOM->phi[row]
                                        + pSPARC_ATOM->XCPotential[row] + pSPARC_ATOM->VJ[row]
                                        + 0.5*l*(l+1)/(r[row]*r[row]);
            }
        }
    }

    // metaGGA
    if ((pSPARC_ATOM->ixc[2]) && (pSPARC_ATOM->countPotentialCalculate > 1)) {
        double spin = 0.5;
        double *VmGGA = (double *)calloc((Nd - 1) * (Nd - 1), sizeof(double));
        mGGA_hamiltonian_term(pSPARC_ATOM, l, spin, VmGGA);
        for (int col = 0; col < Nd - 1; col++) {
            for (int row = 0; row < Nd - 1; row++) {
                Hup[col*(Nd-1) + row] += VmGGA[col*(Nd-1) + row];
            }
        }
        free(VmGGA);
    }

    // hybrid
    if (pSPARC_ATOM->usefock > 1) {
        for (int col = 0; col < Nd - 1; col++) {
            for (int row = 0; row < Nd - 1; row++) {
                Hup[col*(Nd-1) + row] += pSPARC_ATOM->exx_frac*(pSPARC_ATOM->EXXL[l].VexxUp[col*(Nd - 1) + row]);
            }
        }
    }

    if (pSPARC_ATOM->spinFlag == 1) {
        // Form the hamiltonian H_dw
        Hdw = (double *)calloc((Nd - 1) * (Nd - 1), sizeof(double));
        for (int col = 0; col < Nd - 1; col++) {
            for (int row = 0; row < Nd - 1; row++) {
                Hdw[col*(Nd-1) + row] = KEOp[col*(Nd-1) + row] + pSPARC_ATOM->Vnl[l].V[col*(Nd-1) + row];
                if (col == row) {
                    Hdw[col*(Nd-1) + row] += pSPARC_ATOM->phi[row]
                                            + pSPARC_ATOM->XCPotential[row + (Nd - 1)] + pSPARC_ATOM->VJ[row]
                                            + 0.5*l*(l+1)/(r[row]*r[row]);
                }
            }
        }

        // metaGGA
        if ((pSPARC_ATOM->ixc[2]) && (pSPARC_ATOM->countPotentialCalculate > 1)) {
            double spin = -0.5;
            double *VmGGA = (double *)calloc((Nd - 1) * (Nd - 1), sizeof(double));
            mGGA_hamiltonian_term(pSPARC_ATOM, l, spin, VmGGA);
            for (int col = 0; col < Nd - 1; col++) {
                for (int row = 0; row < Nd - 1; row++) {
                    Hdw[col*(Nd-1) + row] += VmGGA[col*(Nd-1) + row];
                }
            }
            free(VmGGA);
        }

        // hybrid
        if (pSPARC_ATOM->usefock > 1) {
            for (int col = 0; col < Nd - 1; col++) {
                for (int row = 0; row < Nd - 1; row++) {
                    Hdw[col*(Nd-1) + row] += pSPARC_ATOM->exx_frac*(pSPARC_ATOM->EXXL[l].VexxDw[col*(Nd - 1) + row]);
                }
            }
        }
    }

    free(KEOp); free(r);

    double* eigvalsUp = (double*)malloc((Nd-1) * sizeof(double));
    double* eigvalsUp_imag = (double*)malloc((Nd-1) * sizeof(double));
    double* eigvecs_left = NULL; // Not needed in this case
    double* eigvecsUp = (double*)malloc((Nd-1) * (Nd-1) * sizeof(double));

    // LAPACK call to compute eigenvalues and right eigenvectors
    int info = LAPACKE_dgeev(LAPACK_COL_MAJOR, 'N', 'V', Nd - 1, Hup, Nd - 1,
                             eigvalsUp, eigvalsUp_imag, eigvecs_left, Nd - 1,
                             eigvecsUp, Nd - 1);

    if (info > 0) {
#ifdef DEBUG
        printf("Error: Failed to compute eigenvalues.\n");
#endif
        free(Hup);
        free(eigvalsUp);
        free(eigvecsUp);
        free(eigvalsUp_imag);
        exit(EXIT_FAILURE);
    }

    sort_eigenpairs(eigvalsUp, eigvecsUp, Nd - 1);

    for (int col = 0; col < Nd - 1; col++) {
        eigvals[col] = eigvalsUp[col];
        for (int row = 0; row < Nd - 1; row++) {
            eigvecs[col*(Nd - 1) + row] = eigvecsUp[col*(Nd - 1) + row];
        }
    }
    free(Hup);
    free(eigvalsUp);
    free(eigvecsUp);
    free(eigvalsUp_imag);

    if (pSPARC_ATOM->spinFlag) {
        double* eigvalsDw = (double*)malloc((Nd-1) * sizeof(double));
        double* eigvalsDw_imag = (double*)malloc((Nd-1) * sizeof(double));
        double* eigvecs_left = NULL; // Not needed in this case
        double* eigvecsDw = (double*)malloc((Nd-1) * (Nd-1) * sizeof(double));
        // LAPACK call to compute eigenvalues and right eigenvectors
        int info = LAPACKE_dgeev(LAPACK_COL_MAJOR, 'N', 'V', Nd - 1, Hdw, Nd - 1,
                                eigvalsDw, eigvalsDw_imag, eigvecs_left, Nd - 1,
                                eigvecsDw, Nd - 1);
        if (info > 0) {
#ifdef DEBUG
            printf("Error: Failed to compute eigenvalues.\n");
#endif
            free(Hdw);
            free(eigvalsDw);
            free(eigvecsDw);
            free(eigvalsDw_imag);
            exit(EXIT_FAILURE);
        }
        sort_eigenpairs(eigvalsDw, eigvecsDw, Nd - 1);

        for (int col = 0; col < Nd - 1; col++) {
            eigvals[col+(Nd - 1)] = eigvalsDw[col];
            for (int row = 0; row < Nd - 1; row++) {
                eigvecs[(Nd - 1)*(Nd - 1) + col*(Nd - 1) + row] = eigvecsDw[col*(Nd - 1) + row];
            }
        }
        free(Hdw);
        free(eigvalsDw);
        free(eigvalsDw_imag);
        free(eigvecsDw);
    }

}

/**
 * @brief Sort the eigenvalues (and corresponding eigenvectors) in ascending order
 */
void sort_eigenpairs(double* eigvals_real, double* eigvecs, int n) {
    // Array to store indices for sorting
    int* indices = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
        indices[i] = i;
    }

    // Sort eigenvalues (real part) and track indices
    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            if (eigvals_real[indices[i]] > eigvals_real[indices[j]]) {
                int temp = indices[i];
                indices[i] = indices[j];
                indices[j] = temp;
            }
        }
    }

    // Create sorted copies of eigenvalues and eigenvectors
    double* sorted_eigvals_real = (double*)malloc(n * sizeof(double));
    double* sorted_eigvecs = (double*)malloc(n * n * sizeof(double));

    for (int i = 0; i < n; i++) {
        sorted_eigvals_real[i] = eigvals_real[indices[i]];
        for (int j = 0; j < n; j++) {
            sorted_eigvecs[j + i * n] = eigvecs[j + indices[i] * n];
        }
    }

    // Copy sorted values back to original arrays
    for (int i = 0; i < n; i++) {
        eigvals_real[i] = sorted_eigvals_real[i];
        for (int j = 0; j < n; j++) {
            eigvecs[j + i * n] = sorted_eigvecs[j + i * n];
        }
    }

    free(indices);
    free(sorted_eigvals_real);
    free(sorted_eigvecs);
}

/**
 * @brief Calculates \int (\tilde{R}_{nl})^2 dr
 */
void integralOrbitalSquared(SPARC_ATOM_OBJ *pSPARC_ATOM, double *eigvec, double *integral) {
    double *w = pSPARC_ATOM->w;
    double *int_scale = pSPARC_ATOM->int_scale;
    int Nd = pSPARC_ATOM->Nd;

    *integral = 0.0;
    for (int i = 0; i < Nd - 1; i++) {
        *integral += w[i+1]*(eigvec[i]*eigvec[i])/int_scale[i+1];
    }
}

/**
 * @brief Calculates SCF error as \frac{norm(rho_{out} - rho_{in})}{norm(rho_{out})}
 */
void SCFerrorAtom(SPARC_ATOM_OBJ *pSPARC_ATOM, double *scf_error) {
    int Nd = pSPARC_ATOM->Nd;
    int col = pSPARC_ATOM->nspin;
    int len = col*(Nd-1);

    double *var_in = NULL, *var_out = NULL;
    var_in = pSPARC_ATOM->mixing_hist_xkm1;
    var_out = pSPARC_ATOM->electronDens + ((pSPARC_ATOM->spinFlag) ? (Nd-1) : 0);

    double sbuf[2] = {0, 0};
    double temp;
    for (int i = 0; i < len; i++) {
        temp = var_out[i] - var_in[i];
        sbuf[0] += var_out[i] * var_out[i];
        sbuf[1] += temp*temp;
    }

    *scf_error = sqrt(sbuf[1]/sbuf[0]);
}

/**
 * @brief Calculates normed relative error as \frac{norm(new - old)}{norm(old)}
 */
void normedRelativeError(double *old, double *new, int len, double *error) {
    double sbuf[2] = {0, 0};
    double temp;
    for (int i = 0; i < len; i++) {
        temp = new[i] - old[i];
        sbuf[0] += old[i]*old[i];
        sbuf[1] += temp*temp;
    }

    *error = sqrt(sbuf[1]/sbuf[0]);
}

//////////////////////////////////////// Mixing Functions ///////////////////////////////////////////
/**
 * @brief   Anderson extrapolation update.
 *
 *          x_{k+1} = (x_k - X * Gamma) + beta * P * (f_k - F * Gamma),
 *          where P is the preconditioner, and Gamma = inv(F^T * F) * F^T * f.
 *          Expanding above equation gives: 
 *          x_{k+1} = x_k + beta * P * f - (X + beta * P * F) * inv(F^T * F) * F^T * f          
 */
void AndersonExtrapolation_atom(
    const int N, const int m, double *x_kp1, const double *x_k,
    const double *f_k, const double *X, const double *F,
    const double beta
) {
    unsigned i;
    double *f_wavg = (double *)malloc(N * sizeof(double));

    // find the weighted average vectors
    AndersonExtrapWtdAvg_atom(N, m, x_k, f_k, X, F, x_kp1, f_wavg);

    // add beta * f to x_{k+1}
    for (i = 0; i < N; i++)
        x_kp1[i] += beta * f_wavg[i];
    
    free(f_wavg);
}

/**
 * @brief   Anderson extrapolation weighted average vectors.
 *
 *          Find x_wavg := x_k - X * Gamma and f_wavg := (f_k - F * Gamma),
 *          where Gamma = inv(F^T * F) * F^T * f.
 */
void AndersonExtrapWtdAvg_atom(
    const int N, const int m, const double *x_k,
    const double *f_k, const double *X, const double *F,
    double *x_wavg, double *f_wavg) {
        double *Gamma = (double *)calloc(m, sizeof(double));
        assert(Gamma != NULL);

        // find extrapolation weigths Gamma = inv(F^T * F) * F^T * f_k
        AndersonExtrapCoeff_atom(N, m, f_k, F, Gamma);

        unsigned i;

        // find weighted average x_{k+1} = x_k - X*Gamma
        for (i = 0; i < N; i++) x_wavg[i] = x_k[i];
        cblas_dgemv(CblasColMajor, CblasNoTrans, N, m, -1.0, X,
                    N, Gamma, 1, 1.0, x_wavg, 1);
        
        // find weighted average f_{k+1} = f_k - F*Gamma
        for (i = 0; i < N; i++) f_wavg[i] = f_k[i];
        cblas_dgemv(CblasColMajor, CblasNoTrans, N, m, -1.0, F, 
                    N, Gamma, 1, 1.0, f_wavg, 1);

        free(Gamma);
    }

/**
 * @brief   Anderson extrapolation coefficiens.
 *
 *          Gamma = inv(F^T * F) * F^T * f.         
 */
void AndersonExtrapCoeff_atom(
    const int N, const int m, const double *f, const double *F, 
    double *Gamma) {
        int matrank;
        double *FtF, *s;
        FtF = (double *)malloc(m*m*sizeof(double));
        s = (double *)malloc(m * sizeof(double));
        assert(FtF != NULL && s != NULL);

        compute_FtF_atom(F, m, N, FtF);
        compute_Ftf_atom(F, f, m, N, Gamma);

        LAPACKE_dgelsd(LAPACK_COL_MAJOR, m, m, 1, FtF, m, Gamma, m, s, -1.0, &matrank);

        free(s); free(FtF);
}

void compute_FtF_atom(const double *F, int m, int N, double *FtF) {
#define FtF(i,j) Ftf[i+j*m]
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m, m, N,
                1.0, F, N, F, N, 0.0, FtF, m);
#undef FtF
}

void compute_Ftf_atom(const double *F, const double *f, int m, int N, double *Ftf) {
    cblas_dgemv(CblasColMajor, CblasTrans, N, m,
                1.0, F, N, f, 1, 0.0, Ftf, 1);
}

void periodicPulayMixAtom(SPARC_ATOM_OBJ *pSPARC_ATOM, int iter_count){
#define R(i,j) (*(R+(j)*N+(i)))
#define F(i,j) (*(F+(j)*N+(i)))

    double *g_k = NULL, *x_kp1 = NULL;
    int Nd = pSPARC_ATOM->Nd;
    int ncol = pSPARC_ATOM->nspden;
    int N = Nd - 1;
    if (pSPARC_ATOM->spinFlag == 1) {
        N *= (ncol - 1);
    }
    int m = pSPARC_ATOM->MixingHistory;
    int p = pSPARC_ATOM->PulayFrequency;
    double beta = pSPARC_ATOM->MixingParameter;
    double omega = pSPARC_ATOM->MixingParameterSimple;
    double beta_mag = pSPARC_ATOM->MixingParameterMag;
    double omega_mag = pSPARC_ATOM->MixingParameterSimpleMag;

    // flag for Pulay (Anderson extrapolation) mixing, otherwise apply simple mixing 
    int Pulay_mixing_flag = (int) ((iter_count+1) % p == 0 && iter_count > 0);

    // decide which mixing parameter to use
    double amix, amix_mag;
    if (Pulay_mixing_flag) { // pulay mixing
        amix = beta; amix_mag = beta_mag;
    } else { // simple (linear) mixing
        amix = omega; amix_mag = omega_mag;
    }

    if (pSPARC_ATOM->spinFlag == 0) {
        g_k = x_kp1 = pSPARC_ATOM->electronDens;
    } else {
        g_k = x_kp1 = (double *)malloc(N*sizeof(double));
        memcpy(g_k, pSPARC_ATOM->electronDens+Nd-1, sizeof(double)*(Nd - 1)); // rho_up
        memcpy(g_k+Nd-1, pSPARC_ATOM->electronDens+2*(Nd - 1), sizeof(double)*(Nd - 1)); // rho_dw
    }

    double *x_k = pSPARC_ATOM->mixing_hist_xk; // the current mixed var x (x^{in}_k)
    double *x_km1 = pSPARC_ATOM->mixing_hist_xkm1; // x_{k-1}
    double *f_k = pSPARC_ATOM->mixing_hist_fk; // f_k = g(x_k) - x_k
    double *f_km1 = pSPARC_ATOM->mixing_hist_fkm1; // f_{k-1}
    double *R = pSPARC_ATOM->X; // [x_{k-m+1} - x_{k-m}, ... , x_k - x_{k-1}]
    double *F = pSPARC_ATOM->F; // [f_{k-m+1} - f_{k-m}, ... , f_k - f_{k-1}]

    // update old residual f_{k-1}
    if (iter_count > 0) {
        for (int  i = 0; i < N; i++) f_km1[i] = f_k[i];
    }

    // compute current residual 
    for (int i = 0; i < N; i++) f_k[i] = g_k[i] - x_k[i];

    // *** store residual & iteration history *** //
    if (iter_count > 0) {
        int i_hist = (iter_count - 1) % m;
        if (pSPARC_ATOM->PulayRestartFlag && i_hist == 0) {
            for (int i = 0; i < N*(m-1); i++) {
                R[N+i] = F[N+i] = 0.0; // set all cols to 0 (except for 1st col)
            }

            // set 1st cols of R and F
            for (int i = 0; i < N; i++) {
                R[i] = x_k[i] - x_km1[i];
                F[i] = f_k[i] - f_km1[i];
            }
        } else {
            for (int i = 0; i < N; i++) {
                R(i, i_hist) = x_k[i] - x_km1[i];
                F(i, i_hist) = f_k[i] - f_km1[i];
            }
        }
    }

    double *f_wavg, *x_wavg;
    x_wavg = (double *)malloc( N * sizeof(double) );
    f_wavg = (double *)malloc( N * sizeof(double) );
    assert(x_wavg != NULL);
    assert(f_wavg != NULL);

    if (Pulay_mixing_flag) {
        AndersonExtrapWtdAvg_atom(N, m, x_k, f_k, R, F, x_wavg, f_wavg);
    } else {
        // Simple (linear) mixing
        for (int i = 0; i < N; i++) {
            x_wavg[i] = x_k[i];
            f_wavg[i] = f_k[i];
        }
    }

    // No preconditioner
    double *Pf = (double *)malloc( N * sizeof(double) );
    if (pSPARC_ATOM->spinFlag == 0) {
        for (int i = 0; i < N; i++) Pf[i] = amix * f_wavg[i];

        for (int i = 0; i < N; i++) x_kp1[i] = x_wavg[i] + Pf[i];
    } else {
        double sum_f_tot = 0.0, sum_f_mag = 0.0, sum_Pf_tot = 0.0, sum_Pf_mag = 0.0;
        for (int i = 0; i < Nd-1; i++) {
            Pf[i] = amix * (f_wavg[i] + f_wavg[i+Nd-1]);
            Pf[i+Nd-1] = amix_mag * (f_wavg[i] - f_wavg[i+Nd-1]);
            sum_f_tot += (f_wavg[i] + f_wavg[i+Nd-1]);
            sum_f_mag += (f_wavg[i] - f_wavg[i+Nd-1]);
            sum_Pf_tot += Pf[i]; 
            sum_Pf_mag += Pf[i+Nd-1];
        }

        for (int i = 0; i < Nd - 1; i++) {
            Pf[i] += (sum_f_tot - sum_Pf_tot)/(Nd - 1);
            Pf[i+Nd-1] += (sum_f_mag - sum_Pf_mag)/(Nd - 1);
        }

        for (int i = 0; i < Nd - 1; i++) {
            x_kp1[i] = x_wavg[i] + (Pf[i] + Pf[i+Nd-1])/2;
            x_kp1[i+Nd-1] = x_wavg[i+Nd-1] + (Pf[i] - Pf[i+Nd-1])/2;
        }
    }

    // Check if rho < 0
    int neg_flag = 0;
    if (pSPARC_ATOM->spinFlag == 0){
        for (int i = 0; i < Nd - 1; i++) {
            if (x_kp1[i] < 0.0) {
                x_kp1[i] = 0.0;
                neg_flag = 1;
            }
        }
    } else {
        double const_C;
        for (int i = 0; i < Nd - 1; i++) {
            if (x_kp1[i]+x_kp1[i+Nd-1] < 0.0) {
                const_C = x_kp1[i] - x_kp1[i+Nd-1];
                x_kp1[i] = const_C/2;
                x_kp1[i+Nd-1] = -const_C/2;
            }
        }
    }

    if (neg_flag > 0) {
#ifdef DEBUG
        printf("WARNING: The density after mixing has negative components!\n");
#endif
        double scalUp = 0.0;
        if (pSPARC_ATOM->spinFlag == 0) {
            for (int i = 1; i < Nd; i++) {
                scalUp += pow(pSPARC_ATOM->r[i],2)*x_kp1[i-1]*pSPARC_ATOM->w[i]/pSPARC_ATOM->int_scale[i];
            }
            scalUp *= 4*M_PI;
            for (int i = 0; i < Nd - 1; i++){
                x_kp1[i] = ((*pSPARC_ATOM->Znucl)/scalUp)*x_kp1[i];
            }
        } else {
            double scalDw = 0.0;
            for (int i = 1; i < Nd; i++) {
                scalUp += pow(pSPARC_ATOM->r[i],2)*x_kp1[i-1]*pSPARC_ATOM->w[i]/pSPARC_ATOM->int_scale[i];
                scalDw += pow(pSPARC_ATOM->r[i],2)*x_kp1[i-1+Nd-1]*pSPARC_ATOM->w[i]/pSPARC_ATOM->int_scale[i];
            }
            scalUp *= 4*M_PI; scalDw *= 4*M_PI;
            for (int i = 0; i < Nd - 1; i++){
                x_kp1[i] = (pSPARC_ATOM->fUpSum/scalUp)*x_kp1[i];
                x_kp1[i+Nd-1] = (pSPARC_ATOM->fDwSum/scalDw)*x_kp1[i+Nd-1];
            }
        }
    }

    if (pSPARC_ATOM->spinFlag == 0) {
        memcpy(pSPARC_ATOM->electronDens, x_kp1, sizeof(double)*N); // rho
    } else {
        memcpy(pSPARC_ATOM->electronDens+Nd-1, x_kp1, sizeof(double)*(Nd-1));
        memcpy(pSPARC_ATOM->electronDens+2*(Nd-1), x_kp1+Nd-1, sizeof(double)*(Nd-1));
        for (int i = 0; i<Nd-1; i++) {
            pSPARC_ATOM->electronDens[i] = pSPARC_ATOM->electronDens[i+Nd-1] 
                                        + pSPARC_ATOM->electronDens[i+2*(Nd-1)];
        }
    }

    free(x_wavg); free(f_wavg); free(Pf); 

    for (int i = 0; i < N; i++) {
        // update x_{k-1} = x_k
        x_km1[i] = x_k[i];
        // update x_k = x_{k+1}
        x_k[i] = x_kp1[i];
    }

    if (pSPARC_ATOM->spinFlag == 1) {
        free(x_kp1); 
    }
#undef R
#undef F
}