#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <mpi.h>
// this is for checking existence of files

#ifdef USE_MKL
    #define MKL_Complex16 double _Complex
    #include <mkl.h>
#else
    #include <cblas.h>
    #include <lapacke.h>
#endif

# include <unistd.h>
#include "initializationAtom.h"
#include "atomicStates.h"
#include "readfiles.h"
#include "tools.h"
#include "isddftAtom.h"
#include "spectralFunctionsAtom.h"
#include "mGGAtoolsAtom.h"
#include "exxToolsAtom.h"
#include "isddft.h"

#define TEMP_TOL 1e-12

#define min(x,y) ((x)<(y)?(x):(y))
#define max(x,y) ((x)>(y)?(x):(y))

//#define N_MEMBR 209
#define N_MEMBR 208

void Initialize_Atom(SPARC_ATOM_OBJ *pSPARC_ATOM, SPARC_OBJ *pSPARC, SPARC_INPUT_OBJ *pSPARC_Input, int ityp) {

    /* Set defaults */
    set_defaults_Atom(pSPARC_ATOM);

    /* Copy pseudopotential */
    copy_PSP_atom(pSPARC_ATOM, pSPARC, ityp);


    // calculate spline derivatives for interpolation
    Calculate_SplineDerivRadFun_atom(pSPARC_ATOM);

    // set up grid, gradient and laplacian
    set_grid_grad_lap(pSPARC_ATOM);

    // find exchange correlation decomposition
    strncpy(pSPARC_ATOM->XC, pSPARC_Input->XC, sizeof(pSPARC_ATOM->XC));
    // strcpy(pSPARC_ATOM->XC, "GGA_PBE"); // To hard code to GGA orbitals
    xc_decomposition_atom(pSPARC_ATOM);

    // Print Z and XC
    printf("Atomic number of element:  %d\n",*pSPARC_ATOM->Zatom);
    printf("Exchange-correlation used:  %s\n",pSPARC_ATOM->XC);

    // check if NLCC is present
    int NLCC_flag = 0;
    if(pSPARC_ATOM->psd->fchrg > TEMP_TOL){
        NLCC_flag = 1;
    }
    pSPARC_ATOM->NLCC_flag = NLCC_flag;

    // spin options
    pSPARC_ATOM->spinFlag = pSPARC->atomSolvSpinFlag;
    if (pSPARC_ATOM->spinFlag == 0){
        pSPARC_ATOM->nspin = 1;
        pSPARC_ATOM->nspinor = 1;
        pSPARC_ATOM->nspden = 1;
        printf("\nSpin unpolarized calculation.\n");
    } else {
        pSPARC_ATOM->nspin = 2;
        pSPARC_ATOM->nspinor = 2;
        pSPARC_ATOM->nspden = 3;
        printf("\nSpin polarized calculation.\n");
    }

    // check if exchange-correlation functional is metaGGA
    if (pSPARC_ATOM->ixc[2]) { // it can be expand, such as adding r2SCAN 
        if (pSPARC_ATOM->NLCC_flag) {
            printf("\nERROR: currently SCAN, RSCAN and R2SCAN functional does not support applying NLCC pseudopotential!\n");
            exit(EXIT_FAILURE);
        }
        initialize_MGGA_atom(pSPARC_ATOM);
        pSPARC_ATOM->SCF_tol = 1e-6;
    }

    // initialize energy values
    pSPARC_ATOM->Eband = 0.0;
    pSPARC_ATOM->Exc = 0.0;
    pSPARC_ATOM->Exc_dc = 0.0;
    pSPARC_ATOM->Eelec_dc = 0.0;
    pSPARC_ATOM->Etot = 0.0;

    // Initialise density and NLCC
    init_rho_NLCC(pSPARC_ATOM);
    pSPARC_ATOM->mag = NULL;

    // Get atomic states
    int net_mag, sum_fup, sum_fdw;
    getAtomicStates(*pSPARC_ATOM->Zatom, &pSPARC_ATOM->states_n,
                    &pSPARC_ATOM->states_l, &pSPARC_ATOM->states_fup, 
                    &pSPARC_ATOM->states_fdw, &pSPARC_ATOM->states_len);
    pSPARC_ATOM->states_ftot = (int *)malloc(pSPARC_ATOM->states_len * sizeof(int));
    for (int jj = 0; jj < pSPARC_ATOM->states_len; jj++){
        pSPARC_ATOM->states_ftot[jj] = pSPARC_ATOM->states_fup[jj] + pSPARC_ATOM->states_fdw[jj];
    }

    // Set valence occupations
    setValence(pSPARC_ATOM);

    sum_fup = sum_fdw = 0;
    for (int jj = 0; jj < pSPARC_ATOM->val_len; jj++){
        sum_fup += pSPARC_ATOM->fUpVal[jj];
        sum_fdw += pSPARC_ATOM->fDwVal[jj];
    }
    net_mag = abs(sum_fup - sum_fdw);
    pSPARC_ATOM->netM = net_mag;

    // Distribute rho if spinFlag is on
    int Nd = pSPARC_ATOM->Nd;
    if (pSPARC_ATOM->spinFlag == 1){
        
        pSPARC_ATOM->mag = (double *)malloc((Nd - 1) * sizeof(double));
        
        double scaling_fac = 0.0; 
        for (int jj = 0; jj < Nd-1; jj++){
            scaling_fac += 4*M_PI*(pSPARC_ATOM->w[jj+1]/pSPARC_ATOM->int_scale[jj+1])*(pow(pSPARC_ATOM->r[jj+1],2)*(pSPARC_ATOM->electronDens[jj]));
        }

        if (*pSPARC_ATOM->Zatom == 1){
            for (int jj = 0; jj < Nd - 1; jj++){
                pSPARC_ATOM->electronDens[jj + (Nd - 1)] = pSPARC_ATOM->electronDens[jj];
                pSPARC_ATOM->mag[jj] = pSPARC_ATOM->electronDens[jj + (Nd - 1)] - pSPARC_ATOM->electronDens[jj + 2*(Nd - 1)];
            }
        } else {
            double frac = (1 + net_mag/scaling_fac)/2;
            for (int jj = 0; jj < Nd - 1; jj++){
                pSPARC_ATOM->electronDens[jj + (Nd - 1)] = frac*pSPARC_ATOM->electronDens[jj];
                pSPARC_ATOM->electronDens[jj + 2*(Nd - 1)] = (1-frac)*pSPARC_ATOM->electronDens[jj];
                pSPARC_ATOM->mag[jj] = pSPARC_ATOM->electronDens[jj + (Nd - 1)] - pSPARC_ATOM->electronDens[jj + 2*(Nd - 1)];
            }
        }
    }

    // Allocate memory phi, e_xc, XC_Potential
    pSPARC_ATOM->phi = (double *)malloc((Nd - 1) * sizeof(double));
    pSPARC_ATOM->e_xc = (double *)malloc((Nd - 1) * sizeof(double));
    if (pSPARC_ATOM->spinFlag == 0) {
        pSPARC_ATOM->XCPotential = (double *)malloc((Nd - 1) * sizeof(double));
        pSPARC_ATOM->Dxcdgrho = NULL;
        if (pSPARC_ATOM->isGradient) {
            pSPARC_ATOM->Dxcdgrho = (double *)malloc((Nd - 1) * sizeof(double));
        }
    } else {
        pSPARC_ATOM->XCPotential = (double *)malloc(2*(Nd - 1) * sizeof(double));
        pSPARC_ATOM->Dxcdgrho = NULL;
        if (pSPARC_ATOM->isGradient) {
            pSPARC_ATOM->Dxcdgrho = (double *)malloc(3*(Nd - 1) * sizeof(double));
        }
    }

    /* hybrid */
    if (pSPARC_ATOM->usefock) exx_initialization_atom(pSPARC_ATOM);
}

void set_defaults_Atom(SPARC_ATOM_OBJ *pSPARC_ATOM) {
    pSPARC_ATOM->localPsd = (int *)malloc(sizeof(int));
    *pSPARC_ATOM->localPsd = 4;

    /* SCF options */
    pSPARC_ATOM->MAXIT_SCF = 100;
    pSPARC_ATOM->SCF_tol = 1e-8;
    pSPARC_ATOM->scf_success = 0;

    /* spin options */
    pSPARC_ATOM->spinFlag = 0;      

    /* Domain description */  
    pSPARC_ATOM->Rmax = 20;

    /* Discretization */
    pSPARC_ATOM->Nd = 400;          
    pSPARC_ATOM->Nq = pSPARC_ATOM->Nd;             
    pSPARC_ATOM->alpha = 1;       
    pSPARC_ATOM->beta = -0.45;    

    /* Spectral grid (Exponential)*/
    pSPARC_ATOM->xmax = pSPARC_ATOM->alpha*(1 - exp(2*pSPARC_ATOM->beta*pSPARC_ATOM->Rmax))/2;

    /* exchange correlation */
    pSPARC_ATOM->xc_rhotol = 1e-14; 
    pSPARC_ATOM->xc_sigmatol = 1e-14;  

    /* Mixing */
    pSPARC_ATOM->MixingHistory = 7;   
    pSPARC_ATOM->PulayFrequency = 1;  
    pSPARC_ATOM->PulayRestartFlag = 0;  
    pSPARC_ATOM->MixingParameter = 1; 
    pSPARC_ATOM->MixingParameterSimple = 1;   
    pSPARC_ATOM->MixingParameterMag = 1;      
    pSPARC_ATOM->MixingParameterSimpleMag = 1;     

    /* metaGGA */     
    pSPARC_ATOM->countPotentialCalculate = 0;
    pSPARC_ATOM->tau = NULL;
    pSPARC_ATOM->vxcMGGA3 = NULL;

    /* hybrid */
    pSPARC_ATOM->MAXIT_FOCK = -1;
    pSPARC_ATOM->MINIT_FOCK = -1;
    pSPARC_ATOM->TOL_FOCK = -1;
    pSPARC_ATOM->TOL_SCF_INIT = -1;
    pSPARC_ATOM->exx_frac = -1;
    pSPARC_ATOM->usefock = 0;
    pSPARC_ATOM->EXXL = NULL;
}

void copy_PSP_atom(SPARC_ATOM_OBJ *pSPARC_ATOM, SPARC_OBJ *pSPARC, int ityp) {
    // allocate memory
    pSPARC_ATOM->psd = (PSD_OBJ *)malloc(sizeof(PSD_OBJ));
    assert(pSPARC_ATOM->psd != NULL);

    // copy items
    pSPARC_ATOM->psd->fchrg = pSPARC->psd[ityp].fchrg;
    pSPARC_ATOM->psd->Vloc_0 = pSPARC->psd[ityp].Vloc_0;
    pSPARC_ATOM->psd->is_r_uniform = pSPARC->psd[ityp].is_r_uniform;
    pSPARC_ATOM->psd->pspxc = pSPARC->psd[ityp].pspxc;
    pSPARC_ATOM->psd->lmax = pSPARC->psd[ityp].lmax;
    pSPARC_ATOM->psd->size = pSPARC->psd[ityp].size;
    pSPARC_ATOM->psd->pspsoc = pSPARC->psd[ityp].pspsoc;
    pSPARC_ATOM->Zatom = (int *)malloc(sizeof(int));
    pSPARC_ATOM->Znucl = (int *)malloc(sizeof(int));
    *pSPARC_ATOM->Zatom = pSPARC->Zatom[ityp];
    *pSPARC_ATOM->Znucl = pSPARC->Znucl[ityp];

    // allocate memory
    pSPARC_ATOM->psd->ppl = (int *)calloc((pSPARC_ATOM->psd->lmax+1), sizeof(int));
    pSPARC_ATOM->psd->rc = (double *)malloc((pSPARC_ATOM->psd->lmax+1) * sizeof(double)); 
    int lmax = pSPARC_ATOM->psd->lmax;
    int *lpos = (int *)calloc((lmax+2), sizeof(int)); // the last value stores the total number of projectors for this l

    // copy items
    lpos[0] = 0;
    for (int l = 0; l <= pSPARC_ATOM->psd->lmax; l++) {
        pSPARC_ATOM->psd->rc[l] = pSPARC->psd[ityp].rc[l];
        pSPARC_ATOM->psd->ppl[l] = pSPARC->psd[ityp].ppl[l];
        lpos[l+1] = lpos[l] + pSPARC_ATOM->psd->ppl[l];
    }
    int nproj = lpos[lmax+1];

    // allocate memory
    int size = pSPARC_ATOM->psd->size;
    pSPARC_ATOM->psd->RadialGrid = (double *)calloc(pSPARC->psd[ityp].size , sizeof(double)); 
    pSPARC_ATOM->psd->UdV = (double *)calloc(nproj * pSPARC->psd[ityp].size , sizeof(double)); 
    pSPARC_ATOM->psd->rVloc = (double *)calloc(pSPARC->psd[ityp].size , sizeof(double)); 
    pSPARC_ATOM->psd->rhoIsoAtom = (double *)calloc(pSPARC->psd[ityp].size , sizeof(double)); 
    pSPARC_ATOM->psd->Gamma = (double *)calloc(nproj , sizeof(double));

    // copy items
    for(int JJ = 0; JJ < size; JJ++) {
        pSPARC_ATOM->psd->RadialGrid[JJ] = pSPARC->psd[ityp].RadialGrid[JJ];
        pSPARC_ATOM->psd->rVloc[JJ] = pSPARC->psd[ityp].rVloc[JJ];
        pSPARC_ATOM->psd->rhoIsoAtom[JJ] = pSPARC->psd[ityp].rhoIsoAtom[JJ];
    }

    for (int JJ = 0; JJ < nproj*size; JJ++) {
        pSPARC_ATOM->psd->UdV[JJ] = pSPARC->psd[ityp].UdV[JJ];
    }

    for (int JJ = 0; JJ < nproj; JJ++){
        pSPARC_ATOM->psd->Gamma[JJ] = pSPARC->psd[ityp].Gamma[JJ];
    }

    // NLCC
    pSPARC_ATOM->psd->rho_c_table = (double *)calloc(pSPARC->psd[ityp].size, sizeof(double));
    if (pSPARC_ATOM->psd->fchrg > TEMP_TOL) {
        for (int JJ = 0; JJ < size; JJ++) {
            pSPARC_ATOM->psd->rho_c_table[JJ] = pSPARC->psd[ityp].rho_c_table[JJ];
        }
    }

    free(lpos);
}

void init_rho_NLCC(SPARC_ATOM_OBJ *pSPARC_ATOM) {
    double *r_grid_rho = (double *)malloc((pSPARC_ATOM->psd->size)*sizeof(double));
    double *rho_psp = (double *)malloc((pSPARC_ATOM->psd->size)*sizeof(double));

    for (int jj = 0; jj < pSPARC_ATOM->psd->size; jj++){
        r_grid_rho[jj] = pSPARC_ATOM->psd->RadialGrid[jj];
        rho_psp[jj] = pSPARC_ATOM->psd->rhoIsoAtom[jj];
    }

    int idx_end = pSPARC_ATOM->psd->size - 1;
    int idx_end_1 = idx_end - 1;
    double rN = r_grid_rho[idx_end];
    double rN_1 = r_grid_rho[idx_end_1];
    double guessN = rho_psp[idx_end];
    double guessN_1 = rho_psp[idx_end_1];
    double C1 = log(guessN/guessN_1)/(rN - rN_1);
    double C2 = guessN/exp(C1*rN);
    double r_rho_cut = rN;

    int Nd = pSPARC_ATOM->Nd;

    // Allocate memory for rho_guess and rho_correct
    double *rho_guess = (double *)malloc((Nd + 1) * sizeof(double));
    double *rho_correct = (double *)malloc((Nd + 1) * sizeof(double));

    // Interpolation: For values where r < r_rho_cut, use spline interpolation
    for (int i = 0; i <= Nd; i++) {
        if (pSPARC_ATOM->r[i] < r_rho_cut) {
            // Perform spline interpolation
            SplineInterp(r_grid_rho, rho_psp, pSPARC_ATOM->psd->size, &pSPARC_ATOM->r[i], &rho_guess[i], 1,
                        pSPARC_ATOM->psd->SplineFitIsoAtomDen);
        } else {
            rho_guess[i] = 0.0;
        }
    }

    // Fill rho_correct for r >= r_rho_cut
    for (int i = 0; i <= Nd; i++) {
        if (pSPARC_ATOM->r[i] < r_rho_cut) {
            rho_correct[i] = 0.0;
        } else {
            rho_correct[i] = C2 * exp(C1 * pSPARC_ATOM->r[i]);
        }
    }

    // Sum rho_guess and rho_correct
    for (int i = 0; i <= Nd; i++) {
        rho_guess[i] += rho_correct[i];
    }

    // Store the result in pSPARC_ATOM->electronDens
    int nspden = pSPARC_ATOM->nspden;
    pSPARC_ATOM->electronDens = (double *)calloc(nspden*(Nd-1), sizeof(double));  // Allocate space for electronDens
    
    // Store the result in pSPARC_ATOM->electronDens (equivalent to MATLAB S.rho(:,1) = rho_guess(2:Nd);)
    for (int i = 1; i < Nd; i++) {  // Store from 1 to Nd-1 (MATLAB equivalent: 2:Nd)
        pSPARC_ATOM->electronDens[i - 1] = rho_guess[i];
    }

    // Clean up memory
    free(rho_psp);
    free(rho_guess);
    free(rho_correct);

    // NLCC part
    if (pSPARC_ATOM->psd->fchrg > TEMP_TOL){
        double *rho_C_psp = (double *)malloc((pSPARC_ATOM->psd->size) * sizeof(double));
        for (int jj = 0; jj < pSPARC_ATOM->psd->size; jj++){
            rho_C_psp[jj] = pSPARC_ATOM->psd->rho_c_table[jj];
        }

        double *rho_Tilde = (double *)malloc((Nd + 1) * sizeof(double));
        // Interpolation: For values where r < r_rho_cut, use spline interpolation
        for (int i = 0; i <= Nd; i++) {
            if (pSPARC_ATOM->r[i] < r_rho_cut) {
                // Perform spline interpolation
                SplineInterp(r_grid_rho, rho_C_psp, pSPARC_ATOM->psd->size, &pSPARC_ATOM->r[i], &rho_Tilde[i], 1,
                            pSPARC_ATOM->psd->SplineRhocD);
            } else {
                rho_Tilde[i] = 0.0;
            }
        }

        pSPARC_ATOM->electronDens_core = (double *)malloc((Nd-1) * sizeof(double));

        for (int i = 1; i < Nd; i++) {  // Store from 1 to Nd-1 (MATLAB equivalent: 2:Nd)
            pSPARC_ATOM->electronDens_core[i - 1] = rho_Tilde[i];
        }

        // Clean up memory
        free(rho_Tilde);
        free(rho_C_psp);
    }

    // Clean up remaining memory
    free(r_grid_rho);
}

void xc_decomposition_atom(SPARC_ATOM_OBJ *pSPARC_ATOM) {
    pSPARC_ATOM->isGradient = 0;
    pSPARC_ATOM->xcoption[0] = pSPARC_ATOM->xcoption[1] = 0;
    int xc = 0;

    if (strcmpi(pSPARC_ATOM->XC, "LDA_PZ") == 0) {
        xc = 2;
        pSPARC_ATOM->ixc[0] = 1; pSPARC_ATOM->ixc[1] = 1; 
        pSPARC_ATOM->ixc[2] = 0; pSPARC_ATOM->ixc[3] = 0;
    } else if (strcmpi(pSPARC_ATOM->XC, "LDA_PW") == 0) {
        xc = 7;
        pSPARC_ATOM->ixc[0] = 1; pSPARC_ATOM->ixc[1] = 2; 
        pSPARC_ATOM->ixc[2] = 0; pSPARC_ATOM->ixc[3] = 0;
    } else if (strcmpi(pSPARC_ATOM->XC, "GGA_PBE") == 0) {
        xc = 11;
        pSPARC_ATOM->ixc[0] = 2; pSPARC_ATOM->ixc[1] = 3; 
        pSPARC_ATOM->ixc[2] = 0; pSPARC_ATOM->ixc[3] = 0;
        pSPARC_ATOM->xcoption[0] = 1; pSPARC_ATOM->xcoption[1] = 1;
        pSPARC_ATOM->isGradient = 1;
    } else if (strcmpi(pSPARC_ATOM->XC, "GGA_PBEsol") == 0) {
        xc = 116;
        pSPARC_ATOM->ixc[0] = 2; pSPARC_ATOM->ixc[1] = 3; 
        pSPARC_ATOM->ixc[2] = 0; pSPARC_ATOM->ixc[3] = 0;
        pSPARC_ATOM->xcoption[0] = 2; pSPARC_ATOM->xcoption[1] = 2;
        pSPARC_ATOM->isGradient = 1;
    } else if (strcmpi(pSPARC_ATOM->XC, "GGA_RPBE") == 0) {
        xc = 15;
        pSPARC_ATOM->ixc[0] = 2; pSPARC_ATOM->ixc[1] = 3; 
        pSPARC_ATOM->ixc[2] = 0; pSPARC_ATOM->ixc[3] = 0;
        pSPARC_ATOM->xcoption[0] = 3; pSPARC_ATOM->xcoption[1] = 3;
        pSPARC_ATOM->isGradient = 1;
    } 
    else if (strcmpi(pSPARC_ATOM->XC, "HF") == 0) {
        xc = 40;
        pSPARC_ATOM->usefock = 1;
        pSPARC_ATOM->ixc[0] = 2; pSPARC_ATOM->ixc[1] = 3; 
        pSPARC_ATOM->ixc[2] = 0; pSPARC_ATOM->ixc[3] = 0;
        pSPARC_ATOM->xcoption[0] = 1; pSPARC_ATOM->xcoption[1] = 1;
        pSPARC_ATOM->isGradient = 1;
        if (pSPARC_ATOM->exx_frac < 0) pSPARC_ATOM->exx_frac = 1;
#ifdef DEBUG
        printf("Note: You are using HF with %.5g exact exchange.\n", pSPARC_ATOM->exx_frac);
#endif
    } else if (strcmpi(pSPARC_ATOM->XC, "PBE0") == 0) {
        xc = 41;
        pSPARC_ATOM->usefock = 1;
        pSPARC_ATOM->ixc[0] = 2; pSPARC_ATOM->ixc[1] = 3; 
        pSPARC_ATOM->ixc[2] = 0; pSPARC_ATOM->ixc[3] = 0;
        pSPARC_ATOM->xcoption[0] = 1; pSPARC_ATOM->xcoption[1] = 1;
        pSPARC_ATOM->isGradient = 1;
        if (pSPARC_ATOM->exx_frac < 0) pSPARC_ATOM->exx_frac = 0.25;
#ifdef DEBUG
        printf("Note: You are using PBE0 with %.5g exact exchange.\n", pSPARC_ATOM->exx_frac);
#endif
    } else if (strcmpi(pSPARC_ATOM->XC, "SCAN") == 0) {
        xc = -263267;
        pSPARC_ATOM->ixc[0] = 4; pSPARC_ATOM->ixc[1] = 4; 
        pSPARC_ATOM->ixc[2] = 1; pSPARC_ATOM->ixc[3] = 0;
        pSPARC_ATOM->isGradient = 1;
    } else if (strcmpi(pSPARC_ATOM->XC, "RSCAN") == 0) {
        xc = -493494;
        pSPARC_ATOM->ixc[0] = 5; pSPARC_ATOM->ixc[1] = 5; 
        pSPARC_ATOM->ixc[2] = 1; pSPARC_ATOM->ixc[3] = 0;
        pSPARC_ATOM->isGradient = 1;
    } else if (strcmpi(pSPARC_ATOM->XC, "R2SCAN") == 0) {
        xc = -497498;
        pSPARC_ATOM->ixc[0] = 6; pSPARC_ATOM->ixc[1] = 6; 
        pSPARC_ATOM->ixc[2] = 1; pSPARC_ATOM->ixc[3] = 0;
        pSPARC_ATOM->isGradient = 1;
    } else {
        printf("WARNING: The chosen XC is not available for atom solve. Defaulting to GGA_PBE...\n");
        strcpy(pSPARC_ATOM->XC, "GGA_PBE");
        xc = 11;
        pSPARC_ATOM->ixc[0] = 2; pSPARC_ATOM->ixc[1] = 3; 
        pSPARC_ATOM->ixc[2] = 0; pSPARC_ATOM->ixc[3] = 0;
        pSPARC_ATOM->xcoption[0] = 1; pSPARC_ATOM->xcoption[1] = 1;
        pSPARC_ATOM->isGradient = 1;
    }

    if (pSPARC_ATOM->psd->pspxc != xc){
#ifdef DEBUG
        printf("\nWARNING: Pseudopotential file for atom type has pspxc = %d,\n"
                    "not equal to input xc = %d (%s). Be careful with the result.\n",
                    pSPARC_ATOM->psd->pspxc, xc, pSPARC_ATOM->XC);
#endif
    }
}

void Calculate_SplineDerivRadFun_atom(SPARC_ATOM_OBJ *pSPARC_ATOM){
    int ityp, l, lcount, lcount2, np, ppl_sum, psd_len;
    int lloc = *pSPARC_ATOM->localPsd;
    psd_len = pSPARC_ATOM->psd->size;
    pSPARC_ATOM->psd->SplinerVlocD = (double *)malloc(sizeof(double)*psd_len);
    pSPARC_ATOM->psd->SplineFitIsoAtomDen = (double *)malloc(sizeof(double)*psd_len);
    pSPARC_ATOM->psd->SplineRhocD = (double *)malloc(sizeof(double)*psd_len);
    getYD_gen(pSPARC_ATOM->psd->RadialGrid,pSPARC_ATOM->psd->rVloc,pSPARC_ATOM->psd->SplinerVlocD,psd_len);
    getYD_gen(pSPARC_ATOM->psd->RadialGrid,pSPARC_ATOM->psd->rhoIsoAtom,pSPARC_ATOM->psd->SplineFitIsoAtomDen,psd_len);
    getYD_gen(pSPARC_ATOM->psd->RadialGrid,pSPARC_ATOM->psd->rho_c_table,pSPARC_ATOM->psd->SplineRhocD,psd_len);

    // note we neglect lloc
    ppl_sum = 0;
    for (l = 0; l <= pSPARC_ATOM->psd->lmax; l++) {
        //if (l == pSPARC_ATOM->localPsd[ityp]) continue; // this fails under -O3, -O2 optimization
        if (l == lloc) continue;
        ppl_sum += pSPARC_ATOM->psd->ppl[l];
    }
    pSPARC_ATOM->psd->SplineFitUdV = (double *)malloc(sizeof(double)*psd_len * ppl_sum);
    if(pSPARC_ATOM->psd->SplineFitUdV == NULL) {
#ifdef DEBUG
        printf("Memory allocation failed!\n");
#endif
        exit(EXIT_FAILURE);
    }

    for (l = lcount = lcount2 = 0; l <= pSPARC_ATOM->psd->lmax; l++) {
        if (l == lloc) {
            lcount2 += pSPARC_ATOM->psd->ppl[l];
            continue;
        }
        for (np = 0; np < pSPARC_ATOM->psd->ppl[l]; np++) {
            // note that UdV is of size (psd_len, lmax+1), while SplineFitUdV has size (psd_len, lmax)
            getYD_gen(pSPARC_ATOM->psd->RadialGrid, pSPARC_ATOM->psd->UdV+lcount2*psd_len, pSPARC_ATOM->psd->SplineFitUdV+lcount*psd_len, psd_len);
            lcount++; lcount2++;
        }
    }

}

void setValence(SPARC_ATOM_OBJ *pSPARC_ATOM) {
    int *n = pSPARC_ATOM->states_n;
    int *l = pSPARC_ATOM->states_l;
    int *f_tot = pSPARC_ATOM->states_ftot;
    int *f_up = pSPARC_ATOM->states_fup;
    int *f_dw = pSPARC_ATOM->states_fdw;

    // Remove core occupations
    int Z_core = *pSPARC_ATOM->Zatom - *pSPARC_ATOM->Znucl;
    int Z_sum, count_val, val_idx;
    Z_sum = count_val = val_idx = 0;

    // count the number of entries required
    for (int i = 0; i < pSPARC_ATOM->states_len; i++) {
        if (Z_sum < Z_core) {
            Z_sum += f_tot[i];
        } else {
            count_val++;
        }
    }

    pSPARC_ATOM->val_len = count_val;

    int *nVal = (int *)malloc(count_val * sizeof(int));
    int *lVal = (int *)malloc(count_val * sizeof(int));
    int *fTotVal = (int *)malloc(count_val * sizeof(int));
    int *fUpVal = (int *)malloc(count_val * sizeof(int));
    int *fDwVal = (int *)malloc(count_val * sizeof(int));

    // Store the valence states
    Z_sum = 0;
    int max_l = 0; int min_l = 0;
    for (int i = 0; i < pSPARC_ATOM->states_len; i++) {
        if (Z_sum < Z_core) {
            Z_sum += f_tot[i];
        } else {
            nVal[val_idx] = n[i];
            lVal[val_idx] = l[i];
            if (l[i] > max_l) {
                max_l = l[i];
            } else if (l[i] < min_l) {
                min_l = l[i];
            }
            fTotVal[val_idx] = f_tot[i];
            fUpVal[val_idx] = f_up[i];
            fDwVal[val_idx] = f_dw[i];
            val_idx++;
        }
    }
    pSPARC_ATOM->max_l = max_l; 
    pSPARC_ATOM->min_l = min_l;

    // Store f_up and f_dw in valence
    pSPARC_ATOM->fUpVal = (int *)malloc(count_val * sizeof(int));
    memcpy(pSPARC_ATOM->fUpVal,fUpVal,count_val *  sizeof(int));
    pSPARC_ATOM->fDwVal = (int *)malloc(count_val * sizeof(int));
    memcpy(pSPARC_ATOM->fDwVal,fDwVal,count_val *  sizeof(int));
    pSPARC_ATOM->fTotVal = (int *)malloc(count_val * sizeof(int));
    memcpy(pSPARC_ATOM->fTotVal,fTotVal,count_val * sizeof(int));

    int count0, count1, count2, count3;
    count0 = count1 = count2 = count3 = 0;
    for (int i = 0; i < count_val; i++){
        if (lVal[i] == 0) {
            count0++;
        } else if (lVal[i] == 1) {
            count1++;
        } else if (lVal[i] == 2) {
            count2++;
        } else if (lVal[i] == 3) {
            count3++;
        }
    }

    // Store sizes
    pSPARC_ATOM->lcount0 = count0;
    pSPARC_ATOM->lcount1 = count1;
    pSPARC_ATOM->lcount2 = count2;
    pSPARC_ATOM->lcount3 = count3;

    // Initialise null pointers for all state related information
    pSPARC_ATOM->n0 = NULL; pSPARC_ATOM->n1 = NULL; pSPARC_ATOM->n2 = NULL; pSPARC_ATOM->n3 = NULL;
    pSPARC_ATOM->f0_up = NULL; pSPARC_ATOM->f1_up = NULL; pSPARC_ATOM->f2_up = NULL; pSPARC_ATOM->f3_up = NULL;
    pSPARC_ATOM->f0_dw = NULL; pSPARC_ATOM->f1_dw = NULL; pSPARC_ATOM->f2_dw = NULL; pSPARC_ATOM->f3_dw = NULL;
    pSPARC_ATOM->f0_tot = NULL; pSPARC_ATOM->f1_tot = NULL; pSPARC_ATOM->f2_tot = NULL; pSPARC_ATOM->f3_tot = NULL;

    // Initialise memory when counter > 0 for l = 0,1,2,3
    if (count0 > 0){
        pSPARC_ATOM->n0 = (int *)malloc(count0 * sizeof(int));
        pSPARC_ATOM->f0_up = (int *)malloc(count0 * sizeof(int));
        pSPARC_ATOM->f0_dw = (int *)malloc(count0 * sizeof(int));
        pSPARC_ATOM->f0_tot = (int *)malloc(count0 * sizeof(int));
    } 
    if (count1 > 0) {
        pSPARC_ATOM->n1 = (int *)malloc(count1 * sizeof(int));
        pSPARC_ATOM->f1_up = (int *)malloc(count1 * sizeof(int));
        pSPARC_ATOM->f1_dw = (int *)malloc(count1 * sizeof(int));
        pSPARC_ATOM->f1_tot = (int *)malloc(count1 * sizeof(int));
    } 
    if (count2 > 0) {
        pSPARC_ATOM->n2 = (int *)malloc(count2 * sizeof(int));
        pSPARC_ATOM->f2_up = (int *)malloc(count2 * sizeof(int));
        pSPARC_ATOM->f2_dw = (int *)malloc(count2 * sizeof(int));
        pSPARC_ATOM->f2_tot = (int *)malloc(count2 * sizeof(int));
    } 
    if (count3 > 0) {
        pSPARC_ATOM->n3 = (int *)malloc(count3 * sizeof(int));
        pSPARC_ATOM->f3_up = (int *)malloc(count3 * sizeof(int));
        pSPARC_ATOM->f3_dw = (int *)malloc(count3 * sizeof(int));
        pSPARC_ATOM->f3_tot = (int *)malloc(count3 * sizeof(int));
    }

    // Distribute the occupations according to l = 0, 1, 2, 3
    count0 = count1 = count2 = count3 = 0;
    for (int i = 0; i < count_val; i++) {
        if (lVal[i] == 0) {
            pSPARC_ATOM->n0[count0] = nVal[i];
            pSPARC_ATOM->f0_up[count0] = fUpVal[i];
            pSPARC_ATOM->f0_dw[count0] = fDwVal[i];
            pSPARC_ATOM->f0_tot[count0] = fTotVal[i];
            count0++;
        } else if (lVal[i] == 1) {
            pSPARC_ATOM->n1[count1] = nVal[i];
            pSPARC_ATOM->f1_up[count1] = fUpVal[i];
            pSPARC_ATOM->f1_dw[count1] = fDwVal[i];
            pSPARC_ATOM->f1_tot[count1] = fTotVal[i];
            count1++;
        } else if (lVal[i] == 2) {
            pSPARC_ATOM->n2[count2] = nVal[i];
            pSPARC_ATOM->f2_up[count2] = fUpVal[i];
            pSPARC_ATOM->f2_dw[count2] = fDwVal[i];
            pSPARC_ATOM->f2_tot[count2] = fTotVal[i];
            count2++;
        } else if (lVal[i] == 3) {
            pSPARC_ATOM->n3[count3] = nVal[i];
            pSPARC_ATOM->f3_up[count3] = fUpVal[i];
            pSPARC_ATOM->f3_dw[count3] = fDwVal[i];
            pSPARC_ATOM->f3_tot[count3] = fTotVal[i];
            count3++;
        }
    }

    // Print the valence occupations
    printf("==============================================");
    printf("\nThe valence occupation:\n");
    printf("n  l  Occ\n");
    for (int i = 0; i < count_val; i++) {
        printf("%d  %d  %d\n",nVal[i], lVal[i], fTotVal[i]);
    }
    printf("==============================================");

    free(nVal); free(lVal); free(fTotVal); free(fUpVal); free(fDwVal);
}