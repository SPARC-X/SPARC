#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#include "finalizeAtom.h"
#include "isddftAtom.h"
#include "exxPotentialEnergyAtom.h"
#include "exchangeCorrelationAtom.h"
#include "toolsAtom.h"

/**
 * @brief Print eigenvalues and energy.
 */
void printResultsAtom(SPARC_ATOM_OBJ *pSPARC_ATOM) {
    char filename[64];
    int Z = *pSPARC_ATOM->Zatom;
    int Nd = pSPARC_ATOM->Nd;
    
    int lcount0 = pSPARC_ATOM->lcount0;
    int lcount1 = pSPARC_ATOM->lcount1;
    int lcount2 = pSPARC_ATOM->lcount2;
    int lcount3 = pSPARC_ATOM->lcount3;

    int *n_final, *l_final;
    int len = lcount0 + lcount1 + lcount2 + lcount3;

    n_final = (int *)malloc(len * sizeof(int));
    l_final = (int *)malloc(len * sizeof(int));  // Separate allocation for l_final
    if (n_final == NULL || l_final == NULL) {
#ifdef DEBUG
        printf("Memory allocation failed for n_final or l_final\n");
#endif
        exit(EXIT_FAILURE);
    }

    if (lcount0 > 0) {
        memcpy(n_final, pSPARC_ATOM->n0, lcount0 * sizeof(int));
        memset(l_final, 0, lcount0 * sizeof(int));
    }
    if (lcount1 > 0) {
        memcpy(n_final + lcount0, pSPARC_ATOM->n1, lcount1 * sizeof(int));
        for (int i = 0; i < lcount1; i++) {
            l_final[lcount0 + i] = 1;  // Set each element to 1
        }

    }
    if (lcount2 > 0) {
        memcpy(n_final + lcount0 + lcount1, pSPARC_ATOM->n2, lcount2 * sizeof(int));
        for (int i = 0; i < lcount1; i++) {
            l_final[lcount0 + lcount1 + i] = 2;  // Set each element to 2
        }
    }
    if (lcount3 > 0) {
        memcpy(n_final + lcount0 + lcount1 + lcount2, pSPARC_ATOM->n3, lcount3 * sizeof(int));
        for (int i = 0; i < lcount1; i++) {
            l_final[lcount0 + lcount1 + lcount2 + i] = 3;  // Set each element to 3
        }
    }

    // Calculate energy before flipping
    evaluateEnergyAtom(pSPARC_ATOM);

    // flip contents
    flipContents(pSPARC_ATOM->r, Nd+1, 1);
    flipContents(pSPARC_ATOM->electronDens, Nd - 1, pSPARC_ATOM->nspden);
    flipContents(pSPARC_ATOM->orbitals, Nd - 1, (pSPARC_ATOM->nspinor)*(pSPARC_ATOM->val_len));

    // write info to file
    if (pSPARC_ATOM->spinFlag == 0) {
        printf("First grid will be printed followed by the density, orbitals and occupations.\n");
        printf("The occupied orbitals for l = 0 are printed (columnwise) first followed by l = 1, 2, 3\n");
        printf("Number of rows: %d\n",Nd - 1);
        printf("Number of columns of density: %d\n",pSPARC_ATOM->nspden);
        printf("Number of columns of orbitals: %d\n",pSPARC_ATOM->val_len);
        printf( "\nGrid\n");
        for (int JJ = 0; JJ < Nd - 1; JJ++) {
            printf("%11.8f\n",pSPARC_ATOM->r[JJ+1]);
        }

        printf("\nDensity\n");
        printColMajor(pSPARC_ATOM->electronDens, Nd - 1, pSPARC_ATOM->nspden);

        printf("\nOrbitals\n");
        for (int JJ = 0; JJ < len; JJ++) {
            printf("%d",n_final[JJ]);
            if (l_final[JJ] == 0) {
                printf("%c ",'s');
            } else if (l_final[JJ] == 1) {
                printf("%c ",'p');
            } else if (l_final[JJ] == 2) {
                printf("%c ",'d');
            } else if (l_final[JJ] == 3) {
                printf("%c ",'f');
            }
        }
        printf("\n");

        printColMajor(pSPARC_ATOM->orbitals, Nd - 1, pSPARC_ATOM->val_len);
    } else {
        printf("First grid will be printed followed by the density, orbitals and occupations.\n");
        printf("The total denisty is printed first (columnwise) followed by spin up and spin dw density.\n");
        printf("The occupied orbitals for l = 0 are printed (columnwise) first followed by l = 1, 2, 3\n");
        printf("The spin-up orbitals are printed (columnwise) first followed by spin-dw orbitals\n\n");
        printf("Number of rows: %d\n",Nd - 1);
        printf("Number of columns of density: %d\n",pSPARC_ATOM->nspden);
        printf("Number of columns of orbitals: %d\n",2*pSPARC_ATOM->val_len);

        printf( "\nGrid\n");
        for (int JJ = 0; JJ < Nd - 1; JJ++) {
            printf("%11.8f\n",pSPARC_ATOM->r[JJ+1]);
        }

        printf("\nDensity\n");
        printColMajor(pSPARC_ATOM->electronDens, Nd - 1, pSPARC_ATOM->nspden);

        printf("\nOrbitals\n");
        for (int spinor = 0; spinor < pSPARC_ATOM->nspinor; spinor++) {
            for (int JJ = 0; JJ < len; JJ++) {
                printf("%d",n_final[JJ]);
                if (l_final[JJ] == 0) {
                    printf("%c ",'s');
                } else if (l_final[JJ] == 1) {
                    printf("%c ",'p');
                } else if (l_final[JJ] == 2) {
                    printf("%c ",'d');
                } else if (l_final[JJ] == 3) {
                    printf("%c ",'f');
                }
            }
        }
        printf("\n");

        printColMajor(pSPARC_ATOM->orbitals, Nd - 1, 2*pSPARC_ATOM->val_len);
    }

    printf("\nAzimuthal qunatum numbers of the printed orbitals\n");
    for (int spinor = 0; spinor < pSPARC_ATOM->nspinor; spinor++) {
        for (int JJ = 0; JJ < len; JJ++) {
            printf("%d ",l_final[JJ]);
        }
    }
    printf("\n");

    printf("\nOccupations\n");
    for (int JJ = 0; JJ < pSPARC_ATOM->nspinor*pSPARC_ATOM->val_len; JJ++)
        printf("%d ",pSPARC_ATOM->occ[JJ]);
    printf("\n");

    int *n_final_sorted, *Ind;
    n_final_sorted = (int *)malloc(len * sizeof(int));
    Ind = (int *)malloc(len * sizeof(int));
    if (n_final_sorted == NULL || Ind == NULL) {
#ifdef DEBUG
        printf("Memory allocation failed for n_final_sorted or Ind\n");
#endif
        free(n_final);
        free(l_final);
        exit(EXIT_FAILURE);
    }

    // Sort and get indices
    Sort_Int(n_final, len, n_final_sorted, Ind);

    // Extract eigenvalues
    double *eigValUp = (double *)malloc(len * sizeof(double));
    if (eigValUp == NULL) {
#ifdef DEBUG
        printf("Memory allocation failed for eigValUp\n");
#endif
        free(n_final);
        free(l_final);
        free(n_final_sorted);
        free(Ind);
        exit(EXIT_FAILURE);
    }
    memcpy(eigValUp, pSPARC_ATOM->eigenVal, len * sizeof(double));

    double *eigValDw = NULL;
    if (pSPARC_ATOM->spinFlag) {
        eigValDw = (double *)malloc(len * sizeof(double));
        if (eigValDw == NULL) {
#ifdef DEBUG
            printf("Memory allocation failed for eigValDw\n");
#endif
            free(n_final);
            free(l_final);
            free(n_final_sorted);
            free(Ind);
            free(eigValUp);
            exit(EXIT_FAILURE);
        }
        memcpy(eigValDw, pSPARC_ATOM->eigenVal + len, len * sizeof(double));
    }

    // Create copies
    double *eigValUp_copy = (double *)malloc(len * sizeof(double));
    int *l_final_copy = (int *)malloc(len * sizeof(int));
    double *eigValDw_copy = NULL;
    memcpy(eigValUp_copy, eigValUp, len * sizeof(double));
    memcpy(l_final_copy,l_final, sizeof(int) * len);
    if (pSPARC_ATOM->spinFlag) {
        eigValDw_copy = (double *)malloc(len * sizeof(double));
        memcpy(eigValDw_copy, eigValDw, len * sizeof(double));
    }

    for (int i = 0; i < len; i++) {
        n_final[i] = n_final_sorted[i];
        l_final[i] = l_final_copy[Ind[i]];
        eigValUp[i] = eigValUp_copy[Ind[i]];
        if (pSPARC_ATOM->spinFlag) {
            eigValDw[i] = eigValDw_copy[Ind[i]];
        }
    }

    printf("\n============================================================\n");
    printf("Eigenvalues (Hartree)\n");
    printf("=============================================================\n");
    if (pSPARC_ATOM->spinFlag == 0) {
        printf("n     l     Occ     Eigenvalue\n");
    } else {
        printf("n     l     s     Occ     Eigenvalue\n");
    }
    printf("=============================================================\n");

    for (int i = 0; i < len; i++) {
        if (pSPARC_ATOM->spinFlag == 0) {
            printf("%d     %d     %d     %0.15f\n", n_final[i], l_final[i], 
                    pSPARC_ATOM->fTotVal[i], eigValUp[i]);
        } else {
            printf("%d     %d     +%0.1f     %d     %0.15f\n", n_final[i], l_final[i], 0.5, 
                    pSPARC_ATOM->fUpVal[i], eigValUp[i]);
            printf("%d     %d     -%0.1f     %d     %0.15f\n", n_final[i], l_final[i], 0.5, 
                    pSPARC_ATOM->fDwVal[i], eigValDw[i]);
        }
    }

    free(n_final);
    free(l_final); free(l_final_copy);
    free(eigValUp); free(eigValUp_copy);
    free(n_final_sorted);
    free(Ind);

    if (pSPARC_ATOM->spinFlag) {
        free(eigValDw); free(eigValDw_copy);
    }

    // Print Energy
    printf("\n============================================================\n");
    printf("Energy (Hartree)\n");
    printf("============================================================\n");
    printf("Free energy:                        %1.10e\n",pSPARC_ATOM->Etot);
    printf("Band Structure energy:              %1.10e\n",pSPARC_ATOM->Eband);
    printf("XC energy:                          %1.10e\n",pSPARC_ATOM->Exc);
    printf("XC Correction energy:               % 1.10e\n",-pSPARC_ATOM->Exc_dc);
    printf("Hartree energy:                     %1.10e\n",pSPARC_ATOM->Eelec_dc);

    if (pSPARC_ATOM->usefock > 1)
        printf("Exact exchange energy:              %1.10e\n", pSPARC_ATOM->Exx);

}


/**
 * @brief Calculate energy.
 */
void evaluateEnergyAtom(SPARC_ATOM_OBJ *pSPARC_ATOM) {
    int Nd = pSPARC_ATOM->Nd;
    double *r = pSPARC_ATOM->r;
    double *w = pSPARC_ATOM->w;
    double *int_scale = pSPARC_ATOM->int_scale;
    int nspinor = pSPARC_ATOM->nspinor;
    int ncol = pSPARC_ATOM->nspden;
    int *occ = pSPARC_ATOM->occ;
    double *eigenVal = pSPARC_ATOM->eigenVal;
    int val_len = pSPARC_ATOM->val_len;
    double *rho = (double *)malloc(ncol * (Nd - 1) * sizeof(double));

    // Band energy
    double Eband = 0.0;
    for (int spinor = 0; spinor < nspinor; spinor++) {
        for (int i = 0; i < val_len; i++)
            Eband += occ[spinor*val_len + i]*eigenVal[spinor*val_len + i];
    }
    pSPARC_ATOM->Eband = Eband;

    // add core electron density if needed
    add_rho_core_atom(pSPARC_ATOM, pSPARC_ATOM->electronDens, rho, ncol);

    // Electrostatic Hartree energy
    double Eelec_dc = 0.0;
    for (int jj = 0; jj < Nd - 1; jj++)
        Eelec_dc += (0.5*4*M_PI)*(w[jj+1]/int_scale[jj+1])*(-pow(r[jj+1], 2)*pSPARC_ATOM->electronDens[jj]*pSPARC_ATOM->phi[jj]);
    pSPARC_ATOM->Eelec_dc = Eelec_dc;
    // printf("Elec_dc=%f\n",Eelec_dc); //debug

    // Exchange correlation energy and correction
    double Exc = 0.0; double Exc_dc = 0.0; double Escan_dc = 0.0;
    for (int jj = 0; jj < Nd - 1; jj++)
        Exc += (4*M_PI)*(w[jj+1]/int_scale[jj+1])*(pow(r[jj+1], 2)*rho[jj]*pSPARC_ATOM->e_xc[jj]);
    
    if (pSPARC_ATOM->spinFlag == 0) {
        for (int jj = 0; jj < Nd - 1; jj++) {
            Exc_dc += (4*M_PI)*(w[jj+1]/int_scale[jj+1])*(pow(r[jj+1], 2)*pSPARC_ATOM->electronDens[jj]*pSPARC_ATOM->XCPotential[jj]);
            if (pSPARC_ATOM->ixc[2]) // metaGGA
                Escan_dc += (4*M_PI)*(w[jj+1]/int_scale[jj+1])*(pow(r[jj+1], 2)*pSPARC_ATOM->tau[jj]*pSPARC_ATOM->vxcMGGA3[jj]);
        }
        Exc_dc += Escan_dc;
    } else {
        int shift = Nd - 1;
        for (int jj = 0; jj < Nd - 1; jj++) {
            Exc_dc += (4*M_PI)*(w[jj+1]/int_scale[jj+1])*(pow(r[jj+1], 2)*pSPARC_ATOM->electronDens[jj+shift]*pSPARC_ATOM->XCPotential[jj]);
            Exc_dc += (4*M_PI)*(w[jj+1]/int_scale[jj+1])*(pow(r[jj+1], 2)*pSPARC_ATOM->electronDens[jj+2*shift]*pSPARC_ATOM->XCPotential[jj+shift]);
            if (pSPARC_ATOM->ixc[2]) { // metaGGA
                Escan_dc += (4*M_PI)*(w[jj+1]/int_scale[jj+1])*(pow(r[jj+1], 2)*pSPARC_ATOM->tau[jj+shift]*pSPARC_ATOM->vxcMGGA3[jj]);
                Escan_dc += (4*M_PI)*(w[jj+1]/int_scale[jj+1])*(pow(r[jj+1], 2)*pSPARC_ATOM->tau[jj+2*shift]*pSPARC_ATOM->vxcMGGA3[jj+shift]);
            }
        }
        Exc_dc += Escan_dc;
    }
    pSPARC_ATOM->Exc = Exc; pSPARC_ATOM->Exc_dc = Exc_dc;
    // printf("Exc=%f\n",Exc); //debug
    // printf("Exc_dc=%f\n",Exc_dc); //debug

    // hybrid
    if (pSPARC_ATOM->usefock < 2) {
        pSPARC_ATOM->Etot = Eband + Exc - Exc_dc + Eelec_dc;
    } else {
        double Exx;
        evaluateExxEnergyAtom(pSPARC_ATOM, &Exx);
        Exx *= pSPARC_ATOM->exx_frac;
        pSPARC_ATOM->Exx = Exx;
        Exc += Exx;
        pSPARC_ATOM->Etot = Eband + Exc - Exc_dc + Eelec_dc - 2*Exx;
        pSPARC_ATOM->Exc = Exc;
    }

    free(rho);
}

/**
* @brief Copy into ATOM_SOLN_OBJ for internal use in the code
*/
void copyAtomSolution(SPARC_ATOM_OBJ *pSPARC_ATOM, SPARC_OBJ *pSPARC, int ityp) {
    // Calculate energy before flipping
    evaluateEnergyAtom(pSPARC_ATOM);

    // Print Energy
    printf("\n============================================================\n");
    printf("Energy (Hartree)\n");
    printf("============================================================\n");
    printf("Free energy:                        %1.10e\n",pSPARC_ATOM->Etot);
    printf("Band Structure energy:              %1.10e\n",pSPARC_ATOM->Eband);
    printf("XC energy:                          %1.10e\n",pSPARC_ATOM->Exc);
    printf("XC Correction energy:               % 1.10e\n",-pSPARC_ATOM->Exc_dc);
    printf("Hartree energy:                     %1.10e\n",pSPARC_ATOM->Eelec_dc);

    if (pSPARC_ATOM->usefock > 1)
        printf("Exact exchange energy:              %1.10e\n", pSPARC_ATOM->Exx);

    int Nd = pSPARC_ATOM->Nd;
    double *r = pSPARC_ATOM->r;
    double *orb = pSPARC_ATOM->orbitals;
    int val_len = pSPARC_ATOM->val_len;
    int nspinor = pSPARC_ATOM->nspinor;
    int orb_size = (Nd)*val_len*nspinor;
    int max_l = pSPARC_ATOM->max_l;
    int min_l = pSPARC_ATOM->min_l;
    // allocate memory
    pSPARC->AtmU[ityp].orbitals = (double *)malloc(orb_size*sizeof(double));
    pSPARC->AtmU[ityp].RadialGrid = (double *)calloc((Nd),sizeof(double));
    pSPARC->AtmU[ityp].ppl = (int *)calloc((max_l+1),sizeof(int));
    pSPARC->AtmU[ityp].occ = (int *)malloc(nspinor*val_len*sizeof(int));
    
    // copy items
    pSPARC->AtmU[ityp].size = Nd;
    pSPARC->AtmU[ityp].max_l = max_l;
    pSPARC->AtmU[ityp].nspinor = pSPARC_ATOM->nspinor;
    pSPARC->AtmU[ityp].val_len = pSPARC_ATOM->val_len;
    // memcpy(pSPARC->AtmU[ityp].orbitals, pSPARC_ATOM->orbitals, orb_size*sizeof(double));
    memcpy(pSPARC->AtmU[ityp].RadialGrid, r+1, (Nd)*sizeof(double));
    pSPARC->AtmU[ityp].RadialGrid[Nd-1] = 0.0;
    memcpy(pSPARC->AtmU[ityp].occ, pSPARC_ATOM->occ, val_len*nspinor*sizeof(int));

    // flip contents
    flipContents(pSPARC->AtmU[ityp].RadialGrid, Nd, 1);
    // flipContents(pSPARC->AtmU[ityp].orbitals, Nd - 1, (pSPARC_ATOM->nspinor)*(pSPARC_ATOM->val_len));
    flipContents(pSPARC_ATOM->orbitals, Nd - 1, (pSPARC_ATOM->nspinor)*(pSPARC_ATOM->val_len));

    for (int col = 0; col < nspinor * val_len; col++) {
        for (int row = 0; row < Nd - 1; row++) {
            // pSPARC->AtmU[ityp].orbitals[col * (Nd - 1) + row] /= pSPARC->AtmU[ityp].RadialGrid[row];
            pSPARC->AtmU[ityp].orbitals[col * Nd + row + 1] = pSPARC_ATOM->orbitals[col*(Nd - 1) + row]/pSPARC->AtmU[ityp].RadialGrid[row+1];
        }
        pSPARC->AtmU[ityp].orbitals[col * Nd] = pSPARC->AtmU[ityp].orbitals[col * Nd + 1];
    }

    // printf( "\nOcc\n");
    // for (int JJ = 0; JJ < val_len*nspinor; JJ++) {
    //     printf("%d\n",pSPARC->AtmU[ityp].occ[JJ]);
    // }

    // printf( "\nOcc_OG\n");
    // for (int JJ = 0; JJ < val_len*nspinor; JJ++) {
    //     printf("%d\n",pSPARC_ATOM->occ[JJ]);
    // }

    // printf( "\nGrid OG\n");
    // for (int JJ = Nd-1; JJ >= 1; JJ--) {
    //     printf("%11.8f\n",r[JJ]);
    // }

    // printColMajor(pSPARC->AtmU[ityp].orbitals, Nd, pSPARC_ATOM->val_len);
    // printf("\nOG:\n");
    // printColMajor(pSPARC_ATOM->orbitals, Nd-1, pSPARC_ATOM->val_len);

    for (int JJ = 0; JJ <= max_l; JJ++) {
        if (JJ == 0) {
            pSPARC->AtmU[ityp].ppl[JJ] = pSPARC_ATOM->lcount0;
        } else if (JJ == 1) {
            pSPARC->AtmU[ityp].ppl[JJ] = pSPARC_ATOM->lcount1;
        } else if (JJ == 2) {
            pSPARC->AtmU[ityp].ppl[JJ] = pSPARC_ATOM->lcount2;
        } else if (JJ == 3) {
            pSPARC->AtmU[ityp].ppl[JJ] = pSPARC_ATOM->lcount3;
        }
    }
}

/**
 * @brief Free memory
 */
void Finalize_Atom(SPARC_ATOM_OBJ *pSPARC_ATOM) {
    if (pSPARC_ATOM->Zatom != NULL) free(pSPARC_ATOM->Zatom);
    if (pSPARC_ATOM->Znucl != NULL) free(pSPARC_ATOM->Znucl);
    if (pSPARC_ATOM->mag != NULL) free(pSPARC_ATOM->mag);
    if (pSPARC_ATOM->x != NULL) free(pSPARC_ATOM->x);
    if (pSPARC_ATOM->r != NULL) free(pSPARC_ATOM->r);
    if (pSPARC_ATOM->w != NULL) free(pSPARC_ATOM->w);
    if (pSPARC_ATOM->int_scale != NULL) free(pSPARC_ATOM->int_scale);
    if (pSPARC_ATOM->D != NULL) free(pSPARC_ATOM->D);
    if (pSPARC_ATOM->grad != NULL) free(pSPARC_ATOM->grad);
    if (pSPARC_ATOM->laplacian != NULL) free(pSPARC_ATOM->laplacian);
    if (pSPARC_ATOM->electronDens != NULL) free(pSPARC_ATOM->electronDens);
    // if (pSPARC_ATOM->electronDens_core != NULL) free(pSPARC_ATOM->electronDens_core);
    if (pSPARC_ATOM->psd->fchrg > 1e-12) free(pSPARC_ATOM->electronDens_core);
    if (pSPARC_ATOM->orbitals != NULL) free(pSPARC_ATOM->orbitals);
    if (pSPARC_ATOM->orbital_l != NULL) free(pSPARC_ATOM->orbital_l);
    if (pSPARC_ATOM->eigenVal != NULL) free(pSPARC_ATOM->eigenVal);
    if (pSPARC_ATOM->occ != NULL) free(pSPARC_ATOM->occ);
    if (pSPARC_ATOM->phi != NULL) free(pSPARC_ATOM->phi);
    if (pSPARC_ATOM->states_n != NULL) free(pSPARC_ATOM->states_n);
    if (pSPARC_ATOM->states_l != NULL) free(pSPARC_ATOM->states_l);
    if (pSPARC_ATOM->states_fup != NULL) free(pSPARC_ATOM->states_fup);
    if (pSPARC_ATOM->states_fdw != NULL) free(pSPARC_ATOM->states_fdw);
    if (pSPARC_ATOM->states_ftot != NULL) free(pSPARC_ATOM->states_ftot);
    if (pSPARC_ATOM->fUpVal != NULL) free(pSPARC_ATOM->fUpVal);
    if (pSPARC_ATOM->fDwVal != NULL) free(pSPARC_ATOM->fDwVal);
    if (pSPARC_ATOM->fTotVal != NULL) free(pSPARC_ATOM->fTotVal);
    if (pSPARC_ATOM->n0 != NULL) free(pSPARC_ATOM->n0);
    if (pSPARC_ATOM->n1 != NULL) free(pSPARC_ATOM->n1);
    if (pSPARC_ATOM->n2 != NULL) free(pSPARC_ATOM->n2);
    if (pSPARC_ATOM->n3 != NULL) free(pSPARC_ATOM->n3);
    if (pSPARC_ATOM->f0_up != NULL) free(pSPARC_ATOM->f0_up);
    if (pSPARC_ATOM->f1_up != NULL) free(pSPARC_ATOM->f1_up);
    if (pSPARC_ATOM->f2_up != NULL) free(pSPARC_ATOM->f2_up);
    if (pSPARC_ATOM->f3_up != NULL) free(pSPARC_ATOM->f3_up);
    if (pSPARC_ATOM->f0_dw != NULL) free(pSPARC_ATOM->f0_dw);
    if (pSPARC_ATOM->f1_dw != NULL) free(pSPARC_ATOM->f1_dw);
    if (pSPARC_ATOM->f2_dw != NULL) free(pSPARC_ATOM->f2_dw);
    if (pSPARC_ATOM->f3_dw != NULL) free(pSPARC_ATOM->f3_dw);
    if (pSPARC_ATOM->f0_tot != NULL) free(pSPARC_ATOM->f0_tot);
    if (pSPARC_ATOM->f1_tot != NULL) free(pSPARC_ATOM->f1_tot);
    if (pSPARC_ATOM->f2_tot != NULL) free(pSPARC_ATOM->f2_tot);
    if (pSPARC_ATOM->f3_tot != NULL) free(pSPARC_ATOM->f3_tot);
    if (pSPARC_ATOM->VJ != NULL) free(pSPARC_ATOM->VJ);
    if (pSPARC_ATOM->XCPotential != NULL) free(pSPARC_ATOM->XCPotential);
    if (pSPARC_ATOM->e_xc != NULL) free(pSPARC_ATOM->e_xc);
    if (pSPARC_ATOM->Dxcdgrho != NULL) free(pSPARC_ATOM->Dxcdgrho);
    if (pSPARC_ATOM->X != NULL) free(pSPARC_ATOM->X);
    if (pSPARC_ATOM->F != NULL) free(pSPARC_ATOM->F);
    if (pSPARC_ATOM->mixing_hist_fkm1 != NULL) free(pSPARC_ATOM->mixing_hist_fkm1);
    if (pSPARC_ATOM->mixing_hist_xkm1 != NULL) free(pSPARC_ATOM->mixing_hist_xkm1);
    if (pSPARC_ATOM->mixing_hist_xk != NULL) free(pSPARC_ATOM->mixing_hist_xk);
    if (pSPARC_ATOM->mixing_hist_fk != NULL) free(pSPARC_ATOM->mixing_hist_fk);

    // non local
    if (pSPARC_ATOM->Vnl != NULL) {
        for (int l = 0; l <= pSPARC_ATOM->psd->lmax; l++) {
            if (l == *pSPARC_ATOM->localPsd) {
                continue; // Skip the local potential channel
            }
            if(pSPARC_ATOM->Vnl[l].V != NULL) {
                free(pSPARC_ATOM->Vnl[l].V);
            }
        }
        free(pSPARC_ATOM->Vnl);
    }

    if (pSPARC_ATOM->psd != NULL) {
        if (pSPARC_ATOM->psd->rVloc != NULL) free(pSPARC_ATOM->psd->rVloc);
        if (pSPARC_ATOM->psd->UdV != NULL) free(pSPARC_ATOM->psd->UdV);
        if (pSPARC_ATOM->psd->rhoIsoAtom != NULL) free(pSPARC_ATOM->psd->rhoIsoAtom);
        if (pSPARC_ATOM->psd->RadialGrid != NULL) free(pSPARC_ATOM->psd->RadialGrid);
        if (pSPARC_ATOM->psd->SplinerVlocD != NULL) free(pSPARC_ATOM->psd->SplinerVlocD);
        if (pSPARC_ATOM->psd->SplineFitUdV != NULL) free(pSPARC_ATOM->psd->SplineFitUdV);
        if (pSPARC_ATOM->psd->SplineFitIsoAtomDen != NULL) free(pSPARC_ATOM->psd->SplineFitIsoAtomDen);
        if (pSPARC_ATOM->psd->SplineRhocD != NULL) free(pSPARC_ATOM->psd->SplineRhocD);
        if (pSPARC_ATOM->psd->rc != NULL) free(pSPARC_ATOM->psd->rc);
        if (pSPARC_ATOM->psd->Gamma != NULL) free(pSPARC_ATOM->psd->Gamma);
        if (pSPARC_ATOM->psd->rho_c_table != NULL) free(pSPARC_ATOM->psd->rho_c_table);
        if (pSPARC_ATOM->psd->ppl != NULL) free(pSPARC_ATOM->psd->ppl);
        free(pSPARC_ATOM->psd);
    }

    if (pSPARC_ATOM->localPsd != NULL) {
        free(pSPARC_ATOM->localPsd);
    }

    // metaGGA
    if (pSPARC_ATOM->tau != NULL) free(pSPARC_ATOM->tau);
    if (pSPARC_ATOM->vxcMGGA3 != NULL) free(pSPARC_ATOM->vxcMGGA3);

    // hybrid
    if (pSPARC_ATOM->EXXL != NULL) {
        for (int l = 0; l <= pSPARC_ATOM->max_l; l++) {
            if (pSPARC_ATOM->EXXL[l].VexxUp != NULL) free(pSPARC_ATOM->EXXL[l].VexxUp);
            if (pSPARC_ATOM->EXXL[l].VexxDw != NULL) free(pSPARC_ATOM->EXXL[l].VexxDw);
        }
        free(pSPARC_ATOM->EXXL);
    }
}
