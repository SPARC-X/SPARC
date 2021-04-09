/**
 * @file    readfiles.c
 * @brief   This file contains the functions for reading files.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <ctype.h>
#include <time.h>

#include "readfiles.h"
#include "atomdata.h"
#include "tools.h"
#include "isddft.h"
#include "initialization.h"

#define TEMP_TOL 1e-12
#define min(x,y) ((x)<(y)?(x):(y))

/**
 * @brief   Find element type from provided atom_type string.
 *
 *          When user provides atom_type, the provided atom_type
 *          sometimes is appended with a number, e.g., C1 or Al3.
 *          This function truncates the number and returns the 
 *          element type only.
 *
 * @param element   Element name. (OUTPUT)
 * @param atom_type User provided atom type name.
 * @param num       Maximum number of characters contained in atom_type.
 */
//void find_element(char *element, char *atom_type, size_t num) 
void find_element(char *element, char *atom_type) 
{
    char *pch, key[]="0123456789", str[64];
    snprintf(str, sizeof(str), "%s", atom_type);
    
    pch = NULL;
    pch = strpbrk(str, key); 
    if (pch != NULL) {
        memcpy(element, str, pch-str);
        element[pch-str] = '\0';
    } else {
        strncpy( element, str, min(sizeof(element),strlen(str)+1) );
    }
}



/**
 * @brief   Read input file.
 */
void read_input(SPARC_INPUT_OBJ *pSPARC_Input, SPARC_OBJ *pSPARC) {
    char *input_filename = malloc(L_STRING * sizeof(char));
    char *str            = malloc(L_STRING * sizeof(char));
    char *temp           = malloc(L_STRING * sizeof(char));
    int i, Flag_smear_typ = 0, Flag_Temp = 0, Flag_elecT = 0, Flag_ionT = 0, Flag_ionT_end = 0; // Flag_eqT = 0,
    int Flag_accuracy = 0;
    int Flag_kptshift = 0;
    int Flag_fdgrid, Flag_ecut, Flag_meshspacing;
    Flag_fdgrid = Flag_ecut = Flag_meshspacing = 0;
    int Flag_tol_relaxcell = 0;

    snprintf(input_filename, L_STRING, "%s.inpt", pSPARC_Input->filename);
    
    FILE *input_fp = fopen(input_filename,"r");
    
    if (input_fp == NULL) {
        printf("\nCannot open file \"%s\"\n",input_filename);
        print_usage();
        exit(EXIT_FAILURE);
    }
#ifdef DEBUG
    printf("Reading input file %s\n",input_filename);
#endif
    while (!feof(input_fp)) {
        fscanf(input_fp,"%s",str);
        
        // enable commenting with '#'
        if (str[0] == '#' || str[0] == '\n'|| strcmpi(str,"undefined") == 0) {
            fscanf(input_fp, "%*[^\n]\n"); // skip current line
            continue;
        }

        // check variable name and assign value
        if (strcmpi(str,"NP_SPIN_PARAL:") == 0) {
            fscanf(input_fp,"%d", &pSPARC_Input->npspin);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"NP_KPOINT_PARAL:") == 0) {
            fscanf(input_fp,"%d", &pSPARC_Input->npkpt);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"NP_BAND_PARAL:") == 0) {
            fscanf(input_fp,"%d", &pSPARC_Input->npband);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"NP_DOMAIN_PARAL:") == 0) {
            fscanf(input_fp,"%d", &pSPARC_Input->npNdx);
            fscanf(input_fp,"%d", &pSPARC_Input->npNdy);
            fscanf(input_fp,"%d", &pSPARC_Input->npNdz);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"NP_DOMAIN_PHI_PARAL:") == 0) {
            fscanf(input_fp,"%d", &pSPARC_Input->npNdx_phi);
            fscanf(input_fp,"%d", &pSPARC_Input->npNdy_phi);
            fscanf(input_fp,"%d", &pSPARC_Input->npNdz_phi);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"EIG_SERIAL_MAXNS:") == 0) {
            fscanf(input_fp,"%d", &pSPARC_Input->eig_serial_maxns);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"EIG_PARAL_BLKSZ:") == 0) {
            fscanf(input_fp,"%d", &pSPARC_Input->eig_paral_blksz);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"CELL:") == 0) {
            fscanf(input_fp,"%lf", &pSPARC_Input->range_x);
            fscanf(input_fp,"%lf", &pSPARC_Input->range_y);
            fscanf(input_fp,"%lf", &pSPARC_Input->range_z);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"LATVEC:") == 0) {
            fscanf(input_fp, "%*[^\n]\n");
            fscanf(input_fp,"%lf %lf %lf", &pSPARC_Input->LatVec[0], &pSPARC_Input->LatVec[1], &pSPARC_Input->LatVec[2]);
            fscanf(input_fp, "%*[^\n]\n");
            fscanf(input_fp,"%lf %lf %lf", &pSPARC_Input->LatVec[3], &pSPARC_Input->LatVec[4], &pSPARC_Input->LatVec[5]);
            fscanf(input_fp, "%*[^\n]\n");
            fscanf(input_fp,"%lf %lf %lf", &pSPARC_Input->LatVec[6], &pSPARC_Input->LatVec[7], &pSPARC_Input->LatVec[8]);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"BOUNDARY_CONDITION:") == 0) { 
            fscanf(input_fp,"%d",&pSPARC_Input->BC);
            fscanf(input_fp, "%*[^\n]\n");
            printf("WARNING: \"BOUNDARY_CONDITION\" is obsolete, use \"BC\" instead!\n");
        } else if (strcmpi(str,"BC:") == 0) { 
            // read BC in 1st lattice direction
            fscanf(input_fp,"%s",temp);
            if (strcmpi(temp,"p") == 0) {
                pSPARC_Input->BCx = 0;
            } else if (strcmpi(temp,"d") == 0) {
                pSPARC_Input->BCx = 1;
            } else {
                printf("Cannot recognize boundary condition: %s\n", temp);
                exit(EXIT_FAILURE);
            }
            // read BC in 2nd lattice direction
            fscanf(input_fp,"%s",temp);
            if (strcmpi(temp,"p") == 0) {
                pSPARC_Input->BCy = 0;
            } else if (strcmpi(temp,"d") == 0) {
                pSPARC_Input->BCy = 1;
            } else {
                printf("Cannot recognize boundary condition: %s\n", temp);
                exit(EXIT_FAILURE);
            }
            // read BC in 3rd lattice direction
            fscanf(input_fp,"%s",temp);
            if (strcmpi(temp,"p") == 0) {
                pSPARC_Input->BCz = 0;
            } else if (strcmpi(temp,"d") == 0) {
                pSPARC_Input->BCz = 1;
            } else {
                printf("Cannot recognize boundary condition: %s\n", temp);
                exit(EXIT_FAILURE);
            }
            snprintf(str, L_STRING, "undefined");    // initialize str
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"POISSON_SOLVER:") == 0){
            // read solver type
            fscanf(input_fp,"%s",temp);
            if (strcmpi(temp,"aar") == 0) {
                pSPARC_Input->Poisson_solver = 0;
            // } else if (strcmpi(temp,"cg") == 0) {
            //     pSPARC_Input->Poisson_solver = 1;
            } else {
                printf("Cannot recognize Poisson solver: %s\n", temp);
                exit(EXIT_FAILURE);
            }
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"FD_ORDER:") == 0) {
            fscanf(input_fp,"%d",&pSPARC_Input->order);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"FD_GRID:") == 0) {
            fscanf(input_fp,"%d",&pSPARC_Input->numIntervals_x);
            fscanf(input_fp,"%d",&pSPARC_Input->numIntervals_y);
            fscanf(input_fp,"%d",&pSPARC_Input->numIntervals_z);
            Flag_fdgrid++; // count number of times FD grid is defined
            snprintf(str, L_STRING, "undefined");    // initialize str
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"MESH_SPACING:") == 0) {
            fscanf(input_fp,"%lf",&pSPARC_Input->mesh_spacing);
            Flag_meshspacing++; // count number of times mesh is defined
            snprintf(str, L_STRING, "undefined");    // initialize str
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"ECUT:") == 0) {
            fscanf(input_fp,"%lf",&pSPARC_Input->ecut);
            Flag_ecut++; // count number of times ecut is defined
            snprintf(str, L_STRING, "undefined");    // initialize str
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"KPOINT_GRID:") == 0) {
            fscanf(input_fp,"%d",&pSPARC_Input->Kx);
            fscanf(input_fp,"%d",&pSPARC_Input->Ky);
            fscanf(input_fp,"%d",&pSPARC_Input->Kz);
            pSPARC_Input->Nkpts = pSPARC_Input->Kx * pSPARC_Input->Ky
                                  * pSPARC_Input->Kz;
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"KPOINT_SHIFT:") == 0) {
            fscanf(input_fp,"%lf %lf %lf",&pSPARC_Input->kptshift[0], &pSPARC_Input->kptshift[1], &pSPARC_Input->kptshift[2]);
            Flag_kptshift = 1;
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"SPIN_TYP:") == 0) {
            fscanf(input_fp,"%d",&pSPARC_Input->spin_typ);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"ELEC_TEMP_TYPE:") == 0) {
            Flag_smear_typ ++;
            fscanf(input_fp,"%s",temp); // read smearing type
            // first convert to all lower case and check
            if (strcmpi(temp,"fd") == 0 || strcmpi(temp,"fermi-dirac") == 0) { 
                pSPARC_Input->elec_T_type = 0;
            } else if (strcmpi(temp,"gaussian") == 0) {
                pSPARC_Input->elec_T_type = 1;
            } else {
                printf("\nCannot recognize electronic temperature (smearing) type: \"%s\"\n",temp);
                printf("Available options: \"fd\" (or \"Fermi-Dirac\"), \"Gaussian\" (case insensitive) \n");
                exit(EXIT_FAILURE);
            }
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"SMEARING:") == 0) {
            Flag_Temp ++;
            Flag_elecT ++;
            double smearing = 0.0;
            fscanf(input_fp,"%lf",&smearing);
            pSPARC_Input->Beta = 1. / smearing;
            pSPARC_Input->elec_T = 1./(CONST_KB * pSPARC_Input->Beta);
            snprintf(str, L_STRING, "undefined");    // initialize str
            fscanf(input_fp, "%*[^\n]\n");
        } /*else if (strcmpi(str,"BETA:") == 0) {
            Flag_Temp ++;
            Flag_elecT ++;
            fscanf(input_fp,"%lf",&pSPARC_Input->Beta);
            pSPARC_Input->elec_T = 1./(CONST_KB * pSPARC_Input->Beta);
            snprintf(str, L_STRING, "undefined");    // initialize str
            fscanf(input_fp, "%*[^\n]\n");
        } */else if (strcmpi(str,"ELEC_TEMP:") == 0) {
            Flag_Temp ++;
            Flag_elecT ++;
            fscanf(input_fp,"%lf",&pSPARC_Input->elec_T);
            pSPARC_Input->Beta = 1./(CONST_KB * pSPARC_Input->elec_T);
            snprintf(str, L_STRING, "undefined");    // initialize str
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"CHEB_DEGREE:") == 0) {
            fscanf(input_fp,"%d",&pSPARC_Input->ChebDegree);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"CHEFSI_OPTMZ:") == 0) {
            fscanf(input_fp,"%d",&pSPARC_Input->CheFSI_Optmz);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"CHEFSI_BOUND_FLAG:") == 0) {
            fscanf(input_fp,"%d",&pSPARC_Input->chefsibound_flag);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"FIX_RAND:") == 0) {
            fscanf(input_fp,"%d",&pSPARC_Input->FixRandSeed);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"RHO_TRIGGER:") == 0) {
            fscanf(input_fp,"%d",&pSPARC_Input->rhoTrigger);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"NSTATES:") == 0){
            fscanf(input_fp,"%d",&pSPARC_Input->Nstates);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"NET_CHARGE:") == 0){
            fscanf(input_fp,"%d",&pSPARC_Input->NetCharge);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"MAXIT_SCF:") == 0) {
            fscanf(input_fp,"%d",&pSPARC_Input->MAXIT_SCF); 
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"MINIT_SCF:") == 0) {
            fscanf(input_fp,"%d",&pSPARC_Input->MINIT_SCF); 
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"MAXIT_POISSON:") == 0) {
            fscanf(input_fp,"%d",&pSPARC_Input->MAXIT_POISSON); 
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"RELAX_NITER:") == 0) {
            fscanf(input_fp,"%d",&pSPARC_Input->Relax_Niter); 
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"RELAX_MAXDILAT:") == 0) {
            fscanf(input_fp,"%lf",&pSPARC_Input->max_dilatation); 
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"ACCURACY:") == 0) { 
            Flag_accuracy++;
            fscanf(input_fp,"%s",temp); // read accuracy level
            if (strcmpi(temp,"minimal") == 0) {
                pSPARC_Input->accuracy_level = 0;
            } else if (strcmpi(temp,"low") == 0) {
                pSPARC_Input->accuracy_level = 1;
            } else if (strcmpi(temp,"medium") == 0) {
                pSPARC_Input->accuracy_level = 2;
            } else if (strcmpi(temp,"high") == 0) {
                pSPARC_Input->accuracy_level = 3;
            } else if (strcmpi(temp,"extreme") == 0) {
                pSPARC_Input->accuracy_level = 4;
            } else {
                printf("\nCannot recognize accuracy level: \"%s\"\n",temp);
                exit(EXIT_FAILURE);
            }
            snprintf(str, L_STRING, "undefined");    // initialize str
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"SCF_FORCE_ACC:") == 0) { 
            Flag_accuracy++;
            fscanf(input_fp,"%lf",&pSPARC_Input->target_force_accuracy);
            snprintf(str, L_STRING, "undefined");    // initialize str
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"SCF_ENERGY_ACC:") == 0) { 
            Flag_accuracy++;
            fscanf(input_fp,"%lf",&pSPARC_Input->target_energy_accuracy);
            snprintf(str, L_STRING, "undefined");    // initialize str
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"TOL_SCF:") == 0) { 
            fscanf(input_fp,"%lf",&pSPARC_Input->TOL_SCF);
            pSPARC_Input->scf_err_type = 0; // can remove since default is 0
            snprintf(str, L_STRING, "undefined");    // initialize str
            fscanf(input_fp, "%*[^\n]\n");
        } /*else if (strcmpi(str,"TOL_SCF_QE:") == 0) { 
            fscanf(input_fp,"%lf",&pSPARC_Input->TOL_SCF);
            pSPARC_Input->scf_err_type = 1;
            snprintf(str, L_STRING, "undefined");    // initialize str
            fscanf(input_fp, "%*[^\n]\n");
        } */else if (strcmpi(str,"TOL_POISSON:") == 0) {
            fscanf(input_fp,"%lf",&pSPARC_Input->TOL_POISSON);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"TOL_RELAX:") == 0) {
            fscanf(input_fp,"%lf",&pSPARC_Input->TOL_RELAX);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"TOL_RELAX_CELL:") == 0) {
            fscanf(input_fp,"%lf",&pSPARC_Input->TOL_RELAX_CELL);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"TOL_LANCZOS:") == 0) {
            fscanf(input_fp,"%lf",&pSPARC_Input->TOL_LANCZOS);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"TOL_PSEUDOCHARGE:") == 0) {
            fscanf(input_fp,"%lf",&pSPARC_Input->TOL_PSEUDOCHARGE);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"TOL_KERKER:") == 0) {
            printf("\n\"TOL_KERKER\" is obsolete, use \"TOL_PRECOND\" instead!\n");
            fscanf(input_fp,"%lf",&pSPARC_Input->TOL_PRECOND);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"TOL_PRECOND:") == 0) {
            fscanf(input_fp,"%lf",&pSPARC_Input->TOL_PRECOND);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"PRECOND_KERKER_KTF:") == 0) {
            fscanf(input_fp,"%lf",&pSPARC_Input->precond_kerker_kTF);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"PRECOND_KERKER_THRESH:") == 0) {
            fscanf(input_fp,"%lf",&pSPARC_Input->precond_kerker_thresh);
            fscanf(input_fp, "%*[^\n]\n");
        } /*else if (strcmpi(str,"PRECOND_RESTA_Q0:") == 0) {
            fscanf(input_fp,"%lf",&pSPARC_Input->precond_resta_q0);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"PRECOND_RESTA_RS:") == 0) {
            fscanf(input_fp,"%lf",&pSPARC_Input->precond_resta_Rs);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"PRECOND_FITPOW:") == 0) {
            fscanf(input_fp,"%d",&pSPARC_Input->precond_fitpow);
            fscanf(input_fp, "%*[^\n]\n");
        } */else if (strcmpi(str,"REFERENCE_CUTOFF:") == 0) {
            fscanf(input_fp,"%lf",&pSPARC_Input->REFERENCE_CUTOFF);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"MIXING_VARIABLE:") == 0) {
            fscanf(input_fp,"%s",temp); // read mixing variable
            if (strcmpi(temp,"density") == 0) {
                pSPARC_Input->MixingVariable = 0;
            } else if (strcmpi(temp,"potential") == 0) {
                pSPARC_Input->MixingVariable = 1;
            } else {
                printf("\nCannot recognize mixing variable: \"%s\"\n",temp);
                printf("Available options: \"density\", \"potential\" (case insensitive)\n");
                exit(EXIT_FAILURE);
            }
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"MIXING_PRECOND:") == 0) {
            fscanf(input_fp,"%s",temp); // read mixing preconditioner
            if (strcmpi(temp,"none") == 0) {
                pSPARC_Input->MixingPrecond = 0;
            } else if (strcmpi(temp,"kerker") == 0) {
                pSPARC_Input->MixingPrecond = 1;
            } else {
                printf("\nCannot recognize mixing preconditioner: \"%s\"\n",temp);
                printf("Available options: \"none\" and \"kerker\" (case insensitive) \n");
                exit(EXIT_FAILURE);
            }
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"MIXING_HISTORY:") == 0) {
            fscanf(input_fp,"%d",&pSPARC_Input->MixingHistory);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"MIXING_PARAMETER:") == 0) {
            fscanf(input_fp,"%lf",&pSPARC_Input->MixingParameter);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"MIXING_PARAMETER_SIMPLE:") == 0) {
            fscanf(input_fp,"%lf",&pSPARC_Input->MixingParameterSimple);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"PULAY_FREQUENCY:") == 0) {
            fscanf(input_fp,"%d",&pSPARC_Input->PulayFrequency);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"PULAY_RESTART:") == 0) {
            fscanf(input_fp,"%d",&pSPARC_Input->PulayRestartFlag);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"TWTIME:") == 0) {
            fscanf(input_fp,"%lf",&pSPARC_Input->TWtime);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"MD_FLAG:") == 0) {
            fscanf(input_fp,"%d",&pSPARC_Input->MDFlag);         
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"MD_METHOD:") == 0) {
            fscanf(input_fp,"%s",pSPARC_Input->MDMeth);
            // convert to upper case
            strupr(pSPARC_Input->MDMeth);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"MD_TIMESTEP:") == 0) {
            fscanf(input_fp,"%lf",&pSPARC_Input->MD_dt);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"MD_NSTEP:") == 0) {
            double MDnstep_temp;
            fscanf(input_fp,"%lf",&MDnstep_temp);
            pSPARC_Input->MD_Nstep = (int) MDnstep_temp;
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"ION_TEMP:") == 0) {
            Flag_ionT ++;
            fscanf(input_fp,"%lf",&pSPARC_Input->ion_T);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"ION_TEMP_END:") == 0) {
            Flag_ionT_end ++;
            fscanf(input_fp,"%lf",&pSPARC_Input->thermos_Tf);
            fscanf(input_fp, "%*[^\n]\n");
        } /*else if (strcmpi(str,"ION_ELEC_EQT:") == 0) {
            Flag_eqT ++;
            fscanf(input_fp,"%d",&pSPARC_Input->ion_elec_eqT);
            fscanf(input_fp, "%*[^\n]\n");
        } */else if (strcmpi(str,"ION_VEL_DSTR:") == 0) {
            fscanf(input_fp,"%d",&pSPARC_Input->ion_vel_dstr);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"ION_VEL_DSTR_RAND:") == 0) {
            fscanf(input_fp,"%d",&pSPARC_Input->ion_vel_dstr_rand);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"QMASS:") == 0) {
            fscanf(input_fp,"%lf",&pSPARC_Input->qmass);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"PRINT_MDOUT:") == 0) {
            fscanf(input_fp,"%d",&pSPARC_Input->PrintMDout);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"RELAX_FLAG:") == 0) {
            fscanf(input_fp,"%d",&pSPARC_Input->RelaxFlag);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"RELAX_METHOD:") == 0) {
            fscanf(input_fp,"%s",pSPARC_Input->RelaxMeth);  
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"NLCG_sigma:") == 0) {
            fscanf(input_fp,"%lf",&pSPARC_Input->NLCG_sigma);  
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"L_HISTORY:") == 0) {
            fscanf(input_fp,"%d",&pSPARC_Input->L_history);  
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"L_FINIT_STP:") == 0) {
            fscanf(input_fp,"%lf",&pSPARC_Input->L_finit_stp);  
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"L_MAXMOV:") == 0) {
            fscanf(input_fp,"%lf",&pSPARC_Input->L_maxmov);  
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"L_AUTOSCALE:") == 0) {
            fscanf(input_fp,"%d",&pSPARC_Input->L_autoscale);  
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"L_LINEOPT:") == 0) {
            fscanf(input_fp,"%d",&pSPARC_Input->L_lineopt);  
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"L_ICURV:") == 0) {
            fscanf(input_fp,"%lf",&pSPARC_Input->L_icurv);  
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"FIRE_dt:") == 0) {
            fscanf(input_fp,"%lf",&pSPARC_Input->FIRE_dt);  
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"FIRE_mass:") == 0) {
            fscanf(input_fp,"%lf",&pSPARC_Input->FIRE_mass);  
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"FIRE_maxmov:") == 0) {
            fscanf(input_fp,"%lf",&pSPARC_Input->FIRE_maxmov);  
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"PRINT_RELAXOUT:") == 0) {
            fscanf(input_fp,"%d",&pSPARC_Input->PrintRelaxout);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"RESTART_FLAG:") == 0) {
            fscanf(input_fp,"%d",&pSPARC_Input->RestartFlag);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"PRINT_RESTART:") == 0) {
            fscanf(input_fp,"%d",&pSPARC_Input->Printrestart);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"PRINT_RESTART_FQ:") == 0) {
            fscanf(input_fp,"%d",&pSPARC_Input->Printrestart_fq);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"EXCHANGE_CORRELATION:") == 0) {
            fscanf(input_fp,"%s",pSPARC_Input->XC);  
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"CALC_STRESS:") == 0) {
            fscanf(input_fp,"%d",&pSPARC_Input->Calc_stress);  
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"CALC_PRES:") == 0) {
            fscanf(input_fp,"%d",&pSPARC_Input->Calc_pres);  
            fscanf(input_fp, "%*[^\n]\n");
        } else if(strcmpi(str,"VERBOSITY:") == 0) {
            fscanf(input_fp,"%d",&pSPARC_Input->Verbosity);
            fscanf(input_fp, "%*[^\n]\n");
        } else if(strcmpi(str,"PRINT_FORCES:") == 0) {
            fscanf(input_fp,"%d",&pSPARC_Input->PrintForceFlag);
            fscanf(input_fp, "%*[^\n]\n");
        } else if(strcmpi(str,"PRINT_ATOMS:") == 0) {
            fscanf(input_fp,"%d",&pSPARC_Input->PrintAtomPosFlag);
            fscanf(input_fp, "%*[^\n]\n");
        } else if(strcmpi(str,"PRINT_VELS:") == 0) {
            fscanf(input_fp,"%d",&pSPARC_Input->PrintAtomVelFlag);
            fscanf(input_fp, "%*[^\n]\n");
        } else if(strcmpi(str,"PRINT_EIGEN:") == 0) {
            fscanf(input_fp,"%d",&pSPARC_Input->PrintEigenFlag);
            fscanf(input_fp, "%*[^\n]\n");
        } else if(strcmpi(str,"PRINT_DENSITY:") == 0) {
            fscanf(input_fp,"%d",&pSPARC_Input->PrintElecDensFlag);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"OUTPUT_FILE:") == 0) {    
            fscanf(input_fp,"%s",pSPARC_Input->filename_out);
            fscanf(input_fp, "%*[^\n]\n");
        } else {
            printf("\nCannot recognize input variable identifier: \"%s\"\n",str);
            exit(EXIT_FAILURE);
        }
    }

    // copy filename into pSPARC struct
    snprintf(pSPARC->filename, L_STRING, "%s", pSPARC_Input->filename);
    
    // Isolated cluster can be in orthogonal cell only
    double mult;
    int j;
    if(pSPARC_Input->BC == 1){
        for(i = 0; i < 2; i++){
            for(j = i+1; j < 3; j++){
                mult = fabs(pSPARC_Input->LatVec[3*i] * pSPARC_Input->LatVec[3*j] + pSPARC_Input->LatVec[3*i+1] * pSPARC_Input->LatVec[3*j+1] + pSPARC_Input->LatVec[3*i+2] * pSPARC_Input->LatVec[3*j+2]);
                if(mult > 1e-12){
                    printf("\nOnly orthogonal cells permitted for isolated clusters\n");
                    exit(EXIT_FAILURE); 
                }
            }
        }
    }

    // Pressure calculation only for crystal and stress calculation non existent for isolated clusters  
    int dim = 3;    
    if (pSPARC_Input->BCx >= 0) 
        dim = 3 - (pSPARC_Input->BCx + pSPARC_Input->BCy + pSPARC_Input->BCz);  

    if((dim == 0 || pSPARC_Input->BC == 1) && pSPARC_Input->Calc_stress == 1){  
        printf("\nStress does not exist in an isolated cluster!\n");    
        exit(EXIT_FAILURE); 
    }   


    if(( dim != 3 || (pSPARC_Input->BC > 0 && pSPARC_Input->BC != 2) ) && pSPARC_Input->Calc_pres == 1){    
        printf("\nPressure exist only in all periodic systems!\n"); 
        exit(EXIT_FAILURE); 
    }   

    if (Flag_tol_relaxcell == 0 && (dim != 3 || (pSPARC_Input->BC > 0 && pSPARC_Input->BC != 2))){  
        if (dim == 2 || pSPARC_Input->BC == 3)  
            pSPARC_Input->TOL_RELAX_CELL = 5e-6;    
        else if (dim == 1 || pSPARC_Input->BC == 4) 
            pSPARC_Input->TOL_RELAX_CELL = 5e-5;    
    }   

    // Restart option unavaiable for RelaxFlag = 2,3    
    if ((pSPARC_Input->RelaxFlag == 2 || pSPARC_Input->RelaxFlag == 3) && pSPARC_Input->RestartFlag == 1) { 
       printf("Restart option unavaiable for cell-relaxation and full relaxation currently\n"); 
       exit(EXIT_FAILURE);  
    }
    
    // check if MD and Relaxation are turned on simultaneously
    if (pSPARC_Input->MDFlag!=0 && pSPARC_Input->RelaxFlag!=0) {
        printf("\nStructural relaxations and MD cannot be turned on simultaneously!\n");
        exit(EXIT_FAILURE);
    }

    // check if both smearing and temperature are specified
    if (Flag_Temp > 1) {
        printf("\nElectronic smearing and electronic temperature cannot be simultaneously specified!\n");
        exit(EXIT_FAILURE);
    }

    // check if both smearing and temperature are specified
    if (Flag_accuracy > 1) {
        printf("\nTOL_SCF, ACCURACY, SCF_ENERGY_ACC, and SCF_FORCE_ACC cannot be simultaneously specified!\n");
        exit(EXIT_FAILURE);
    }

    // check if mesh size is defined using multiple ways
    int mesh_count = (Flag_fdgrid > 0) + (Flag_meshspacing > 0) + (Flag_ecut > 0);
    if (mesh_count > 1) {
        printf("\nFD_GRID, MESH_SPACING, ECUT cannot be simultaneously specified!\n");
        exit(EXIT_FAILURE);    
    } else if (mesh_count == 0) {
        printf("\nPlease specify mesh spacing by: FD_GRID, MESH_SPACING, or ECUT!\n");
        exit(EXIT_FAILURE);    
    }

    // Check whether electronic and ionic temp. are provided and choose the isEqual option accordingly
    if(pSPARC_Input->MDFlag == 1){
        if(Flag_ionT == 0){
            printf("\nIonic temperature must be specified for MD!\n");
            exit(EXIT_FAILURE);    
        }
        
        /*if (Flag_eqT == 0){
            if (Flag_elecT == 0 && Flag_ionT == 1){
                pSPARC_Input->elec_T = pSPARC_Input->ion_T;
                pSPARC_Input->ion_elec_eqT = 1; 
            } else if (Flag_elecT == 1 && Flag_ionT == 0){
                pSPARC_Input->ion_elec_eqT = 0;
            } else if (Flag_elecT == 1 && Flag_ionT == 1){
                pSPARC_Input->ion_elec_eqT = (pSPARC_Input->elec_T == pSPARC_Input->ion_T);
            }
        } else{
            if (Flag_elecT == 0 && Flag_ionT == 0){
                if(pSPARC_Input->ion_elec_eqT == 0){
                    printf("\nInvalid value of input ION_ELEC_EQT specified!\n");
                    exit(EXIT_FAILURE);
                }    
            } else if (Flag_elecT == 0 && Flag_ionT == 1){
                if(pSPARC_Input->ion_elec_eqT == 1)
                    pSPARC_Input->elec_T = pSPARC_Input->ion_T;
            } else if (Flag_elecT == 1 && Flag_ionT == 0){
                if(pSPARC_Input->ion_elec_eqT == 1)
                    pSPARC_Input->ion_T = pSPARC_Input->elec_T;
            } else if (Flag_elecT == 1 && Flag_ionT == 1){
                if(pSPARC_Input->ion_elec_eqT == 1 && pSPARC_Input->ion_T != pSPARC_Input->elec_T){
                    printf("\nInvalid value of Flag ion_elec_eqT specified!\n");
                    exit(EXIT_FAILURE);
                }
            }
        }*/
        
        /*
        if (Flag_eqT == 0){
            if (Flag_elecT == 0){
                pSPARC_Input->elec_T = pSPARC_Input->ion_T;
                pSPARC_Input->Beta = 1./(CONST_KB * pSPARC_Input->elec_T);
                pSPARC_Input->ion_elec_eqT = 1; 
            } else if (Flag_elecT == 1 && Flag_ionT == 1){
                pSPARC_Input->ion_elec_eqT = (pSPARC_Input->elec_T == pSPARC_Input->ion_T);
            }
        } else{
            if (Flag_elecT == 0){
                if(pSPARC_Input->ion_elec_eqT == 1){
                    pSPARC_Input->elec_T = pSPARC_Input->ion_T;
                    pSPARC_Input->Beta = 1./(CONST_KB * pSPARC_Input->elec_T);
                }    
            } else if (Flag_elecT == 1 && Flag_ionT == 1){
                if(pSPARC_Input->ion_elec_eqT == 1 && pSPARC_Input->ion_T != pSPARC_Input->elec_T){
                    printf("\nInvalid value of Flag ion_elec_eqT specified!\n");
                    exit(EXIT_FAILURE);
                }
            }
        }

        if(Flag_smear_typ == 0){
            if(pSPARC_Input->ion_elec_eqT == 1){
                if(pSPARC_Input->ion_T >= 1./(CONST_KB * (CONST_EH / 0.2)))
                    pSPARC_Input->elec_T_type = 0; //fermi-dirac
                else
                    pSPARC_Input->elec_T_type = 1; // gaussian
            }               
        }

        if (pSPARC_Input->Beta < 0) {
            if (pSPARC_Input->elec_T_type == 1) { // gaussian
                // The electronic temperature corresponding to 0.2 eV is 2320.904422 K
                pSPARC_Input->Beta = CONST_EH / 0.2; // smearing = 0.2 eV = 0.00734986450 Ha, Beta := 1 / smearing
            } else { // fermi-dirac
                // The electronic temperature corresponding to 0.1 eV is 1160.452211 K
                pSPARC_Input->Beta = CONST_EH / 0.1; // smearing = 0.1 eV = 0.00367493225 Ha, Beta := 1 / smearing
            }
            pSPARC_Input->elec_T = 1./(CONST_KB * pSPARC_Input->Beta); 
        }
        */

        if (Flag_elecT == 0){   
            pSPARC_Input->elec_T = pSPARC_Input->ion_T; 
            pSPARC_Input->Beta = 1./(CONST_KB * pSPARC_Input->elec_T);  
            pSPARC_Input->ion_elec_eqT = 0;     
        } else if (Flag_elecT == 1 && pSPARC_Input->elec_T > 0){    
            pSPARC_Input->ion_elec_eqT = 0; 
        } else if (Flag_elecT == 1 && pSPARC_Input->elec_T <= 0){   
            pSPARC_Input->elec_T = pSPARC_Input->ion_T; 
            pSPARC_Input->Beta = 1./(CONST_KB * pSPARC_Input->elec_T);  
            pSPARC_Input->ion_elec_eqT = 1; 
        }   

        if(Flag_smear_typ == 0){    
            pSPARC_Input->elec_T_type = 0; //fermi-dirac                
        }           

        // Start and end temperatures of thermostat
        if(strcmpi(pSPARC_Input->MDMeth,"NVT_NH") == 0){
            if(Flag_ionT_end == 0)
                pSPARC_Input->thermos_Tf = pSPARC_Input->ion_T;
        }
    }
    
    // Set default for KPOINT_SHIFT
    if(Flag_kptshift == 0){
        if((pSPARC_Input->Kx%2) == 0){
            pSPARC_Input->kptshift[0] = 0.5;
        }
        if((pSPARC_Input->Ky%2) == 0){
            pSPARC_Input->kptshift[1] = 0.5;
        }
        if((pSPARC_Input->Kz%2) == 0){
            pSPARC_Input->kptshift[2] = 0.5;
        }
    }
    free(input_filename);
    free(str);
    free(temp);
    fclose(input_fp);
} 



/**
 * @brief   Read ion file.
 */
 //TODO: add n cell replica
void read_ion(SPARC_INPUT_OBJ *pSPARC_Input, SPARC_OBJ *pSPARC) {
    char *ion_filename = malloc(L_STRING * sizeof(char));
    char *str          = malloc(L_STRING * sizeof(char));
    int i, ityp, typcnt, atmcnt_coord, atmcnt_relax, atmcnt_spin, *atmcnt_cum, n_atom;
    snprintf(ion_filename, L_STRING, "%s.ion", pSPARC->filename);
    FILE *ion_fp = fopen(ion_filename,"r");
    
    if (ion_fp == NULL) {
        printf("\nCannot open file \"%s\"\n",ion_filename);
        print_usage();
        exit(EXIT_FAILURE);
    }
#ifdef DEBUG
    printf("Reading ion file %s\n",ion_filename);
#endif
    
    /* first identify total number of atom types */
    typcnt = 0;
    while (!feof(ion_fp)) {
        fscanf(ion_fp,"%s",str);
        if (strcmpi(str, "ATOM_TYPE:") == 0) {
            typcnt++;
            fscanf(ion_fp, "%*[^\n]\n"); 
        } else {
            // skip current line
            fscanf(ion_fp, "%*[^\n]\n"); 
        }
    }
    
#ifdef DEBUG
    printf("Number of atom types : %d.\n",typcnt);    
#endif
    
    if (typcnt < 1) {
        printf("\nPlease provide at least one type of atoms!\n");
        exit(EXIT_FAILURE);
    }
    pSPARC->Ntypes = typcnt;
    
    // reset file pointer to the start of the file
    fseek(ion_fp, 0L, SEEK_SET);  // returns 0 if succeeded, can check status
    
    // allocate memory for nAtomv and atmcnt_cum
    atmcnt_cum = (int *)malloc( (pSPARC->Ntypes+1) * sizeof(int));
    if (atmcnt_cum == NULL) {
        printf("\nCannot allocate memory for \"atmcnt_cum\"!\n");
        exit(EXIT_FAILURE);
    }
    
    // allocate memory 
    pSPARC->localPsd = (int *)malloc( pSPARC->Ntypes * sizeof(int) );
    pSPARC->Mass = (double *)malloc( pSPARC->Ntypes * sizeof(double) );
    pSPARC->atomType = (char *)calloc( pSPARC->Ntypes * L_ATMTYPE, sizeof(char) ); 
    pSPARC->Znucl = (int *)malloc( pSPARC->Ntypes * sizeof(int) );
    pSPARC->nAtomv = (int *)malloc( pSPARC->Ntypes * sizeof(int) );
    pSPARC->psdName = (char *)calloc( pSPARC->Ntypes * L_PSD, sizeof(char) );
    if (pSPARC->localPsd == NULL || pSPARC->Mass == NULL || 
        pSPARC->atomType == NULL || pSPARC->psdName == NULL ||
        pSPARC->Znucl == NULL || pSPARC->nAtomv == NULL) {
        printf("\nmemory cannot be allocated\n");
        exit(EXIT_FAILURE);
    }
     
    // set default local components of pseudopotentials
    for (i = 0; i < pSPARC->Ntypes; i++) {
        pSPARC->localPsd[i] = 4; // default is 4
    }

    /* find total number of atoms */
    n_atom = 0;    // totoal num of atoms
    typcnt = -1;    // atom type count    
    atmcnt_cum[0] = 0;
    while (!feof(ion_fp)) {
        fscanf(ion_fp,"%s",str);
        if (strcmpi(str, "ATOM_TYPE:") == 0) {
            typcnt++;
            fscanf(ion_fp, "%s", &pSPARC->atomType[L_ATMTYPE*typcnt]);
        } else if (strcmpi(str, "N_TYPE_ATOM:") == 0) {
            fscanf(ion_fp, "%d", &pSPARC->nAtomv[typcnt]);
            fscanf(ion_fp, "%*[^\n]\n"); 
            n_atom += pSPARC->nAtomv[typcnt];
            atmcnt_cum[typcnt+1] = n_atom;
        } else {
            // skip current line
            fscanf(ion_fp, "%*[^\n]\n"); 
            continue;
        }
    }

#ifdef DEBUG
    printf("Total number of atoms: %d.\n",n_atom);
#endif

    if (n_atom < 1) {
        printf("\nPlease provide at least one atom!\n");
        exit(EXIT_FAILURE);
    }    

    pSPARC->n_atom = n_atom;
    
    // allocate memory for atom positions, atom relax constraints and atom spin
    pSPARC->atom_pos = (double *)malloc(3*n_atom*sizeof(double));
    pSPARC->mvAtmConstraint = (int *)malloc(3*n_atom*sizeof(int));
    pSPARC->atom_spin = (double *)calloc(n_atom, sizeof(double));
    if (pSPARC->atom_pos == NULL || pSPARC->mvAtmConstraint == NULL || pSPARC->atom_spin == NULL) {
        printf("\nCannot allocate memory for atom positions, atom relax constraints and atom spin!\n");
        exit(EXIT_FAILURE);
    }
    
    // set default atom relax constraints to be all on (move in all DOFs)
    for (i = 0; i < 3*n_atom; i++) {
        pSPARC->mvAtmConstraint[i] = 1;
    }
    
    double t1, t2;
    // set default atomic masses based on atom types (for MD)
    if (1) { // atomic mass is only needed for MD
        char elemType[8];
        for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
            t1 = MPI_Wtime();
            // first identify element type
            find_element(elemType, &pSPARC->atomType[L_ATMTYPE*ityp]);
            t2 = MPI_Wtime();
#ifdef DEBUG
            printf(GRN"\nTime for finding element is %.3f ms\n",(t2-t1)*1000);
            printf(GRN"Element type for atom_type %s is %s\n"RESET, &pSPARC->atomType[L_ATMTYPE*ityp], elemType);
#endif            
            // find default atomic mass
            atomdata_mass(elemType, &pSPARC->Mass[ityp]);
#ifdef DEBUG
            printf(GRN"Default atomic mass for %s is %f\n"RESET,elemType,pSPARC->Mass[ityp]);
#endif
        }
    }
    
    // reset file pointer to the start of the file
    fseek(ion_fp, 0L, SEEK_SET);  // returns 0 if succeeded, can check status

    // reset temp var
    typcnt = -1;
    atmcnt_coord = 0;
    atmcnt_relax = 0;
    atmcnt_spin = 0;
    
    // allocate the size of the Isfrac vector which stores the coordinate type of each atom type
    pSPARC->IsFrac = (int *)calloc( pSPARC->Ntypes, sizeof(int) );
    
    // allocate the size of the Isspin vector which stores the spin value of each atom type
    pSPARC->IsSpin = (int *)calloc( pSPARC->Ntypes, sizeof(int) );
    
    while (fscanf(ion_fp,"%s",str) != EOF) {
        // enable commenting with '#'
        if (str[0] == '#' || str[0] == '\n') {
            fscanf(ion_fp, "%*[^\n]\n"); // skip current line
            continue;
        }

        if (strcmpi(str, "ATOM_TYPE:") == 0) {
            typcnt++; 
            fscanf(ion_fp, "%*[^\n]\n"); // skip current line
        } else if (strcmpi(str, "N_TYPE_ATOM:") == 0) {
            fscanf(ion_fp, "%*[^\n]\n"); // skip current line
        } else if (strcmpi(str, "COORD:") == 0) {
            fscanf(ion_fp, "%*[^\n]\n"); // skip current line
            for (i = 0; i < pSPARC->nAtomv[typcnt]; i++) {
                fscanf(ion_fp,"%lf", pSPARC->atom_pos+3*atmcnt_coord);
                fscanf(ion_fp,"%lf", pSPARC->atom_pos+3*atmcnt_coord+1);
                fscanf(ion_fp,"%lf", pSPARC->atom_pos+3*atmcnt_coord+2);
                fscanf(ion_fp, "%*[^\n]\n"); // skip current line
                atmcnt_coord++;  
            }
        } else if (strcmpi(str, "COORD_FRAC:") == 0) {
            fscanf(ion_fp, "%*[^\n]\n"); // skip current line
            for (i = 0; i < pSPARC->nAtomv[typcnt]; i++) {
                fscanf(ion_fp,"%lf", pSPARC->atom_pos+3*atmcnt_coord);
                pSPARC->atom_pos[3*atmcnt_coord] *= pSPARC_Input->range_x;
                fscanf(ion_fp,"%lf", pSPARC->atom_pos+3*atmcnt_coord+1);
                pSPARC->atom_pos[3*atmcnt_coord+1] *= pSPARC_Input->range_y;
                fscanf(ion_fp,"%lf", pSPARC->atom_pos+3*atmcnt_coord+2);
                pSPARC->atom_pos[3*atmcnt_coord+2] *= pSPARC_Input->range_z;
                fscanf(ion_fp, "%*[^\n]\n"); // skip current line
                atmcnt_coord++;  
            }
            pSPARC->IsFrac[typcnt] = 1;
        } else if (strcmpi(str, "RELAX:") == 0) {
            fscanf(ion_fp, "%*[^\n]\n"); // skip current line
            atmcnt_relax = atmcnt_cum[typcnt];
            for (i = 0; i < pSPARC->nAtomv[typcnt]; i++) {
                fscanf(ion_fp,"%d", pSPARC->mvAtmConstraint+3*atmcnt_relax);
                fscanf(ion_fp,"%d", pSPARC->mvAtmConstraint+3*atmcnt_relax+1);
                fscanf(ion_fp,"%d", pSPARC->mvAtmConstraint+3*atmcnt_relax+2);
                fscanf(ion_fp, "%*[^\n]\n"); // skip current line
                atmcnt_relax++;
            }
        } else if (strcmpi(str, "SPIN:") == 0) {
            fscanf(ion_fp, "%*[^\n]\n"); // skip current line
            atmcnt_spin = atmcnt_cum[typcnt];
            for (i = 0; i < pSPARC->nAtomv[typcnt]; i++) {
                fscanf(ion_fp,"%lf", pSPARC->atom_spin + atmcnt_spin);
                fscanf(ion_fp, "%*[^\n]\n"); // skip current line
                atmcnt_spin++;
            }
            pSPARC->IsSpin[typcnt] = 1;
        } else if (strcmpi(str, "PSEUDO_POT:") == 0) {
            #define STR_(X) #X
            #define STR(X) STR_(X)
            #define MAX_PATH_LEN 10240
            char *str_tmp = malloc(MAX_PATH_LEN * sizeof(char));
            fscanf(ion_fp,"%" STR(MAX_PATH_LEN) "s",str_tmp); // read at most MAX_PATH_LEN chars
            if (strlen(str_tmp) > L_PSD) {
                printf("\n[FATAL] PSEUDO_POT: path length (%ld) exceeds maximum length (%d)\n", strlen(str_tmp), L_PSD);
                printf("Please provide a shorter path to your pseudopotential\n");
                exit(EXIT_FAILURE);
            }
            //simplifyPath(str_tmp, &pSPARC->psdName[typcnt*L_PSD], L_PSD);
            snprintf(&pSPARC->psdName[typcnt*L_PSD], L_PSD, "%s", str_tmp);
            free(str_tmp);
            #undef STR
            #undef STR_
            #undef MAX_PATH_LEN
            fscanf(ion_fp, "%*[^\n]\n"); // skip current line 
            pSPARC->is_default_psd = 0; // switch off default psedopots
#ifdef DEBUG
            printf("pseudo_dir # %d = %s\n",typcnt+1,&pSPARC->psdName[typcnt*L_PSD]);
#endif
        } else if (strcmpi(str, "ATOMIC_MASS:") == 0) {
            fscanf(ion_fp, "%lf", &pSPARC->Mass[typcnt]);  
            fscanf(ion_fp, "%*[^\n]\n"); // skip current line
        } else if ( isdigit(str[0]) ) {
            printf("\nPlease specify the identifier before numbers!\n"
                   "Reminder: check if the number of atoms specified is inconsistent\n"
                   "          with the number of coordinates provided\n"); 
            exit(EXIT_FAILURE);
        } else {
            printf("\nCannot recognize input variable identifier: \"%s\"",str);
            exit(EXIT_FAILURE);
        }
    }
    
    if (atmcnt_coord != n_atom) {
        printf("the number of coordinates provided is inconsistent "
               "with the given number of atoms!\n");
        exit(EXIT_FAILURE);
    }

    free(atmcnt_cum);
    free(ion_filename);
    free(str);
    fclose(ion_fp);
}



/**
 * @brief   Read pseudopotential files (psp format).
 */
void read_pseudopotential_PSP(SPARC_INPUT_OBJ *pSPARC_Input, SPARC_OBJ *pSPARC)
{
#ifdef DEBUG
    printf("Reading pseudopotential (PSP) file.\n");
#endif
    int jj, kk, l, ityp, *lpos, lmax, nproj;
    char *str          = malloc(L_PSD * sizeof(char));
    char *INPUT_DIR    = malloc(L_PSD * sizeof(char));
    char *psd_filename = malloc(L_PSD * sizeof(char));
    char *simp_path    = malloc(L_PSD * sizeof(char));
    double vtemp, vtemp2;

    FILE *psd_fp;

    // allocate memory
    pSPARC->psd = (PSD_OBJ *)malloc(pSPARC->Ntypes * sizeof(PSD_OBJ));
    assert(pSPARC->psd != NULL);

    //char *inpt_path = pSPARC->filename;
    char *inpt_path = pSPARC_Input->filename;

    // extract INPUT_DIR from filename
    char *pch = strrchr(inpt_path,'/'); // find last occurrence of '/'
    if (pch == NULL) { // in case '/' is not found
        snprintf(INPUT_DIR, L_PSD, "%s", ".");
    } else {
        memcpy(INPUT_DIR, inpt_path, pch-inpt_path);
        INPUT_DIR[(int)(pch-inpt_path)] = '\0';
    }

    // loop over all atom types
    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        if (pSPARC->is_default_psd) { 
            // use default pseudopotential files
            snprintf(psd_filename, L_PSD, "%s/psdpots/%s.psp8", pSPARC_Input->SPARCROOT, 
                                                    &pSPARC->atomType[ityp*L_ATMTYPE]);
            snprintf(&pSPARC->psdName[ityp*L_PSD], L_PSD, "%s", psd_filename);
        } else {
            if (pSPARC->psdName[ityp*L_PSD] == '/') { // absolute path 
                snprintf(psd_filename, L_PSD, "%s", &pSPARC->psdName[ityp*L_PSD]);
            } else { // relative path to where the .ion file is located
                // use user-provided pseudopotential files
                snprintf(psd_filename, L_PSD, "%s/%s", INPUT_DIR, &pSPARC->psdName[ityp*L_PSD]);
            }
            snprintf(&pSPARC->psdName[ityp*L_PSD], L_PSD, "%s", psd_filename);
        }

        // simplify input path (this will be printed to .out file)
        simplifyPath(&pSPARC->psdName[ityp*L_PSD], simp_path, L_PSD);
        snprintf(&pSPARC->psdName[ityp*L_PSD], L_PSD, "%s", simp_path);

        // simplify contatenated path
        simplifyPath(psd_filename, simp_path, L_PSD);
        snprintf(psd_filename, L_PSD, "%s", simp_path);

#ifdef DEBUG
        printf("Reading pseudopotential: %s\n", psd_filename);
#endif
        psd_fp = fopen(psd_filename,"r"); 
        if (psd_fp == NULL) {
            printf("\nCannot open file \"%s\"\n",psd_filename);
            exit(EXIT_FAILURE);
        }

        /* first check the pseudopotential file and find size of pseudopotential data */
        // check atom type
        fscanf(psd_fp, "%s", str);
        fscanf(psd_fp, "%*[^\n]\n"); // skip current line
        // first identify element type
        char elemType[8];
        find_element(elemType, &pSPARC->atomType[L_ATMTYPE*ityp]);
#ifdef DEBUG
        printf("Input element type: %s\n", elemType);
#endif
        if (strcmpi(str,elemType) != 0) {
            printf("\nError: Pseudopotential file %s does not match with input atom type %s!\n",
            psd_filename, &pSPARC->atomType[ityp*L_ATMTYPE]);
            exit(EXIT_FAILURE);
        }
        
        // read valence charge zion
        fscanf(psd_fp,"%lf %lf",&vtemp,&vtemp2);
        pSPARC->Znucl[ityp] = (int)vtemp2;
        fscanf(psd_fp, "%*[^\n]\n"); // skip current line
        
        int pspcod, pspxc;
        fscanf(psd_fp,"%d",&pspcod); // pspcod
        fscanf(psd_fp,"%d %d %d %d",&pspxc,&pSPARC->psd[ityp].lmax,&pSPARC->localPsd[ityp],&pSPARC->psd[ityp].size); 
        pSPARC->psd[ityp].pspxc = pspxc;
#ifdef DEBUG
        printf("pspcod = %d, pspxc = %d\n", pspcod, pspxc);
#endif
        double rchrg, fchrg, qchrg;
        fscanf(psd_fp, "%*[^\n]\n"); // skip current line
        fscanf(psd_fp,"%lf %lf %lf",&rchrg,&fchrg,&qchrg);
        
        pSPARC->psd[ityp].fchrg = fchrg; // save for nonlinear core correction
        if (fabs(fchrg) > TEMP_TOL) {
            printf("\nfchrg = %.8f > 0.0 (icmod != 0)\n", fchrg);
            printf("This pseudopotential contains non-linear core correction. \n");
        }

        lmax = pSPARC->psd[ityp].lmax;
        pSPARC->psd[ityp].ppl = (int *)calloc((lmax+1), sizeof(int));
        lpos = (int *)calloc((lmax+2), sizeof(int)); // the last value stores the total number of projectors for this l
        pSPARC->psd[ityp].rc = (double *)malloc((lmax+1) * sizeof(double));        
        
        // reset file pointer to the start of the file
        fseek(psd_fp, 0L, SEEK_SET);  // returns 0 if succeeded, can use to check status
        
        // read rc
        do {
            fscanf(psd_fp,"%s",str);
        } while (strcmpi(str,"r_core="));

        for (l = 0; l <= pSPARC->psd[ityp].lmax; l++) {
            fscanf(psd_fp,"%lf",&vtemp);
            pSPARC->psd[ityp].rc[l] = vtemp;
        }
        
        // read number of projectors 
        do {
            fscanf(psd_fp,"%s",str);
        } while (strcmpi(str,"qchrg"));

        lpos[0] = 0;
        for (l = 0; l <= lmax; l++) {
            fscanf(psd_fp,"%d",&pSPARC->psd[ityp].ppl[l]);
            lpos[l+1] = lpos[l] + pSPARC->psd[ityp].ppl[l];
        }
        nproj = lpos[lmax+1]; 
        
        do {
            fscanf(psd_fp,"%s",str);
        } while (strcmpi(str,"extension_switch"));
        
        // allocate memory
        pSPARC->psd[ityp].RadialGrid = (double *)calloc(pSPARC->psd[ityp].size , sizeof(double)); 
        pSPARC->psd[ityp].UdV = (double *)calloc(nproj * pSPARC->psd[ityp].size , sizeof(double)); 
        pSPARC->psd[ityp].rVloc = (double *)calloc(pSPARC->psd[ityp].size , sizeof(double)); 
        pSPARC->psd[ityp].rhoIsoAtom = (double *)calloc(pSPARC->psd[ityp].size , sizeof(double)); 
        pSPARC->psd[ityp].Gamma = (double *)calloc(nproj , sizeof(double)); 
        
        // start reading projectors
        int l_read = 0;
        fscanf(psd_fp,"%d",&l_read); 
        for (l = 0; l <= pSPARC->psd[ityp].lmax; l++) {
            if (l != pSPARC->localPsd[ityp]) {
                for (kk = 0; kk < pSPARC->psd[ityp].ppl[l]; kk++) {
                    fscanf(psd_fp,"%lf", &vtemp);
                    pSPARC->psd[ityp].Gamma[lpos[l]+kk] = vtemp;
                }
                for (jj = 0; jj < pSPARC->psd[ityp].size; jj++) {
                    fscanf(psd_fp,"%lf", &vtemp);
                    fscanf(psd_fp,"%lf", &vtemp);
                    pSPARC->psd[ityp].RadialGrid[jj] = vtemp;
                    for (kk = 0; kk < pSPARC->psd[ityp].ppl[l]; kk++) {
                        fscanf(psd_fp,"%lf",&vtemp);
                        pSPARC->psd[ityp].UdV[(lpos[l]+kk)*pSPARC->psd[ityp].size+jj] = vtemp/pSPARC->psd[ityp].RadialGrid[jj];
                    }
                }
                for (kk = 0; kk < pSPARC->psd[ityp].ppl[l]; kk++)
                    pSPARC->psd[ityp].UdV[(lpos[l]+kk)*pSPARC->psd[ityp].size] = pSPARC->psd[ityp].UdV[(lpos[l]+kk)*pSPARC->psd[ityp].size+1];
            } else {
                // first read Vloc(r = 0)
                fscanf(psd_fp, "%*[^\n]\n"); // skip current line
                fscanf(psd_fp, "%lf", &vtemp); // skip index
                fscanf(psd_fp,"%lf %lf", &vtemp, &vtemp2);
                pSPARC->psd[ityp].RadialGrid[0] = vtemp;
                pSPARC->psd[ityp].rVloc[0] = vtemp * vtemp2;
                pSPARC->psd[ityp].Vloc_0 = vtemp2;
                // read r and Vloc, store rVloc
                for (jj = 1; jj < pSPARC->psd[ityp].size; jj++) {
                    fscanf(psd_fp, "%lf", &vtemp);
                    fscanf(psd_fp,"%lf %lf", &vtemp, &vtemp2);
                    pSPARC->psd[ityp].RadialGrid[jj] = vtemp;
                    pSPARC->psd[ityp].rVloc[jj] = vtemp * vtemp2;
                }
            }
            fscanf(psd_fp,"%d",&l_read); 
        }
        
        // read Vloc if lloc > lmax
        //if (pSPARC->localPsd[ityp] > pSPARC->psd[ityp].lmax || l > pSPARC->psd[ityp].lmax) {
        if (pSPARC->localPsd[ityp] > pSPARC->psd[ityp].lmax || l_read == 4) {
            // first read Vloc(r = 0)
            fscanf(psd_fp, "%lf", &vtemp);
            fscanf(psd_fp,"%lf %lf", &vtemp, &vtemp2);
            pSPARC->psd[ityp].rVloc[0] = vtemp * vtemp2;
            pSPARC->psd[ityp].Vloc_0 = vtemp2;
            
            // read r and Vloc, store rVloc
            for (jj = 1; jj < pSPARC->psd[ityp].size; jj++) {
                fscanf(psd_fp, "%lf", &vtemp);
                fscanf(psd_fp,"%lf %lf", &vtemp, &vtemp2);
                pSPARC->psd[ityp].rVloc[jj] = vtemp * vtemp2;
            }
        } else {
            fseek (psd_fp, -1*4, SEEK_CUR ); // move back 4 columns
        }  

        // read model core charge for NLCC
        pSPARC->psd[ityp].rho_c_table = (double *)calloc(pSPARC->psd[ityp].size, sizeof(double));
        if (fchrg > TEMP_TOL) {
            printf("\nfchrg = %f, READING MODEL CORE CHARGE!\n\n", fchrg);
            for (jj = 0; jj < pSPARC->psd[ityp].size;jj++) {
                fscanf(psd_fp,"%lf %lf", &vtemp, &vtemp2);              
                fscanf(psd_fp,"%lf",&vtemp);
                pSPARC->psd[ityp].rho_c_table[jj] = vtemp / (4.0 * M_PI);
                fscanf(psd_fp, "%*[^\n]\n"); // skip current line
            }
        }

        // read isolated atom electron density (the 3rd number of each line)
        for (jj = 0; jj < pSPARC->psd[ityp].size;jj++) {
            fscanf(psd_fp,"%lf %lf", &vtemp, &vtemp2);
            fscanf(psd_fp,"%lf",&vtemp);
            pSPARC->psd[ityp].rhoIsoAtom[jj] = vtemp / (4.0 * M_PI);
            fscanf(psd_fp,"%lf %lf", &vtemp, &vtemp2);
        }
        
        //// read rc from <INPUT> parameters
        //do {
        //  fscanf(psd_fp,"%s",str);
        //} while (strcmpi(str,"qcut"));

        //for (l = 0; l <= pSPARC->psd[ityp].lmax; l++) {
        //    fscanf(psd_fp,"%lf %lf",&vtemp2,&vtemp);
        //    pSPARC->psd[ityp].rc[l] = vtemp;
        //    fscanf(psd_fp, "%*[^\n]\n"); // go to next line
        //}
        
        /* check pseudopotential data */
        // check radial grid size
        if (pSPARC->psd[ityp].size < 2) {
            printf("Radial grid too small: mmax = %d\n", pSPARC->psd[ityp].size);
            exit(EXIT_FAILURE);
        }

        // check if radial grid mesh is uniform
        double dr0 = pSPARC->psd[ityp].RadialGrid[1] - pSPARC->psd[ityp].RadialGrid[0];
        double dr_j;
        int is_r_uniform = 1;
        for (jj = 1; jj < pSPARC->psd[ityp].size; jj++) {
            dr_j = pSPARC->psd[ityp].RadialGrid[jj] - pSPARC->psd[ityp].RadialGrid[jj-1];
            // check if dr is 0
            if (fabs(dr_j) < TEMP_TOL) {
                printf("\nError: repeated radial grid values in pseudopotential (%s)!\n\n", psd_filename);
                exit(EXIT_FAILURE);
            }
            // check if mesh is uniform
            if (fabs(dr_j - dr0) > TEMP_TOL) {
                is_r_uniform = 0;
                #ifdef DEBUG
                printf("r[%d] = %.6E\n",jj-1, pSPARC->psd[ityp].RadialGrid[jj-1]);
                printf("r[%d] = %.6E\n",jj, pSPARC->psd[ityp].RadialGrid[jj]);
                printf("r(%d)-r(%d) = %.6E, r(2)-r(1) = %.6E\n", jj+1, jj, dr_j, dr0);
                #endif
                break;
            }
        }

        pSPARC->psd[ityp].is_r_uniform = is_r_uniform;

        if (pspcod == 8 && is_r_uniform == 0) {
            printf("\nWARNING: radial grid is non-uniform for psp8 format! (%s)\n\n", psd_filename);
        }

        // close the file
        fclose(psd_fp);
        free(lpos);
    }
    free(str);
    free(INPUT_DIR);
    free(psd_filename);
    free(simp_path);
}
