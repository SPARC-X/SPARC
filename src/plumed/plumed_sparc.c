/**
 * @file    plumed_sparc.c
 * @brief   PLUMED C API hooks for SPARC MD (rank 0 holds plumed object in pSPARC->PlumedHandle).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "isddft.h"
#include "plumed_sparc.h"

#include <mpi.h>

#ifdef USE_PLUMED
#include <Plumed.h>

// PLUMED setMD*Units: scale quantities from SPARC into PLUMED (kJ/mol, nm, ps, |e|, amu).
// Matches isddft.h and SPARC after Initialize_MD (md.c).
// 1 eV per mole -> kJ/mol; with CONST_EH gives Hartree -> kJ/mol
#ifndef CONST_EV_TO_KJMOL
#define CONST_EV_TO_KJMOL 96.4853321233100184
#endif
#ifndef HARTREE_TO_KJMOL
#define HARTREE_TO_KJMOL (CONST_EH * CONST_EV_TO_KJMOL)
#endif
// CONST_BOHR is Angstrom/Bohr; Bohr -> nm for PLUMED
#ifndef SPARC_BOHR_TO_NM
#define SPARC_BOHR_TO_NM (CONST_BOHR * 0.1)
#endif
// CONST_FS2ATU: multiply fs by this to get atu -> 1 atu = 1/CONST_FS2ATU fs = (1e-3/CONST_FS2ATU) ps
#ifndef ATU_TO_PS
#define ATU_TO_PS (1.0e-3 / CONST_FS2ATU)
#endif
#endif

void Plumed_Init(SPARC_OBJ *pSPARC)
{
    if (pSPARC == NULL)
        return;
#ifdef USE_PLUMED
    if (!pSPARC->PlumedFlag)
        return;

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0)
        return;

    if (pSPARC->PlumedHandle != NULL) {
        fprintf(stderr, "[PLUMED] Plumed_Init: PlumedHandle already set, skipping.\n");
        return;
    }

    if (!plumed_installed()) {
        fprintf(stderr, "[PLUMED] plumed_installed() == 0. Check PLUMED install and linker path (e.g. -lplumedKernel).\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    plumed *ph = (plumed *)malloc(sizeof(plumed));
    if (ph == NULL) {
        fprintf(stderr, "[PLUMED] Plumed_Init: malloc(sizeof(plumed)) failed.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Create the plumed object
    *ph = plumed_create();

    {
        int real_precision;
        double energy_units;
        double length_units;
        double time_units;
        double charge_units;
        double mass_units;
        int natoms;
        double dt_atu;
        double kbt_kjmol;
        int restart_flag;
        MPI_Comm comm_plumed;

        // sizeof(double) in bytes (8); PLUMED accepts 4 or 8
        real_precision = (int)sizeof(double);
        energy_units = HARTREE_TO_KJMOL;
        length_units = SPARC_BOHR_TO_NM;
        // setTimestep below is in atu (after Initialize_MD); PLUMED converts with Atoms::getTimeStep: * MDUnits/time
        time_units = ATU_TO_PS;
        charge_units = 1.0;
        mass_units = 1.0 / CONST_AMU2AU;
        natoms = pSPARC->n_atom;
        dt_atu = pSPARC->MD_dt;
        kbt_kjmol = (CONST_KB * pSPARC->ion_T) * HARTREE_TO_KJMOL;
        restart_flag = (pSPARC->RestartFlag != 0) ? 1 : 0;
        // Only rank 0 calls PLUMED; MPI_COMM_WORLD would deadlock. COMM_SELF is local to rank 0.
        comm_plumed = MPI_COMM_SELF;

        // Pass a pointer to an integer: size of a real in bytes (4 or 8)
        plumed_cmd(*ph, "setRealPrecision", &real_precision);
        // Pass a pointer to the conversion factor between Hartree (SPARC) and kJ mol-1 (PLUMED)
        plumed_cmd(*ph, "setMDEnergyUnits", &energy_units); // 1 Ha -> 2625.499639479 kJ/mol
        // Pass a pointer to the conversion factor between Bohr (e.g. atom_pos) and nm
        plumed_cmd(*ph, "setMDLengthUnits", &length_units); // 1 Bohr -> 0.052917721067 nm
        // Pass a pointer to the conversion factor between atu and ps
        plumed_cmd(*ph, "setMDTimeUnits", &time_units);     // 1 atu -> 2.418884326509e-05 ps
        // Pass a pointer to the conversion factor between |e| (valence Z) and e (API > 3)
        plumed_cmd(*ph, "setMDChargeUnits", &charge_units); // charge_units = 1.0 (same)
        // Pass a pointer to the conversion factor between electron-mass atomic units (Mass[] after Initialize_MD) and amu (API > 3)
        plumed_cmd(*ph, "setMDMassUnits", &mass_units);     // 1 au = 0.00054857991 amu

        // Pass the plumed input file path from SPARC
        plumed_cmd(*ph, "setPlumedDat", pSPARC->PlumedFile);
        // Pass PLUMED log path (set in SPARC_copy_input like MDFilename: filename_out.plumed or ...plumed_NN)
        plumed_cmd(*ph, "setLogFile", pSPARC->PlumedLogFilename);     // PLUMED opens/creates
        // Pass the MPI communicator (SPARC: MPI_COMM_SELF on rank 0, not MPI_COMM_WORLD)
        plumed_cmd(*ph, "setMPIComm", &comm_plumed);
        // Pass a pointer to the number of atoms
        plumed_cmd(*ph, "setNatoms", &natoms);
        // MD engine label for PLUMED (SPARC MD_METHOD: e.g. NVT_NH, NPT_NH)
        plumed_cmd(*ph, "setMDEngine", pSPARC->MDMeth);
        // MD timestep in atu (same as MD_dt after Initialize_MD in md.c)
        plumed_cmd(*ph, "setTimestep", &dt_atu);
        // Pass k_B*T in kJ mol-1 (API > 1); CONST_KB*ion_T is Ha, then Hartree -> kJ/mol
        plumed_cmd(*ph, "setKbT", &kbt_kjmol);
        // Pass restart flag: 1 yes, 0 no (API > 2)
        plumed_cmd(*ph, "setRestart", &restart_flag);
        // Virial each step: same formula as socket/driver.c stress_to_virial (3D PBC block).
        // All setup above must precede init; read runs inside init
        plumed_cmd(*ph, "init", NULL);
    }

    pSPARC->PlumedHandle = (void *)ph;
#else
    (void)pSPARC;
#endif
}

void Plumed_Evaluate(SPARC_OBJ *pSPARC)
{
    if (pSPARC == NULL)
        return;
#ifdef USE_PLUMED
    if (!pSPARC->PlumedFlag)
        return;

    int rank = 0;
    int apply_virial_stress = 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        if (pSPARC->PlumedHandle == NULL) {
            fprintf(stderr, "[PLUMED] Plumed_Evaluate: PlumedHandle is NULL on rank 0.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        int n = pSPARC->n_atom;
        plumed *ph = (plumed *)pSPARC->PlumedHandle;
        double *mass;
        double *charge;
        double box[3][3];
        double poteng;
        int step;
        int ityp, atm, count;
        double cell[3];

        if (n > 0 && pSPARC->atom_pos != NULL && pSPARC->forces != NULL) {

            mass = (double *)malloc((size_t)n * sizeof(double));
            charge = (double *)malloc((size_t)n * sizeof(double));
            if (mass == NULL || charge == NULL) {
                free(mass);
                free(charge);
                fprintf(stderr, "[PLUMED] Plumed_Evaluate: malloc failed.\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            count = 0;
            for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
                for (atm = 0; atm < pSPARC->nAtomv[ityp]; atm++) {
                    mass[count] = pSPARC->Mass[ityp];
                    charge[count] = (double)pSPARC->Znucl[ityp];
                    count++;
                }
            }

            /* Cell matrix in Bohr (same Cartesian frame as atom_pos); PLUMED converts using setMDLengthUnits. */
            cell[0] = pSPARC->range_x;
            cell[1] = pSPARC->range_y;
            cell[2] = pSPARC->range_z;
            for (count = 0; count < 3; count++) {
                box[count][0] = pSPARC->LatUVec[count * 3 + 0] * cell[count];
                box[count][1] = pSPARC->LatUVec[count * 3 + 1] * cell[count];
                box[count][2] = pSPARC->LatUVec[count * 3 + 2] * cell[count];
            }

            /* Potential energy (Ha); PLUMED converts via setMDEnergyUnits (QE/GROMACS patches). */
            poteng = pSPARC->Etot;

            double volCell = pSPARC->Jacbdet * pSPARC->range_x * pSPARC->range_y * pSPARC->range_z;

            // W = -σ V, same as socket/driver.c stress_to_virial. i-PI and SPARC
            // pressure are both -trace(σ)/3; dropping the minus reverses the barostat.
            double virial_calc[9] = {
                -pSPARC->stress[0] * volCell, -pSPARC->stress[1] * volCell, -pSPARC->stress[2] * volCell,
                -pSPARC->stress[1] * volCell, -pSPARC->stress[3] * volCell, -pSPARC->stress[4] * volCell,
                -pSPARC->stress[2] * volCell, -pSPARC->stress[4] * volCell, -pSPARC->stress[5] * volCell
            };

            if (pSPARC->Calc_stress != 1 && pSPARC->Calc_pres != 1)
                memset(virial_calc, 0, sizeof(virial_calc));

            /* Matches md.c: elecgs_Count is incremented after this call (one-based SCF index for this MD step). */
            step = pSPARC->elecgs_Count;

            // Pass a pointer to the current step index to plumed
            plumed_cmd(*ph, "setStep", &step);
            // Pass a pointer to the first element in the atomic positions array to plumed, assuming they are stored in a x1,y1,z1,x2,y2,z2 ... kind of ordering
            plumed_cmd(*ph, "setPositions", pSPARC->atom_pos);
            // Pass a pointer to the first element in the masses array to plumed
            plumed_cmd(*ph, "setMasses", mass);
            // Pass a pointer to the first element in the charges array to plumed
            plumed_cmd(*ph, "setCharges", charge);
            // Pass a pointer to the box shape array to plumed
            plumed_cmd(*ph, "setBox", &box[0][0]);
            // Pass a pointer to the current potential energy to plumed
            plumed_cmd(*ph, "setEnergy", &poteng);
            // Pass a pointer to the first element in the virial array to plumed
            plumed_cmd(*ph, "setVirial", &virial_calc[0]);
            // DFT forces in Ha/Bohr; PLUMED adds bias into this buffer.
            plumed_cmd(*ph, "setForces", pSPARC->forces);
            // calc() writes bias forces back into the MD force buffer; PLUMED requires setForces first.
            plumed_cmd(*ph, "calc", NULL);

            /* Inverse of socket/driver.c stress_to_virial init: σ = -W/V (Ha/Bohr^3). PLUMED adds bias into virial_calc in place. */
            if (volCell > 0.0 && (pSPARC->Calc_stress == 1 || pSPARC->Calc_pres == 1)) {
                double inv_vol = 1.0 / volCell;

                pSPARC->stress[0] = -virial_calc[0] * inv_vol;
                pSPARC->stress[1] = -virial_calc[1] * inv_vol;
                pSPARC->stress[2] = -virial_calc[2] * inv_vol;
                pSPARC->stress[3] = -virial_calc[4] * inv_vol;
                pSPARC->stress[4] = -virial_calc[5] * inv_vol;
                pSPARC->stress[5] = -virial_calc[8] * inv_vol;
                if (pSPARC->BC == 2)
                    pSPARC->pres = -1.0 * (pSPARC->stress[0] + pSPARC->stress[3] + pSPARC->stress[5]) / 3.0;
                apply_virial_stress = 1;
            }

            free(mass);
            free(charge);
        }
    }

    /* Electronic (+ PLUMED virial) stress lives on rank 0 after Calculate_Properties; replicate for MD_QOI / NPT on all ranks. */
    if (pSPARC->Calc_stress == 1 || pSPARC->Calc_pres == 1) {
        MPI_Bcast(&apply_virial_stress, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (apply_virial_stress) {
            MPI_Bcast(pSPARC->stress, 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            /* Same collective on every rank; do not gate on BC (avoids MPI mismatch). */
            MPI_Bcast(&pSPARC->pres, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
    }
#else
    (void)pSPARC;
#endif
}

void Plumed_Finalize(SPARC_OBJ *pSPARC)
{
    if (pSPARC == NULL)
        return;
#ifdef USE_PLUMED
    if (!pSPARC->PlumedFlag || pSPARC->PlumedHandle == NULL)
        return;

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0)
        return;

    plumed *ph = (plumed *)pSPARC->PlumedHandle;
    plumed_finalize(*ph);
    free(ph);
    pSPARC->PlumedHandle = NULL;
#else
    (void)pSPARC;
#endif
}
