/**
 * @brief    Main file of sparc-atom, the radial single atom code using spectral scheme
 *           Refer to Bhowmik, S. et al. Comput. Phys. Commun. 308 (2025) 109448
 * 
 *           The contributors to the files in this sub-folder and its other sub-folders:
 * @authors  Sayan Bhowmik <sbhowmik9@gatech.edu>
 *           Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 *           Andrew J. Medford <ajm@gatech.edu>  
 *           John E. Pask, Lawrence Livermore National Laboratory
 * 
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#include "initializationAtom.h"
#include "isddftAtom.h"
#include "finalizeAtom.h"
#include "toolsAtom.h"
#include "electronicGroundStateAtom.h"
#include "isddft.h"

void sparc_atom(SPARC_OBJ *pSPARC, int ityp, SPARC_INPUT_OBJ *pSPARC_Input) {
    SPARC_ATOM_OBJ pSPARC_ATOM;
    double t1_tot, t2_tot;

    t1_tot = MPI_Wtime();
    // Initialize 
    Initialize_Atom(&pSPARC_ATOM, pSPARC, pSPARC_Input, ityp);

    // Calculate Electronic Ground State
    electronicGroundState_atom(&pSPARC_ATOM);

    // Printing
    if (!pSPARC_Input->is_hubbard){
        printResultsAtom(&pSPARC_ATOM);
    } 

    // Copy solution
    if (pSPARC_Input->is_hubbard) copyAtomSolution(&pSPARC_ATOM, pSPARC, ityp);

    // Free memory
    Finalize_Atom(&pSPARC_ATOM);

    t2_tot = MPI_Wtime();
    printf("*************************************************************\n");
    printf("The sparc-atom program took:  %0.10f sec.\n",t2_tot-t1_tot);
    printf("*************************************************************\n");

}

