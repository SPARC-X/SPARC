#ifndef ISDDFTATOM_H
#define ISDDFTATOM_H

#include <stdio.h>
#include <complex.h>
#include <math.h>
#include "isddft.h"

// max length of pseudopotential path
#ifndef L_PSD
#define L_PSD 4096
#endif

// max length of input strings
#ifndef L_STRING
#define L_STRING 512
#endif 

#ifndef L_ATMTYPE
#define L_ATMTYPE 8
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct _VNLOC_OBJ_ATOM {
    double *V;               // to store the non local matrix for each azimuthal quantum number
}VNLOC_OBJ_ATOM;

typedef struct _EXX_POT_OBJ_ATOM {
    double *VexxUp;
    double *VexxDw;
}EXX_POT_OBJ_ATOM;


typedef struct _SPARC_ATOM_OBJ{
    /* File names */
    char filename[L_STRING];
    
    /* Pseudopotential */
    // int is_default_psd;     // flag for reading psp
    int NLCC_flag;          // flag for Non-Linear Core Correction (NLCC)
    int *localPsd;          // respective local component of psp for
                            // each type of atom. 0 for s, 1 for p, 2 for d, 3 for f
    int *Zatom;             // atomic number
    int *Znucl;             // valence charge of each type
    // char *psdName;          // psp file name for each type of atom
    PSD_OBJ *psd;           // struct array storing psp info

    /* SCF options */
    int MAXIT_SCF;          // maximum iterations of SCF
    double SCF_tol;         // SCF tolerance for convergence
    int scf_success;        // to store scf success

    /* spin options */
    int spinFlag;           // Flag to indicate spin polarized or un-polarized calculations
    int nspin;              // number of spin in a calculation. 1 - spin unpolarized and 2 - spin polarized
    int nspinor;            // number of spinor in wavefunction
    int nspden;             // number of columns of spin density (including total)
    double *mag;            // magnetization
    double netM;            // net magnetization
    int fUpSum;             // To be used in mixing
    int fDwSum;             // To be used in mixing

    /* Domain description */
    double Rmax;            // expected half of domain (in Bohr), domain = [0, 2Rmax]

    /* Discretization */
    int Nd;                 // number of discretization grid points
    int Nq;                 // number of quadrature grid points
    double alpha;           // parameter for exponential grid
    double beta;            // parameter for exponential grid

    /* Spectral grid (Exponential)*/
    double xmax;            // xmax = 0.5[1 - exp(beta * 2Rmax)]
    double *x;              // x = xmax * (y + 1), y belongs to (-1, 1) via the cosine function (chebD)
                            // y grid described in paper (Comput. Phys. Commun. 308 (2025) 109448)
                            // in eqn 14, 15
    double *r;              // store the r grid [0, 2Rmax]
                            // r = (1/beta)log(1 - x), eqn. 15 from paper
    double *w;              // storing the clenshaw curtis weights
    double *int_scale;      // dx = int_scale * dr, int_scale = -beta(1 - x)
    double *D;              // storing the derivative matrix for x grid
    double *grad;           // storing d/dr, grad = int_scale * D
    double *laplacian;      // storing the laplacian matrix

    /* System */
    double *electronDens;           // electron density
    double *electronDens_core;      // model core electron density for NLCC
    double *orbitals;               // Kohn Sham orbitals
    double *eigenVal;               // Eigen values corresponding to the KS orbitals
    int *orbital_l;               // azimuthal quantum number corresponding to the KS orbital
    int *occ;                       // occupation of each KS orbital
    double *phi;                     // store the electrostatic potential

    /* Atomic states */
    int *states_n;                  // principal quantum numbers of occupied states
    int *states_l;                  // azimuthal quantum numbers of occupied states
    int *states_fup;                // spin up occupations in each state
    int *states_fdw;                // spin down occupations in each state
    int *states_ftot;               // total spin up + spin down occupations in each state
    int states_len;                 // length of states array
    int val_len;
    int *fUpVal;
    int *fDwVal;
    int *fTotVal;
    int *n0;
    int *n1;
    int *n2;
    int *n3;
    int *f0_up;
    int *f1_up;
    int *f2_up;
    int *f3_up;
    int *f0_dw;
    int *f1_dw;
    int *f2_dw;
    int *f3_dw;
    int *f0_tot;
    int *f1_tot;
    int *f2_tot;
    int *f3_tot;
    int max_l;
    int min_l;
    int lcount0;
    int lcount1;
    int lcount2;
    int lcount3;

    /* nonlocal */
    VNLOC_OBJ_ATOM *Vnl;                 // to store the non local potential for each azimuthal quantum number

    /* local */
    double *VJ;                     // to store the local potential

    /* exchange correlation */
    char XC[32];            // exchange correlation name
    double xc_rhotol;       // minimum value of rho below which it is made equal to xc_rhotol
    double xc_magtol;       // minimum value of mag below which it is made equal to xc_magtol
    double xc_sigmatol;     // minimum value of sgima below which it is made equal to xc_sigmatol
    int xc;                 // xc identifier tag
    int isGradient;         // if gradient of rho is required for xc
    int ixc[4];             // decomposition of xc
    int xcoption[2];        // option flag in xc 
    double *XCPotential;    // store the xc potential
    double *e_xc;           // xc energy per particle
    double *Dxcdgrho;       // derivative of exchange-correlation energy per particle wrt to norm of the gradient

    /* metaGGA */
    int countPotentialCalculate;
    double *tau;
    double *vxcMGGA3;

    /* Exact Exchange */
    int usefock;
    int MAXIT_FOCK;
    int MINIT_FOCK;
    double TOL_FOCK;
    double TOL_SCF_INIT;
    double exx_frac;
    EXX_POT_OBJ_ATOM *EXXL;

    /* Mixing */
    int MixingHistory;      // number of history vectors to keep
    int PulayFrequency;     // Pulay frequency in periodic pulay method
    int PulayRestartFlag;   // Pulay restart flag
    double MixingParameter; // mixing parameter, often denoted as beta
    double MixingParameterSimple;   // mixing parameter for simple mixing step, often denoted as omega
    double MixingParameterMag;      // mixing parameter for the magnetization density, denoted as beta_mag
    double MixingParameterSimpleMag;    // mixing parameter for the magnetixation density in simple mixing step, often denoted as omega_mag
    double *X;                      // residual matrix of Veff_loc, for mixing 
    double *F;                      // residual matrix of the residual of Veff_loc
    double *mixing_hist_fkm1;       // previous residual of Veff_loc 
    double *mixing_hist_xkm1;       // residual of mixed Veff_loc (between new mixed and previous mixed)
    double *mixing_hist_xk;         // current mixed variable
    double *mixing_hist_fk;         // f_k = g(x_k) - x_k

    /* Energies */
    double Eband;
    double Exc;
    double Exc_dc;
    double Eelec_dc;
    double Etot;
    double Exx;

    /* Energy density */

    /* Timing */
    // double tTotal;
    double tSCF;
} SPARC_ATOM_OBJ;

#endif