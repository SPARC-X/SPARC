#ifndef ELECTRONICGROUNDSTATEATOM_H
#define ELECTRONICGROUNDSTATEATOM_H

#include "isddftAtom.h"

/**
 * @brief Calculate the electronic ground state
 * Refer to Eq. 7 of Bhowmik, S. et al. Comput. Phys. Commun. 308 (2025) 109448
 * 
 */
void electronicGroundState_atom(SPARC_ATOM_OBJ *pSPARC_ATOM);

/**
 * @brief Computes the nonlocal potential matrices from pseudopotential
 * Refer to Eq. 5b, 24b of Bhowmik, S. et al. Comput. Phys. Commun. 308 (2025) 109448
 */
void compute_nonlocal_potential(SPARC_ATOM_OBJ *pSPARC_ATOM);

/**
 * @brief Local part of pseudopotential V_{loc}
 * Refer to Eq. 5c, 24c of Bhowmik, S. et al. Comput. Phys. Commun. 308 (2025) 109448
 * 
 */
void compute_local_potential(SPARC_ATOM_OBJ *pSPARC_ATOM);

/**
 * @brief Poisson solve to get the Hartree Potential
 * Refer to Eq. 8b, 17b, and 24c of Bhowmik, S. et al. Comput. Phys. Commun. 308 (2025) 109448
 * 
 */
void poissonSolve_atom(SPARC_ATOM_OBJ *pSPARC_ATOM);

/**
 * @brief Calculate XC Potential for a given electron sensity (and set of wavefunctions)
 * Refer to Eq. 11a-c of Bhowmik, S. et al. Comput. Phys. Commun. 308 (2025) 109448
 * 
 */
void Calculate_Vxc_atom(SPARC_ATOM_OBJ *pSPARC_ATOM);

/**
* @brief add core electron density if needed
*/
void add_rho_core_atom(SPARC_ATOM_OBJ *pSPARC_ATOM, double *rho_in, double *rho_out, int ncol);

/**
 * @brief SCF Loop 
 * 1. Poisson Solve to calculate Hartree potential
 * 2. Calculate XC potential
 * 3. Perform Eigen solve
 * 4. Update wavefunctions
 * 5. Calculate electron density (Refer to Eq. 2, 18, and 23 of 
 * Bhowmik, S. et al. Comput. Phys. Commun. 308 (2025) 109448)
 */
void scfLoopAtom(SPARC_ATOM_OBJ *pSPARC_ATOM);

/**
 * @brief Form the electronic Hamiltonian and perform the Eigen Solve
 * 
 * H_{el} \tilde{R}_{nl} = E_{el} \tilde{R}_{nl}
 * Refer to Eq. 8a (and Eq. 17a) of Bhowmik, S. et al. Comput. Phys. Commun. 308 (2025) 109448
 * 
 */
void eigenSolve(SPARC_ATOM_OBJ *pSPARC_ATOM, int l, double *eigvecs, double *eigvals);

/**
 * @brief Sort the eigenvalues (and corresponding eigenvectors) in ascending order
 */
void sort_eigenpairs(double* eigvals_real, double* eigvecs, int n);

/**
 * @brief Calculates \int (\tilde{R}_{nl})^2 dr
 */
void integralOrbitalSquared(SPARC_ATOM_OBJ *pSPARC_ATOM, double *eigvec, double *integral);

/**
 * @brief Calculates SCF error as \frac{norm(rho_{out} - rho_{in})}{norm(rho_{out})}
 */
void SCFerrorAtom(SPARC_ATOM_OBJ *pSPARC_ATOM, double *scf_error);

/**
 * @brief Calculates normed relative error as \frac{norm(new - old)}{norm(old)}
 */
void normedRelativeError(double *old, double *new, int len, double *error);

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
);

/**
 * @brief   Anderson extrapolation weighted average vectors.
 *
 *          Find x_wavg := x_k - X * Gamma and f_wavg := (f_k - F * Gamma),
 *          where Gamma = inv(F^T * F) * F^T * f.
 */
void AndersonExtrapWtdAvg_atom(
    const int N, const int m, const double *x_k,
    const double *f_k, const double *X, const double *F,
    double *x_wavg, double *f_wavg);

/**
 * @brief   Anderson extrapolation coefficiens.
 *
 *          Gamma = inv(F^T * F) * F^T * f.         
 */
void AndersonExtrapCoeff_atom(
    const int N, const int m, const double *f, const double *F, 
    double *Gamma);

void compute_FtF_atom(const double *F, int m, int N, double *FtF);

void compute_Ftf_atom(const double *F, const double *f, int m, int N, double *Ftf);

void periodicPulayMixAtom(SPARC_ATOM_OBJ *pSPARC_ATOM, int iter_count);


#endif