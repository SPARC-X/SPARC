/**
 * @file    finiteTempXC.h
 * @brief   Finite-temperature exchange-correlation free-energy functionals. THREE separate functionals:
 *          KSDT     (LDA): V.V. Karasiev, T. Sjostrom, J. Dufty, S.B. Trickey, PRL 112, 076403 (2014);
 *                          original parameters; spin-unpolarized and spin-polarized.
 *          corrKSDT (LDA): corrected KSDT, V.V. Karasiev, J.W. Dufty, S.B. Trickey, PRL 120, 076401 (2018),
 *                          Supplemental Material Table S2; same equations and exchange as KSDT, refitted zeta = 0
 *                          correlation parameters; spin-unpolarized only. Selected with corr = 1 in ksdtc and
 *                          ksdt_TdfdT (exchange: ksdtx, identical to KSDT).
 *          KDT16    (GGA): V.V. Karasiev, J.W. Dufty, S.B. Trickey, PRL 120, 076401 (2018); built on corrKSDT;
 *                          spin-unpolarized only.
 *          T is the electronic temperature k_B*T in Ha, i.e. 1.0/pSPARC->Beta. T below 1e-12 K (KSDT, corrKSDT)
 *          or 1e-6 K (KDT16) is raised to that value (T = max(T, floor)), which gives the T -> 0 limit.
 */

#ifndef FINITETEMPXC_H
#define FINITETEMPXC_H

/**
 * @brief   KSDT and corrKSDT finite-temperature LDA exchange free energy per electron and potential (PRL Eq. 15);
 *          identical for the two functionals
 *
 * @param DMnd  number of local grid points
 * @param rho   electron density (DMnd)
 * @param T     electronic temperature k_B*T (Ha)
 * @param ex    (out) f_x (DMnd)
 * @param vx    (out) d(rho f_x)/d(rho) at fixed T (DMnd)
 */
void ksdtx(int DMnd, double *rho, double T, double *ex, double *vx);

/**
 * @brief   KSDT finite-temperature LSDA exchange - spin polarized (PRL Eq. 20); KSDT only
 *
 * @param rho   [total | up | down] densities (3*DMnd)
 * @param ex    (out) f_x (DMnd)
 * @param vx    (out) [up | down] potentials (2*DMnd)
 */
void ksdtx_spin(int DMnd, double *rho, double T, double *ex, double *vx);

/**
 * @brief   KSDT or corrKSDT finite-temperature LDA correlation f_c = f_xc - f_x (PRL Eqs. 9, 21),
 *          spin-unpolarized, inputs/outputs as ksdtx
 *
 * @param corr  0 = KSDT (PRL 112, 076403 Table I), 1 = corrKSDT (PRL 120, 076401 SM Table S2)
 */
void ksdtc(int DMnd, double *rho, double T, int corr, double *ec, double *vc);

/**
 * @brief   KSDT finite-temperature LSDA correlation - spin polarized (PRL Eqs. 17, 21), inputs/outputs as ksdtx_spin;
 *          KSDT only (corrKSDT is spin-unpolarized only)
 */
void ksdtc_spin(int DMnd, double *rho, double T, double *ec, double *vc);

/**
 * @brief   KDT16 finite-temperature GGA exchange (spin-unpolarized), outputs as in pbex
 *
 * @param sigma |grad rho|^2 (DMnd)
 * @param ex    (out) f_x (DMnd)
 * @param vx    (out) d(rho f_x)/d(rho) at fixed sigma and T (DMnd)
 * @param v2x   (out) 2 d(rho f_x)/d(sigma) (DMnd)
 */
void kdt16x(int DMnd, double *rho, double *sigma, double T, double *ex, double *vx, double *v2x);

/**
 * @brief   KDT16 finite-temperature GGA correlation (spin-unpolarized), outputs as in pbec
 */
void kdt16c(int DMnd, double *rho, double *sigma, double T, double *ec, double *vc, double *v2c);

/**
 * @brief   XC entropy term per electron, -T s_xc = T d(f_xc)/dT at fixed rho (and sigma), KSDT PRL Eq. 5;
 *          -T S_xc = int rho tdfdt is the XC part of -TS in the internal energy U = F + TS
 *
 * @param tdfdt (out) T d(f_x + f_c)/dT (DMnd); rho, sigma as in ksdtx / ksdtx_spin / kdt16x
 * @param corr  (ksdt_TdfdT) 0 = KSDT, 1 = corrKSDT; ksdt_spin_TdfdT is KSDT only, kdt16_TdfdT is KDT16
 */
void ksdt_TdfdT(int DMnd, double *rho, double T, int corr, double *tdfdt);
void ksdt_spin_TdfdT(int DMnd, double *rho, double T, double *tdfdt);
void kdt16_TdfdT(int DMnd, double *rho, double *sigma, double T, double *tdfdt);

#endif // FINITETEMPXC_H
