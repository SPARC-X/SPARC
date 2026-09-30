/**
 * @file    finiteTempXC.c
 * @brief   Finite-temperature exchange-correlation free-energy functionals. This module implements THREE
 *          separate functionals (input option EXCHANGE_CORRELATION):
 *
 *          KSDT     -- LDA, V.V. Karasiev, T. Sjostrom, J. Dufty, S.B. Trickey, PRL 112, 076403 (2014).
 *                      Original parameters (PRL Table I, zeta = 0 and zeta = 1 fits + spin interpolation).
 *                      Spin-unpolarized and spin-polarized.
 *                      Functions: ksdtx, ksdtx_spin, ksdtc (corr = 0), ksdtc_spin, ksdt_TdfdT (corr = 0),
 *                      ksdt_spin_TdfdT.
 *          corrKSDT -- LDA, corrected KSDT: V.V. Karasiev, J.W. Dufty, S.B. Trickey, PRL 120, 076401 (2018),
 *                      Supplemental Material Table S2. A separate functional: same equations and same exchange
 *                      as KSDT, but different (refitted) zeta = 0 correlation parameters. The authors refitted
 *                      only the spin-unpolarized case, so corrKSDT is SPIN-UNPOLARIZED ONLY.
 *                      Functions: ksdtx, ksdtc (corr = 1), ksdt_TdfdT (corr = 1).
 *          KDT16    -- GGA, V.V. Karasiev, J.W. Dufty, S.B. Trickey, PRL 120, 076401 (2018); uses corrKSDT as
 *                      its LDA part. SPIN-UNPOLARIZED ONLY.
 *                      Functions: kdt16x, kdt16c, kdt16_TdfdT.
 */

#include <math.h>

#include "finiteTempXC.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* =====================================================================================================
 * EQUATIONS, IN THE ORDER THEY ARE EVALUATED   (Hartree a.u.; n = density; T = k_B*T [Ha] = 1/Beta)
 * [name] = the function below that implements the step. All derivatives are analytic, taken at fixed T.
 *
 * KSDT  -- V.V. Karasiev, T. Sjostrom, J. Dufty, S.B. Trickey, PRL 112, 076403 (2014)
 *  1. r_s = (3/(4 pi n))^(1/3);  t = T/T_F = 2 lambda^2 T r_s^2,  lambda = (4/(9 pi))^(1/3)         [ft_rs_t]
 *  2. Eq. 10  a(t) = 0.610887 tanh(1/t) (0.75 + 3.04363 t^2 - 0.09227 t^3 + 1.7035 t^4)
 *                    / (1 + 8.31051 t^2 + 5.1105 t^4)                                               [ksdt_a]
 *     Eq. 11  b(t) = tanh(1/sqrt(t)) (b1 + b2 t^2 + b3 t^4) / (1 + b4 t^2 + b5 t^4)                 [ksdt_pade, s = 1/2]
 *     Eq. 12  c(t) = [c1 + c2 exp(-c3/t)] e(t)                                                      [ksdt_c]
 *     Eq. 13  d(t) = tanh(1/sqrt(t)) (d1 + d2 t^2 + d3 t^4) / (1 + d4 t^2 + d5 t^4)                 [ksdt_pade, s = 1/2]
 *     Eq. 14  e(t) = tanh(1/t) (e1 + e2 t^2 + e3 t^4) / (1 + e4 t^2 + e5 t^4)                       [ksdt_pade, s = 1]
 *  3. Eq. 9   f_xc^zeta(r_s,t) = -(1/r_s) [omega a + b sqrt(r_s) + c r_s] / [1 + d sqrt(r_s) + e r_s]
 *             omega_0 = 1, omega_1 = 2^(1/3);  b..e from Table I (zeta = 0, 1);  b5 = sqrt(3/2) omega b3/lambda
 *                                                                                                   [ksdt_fxc]
 *  4. spin-polarized only, zeta = (n_up - n_dn)/n:
 *     Eq. 19  alpha(r_s,t) = 2 - g(r_s) exp[-t lambda(r_s,t)],  g = (g1 + g2 r_s)/(1 + g3 r_s),
 *             lambda(r_s,t) = lambda1 + lambda2 t sqrt(r_s);  g_i, lambda_i from Table III           [ksdt_alpha]
 *     Eq. 18  phi = [(1 + zeta)^alpha + (1 - zeta)^alpha - 2] / (2^alpha - 2)                       [ksdt_phi]
 *     Eq. 17  f_xc(r_s,t,zeta) = f_xc^0(r_s,t) + [f_xc^1(r_s, 2^(-2/3) t) - f_xc^0(r_s,t)] phi      [ksdt_fxc_spin]
 *  5. v_xc = d(n f_xc)/dn = f_xc - (r_s/3) df/dr_s - (2t/3) df/dt   (n dr_s/dn = -r_s/3, n dt/dn = -2t/3)
 *     spin: v_up/dn = v_xc - zeta df/dzeta +/- df/dzeta                                             [ksdtc, ksdtc_spin]
 *  6. exchange:
 *     Eq. 15  f_x^0(r_s,t) = -a(t)/r_s;  v_x = (4/3) f_x + (2t/3) a'(t)/r_s                         [ksdt_fx, ksdtx]
 *     Eq. 20  f_x(r_s,T,zeta) = 1/2 [(1+zeta)^(4/3) f_x^0(r_s,t_up) + (1-zeta)^(4/3) f_x^0(r_s,t_dn)],
 *             t_up/dn = t(2 n_up/dn, T);  = sum_s (n_s/n) f_x^0 at density 2 n_s, so v_x,s = v_x^0(2 n_s)
 *                                                                                                   [ksdt_fx_spin, ksdtx_spin]
 *  7. correlation:
 *     Eq. 21  f_c = f_xc - f_x  (Eq. 9 or 17 minus Eq. 15 or 20);  v_c = v_xc - v_x                 [ksdtc, ksdtc_spin]
 *  8. SPARC: e_x = f_x, e_c = f_c  (e_x + e_c = f_xc;  Exc = int n (e_x + e_c))
 *  9. Eq. 5   XC entropy term per electron (internal energy U = F + TS):  -T s_xc = f_xc - e_xc = t df_xc/dt,
 *             at fixed n, zeta                                                                      [ksdt_TdfdT, ksdt_spin_TdfdT]
 *
 * corrKSDT -- V.V. Karasiev, J.W. Dufty, S.B. Trickey, PRL 120, 076401 (2018), Supplemental Material Table S2
 *  Same steps 1-3 and 5-9 as KSDT, spin-unpolarized only (zeta = 0; no step 4). The only difference:
 *  in step 3 the zeta = 0 parameters b..e are those of SM Table S2 (parameter set 2 of ksdt_fxc) instead of
 *  KSDT Table I. Exchange (step 6, a(t)) is identical to KSDT.                     [ksdtc, ksdt_TdfdT with corr = 1]
 *
 * KDT16 -- V.V. Karasiev, J.W. Dufty, S.B. Trickey, PRL 120, 076401 (2018) + its Supplemental Material (SM);
 *          Ax, Bx fits from V.V. Karasiev, D. Chakraborty, S.B. Trickey, CPC 192, 114 (2015). Spin-unpolarized.
 *  exchange
 *  1. r_s, t as above;  s^2 = sigma / [4 (3 pi^2)^(2/3) n^(8/3)],  sigma = |grad n|^2
 *  2. Eqs. 3-4  f_x^LDA = eps_x^LDA Ax(t),  eps_x^LDA = -(3/4)(3/pi)^(1/3) n^(1/3);
 *               Ax(t): fit CPC Eq. 39, Table 9 (as cited in SM Sec. V),
 *               y = 2/(3 t^(3/2)), u = y^(2/3), v = y^(4/3)                                         [kdt16_Ax]
 *  3. Eq. 6     Bx(t): fit from CPC 192, 114 (PRL Ref. 76), u = y^(2/3)                             [kdt16_Bx]
 *  4. Eq. 7     s_2x = s^2 Bx(t)/Ax(t)                                                              [kdt16x]
 *  5. Eq. 10    Fx(s_2x) = 1 + nu_x s_2x / (1 + alpha |s_2x|),  nu_x = 0.21951, alpha = nu_x/(1.804 - 1)
 *                                                                                                   [kdt16_Fx]
 *  6. Eq. 8     f_x = eps_x^LDA Ax(t) Fx(s_2x)                                                      [kdt16x]
 *  correlation
 *  7. SM S8     f_c^LDA = f_xc^corrKSDT(r_s,t) - eps_x^LDA Ax(t);  corrKSDT = Eq. 9 with SM Table S2 [kdt16_fc_lda]
 *  8. SM S2     Bc(r_s,t) = [1 + sum_{i=1}^4 (a_i + b_i sqrt(r_s) + c_i r_s) u^i]
 *                         / [1 + sum_{i=1}^5 (d_i + e_i r_s^(3/2) + f_i r_s^3) u^i],  u = t^(13/4);  SM Table S1
 *                                                                                                   [kdt16_Bc]
 *  9. Eq. 11    q_c^2 = q^2 Bc(r_s,t),  q^2 = sigma / [16 (3/pi)^(1/3) n^(7/3)]  (PBE variable t^2) [kdt16c]
 * 10. SM S4-S5  H = gamma ln[1 + (beta/gamma) q_c^2 (1 + A q_c^2) / (1 + A q_c^2 + A^2 q_c^4)],
 *               A = (beta/gamma) / [exp(-f_c^LDA/gamma) - 1],  beta = 0.066725, gamma = (1 - ln 2)/pi^2
 *                                                                                                   [kdt16_H]
 * 11. Eq. 12    f_c = f_c^LDA + H                                                                   [kdt16c]
 * 12. e_x = f_x, e_c = f_c;  v = d(n f)/dn at fixed sigma, T;  v2 = 2 d(n f)/d(sigma)   (as pbex/pbec;
 *     SPARC adds -div(v2 grad n) to v);  n ds^2/dn = -(8/3) s^2,  n dq^2/dn = -(7/3) q^2            [kdt16x, kdt16c]
 * 13. XC entropy term (as KSDT Eq. 5):  -T s_xc = t d(f_x + f_c)/dt  at fixed n, sigma                [kdt16_TdfdT]
 * ===================================================================================================== */


/* ---- KSDT Table I (rows 0, 1: zeta = 0, zeta = 1) and corrKSDT SM Table S2 (row 2: zeta = 0; used by corrKSDT
 *      and by KDT16) ---- */
/* b5 = sqrt(3/2) omega b3/lambda at full precision, as in libxc; the tables print 0.871837, 1.590438, 1.054151 */
/* (formula: 0.8718374, 1.5904386, 1.0541500).  SM Table S2 labels c3 = 0.953988 as "c4" */
static const double KSDT_b[3][5] = {{0.283997,  48.932154, 0.370919,  61.095357, 0.871837422702768},
                                    {0.329001, 111.598308, 0.537053, 105.086663, 1.590438591727009},
                                    {0.342554,   9.141315, 0.448483,  18.553096, 1.054149997293224}};
static const double KSDT_c[3][3] = {{0.870089,  0.193077, 2.414644},
                                    {0.848930,  0.167952, 0.088820},
                                    {0.875130, -0.256320, 0.953988}};
static const double KSDT_d[3][5] = {{0.579824,  94.537454,  97.839603,  59.939999, 24.388037},
                                    {0.551330, 180.213159, 134.486231, 103.861695, 17.750710},
                                    {0.725917,   2.237347,   0.280748,   4.185911,  0.692183}};
static const double KSDT_e[3][5] = {{0.212036, 16.731249, 28.485792,  34.028876, 17.235515},
                                    {0.153124, 19.543945, 43.400337, 120.255145, 15.662836},
                                    {0.255415,  0.931933,  0.115398,  17.234117,  0.451437}};


/* ---- Step 1: r_s and t = T/T_F ---- */
/* T = k_B T (Ha) below FT_T_MIN (1e-12 K) is raised to FT_T_MIN, so that T = 0 gives the T -> 0 limit: at t = 0
 * exactly the fits have 0/0 terms (d tanh(1/t)/dt, y = 2/(3 t^(3/2)) in Ax, Bx) that make the potentials NaN.
 * KDT16 uses the higher floor KDT16_T_MIN (1e-6 K; kdt16x, kdt16c, kdt16_TdfdT): for t < ~3e-17 (T = 1e-12 K,
 * n > ~4e-3) the Bx fit derivative dBx/dt overflows (terms ~ t^-19) and v_x and T df/dT become NaN.
 * k_B = 3.1668115634556e-6 Ha/K, as SPARC's CONST_KB. */
#define FT_T_MIN    (1e-12 * 3.1668115634556e-6)
#define KDT16_T_MIN (1e-6  * 3.1668115634556e-6)
static void ft_rs_t(double n, double T, double *rs, double *t) {
    const double C31 = 0.6203504908993999;      // (3/(4 pi))^(1/3)
    const double lambda = cbrt(4.0/(9.0*M_PI));
    if (T < FT_T_MIN) T = FT_T_MIN;
    *rs = C31 / cbrt(n);
    *t = 2.0 * lambda * lambda * T * (*rs) * (*rs);
}


/* ---- Eq. 10: a(t) and da/dt; 0.610887 = 1/(pi lambda) ---- */
static void ksdt_a(double t, double *a, double *da_dt) {
    const double lambda = cbrt(4.0/(9.0*M_PI));
    double t2 = t*t, t3 = t2*t, t4 = t2*t2, th = tanh(1.0/t);
    double N = 0.75 + 3.04363*t2 - 0.09227*t3 + 1.7035*t4, dN = 2.0*3.04363*t - 3.0*0.09227*t2 + 4.0*1.7035*t3;
    double D = 1.0 + 8.31051*t2 + 5.1105*t4,               dD = 2.0*8.31051*t + 4.0*5.1105*t3;
    *a = th * N / D / (M_PI*lambda);
    // d tanh(1/t)/dt = -sech^2(1/t)/t^2
    *da_dt = (-(1.0 - th*th) / t2 * N / D + th * (dN*D - N*dD) / (D*D)) / (M_PI*lambda);
}


/* ---- Eqs. 11, 13, 14: y(t) = tanh(t^-s) (p1 + p2 t^2 + p3 t^4)/(1 + p4 t^2 + p5 t^4) and dy/dt ---- */
/* s = 1/2 for b(t) and d(t), s = 1 for e(t) */
static void ksdt_pade(double t, double s, const double *p, double *y, double *dy_dt) {
    double t2 = t*t, t4 = t2*t2, x = pow(t, -s), th = tanh(x);
    double N = p[0] + p[1]*t2 + p[2]*t4, dN = 2.0*p[1]*t + 4.0*p[2]*t2*t;
    double D = 1.0 + p[3]*t2 + p[4]*t4,  dD = 2.0*p[3]*t + 4.0*p[4]*t2*t;
    *y = th * N / D;
    // d tanh(t^-s)/dt = -sech^2(t^-s) s t^(-s-1)
    *dy_dt = -(1.0 - th*th) * s * x / t * N / D + th * (dN*D - N*dD) / (D*D);
}


/* ---- Eq. 12: c(t) = [c1 + c2 exp(-c3/t)] e(t) and dc/dt ---- */
static void ksdt_c(double t, const double *c, double e, double de_dt, double *cval, double *dc_dt) {
    double ex = exp(-c[2]/t);
    *cval = (c[0] + c[1]*ex) * e;
    *dc_dt = c[1]*c[2]/(t*t)*ex * e + (c[0] + c[1]*ex) * de_dt;
}


/* ---- Eq. 9: f_xc^zeta(r_s,t), df/dr_s (fixed t), df/dt (fixed r_s) ---- */
/* set = 0: KSDT zeta = 0;  1: KSDT zeta = 1;  2: corrKSDT zeta = 0 (used by corrKSDT, i.e. corr = 1, and by KDT16) */
static void ksdt_fxc(int set, double rs, double t, double *f, double *df_drs, double *df_dt) {
    double omega = (set == 1) ? cbrt(2.0) : 1.0;
    double a, da, b, db, c, dc, d, dd, e, de;
    ksdt_a(t, &a, &da);                          // Eq. 10
    ksdt_pade(t, 0.5, KSDT_b[set], &b, &db);     // Eq. 11
    ksdt_pade(t, 0.5, KSDT_d[set], &d, &dd);     // Eq. 13
    ksdt_pade(t, 1.0, KSDT_e[set], &e, &de);     // Eq. 14
    ksdt_c(t, KSDT_c[set], e, de, &c, &dc);      // Eq. 12

    double rs_sqrt = sqrt(rs);
    double N = omega*a + b*rs_sqrt + c*rs, D = 1.0 + d*rs_sqrt + e*rs;
    *f = -N / (rs*D);
    // quotient rule; dN/dr_s = b/(2 sqrt(r_s)) + c, dD/dr_s = d/(2 sqrt(r_s)) + e
    *df_drs = -((0.5*b/rs_sqrt + c)*rs*D - N*(D + rs*(0.5*d/rs_sqrt + e))) / (rs*rs*D*D);
    *df_dt  = -((omega*da + db*rs_sqrt + dc*rs)*D - N*(dd*rs_sqrt + de*rs)) / (rs*D*D);
}


/* ---- Eq. 19: alpha(r_s,t) and its derivatives; Table III parameters ---- */
static void ksdt_alpha(double rs, double t, double *alpha, double *dalpha_drs, double *dalpha_dt) {
    const double g1 = 2.0/3.0, g2 = -0.0139261, g3 = 0.183208;
    const double lambda1 = 1.064009, lambda2 = 0.572565;
    double rs_sqrt = sqrt(rs);
    double g = (g1 + g2*rs) / (1.0 + g3*rs), dg_drs = (g2 - g1*g3) / ((1.0 + g3*rs)*(1.0 + g3*rs));
    double ex = exp(-t*(lambda1 + lambda2*t*rs_sqrt));      // exp[-t lambda(r_s,t)]
    *alpha = 2.0 - g*ex;
    *dalpha_drs = -dg_drs*ex + g*ex*lambda2*t*t/(2.0*rs_sqrt);
    *dalpha_dt  = g*ex*(lambda1 + 2.0*lambda2*t*rs_sqrt);
}


/* ---- Eq. 18: phi(alpha, zeta), dphi/dalpha, dphi/dzeta ---- */
static void ksdt_phi(double alpha, double zeta, double *phi, double *dphi_dalpha, double *dphi_dzeta) {
    double zp = pow(1.0 + zeta, alpha), zm = pow(1.0 - zeta, alpha), D = pow(2.0, alpha) - 2.0;
    *phi = (zp + zm - 2.0) / D;
    *dphi_dalpha = (zp*log(1.0 + zeta) + zm*log(1.0 - zeta) - (*phi)*pow(2.0, alpha)*log(2.0)) / D;
    *dphi_dzeta  = alpha * (zp/(1.0 + zeta) - zm/(1.0 - zeta)) / D;
}


/* ---- zeta = (n_up - n_dn)/n, kept inside (-1, 1) so that (1 -/+ zeta)^alpha and its log stay finite ---- */
static double ksdt_zeta(double n, double n_up, double n_dn) {
    return fmin(fmax((n_up - n_dn) / n, -1.0 + 1e-12), 1.0 - 1e-12);
}


/* ---- Eq. 17: f_xc(r_s,t,zeta) and its partial derivatives ---- */
static void ksdt_fxc_spin(double rs, double t, double zeta, double *f, double *df_drs, double *df_dt, double *df_dzeta) {
    const double s = pow(2.0, -2.0/3.0);   // t of the polarized gas is 2^(-2/3) t, Eq. 17
    double f0, df0_drs, df0_dt, f1, df1_drs, df1_dt;
    ksdt_fxc(0, rs, t, &f0, &df0_drs, &df0_dt);                 // Eq. 9, zeta = 0
    ksdt_fxc(1, rs, s*t, &f1, &df1_drs, &df1_dt);               // Eq. 9, zeta = 1, at 2^(-2/3) t
    df1_dt *= s;

    double alpha, dalpha_drs, dalpha_dt, phi, dphi_dalpha, dphi_dzeta;
    ksdt_alpha(rs, t, &alpha, &dalpha_drs, &dalpha_dt);         // Eq. 19
    ksdt_phi(alpha, zeta, &phi, &dphi_dalpha, &dphi_dzeta);     // Eq. 18

    *f = f0 + (f1 - f0)*phi;                                    // Eq. 17
    *df_drs = df0_drs + (df1_drs - df0_drs)*phi + (f1 - f0)*dphi_dalpha*dalpha_drs;
    *df_dt  = df0_dt  + (df1_dt  - df0_dt )*phi + (f1 - f0)*dphi_dalpha*dalpha_dt;
    *df_dzeta = (f1 - f0)*dphi_dzeta;
}


/* ---- Eq. 15: f_x^0(r_s,t) = -a(t)/r_s at density n, and v_x = d(n f_x)/dn ---- */
static void ksdt_fx(double n, double T, double *fx, double *vx) {
    double rs, t, a, da_dt;
    ft_rs_t(n, T, &rs, &t);
    ksdt_a(t, &a, &da_dt);
    *fx = -a / rs;
    *vx = (4.0/3.0)*(*fx) + (2.0*t/3.0)*da_dt/rs;             // f - (r_s/3) df/dr_s - (2t/3) df/dt
}


/* ---- Eq. 20: f_x(r_s,T,zeta) = sum_s (n_s/n) f_x^0 at density 2 n_s;  v_x,s = v_x^0(2 n_s) ---- */
static void ksdt_fx_spin(double n, double n_up, double n_dn, double T, double *fx, double *vx_up, double *vx_dn) {
    double fx_up, fx_dn;
    ksdt_fx(2.0*n_up, T, &fx_up, vx_up);                        // t_up = t(2 n_up, T)
    ksdt_fx(2.0*n_dn, T, &fx_dn, vx_dn);                        // t_dn = t(2 n_dn, T)
    *fx = (n_up*fx_up + n_dn*fx_dn) / n;
}


/* ---- KSDT and corrKSDT exchange (identical), spin-unpolarized: steps 1, 6 ---- */
void ksdtx(int DMnd, double *rho, double T, double *ex, double *vx) {
    for (int i = 0; i < DMnd; i++)
        ksdt_fx(rho[i], T, &ex[i], &vx[i]);                         // Eq. 15
}


/* ---- KSDT exchange, spin-polarized (KSDT only; corrKSDT is spin-unpolarized only): steps 1, 6 ---- */
void ksdtx_spin(int DMnd, double *rho, double T, double *ex, double *vx) {
    for (int i = 0; i < DMnd; i++)
        ksdt_fx_spin(rho[i], rho[DMnd+i], rho[2*DMnd+i], T, &ex[i], &vx[i], &vx[DMnd+i]);   // Eq. 20
}


/* ---- KSDT (corr = 0) or corrKSDT (corr = 1) correlation, spin-unpolarized: steps 1, 3, 5, 7 ---- */
void ksdtc(int DMnd, double *rho, double T, int corr, double *ec, double *vc) {
    for (int i = 0; i < DMnd; i++) {
        double rs, t, f, df_drs, df_dt, fx, vx;
        ft_rs_t(rho[i], T, &rs, &t);                                // step 1
        ksdt_fxc(corr ? 2 : 0, rs, t, &f, &df_drs, &df_dt);         // Eq. 9, zeta = 0 (corrKSDT: SM Table S2)
        ksdt_fx(rho[i], T, &fx, &vx);                               // Eq. 15
        ec[i] = f - fx;                                             // Eq. 21
        vc[i] = f - (rs/3.0)*df_drs - (2.0*t/3.0)*df_dt - vx;       // step 5, minus v_x
    }
}


/* ---- KSDT correlation, spin-polarized (KSDT only): steps 1, 3, 4, 5, 7 ---- */
void ksdtc_spin(int DMnd, double *rho, double T, double *ec, double *vc) {
    for (int i = 0; i < DMnd; i++) {
        double rs, t, f, df_drs, df_dt, df_dzeta;
        ft_rs_t(rho[i], T, &rs, &t);                                // step 1
        double zeta = ksdt_zeta(rho[i], rho[DMnd+i], rho[2*DMnd+i]);
        ksdt_fxc_spin(rs, t, zeta, &f, &df_drs, &df_dt, &df_dzeta); // Eqs. 17-19

        double v = f - (rs/3.0)*df_drs - (2.0*t/3.0)*df_dt - zeta*df_dzeta;   // step 5
        double fx, vx_up, vx_dn;
        ksdt_fx_spin(rho[i], rho[DMnd+i], rho[2*DMnd+i], T, &fx, &vx_up, &vx_dn);   // Eq. 20
        ec[i] = f - fx;                                                        // Eq. 21
        vc[i]      = v + df_dzeta - vx_up;   // up
        vc[DMnd+i] = v - df_dzeta - vx_dn;   // down
    }
}


/* ---- KSDT (corr = 0) or corrKSDT (corr = 1) XC entropy term, spin-unpolarized: step 9 ---- */
void ksdt_TdfdT(int DMnd, double *rho, double T, int corr, double *tdfdt) {
    for (int i = 0; i < DMnd; i++) {
        double rs, t, f, df_drs, df_dt;
        ft_rs_t(rho[i], T, &rs, &t);                                // step 1
        ksdt_fxc(corr ? 2 : 0, rs, t, &f, &df_drs, &df_dt);         // Eq. 9, zeta = 0 (corrKSDT: SM Table S2)
        tdfdt[i] = t*df_dt;                                         // Eq. 5
    }
}


/* ---- KSDT XC entropy term, spin-polarized (KSDT only): step 9 ---- */
void ksdt_spin_TdfdT(int DMnd, double *rho, double T, double *tdfdt) {
    for (int i = 0; i < DMnd; i++) {
        double rs, t, f, df_drs, df_dt, df_dzeta;
        ft_rs_t(rho[i], T, &rs, &t);                                // step 1
        double zeta = ksdt_zeta(rho[i], rho[DMnd+i], rho[2*DMnd+i]);
        ksdt_fxc_spin(rs, t, zeta, &f, &df_drs, &df_dt, &df_dzeta); // Eqs. 17-19
        tdfdt[i] = t*df_dt;                                         // Eq. 5
    }
}


/* ---- KDT16 Eqs. 3-4: Ax(t) = f_x^LDA/eps_x^LDA and dAx/dt; fit CPC Eq. 39, Table 9 (same as ABINIT tildeAx) ---- */
/* Ax(y) = [a_ln y^4 ln(y) + a_2.5 u^(5/2) + sum_{i=1}^8 a_i u^i] / [1 + sum_{i=1}^4 b_i v^i], */
/* y = 2/(3 t^(3/2)), u = y^(2/3), v = y^(4/3) */
static void kdt16_Ax(double t, double *Ax, double *dAx_dt) {
    const double a_ln = -0.0475410604245741, a_25 = -0.1065378473507800;
    const double a[8] = {0.5823869764908659, -0.0068339509356661, 11.5469239288490009, -0.8465428870889800,
                         -0.1212525366470300, 1.9902818786101000, 0.0, 0.0744389046707120};
    const double b[4] = {19.9256144707979992, 5.1663994545590004, 2.0463164858237000, 0.0744389046707120};
    double y = 2.0/(3.0*t*sqrt(t)), u = cbrt(y*y), v = u*u;
    double N = a_ln*y*y*y*y*log(y) + a_25*u*u*sqrt(u), dN_du = 2.5*a_25*u*sqrt(u);
    double D = 1.0, dD_dv = 0.0, ui = 1.0, vi = 1.0;
    for (int i = 0; i < 8; i++) { dN_du += (i+1)*a[i]*ui; ui *= u; N += a[i]*ui; }
    for (int i = 0; i < 4; i++) { dD_dv += (i+1)*b[i]*vi; vi *= v; D += b[i]*vi; }
    // chain rule: du/dy = 2u/(3y), dv/dy = 4v/(3y), dy/dt = -3y/(2t)
    double dN_dy = a_ln*y*y*y*(4.0*log(y) + 1.0) + dN_du*2.0*u/(3.0*y);
    double dD_dy = dD_dv*4.0*v/(3.0*y);
    *Ax = N / D;
    *dAx_dt = (dN_dy*D - N*dD_dy) / (D*D) * (-1.5*y/t);
}


/* ---- KDT16 Eq. 6: Bx(t) and dBx/dt; fit from CPC 192, 114 (PRL Ref. 76; same as ABINIT tildeBx) ---- */
/* Bx(y) = sum_{i=2}^10 a_i u^i / [1 + sum_{i=1}^10 b_i u^i],  u = y^(2/3) */
/* a2 = -2 (3/2)^(4/3) = -3^(4/3) 2^(-1/3) is the exact classical-limit coefficient (I_k -> Gamma(k+1) e^eta) */
static void kdt16_Bx(double t, double *Bx, double *dBx_dt) {
    const double a[10] = {0.0, -3.4341427276599950, -0.9066069544311700, 2.2386316137237001, 2.4232553178542000,
                          -0.1339278564306200, 0.4392739633708200, -0.0497109675177910, 0.0, 0.0028609701106953};
    const double b[10] = {0.7098198258073800, 4.6311326377185997, -2.9243190977647000, 6.1688157841895004, -1.3435764191535999,
                          0.1576046383295400, 0.4365792821186800, -0.0620444574606262, 0.0, 0.0028609701106953};
    double y = 2.0/(3.0*t*sqrt(t)), u = cbrt(y*y);
    double N = 0.0, dN_du = 0.0, D = 1.0, dD_du = 0.0, ui = 1.0;
    for (int i = 0; i < 10; i++) {
        dN_du += (i+1)*a[i]*ui; dD_du += (i+1)*b[i]*ui;
        ui *= u; N += a[i]*ui; D += b[i]*ui;
    }
    *Bx = N / D;
    *dBx_dt = (dN_du*D - N*dD_du) / (D*D) * (-u/t);   // du/dt = -u/t
}


/* ---- KDT16 Eq. 10: Fx(s_2x) = 1 + nu_x s_2x/(1 + alpha |s_2x|); returns Fx and dFx/ds_2x ---- */
static double kdt16_Fx(double s2x, double *dFx_ds2x) {
    const double nu = 0.21951, alpha = 0.21951/0.804;   // nu_x = PBE mu; alpha = nu_x/(Fx_max - 1), Fx_max = 1.804
    double den = 1.0 + alpha*fabs(s2x);
    *dFx_ds2x = nu / (den*den);
    return 1.0 + nu*s2x/den;
}


/* ---- KDT16 SM Eq. S2: Bc(r_s,t), dBc/dr_s, dBc/dt; SM Table S1 ---- */
static void kdt16_Bc(double rs, double t, double *Bc, double *dBc_drs, double *dBc_dt) {
    const double a[4] = { 0.30047773E+03, -0.38706401E+03,  0.25112237E+04,  0.52243427E+03};
    const double b[4] = {-0.11166044E+03, -0.45327975E+02, -0.14507109E+04, -0.30665095E+02};
    const double c[4] = { 0.32175261E+02,  0.61853048E+02,  0.33585054E+03,  0.12874241E+03};
    const double d[5] = { 0.11077393E+03,  0.32355494E+03,  0.45509212E+03,  0.10884352E+04,  0.36112605E+00};
    const double e[5] = { 0.12854960E+01,  0.13482659E+02,  0.23416018E+02,  0.24480831E+02,  0.32161372E-08};
    const double f[5] = { 0.41006057E-02,  0.18933118E-01,  0.24295413E-04,  0.18369776E-07,  0.69274681E-10};
    double u = pow(t, 3.25), rs_sqrt = sqrt(rs), rs_32 = rs*rs_sqrt, rs_3 = rs*rs*rs;
    double N = 1.0, dN_drs = 0.0, dN_du = 0.0, D = 1.0, dD_drs = 0.0, dD_du = 0.0, ui = 1.0;
    for (int i = 0; i < 4; i++) {
        double p = a[i] + b[i]*rs_sqrt + c[i]*rs;
        dN_du += (i+1)*p*ui; ui *= u; N += p*ui; dN_drs += (0.5*b[i]/rs_sqrt + c[i])*ui;
    }
    ui = 1.0;
    for (int i = 0; i < 5; i++) {
        double q = d[i] + e[i]*rs_32 + f[i]*rs_3;
        dD_du += (i+1)*q*ui; ui *= u; D += q*ui; dD_drs += (1.5*e[i]*rs_sqrt + 3.0*f[i]*rs*rs)*ui;
    }
    *Bc = N / D;                                          // N, D > 0: log-derivative avoids D^2 overflow at large t
    *dBc_drs = (*Bc) * (dN_drs/N - dD_drs/D);
    *dBc_dt  = (*Bc) * (dN_du/N - dD_du/D) * 3.25*u/t;    // du/dt = (13/4) u/t
}


/* ---- KDT16 SM Eq. S8: f_c^LDA = f_xc^corrKSDT - eps_x^LDA Ax(t), with df/dr_s and df/dt ---- */
static void kdt16_fc_lda(double rs, double t, double *fc, double *dfc_drs, double *dfc_dt) {
    double fxc, dfxc_drs, dfxc_dt, Ax, dAx_dt;
    ksdt_fxc(2, rs, t, &fxc, &dfxc_drs, &dfxc_dt);      // corrKSDT, Eq. 9 with SM Table S2
    kdt16_Ax(t, &Ax, &dAx_dt);
    double epsx = -0.458165293283143 / rs;               // eps_x^LDA = -(3/(4 pi lambda))/r_s
    *fc = fxc - epsx*Ax;
    *dfc_drs = dfxc_drs + epsx/rs*Ax;                    // d(eps_x)/dr_s = -eps_x/r_s
    *dfc_dt  = dfxc_dt - epsx*dAx_dt;
}


/* ---- KDT16 SM Eqs. S4-S5: PBE H(f_c^LDA, zeta = 0, q_c) with Q = q_c^2; dH/df_c and dH/dQ ---- */
static void kdt16_H(double fc, double Q, double *H, double *dH_dfc, double *dH_dQ) {
    const double beta = 0.066725, gamma = (1.0 - log(2.0)) / (M_PI*M_PI);
    double expf = exp(-fc/gamma);
    double A = beta/gamma / (expf - 1.0), dA_dfc = A*A*expf/beta;
    double D = 1.0 + A*Q + A*A*Q*Q;
    double P = Q*(1.0 + A*Q)/D, dP_dQ = (1.0 + 2.0*A*Q)/(D*D), dP_dA = -Q*Q*Q*A*(2.0 + A*Q)/(D*D);
    double dH_dP = beta / (1.0 + beta/gamma*P);
    *H = gamma * log(1.0 + beta/gamma*P);
    *dH_dfc = dH_dP * dP_dA * dA_dfc;
    *dH_dQ  = dH_dP * dP_dQ;
}


/* ---- KDT16 exchange: steps 1-6, 12 ---- */
void kdt16x(int DMnd, double *rho, double *sigma, double T, double *ex, double *vx, double *v2x) {
    const double Cx = -0.75 * cbrt(3.0/M_PI);                    // eps_x^LDA = Cx n^(1/3)
    const double Cs = 1.0 / (4.0 * pow(3.0*M_PI*M_PI, 2.0/3.0)); // s^2 = Cs sigma / n^(8/3)
    for (int i = 0; i < DMnd; i++) {
        double n = rho[i], n13 = cbrt(n), n83 = n*n*n13*n13;
        double rs, t;
        ft_rs_t(n, fmax(T, KDT16_T_MIN), &rs, &t);              // step 1; T = max(T, 1e-6 K)
        double Ax, dAx_dt, Bx, dBx_dt;
        kdt16_Ax(t, &Ax, &dAx_dt);                               // Eq. 4
        kdt16_Bx(t, &Bx, &dBx_dt);                               // Eq. 6
        double epsx = Cx*n13;
        double s2 = Cs*sigma[i]/n83;
        double R = Bx/Ax, dR_dt = (dBx_dt*Ax - Bx*dAx_dt)/(Ax*Ax);
        double s2x = s2*R;                                       // Eq. 7
        double dFx_ds2x, Fx = kdt16_Fx(s2x, &dFx_ds2x);          // Eq. 10
        ex[i] = epsx*Ax*Fx;                                      // Eq. 8
        // n d/dn at fixed sigma, T:  n d(eps_x)/dn = eps_x/3,  n dt/dn = -2t/3,  n ds^2/dn = -8 s^2/3
        vx[i] = (4.0/3.0)*ex[i] + epsx*Fx*dAx_dt*(-2.0*t/3.0)
              + epsx*Ax*dFx_ds2x*s2*(-8.0/3.0*R - 2.0*t/3.0*dR_dt);
        v2x[i] = 2.0*n*epsx*Ax*dFx_ds2x*R*Cs/n83;                // 2 d(n f_x)/d(sigma)
    }
}


/* ---- KDT16 correlation: steps 1, 7-12 ---- */
void kdt16c(int DMnd, double *rho, double *sigma, double T, double *ec, double *vc, double *v2c) {
    const double Cq = 1.0 / (16.0 * cbrt(3.0/M_PI));             // q^2 = Cq sigma / n^(7/3)
    for (int i = 0; i < DMnd; i++) {
        double n = rho[i], n73 = n*n*cbrt(n);
        double rs, t;
        ft_rs_t(n, fmax(T, KDT16_T_MIN), &rs, &t);              // step 1; T = max(T, 1e-6 K)
        double fc, dfc_drs, dfc_dt;
        kdt16_fc_lda(rs, t, &fc, &dfc_drs, &dfc_dt);             // SM S8
        double Bc, dBc_drs, dBc_dt;
        kdt16_Bc(rs, t, &Bc, &dBc_drs, &dBc_dt);                 // SM S2
        double q2 = Cq*sigma[i]/n73, Q = q2*Bc;                  // Eq. 11: Q = q_c^2
        double H, dH_dfc, dH_dQ;
        kdt16_H(fc, Q, &H, &dH_dfc, &dH_dQ);                     // SM S4-S5
        ec[i] = fc + H;                                          // Eq. 12
        // n d/dn at fixed sigma, T:  n dr_s/dn = -r_s/3,  n dt/dn = -2t/3,  n dq^2/dn = -7 q^2/3
        double n_dfc = -rs/3.0*dfc_drs - 2.0*t/3.0*dfc_dt;
        double n_dQ  = q2*(-7.0/3.0*Bc - rs/3.0*dBc_drs - 2.0*t/3.0*dBc_dt);
        vc[i] = ec[i] + (1.0 + dH_dfc)*n_dfc + dH_dQ*n_dQ;
        v2c[i] = 2.0*n*dH_dQ*Bc*Cq/n73;                          // 2 d(n f_c)/d(sigma)
    }
}


/* ---- KDT16 XC entropy term: step 13, -T s_xc = t d(f_x + f_c)/dt at fixed n, sigma ---- */
void kdt16_TdfdT(int DMnd, double *rho, double *sigma, double T, double *tdfdt) {
    const double Cx = -0.75 * cbrt(3.0/M_PI);                    // eps_x^LDA = Cx n^(1/3)
    const double Cs = 1.0 / (4.0 * pow(3.0*M_PI*M_PI, 2.0/3.0)); // s^2 = Cs sigma / n^(8/3)
    const double Cq = 1.0 / (16.0 * cbrt(3.0/M_PI));             // q^2 = Cq sigma / n^(7/3)
    for (int i = 0; i < DMnd; i++) {
        double n = rho[i], n13 = cbrt(n), rs, t;
        ft_rs_t(n, fmax(T, KDT16_T_MIN), &rs, &t);              // step 1; T = max(T, 1e-6 K)
        double Ax, dAx_dt, Bx, dBx_dt, dFx_ds2x;
        kdt16_Ax(t, &Ax, &dAx_dt);                               // Eq. 4
        kdt16_Bx(t, &Bx, &dBx_dt);                               // Eq. 6
        double s2 = Cs*sigma[i]/(n*n*n13*n13), R = Bx/Ax, dR_dt = (dBx_dt*Ax - Bx*dAx_dt)/(Ax*Ax);
        double Fx = kdt16_Fx(s2*R, &dFx_ds2x);                   // Eqs. 7, 10
        double dfx_dt = Cx*n13*(dAx_dt*Fx + Ax*dFx_ds2x*s2*dR_dt);          // Eq. 8
        double fc, dfc_drs, dfc_dt, Bc, dBc_drs, dBc_dt, H, dH_dfc, dH_dQ;
        kdt16_fc_lda(rs, t, &fc, &dfc_drs, &dfc_dt);             // SM S8
        kdt16_Bc(rs, t, &Bc, &dBc_drs, &dBc_dt);                 // SM S2
        double q2 = Cq*sigma[i]/(n*n*n13);
        kdt16_H(fc, q2*Bc, &H, &dH_dfc, &dH_dQ);                 // Eq. 11, SM S4-S5
        double dfc_tot_dt = (1.0 + dH_dfc)*dfc_dt + dH_dQ*q2*dBc_dt;         // Eq. 12
        tdfdt[i] = t*(dfx_dt + dfc_tot_dt);
    }
}
