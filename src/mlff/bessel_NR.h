#ifndef BESSEL_NR_
#define BESSEL_NR_


/*
Returns the Bessel function J0(x) for any real x.
*/
double bessj0(double x);


/*
Returns the Bessel function Y0(x) for positive x.
*/
double bessy0(double x);

/*
Returns the Bessel function J1(x) for any real x.
*/
double bessj1(double x);

/*
Returns the Bessel function Y1(x) for positive x.
*/
double bessy1(double x);

/*
Returns the Bessel function Yn(x) for positive x and n ≥ 2.
*/
double bessy(int n, double x);


/*
Returns the Bessel function Jn(x) for any real x and n ≥ 2.
*/
double bessj(int n, double x);

/*
Returns the modified Bessel function I0(x) for any real x.
*/
double bessi0(double x);


/*
Returns the modified Bessel function K0(x) for positive real x.
*/
double bessk0(double x);


/*
Returns the modified Bessel function I1(x) for any real x.
*/
double bessi1(double x);

/*
Returns the modified Bessel function K1(x) for positive real x
*/
double bessk1(double x);

/*
Returns the modified Bessel function Kn(x) for positive x and n ≥ 2.
*/
double bessk(int n, double x);

/*
Returns the modified Bessel function In(x) for any real x and n ≥ 2.
*/
double bessi(int n, double x);

double chebev(double a, double b, double c[], int m, double x);


/*Evaluates Gamma1 and Gamm2 by Chebyshev expansion for |x| <= 1/2. Also returns 1/Gamma(1 + x) and
1/Gamm(1 − x). If converting to double precision, set NUSE1 = 7, NUSE2 = 8.
*/
void beschb(double x, double *gam1, double *gam2, double *gampl, double *gammi);


/*
Returns the Bessel functions rj = Jν, ry = Yν and their derivatives rjp = Jv', ryp = Yν'' , for positive x and for xnu = ν ≥ 0. The relative accuracy is within one or two significant digits
of EPS, except near a zero of one of the functions, where EPS controls its absolute accuracy.
FPMIN is a number close to the machine’s smallest doubleing-point number. All internal arithmetic
is in double precision. To convert the entire routine to double precision, change the double
declarations above to double and decrease EPS to 10−16. Also convert the function beschb
*/
void bessjy(double x, double xnu, double *rj, double *ry, double *rjp, double *ryp);


/*
Returns the modified Bessel functions ri = Iν, rk = Kν and their derivatives rip = Iv, rkp = Kν,
 for positive x and for xnu = ν ≥ 0. The relative accuracy is within one or two
significant digits of EPS. FPMIN is a number close to the machine’s smallest doubleing-point
number. All internal arithmetic is in double precision. To convert the entire routine to double
precision, change the double declarations above to double and decrease EPS to 10−16. Also
convert the function beschb.
*/
void bessik(double x, double xnu, double *ri, double *rk, double *rip, double *rkp);


/*
Returns spherical Bessel functions jn(x), yn(x), and their derivatives j'
n(x), y' n(x) for integer n
*/

void sphbes(int n, double x, double *sj, double *sy, double *sjp, double *syp);

void sphbes_mod(int n, double x, double *sj, double *sy, double *sjp, double *syp);

void roots_sph_bessel(int order, double *roots);
#endif