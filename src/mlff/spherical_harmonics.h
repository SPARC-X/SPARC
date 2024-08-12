#ifndef SPHERICAL_HARMONICS_H
#define SPHERICAL_HARMONICS_H

/*
computeP function calculates the normalized associated Legendre's polynomial. The normalization is done such that
the corresponding spherical harmonics functions are orthonormal. 

[Input]
1. L: maximum 'l' index (orbital angular momentum number)
2. A,B: coefficients for the recursion precomputed
3. x: cos(theta)
[Output]
1. P: pointer to the array containing the Legendre polynomial. Values for only m>0 are calculated.
*/

void computeP ( const size_t L ,
			const double * const A , const double * const B ,
			double * const P , const double x , const double sintheta);

/*
computeY function calculates the spherical harmonics. 

[Input]
1. L: maximum 'l' index (orbital angular momentum number)
2. P: pointer to the array containing the Legendre polynomial. Values for only m>0 are stored.
3. phi
[Output]
1. Y: pointer to the array containing the Spherical harmonics for all combinations 0 <= l <= L and -l <=m <= l. 
*/


double computeP_Ylim_theta_zero( const size_t L ,
			const double * const A , const double * const B);


void computeY( const size_t L , const double * const P ,
	double complex * const Y ,  const double phi );



/*
sph_harmonics function calculates the spherical harmonics and its derivatives with respect to theta and phi. 

[Input]
1. theta, phi: spherical coordinate of the point on sphere
2. LL: maximum 'l' index (orbital angular momentum number)
[Output]
1. Y: pointer to the array containing the Spherical harmonics for all combinations 0 <= l <= L and -l <=m <= l. 
1. dY_theta: pointer to the array containing the derrivative of  Spherical harmonics w.r.t theta for all combinations 0 <= l <= L and -l <=m <= l. 
1. dY_phi: pointer to the array containing the derrivative of  Spherical harmonics w.r.t phi for all combinations 0 <= l <= L and -l <=m <= l. 
*/

void sph_harmonics(const double theta, const double phi, const int LL,
				double complex * const Y, double complex * const dY_theta,
				double complex * const dY_phi );

#endif
