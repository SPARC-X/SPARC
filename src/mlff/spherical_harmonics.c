/* 
Implementation of Spherical Harmonics and its derivatives with respect to Theta
and Phi as implemented in the book "Numerical recipes, 3rd ed" and in the following paper:
https://arxiv.org/abs/1410.1748v1
*/

#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <time.h>

#include "spherical_harmonics.h"
#include "tools.h"

#define PT(l1, m1) ((m1) +((l1) *((l1) +1))/2)
#define YR(l2 , m2 ) (( m2 ) +( l2 ) +(( l2 ) *( l2) ) )
#define temp_tol 1.0E-10

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

void computeP( const size_t L ,
			const double * const A , const double * const B ,
			double * const P , const double x, const double sintheta ) {

	// const double sintheta = sin (theta) ;
	double temp = 0.282094791773878 ; // = sqrt (1/ 4 M_PI )
	P[ PT(0,0) ] = temp ;
	if ( L > 0) {
		const double SQRT3 = 1.732050807568877;
		P[ PT(1,0) ] = x * SQRT3 * temp ;
		const double SQRT3DIV2 = -1.2247448713915890491;
		temp = SQRT3DIV2 * sintheta * temp ;
		P[ PT(1,1) ] = temp ;

		for ( size_t l =2; l <= L ; l ++) {
			for ( size_t m =0; m <l -1; m ++) {
				P[ PT(l,m) ] = A[ PT(l,m) ]*( x * P[ PT(l-1,m) ]
				+ B[ PT(l,m) ]* P[ PT(l-2,m) ]) ;
			}
			P[ PT(l,l-1) ] = x * sqrt (2*( l-1) +3) * temp ;
			temp = - sqrt(1.0+0.5/l) * sintheta * temp ;
			P[ PT(l,l) ] = temp ;
		}
	}
}


double computeP_Ylim_theta_zero( const size_t L ,
			const double * const A , const double * const B) {

	int memsize_A_B_P = ((L+1)*(L+2))/2;
	double *P = (double *) calloc(sizeof(double), memsize_A_B_P);

	const double sintheta = 1.0 ;
	double temp = 0.282094791773878 ; // = sqrt (1/ 4 M_PI )
	P[ PT(0,0) ] = temp ;
	if ( L > 0) {
		const double SQRT3 = 1.732050807568877;
		P[ PT(1,0) ] =SQRT3 * temp ;
		const double SQRT3DIV2 = -1.2247448713915890491;

		// double sintheta = 1.0;

		temp = SQRT3DIV2 * sintheta * temp ;
		P[ PT(1,1) ] = temp ;

		for ( size_t l =2; l <= L ; l ++) {
			int m=1;
			P[ PT(l,m) ] = A[ PT(l,m) ]*(P[ PT(l-1,m) ]
			+ B[ PT(l,m) ]* P[ PT(l-2,m) ]) ;

			P[ PT(l,l-1) ] = sqrt (2*( l-1) +3) * temp ;
			temp = - sqrt(1.0+0.5/l) * sintheta * temp ;
			P[ PT(l,l) ] = temp ;
		}
	}

	double val = P[ PT(L,1) ];

	free(P);

	return val;

}



/*
computeY function calculates the spherical harmonics. 

[Input]
1. L: maximum 'l' index (orbital angular momentum number)
2. P: pointer to the array containing the Legendre polynomial. Values for only m>0 are stored.
3. phi
[Output]
1. Y: pointer to the array containing the Spherical harmonics for all combinations 0 <= l <= L and -l <=m <= l. 
*/

void computeY( const size_t L , const double * const P ,
	double complex * const Y ,  const double phi ) {

	for ( size_t l =0; l <= L ; l ++){
		Y[ YR(l,0) ] = P[ PT(l, 0) ]  ;
	}
	
	double complex temp1, temp2;
	double c1 = 1.0 , c2 = cos ( phi ) ;
	double s1 = 0.0 , s2 = - sin ( phi ) ;
	double tc = 2.0 * c2 ;
	for ( size_t m =1; m <= L ; m ++) {
		double s = tc * s1 - s2 ;
		double c = tc * c1 - c2 ;
		s2 = s1 ; s1 = s ; c2 = c1 ; c1 = c ;
		for ( size_t l = m ; l <= L ; l ++) {
			
			temp1 = P[ PT(l,m) ] * c + P[ PT(l,m) ] * s * I ;
			if (m%2 == 0){
				temp2 =  P[ PT(l,m) ] * c - P[ PT(l,m) ] * s * I ;
			} else {
				temp2 =  -P[ PT(l,m) ] * c + P[ PT(l,m) ] * s * I ;
			}
			
			Y[ YR(l, m) ] = temp1 ;
			Y[ YR(l, -m) ] = temp2 ;

		}
	}
}

/*
sph_harmonics function calculates the spherical harmonics and its derivatives with respect to theta and phi. 

[Input]
1. theta, phi: spherical coordinate of the point on sphere [ theta \in [0 \pi], phi \in [0 2\pi] ]
2. LL: maximum 'l' index (orbital angular momentum number)
[Output]
1. Y: pointer to the array containing the Spherical harmonics for all combinations 0 <= l <= L and -l <=m <= l. 
2. dY_theta: pointer to the array containing the derrivative of  Spherical harmonics w.r.t theta for all combinations 0 <= l <= L and -l <=m <= l. 
3. dY_phi: pointer to the array containing the derrivative of  Spherical harmonics w.r.t phi for all combinations 0 <= l <= L and -l <=m <= l. 
*/

void sph_harmonics(const double theta, const double phi, const int LL,
				double complex * const Y, double complex * const dY_theta,
				double complex * const dY_phi ) {
  	
	int memsize_A_B_P, memsize_Y;
	memsize_A_B_P = ((LL+1)*(LL+2))/2;

	double A[memsize_A_B_P], B[memsize_A_B_P];
	double *P;
	
	
	memsize_Y = (LL+1)*(LL+1) ;

	P = (double *) malloc(sizeof(double)*memsize_A_B_P);
	for ( size_t l =2; l <= LL ; l++) {
		double ls = l *l , lm1s = (l -1) *( l -1) ;
		for ( size_t m =0; m <l -1; m ++) {
			double ms = m * m ;
			A[ PT(l, m) ] = sqrt ((4* ls -1.0) /( ls - ms ) ) ;
			B[ PT(l, m) ] = - sqrt (( lm1s - ms ) /(4* lm1s -1.0) ) ;
		}

	}
	
	double sintheta = sin(theta); // cos function not accurate below 1e-8 gives 1
	double x = cos(theta);  // to ensure correct sign for theta = 0 and pi
	computeP( LL , &A[0] , &B[0] , P , x, sintheta );
	computeY( LL , P , Y ,  phi );

	

	
	double constant;
	constant=1;
	if (theta<0.0) constant = -1;
	double theta_abs;
	theta_abs = fabs(theta);
	dY_phi[ YR(0, 0) ] = 0.0 ;
	dY_theta[ YR(0, 0) ] = 0.0;

	// double theta_cut = 1.0E-14;
	double c = cos(theta_abs), s = sin(theta_abs), ct=0.0;
	double complex emph = cos(phi) - sin(phi) *I;
	if (fabs(theta_abs) > temp_tol){
		ct = c/s;
	}
	
	for ( int l =1; l <= LL ; l++) {
		for ( int m =-l; m <=l; m++) {
			dY_phi[ YR(l, m) ] = Y[YR(l, m)] * m * I;
			if (fabs(theta_abs) > temp_tol && fabs(theta_abs - M_PI) > temp_tol){
				if (m<l){
					dY_theta[ YR(l, m) ] = constant*(m*ct*Y[ YR(l,m) ] + sqrt((l-m)*(l+m+1)) * emph * Y[ YR(l,m+1)]);		
				} else {
					dY_theta[ YR(l, m) ] = constant*(m*ct*Y[ YR(l,m) ]);
				}
			} else {
				// emph = 1.0 - 1.0 *I;
				if (abs(m) == 1){
					// Y[YR(l, m)] = 0.0 + 0.0 *I;
					dY_theta[ YR(l, 1) ] = computeP_Ylim_theta_zero(l, &A[0], &B[0]) *(1.0+0.0*I);
					// printf("dY_theta[ YR(%d, 1) ]: %.10f + %.10f i\n", l, creal(dY_theta[ YR(l, 1) ] ), cimag(dY_theta[ YR(l, 1) ]));
					dY_theta[ YR(l, -1) ] = -conj(dY_theta[ YR(l, 1) ]);
				}else {
					dY_theta[ YR(l, m) ] = 0.0 + 0.0 *I; 
				}
				
			}
			
		}
	} 
	free(P);
}

// int main(){
// 	double theta, phi;
// 	double complex *Y, *dY_th, *dY_phi;
// 	int Lmax, memsize_Y;
	
	
// 	Lmax=2;
// 	theta=1.2;
// 	phi = 2.1;
// 	memsize_Y = (Lmax+1)*(Lmax+1) ;
// 	Y = (double complex *) malloc(sizeof(double complex)*memsize_Y);
// 	dY_phi = (double complex *) malloc(sizeof(double complex)*memsize_Y);
// 	dY_th = (double complex *) malloc(sizeof(double complex)*memsize_Y);
	
	
// 	sph_harmonics(theta, phi, Lmax, Y, dY_th, dY_phi);
	
	
	
// 	for (int l=0; l <= Lmax; l++){
// 		for (int m=-l; m<=l; m++){
// 			printf("Y(%d,%d,%f,%f): %f + %f i,\n dY_theta(%d,%d,%f,%f): %f + %f i,\n dY_phi(%d,%d,%f,%f): %f + %f i\n",
// 				  l,m,theta,phi, creal(Y[YR(l,m)]), cimag(Y[YR(l,m)]),  l,m,theta,phi, creal(dY_th[YR(l,m)]), cimag(dY_th[YR(l,m)]),
// 				  l,m,theta,phi, creal(dY_phi[YR(l,m)]), cimag(dY_phi[YR(l,m)]));
// 				  printf("\n");
// 		}
// 	}
	
	
	
				  
// 	free(Y);
// 	free(dY_phi);
// 	free(dY_th);		  
// 	return 0;
// }
