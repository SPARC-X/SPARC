#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <time.h>
#include <mpi.h>

#define ACC 40.0
#define BIGNO 1.0e10
#define BIGNI 1.0e-10

#define EPS 1.0e-10
#define FPMIN 1.0e-30
#define MAXIT 10000
#define XMIN 2.0
#define PI 3.141592653589793
#define RTPIO2 1.2533141

#define NUSE1 5
#define NUSE2 5

#define NRANSI
#include "nrutils.h"

// Taken from https://www.astro.umd.edu/~ricotti/NEWWEB/teaching/ASTR415/InClassExamples/NR3/legacy/nr2/C_211/recipes/ 

/*
Returns the Bessel function J0(x) for any real x.
*/
double bessj0(double x)
{
	double ax,z;
	double xx,y,ans,ans1,ans2;

	if ((ax=fabs(x)) < 8.0) {
		y=x*x;
		ans1=57568490574.0+y*(-13362590354.0+y*(651619640.7
			+y*(-11214424.18+y*(77392.33017+y*(-184.9052456)))));
		ans2=57568490411.0+y*(1029532985.0+y*(9494680.718
			+y*(59272.64853+y*(267.8532712+y*1.0))));
		ans=ans1/ans2;
	} else {
		z=8.0/ax;
		y=z*z;
		xx=ax-0.785398164;
		ans1=1.0+y*(-0.1098628627e-2+y*(0.2734510407e-4
			+y*(-0.2073370639e-5+y*0.2093887211e-6)));
		ans2 = -0.1562499995e-1+y*(0.1430488765e-3
			+y*(-0.6911147651e-5+y*(0.7621095161e-6
			-y*0.934945152e-7)));
		ans=sqrt(0.636619772/ax)*(cos(xx)*ans1-z*sin(xx)*ans2);
	}
	return ans;
}

/*
Returns the Bessel function Y0(x) for positive x.
*/
double bessy0(double x)
{
	double z;
	double xx,y,ans,ans1,ans2;

	if (x < 8.0) {
		y=x*x;
		ans1 = -2957821389.0+y*(7062834065.0+y*(-512359803.6
			+y*(10879881.29+y*(-86327.92757+y*228.4622733))));
		ans2=40076544269.0+y*(745249964.8+y*(7189466.438
			+y*(47447.26470+y*(226.1030244+y*1.0))));
		ans=(ans1/ans2)+0.636619772*bessj0(x)*log(x);
	} else {
		z=8.0/x;
		y=z*z;
		xx=x-0.785398164;
		ans1=1.0+y*(-0.1098628627e-2+y*(0.2734510407e-4
			+y*(-0.2073370639e-5+y*0.2093887211e-6)));
		ans2 = -0.1562499995e-1+y*(0.1430488765e-3
			+y*(-0.6911147651e-5+y*(0.7621095161e-6
			+y*(-0.934945152e-7))));
		ans=sqrt(0.636619772/x)*(sin(xx)*ans1+z*cos(xx)*ans2);
	}
	return ans;
}

/*
Returns the Bessel function J1(x) for any real x.
*/
double bessj1(double x)
{
	double ax,z;
	double xx,y,ans,ans1,ans2;

	if ((ax=fabs(x)) < 8.0) {
		y=x*x;
		ans1=x*(72362614232.0+y*(-7895059235.0+y*(242396853.1
			+y*(-2972611.439+y*(15704.48260+y*(-30.16036606))))));
		ans2=144725228442.0+y*(2300535178.0+y*(18583304.74
			+y*(99447.43394+y*(376.9991397+y*1.0))));
		ans=ans1/ans2;
	} else {
		z=8.0/ax;
		y=z*z;
		xx=ax-2.356194491;
		ans1=1.0+y*(0.183105e-2+y*(-0.3516396496e-4
			+y*(0.2457520174e-5+y*(-0.240337019e-6))));
		ans2=0.04687499995+y*(-0.2002690873e-3
			+y*(0.8449199096e-5+y*(-0.88228987e-6
			+y*0.105787412e-6)));
		ans=sqrt(0.636619772/ax)*(cos(xx)*ans1-z*sin(xx)*ans2);
		if (x < 0.0) ans = -ans;
	}
	return ans;
}

/*
Returns the Bessel function Y1(x) for positive x.
*/
double bessy1(double x)
{
	double z;
	double xx,y,ans,ans1,ans2;

	if (x < 8.0) {
		y=x*x;
		ans1=x*(-0.4900604943e13+y*(0.1275274390e13
			+y*(-0.5153438139e11+y*(0.7349264551e9
			+y*(-0.4237922726e7+y*0.8511937935e4)))));
		ans2=0.2499580570e14+y*(0.4244419664e12
			+y*(0.3733650367e10+y*(0.2245904002e8
			+y*(0.1020426050e6+y*(0.3549632885e3+y)))));
		ans=(ans1/ans2)+0.636619772*(bessj1(x)*log(x)-1.0/x);
	} else {
		z=8.0/x;
		y=z*z;
		xx=x-2.356194491;
		ans1=1.0+y*(0.183105e-2+y*(-0.3516396496e-4
			+y*(0.2457520174e-5+y*(-0.240337019e-6))));
		ans2=0.04687499995+y*(-0.2002690873e-3
			+y*(0.8449199096e-5+y*(-0.88228987e-6
			+y*0.105787412e-6)));
		ans=sqrt(0.636619772/x)*(sin(xx)*ans1+z*cos(xx)*ans2);
	}
	return ans;
}

/*
Returns the Bessel function Yn(x) for positive x and n ≥ 2.
*/
double bessy(int n, double x)
{
	int j;
	double by,bym,byp,tox;

	if (n < 2) nrerror("Index n less than 2 in bessy");
	tox=2.0/x;
	by=bessy1(x);
	bym=bessy0(x);
	for (j=1;j<n;j++) {
		byp=j*tox*by-bym;
		bym=by;
		by=byp;
	}
	return by;
}

/*
Returns the Bessel function Jn(x) for any real x and n ≥ 2.
*/
double bessj(int n, double x)
{
	int j,jsum,m;
	double ax,bj,bjm,bjp,sum,tox,ans;

	if (n < 2) nrerror("Index n less than 2 in bessj");
	ax=fabs(x);
	if (ax == 0.0)
		return 0.0;
	else if (ax > (double) n) {
		tox=2.0/ax;
		bjm=bessj0(ax);
		bj=bessj1(ax);
		for (j=1;j<n;j++) {
			bjp=j*tox*bj-bjm;
			bjm=bj;
			bj=bjp;
		}
		ans=bj;
	} else {
		tox=2.0/ax;
		m=2*((n+(int) sqrt(ACC*n))/2);
		jsum=0;
		bjp=ans=sum=0.0;
		bj=1.0;
		for (j=m;j>0;j--) {
			bjm=j*tox*bj-bjp;
			bjp=bj;
			bj=bjm;
			if (fabs(bj) > BIGNO) {
				bj *= BIGNI;
				bjp *= BIGNI;
				ans *= BIGNI;
				sum *= BIGNI;
			}
			if (jsum) sum += bj;
			jsum=!jsum;
			if (j == n) ans=bjp;
		}
		sum=2.0*sum-bj;
		ans /= sum;
	}
	return x < 0.0 && (n & 1) ? -ans : ans;
}

/*
Returns the modified Bessel function I0(x) for any real x.
*/
double bessi0(double x)
{
	double ax,ans;
	double y;

	if ((ax=fabs(x)) < 3.75) {
		y=x/3.75;
		y*=y;
		ans=1.0+y*(3.5156229+y*(3.0899424+y*(1.2067492
			+y*(0.2659732+y*(0.360768e-1+y*0.45813e-2)))));
	} else {
		y=3.75/ax;
		ans=(exp(ax)/sqrt(ax))*(0.39894228+y*(0.1328592e-1
			+y*(0.225319e-2+y*(-0.157565e-2+y*(0.916281e-2
			+y*(-0.2057706e-1+y*(0.2635537e-1+y*(-0.1647633e-1
			+y*0.392377e-2))))))));
	}
	return ans;
}

/*
Returns the modified Bessel function K0(x) for positive real x.
*/
double bessk0(double x)
{
	double y,ans;

	if (x <= 2.0) {
		y=x*x/4.0;
		ans=(-log(x/2.0)*bessi0(x))+(-0.57721566+y*(0.42278420
			+y*(0.23069756+y*(0.3488590e-1+y*(0.262698e-2
			+y*(0.10750e-3+y*0.74e-5))))));
	} else {
		y=2.0/x;
		ans=(exp(-x)/sqrt(x))*(1.25331414+y*(-0.7832358e-1
			+y*(0.2189568e-1+y*(-0.1062446e-1+y*(0.587872e-2
			+y*(-0.251540e-2+y*0.53208e-3))))));
	}
	return ans;
}

/*
Returns the modified Bessel function I1(x) for any real x.
*/
double bessi1(double x)
{
	double ax,ans;
	double y;

	if ((ax=fabs(x)) < 3.75) {
		y=x/3.75;
		y*=y;
		ans=ax*(0.5+y*(0.87890594+y*(0.51498869+y*(0.15084934
			+y*(0.2658733e-1+y*(0.301532e-2+y*0.32411e-3))))));
	} else {
		y=3.75/ax;
		ans=0.2282967e-1+y*(-0.2895312e-1+y*(0.1787654e-1
			-y*0.420059e-2));
		ans=0.39894228+y*(-0.3988024e-1+y*(-0.362018e-2
			+y*(0.163801e-2+y*(-0.1031555e-1+y*ans))));
		ans *= (exp(ax)/sqrt(ax));
	}
	return x < 0.0 ? -ans : ans;
}


/*
Returns the modified Bessel function K1(x) for positive real x
*/
double bessk1(double x)
{
	double y,ans;

	if (x <= 2.0) {
		y=x*x/4.0;
		ans=(log(x/2.0)*bessi1(x))+(1.0/x)*(1.0+y*(0.15443144
			+y*(-0.67278579+y*(-0.18156897+y*(-0.1919402e-1
			+y*(-0.110404e-2+y*(-0.4686e-4)))))));
	} else {
		y=2.0/x;
		ans=(exp(-x)/sqrt(x))*(1.25331414+y*(0.23498619
			+y*(-0.3655620e-1+y*(0.1504268e-1+y*(-0.780353e-2
			+y*(0.325614e-2+y*(-0.68245e-3)))))));
	}
	return ans;
}


/*
Returns the modified Bessel function Kn(x) for positive x and n ≥ 2.
*/
double bessk(int n, double x)
{
	int j;
	double bk,bkm,bkp,tox;

	if (n < 2) nrerror("Index n less than 2 in bessk");
	tox=2.0/x;
	bkm=bessk0(x);
	bk=bessk1(x);
	for (j=1;j<n;j++) {
		bkp=bkm+j*tox*bk;
		bkm=bk;
		bk=bkp;
	}
	return bk;
}

/*
Returns the modified Bessel function In(x) for any real x and n ≥ 2.
*/
double bessi(int n, double x)
{
	int j;
	double bi,bim,bip,tox,ans;

	if (n < 2) nrerror("Index n less than 2 in bessi");
	if (x == 0.0)
		return 0.0;
	else {
		tox=2.0/fabs(x);
		bip=ans=0.0;
		bi=1.0;
		for (j=2*(n+(int) sqrt(ACC*n));j>0;j--) {
			bim=bip+j*tox*bi;
			bip=bi;
			bi=bim;
			if (fabs(bi) > BIGNO) {
				ans *= BIGNI;
				bi *= BIGNI;
				bip *= BIGNI;
			}
			if (j == n) ans=bip;
		}
		ans *= bessi0(x)/bi;
		return x < 0.0 && (n & 1) ? -ans : ans;
	}
}

double chebev(double a, double b, double c[], int m, double x)
{
	void nrerror(char error_text[]);
	double d=0.0,dd=0.0,sv,y,y2;
	int j;

	if ((x-a)*(x-b) > 0.0) nrerror("x not in range in routine chebev");
	y2=2.0*(y=(2.0*x-a-b)/(b-a));
	for (j=m-1;j>=1;j--) {
		sv=d;
		d=y2*d-dd+c[j];
		dd=sv;
	}
	return y*d-dd+0.5*c[0];
}

/*Evaluates Gamma1 and Gamm2 by Chebyshev expansion for |x| <= 1/2. Also returns 1/Gamma(1 + x) and
1/Gamm(1 − x). If converting to double precision, set NUSE1 = 7, NUSE2 = 8.
*/
void beschb(double x, double *gam1, double *gam2, double *gampl, double *gammi)
{
	double chebev(double a, double b, double c[], int m, double x);
	double xx;
	static double c1[] = {
		-1.142022680371168e0,6.5165112670737e-3,
		3.087090173086e-4,-3.4706269649e-6,6.9437664e-9,
		3.67795e-11,-1.356e-13};
	static double c2[] = {
		1.843740587300905e0,-7.68528408447867e-2,
		1.2719271366546e-3,-4.9717367042e-6,-3.31261198e-8,
		2.423096e-10,-1.702e-13,-1.49e-15};

	xx=8.0*x*x-1.0;
	*gam1=chebev(-1.0,1.0,c1,NUSE1,xx);
	*gam2=chebev(-1.0,1.0,c2,NUSE2,xx);
	*gampl= *gam2-x*(*gam1);
	*gammi= *gam2+x*(*gam1);
}

/*
Returns the Bessel functions rj = Jν, ry = Yν and their derivatives rjp = Jv', ryp = Yν'' , for positive x and for xnu = ν ≥ 0. The relative accuracy is within one or two significant digits
of EPS, except near a zero of one of the functions, where EPS controls its absolute accuracy.
FPMIN is a number close to the machine’s smallest doubleing-point number. All internal arithmetic
is in double precision. To convert the entire routine to double precision, change the double
declarations above to double and decrease EPS to 10−16. Also convert the function beschb
*/
void bessjy(double x, double xnu, double *rj, double *ry, double *rjp, double *ryp)
{

	int i,isign,l,nl;
	double a,b,br,bi,c,cr,ci,d,del,del1,den,di,dlr,dli,dr,e,f,fact,fact2,
		fact3,ff,gam,gam1,gam2,gammi,gampl,h,p,pimu,pimu2,q,r,rjl,
		rjl1,rjmu,rjp1,rjpl,rjtemp,ry1,rymu,rymup,rytemp,sum,sum1,
		temp,w,x2,xi,xi2,xmu,xmu2;

	if (x <= 0.0 || xnu < 0.0) nrerror("bad arguments in bessjy");
	nl=(x < XMIN ? (int)(xnu+0.5) : IMAX(0,(int)(xnu-x+1.5)));
	xmu=xnu-nl;
	xmu2=xmu*xmu;
	xi=1.0/x;
	xi2=2.0*xi;
	w=xi2/PI;
	isign=1;
	h=xnu*xi;
	if (h < FPMIN) h=FPMIN;
	b=xi2*xnu;
	d=0.0;
	c=h;
	for (i=1;i<=MAXIT;i++) {
		b += xi2;
		d=b-d;
		if (fabs(d) < FPMIN) d=FPMIN;
		c=b-1.0/c;
		if (fabs(c) < FPMIN) c=FPMIN;
		d=1.0/d;
		del=c*d;
		h=del*h;
		if (d < 0.0) isign = -isign;
		if (fabs(del-1.0) < EPS) break;
	}
	if (i > MAXIT) nrerror("x too large in bessjy; try asymptotic expansion");
	rjl=isign*FPMIN;
	rjpl=h*rjl;
	rjl1=rjl;
	rjp1=rjpl;
	fact=xnu*xi;
	for (l=nl;l>=1;l--) {
		rjtemp=fact*rjl+rjpl;
		fact -= xi;
		rjpl=fact*rjtemp-rjl;
		rjl=rjtemp;
	}
	if (rjl == 0.0) rjl=EPS;
	f=rjpl/rjl;
	if (x < XMIN) {
		x2=0.5*x;
		pimu=PI*xmu;
		fact = (fabs(pimu) < EPS ? 1.0 : pimu/sin(pimu));
		d = -log(x2);
		e=xmu*d;
		fact2 = (fabs(e) < EPS ? 1.0 : sinh(e)/e);
		beschb(xmu,&gam1,&gam2,&gampl,&gammi);
		ff=2.0/PI*fact*(gam1*cosh(e)+gam2*fact2*d);
		e=exp(e);
		p=e/(gampl*PI);
		q=1.0/(e*PI*gammi);
		pimu2=0.5*pimu;
		fact3 = (fabs(pimu2) < EPS ? 1.0 : sin(pimu2)/pimu2);
		r=PI*pimu2*fact3*fact3;
		c=1.0;
		d = -x2*x2;
		sum=ff+r*q;
		sum1=p;
		for (i=1;i<=MAXIT;i++) {
			ff=(i*ff+p+q)/(i*i-xmu2);
			c *= (d/i);
			p /= (i-xmu);
			q /= (i+xmu);
			del=c*(ff+r*q);
			sum += del;
			del1=c*p-i*del;
			sum1 += del1;
			if (fabs(del) < (1.0+fabs(sum))*EPS) break;
		}
		if (i > MAXIT) nrerror("bessy series failed to converge");
		rymu = -sum;
		ry1 = -sum1*xi2;
		rymup=xmu*xi*rymu-ry1;
		rjmu=w/(rymup-f*rymu);
	} else {
		a=0.25-xmu2;
		p = -0.5*xi;
		q=1.0;
		br=2.0*x;
		bi=2.0;
		fact=a*xi/(p*p+q*q);
		cr=br+q*fact;
		ci=bi+p*fact;
		den=br*br+bi*bi;
		dr=br/den;
		di = -bi/den;
		dlr=cr*dr-ci*di;
		dli=cr*di+ci*dr;
		temp=p*dlr-q*dli;
		q=p*dli+q*dlr;
		p=temp;
		for (i=2;i<=MAXIT;i++) {
			a += 2*(i-1);
			bi += 2.0;
			dr=a*dr+br;
			di=a*di+bi;
			if (fabs(dr)+fabs(di) < FPMIN) dr=FPMIN;
			fact=a/(cr*cr+ci*ci);
			cr=br+cr*fact;
			ci=bi-ci*fact;
			if (fabs(cr)+fabs(ci) < FPMIN) cr=FPMIN;
			den=dr*dr+di*di;
			dr /= den;
			di /= -den;
			dlr=cr*dr-ci*di;
			dli=cr*di+ci*dr;
			temp=p*dlr-q*dli;
			q=p*dli+q*dlr;
			p=temp;
			if (fabs(dlr-1.0)+fabs(dli) < EPS) break;
		}
		if (i > MAXIT) nrerror("cf2 failed in bessjy");
		gam=(p-f)/q;
		rjmu=sqrt(w/((p-f)*gam+q));
		rjmu=SIGN(rjmu,rjl);
		rymu=rjmu*gam;
		rymup=rymu*(p+q/gam);
		ry1=xmu*xi*rymu-rymup;
	}
	fact=rjmu/rjl;
	*rj=rjl1*fact;
	*rjp=rjp1*fact;
	for (i=1;i<=nl;i++) {
		rytemp=(xmu+i)*xi2*ry1-rymu;
		rymu=ry1;
		ry1=rytemp;
	}
	*ry=rymu;
	*ryp=xnu*xi*rymu-ry1;
}


/*
Returns the modified Bessel functions ri = Iν, rk = Kν and their derivatives rip = Iv, rkp = Kν,
 for positive x and for xnu = ν ≥ 0. The relative accuracy is within one or two
significant digits of EPS. FPMIN is a number close to the machine’s smallest doubleing-point
number. All internal arithmetic is in double precision. To convert the entire routine to double
precision, change the double declarations above to double and decrease EPS to 10−16. Also
convert the function beschb.
*/
void bessik(double x, double xnu, double *ri, double *rk, double *rip, double *rkp)
{
	int i,l,nl;
	double a,a1,b,c,d,del,del1,delh,dels,e,f,fact,fact2,ff,gam1,gam2,
		gammi,gampl,h,p,pimu,q,q1,q2,qnew,ril,ril1,rimu,rip1,ripl,
		ritemp,rk1,rkmu,rkmup,rktemp,s,sum,sum1,x2,xi,xi2,xmu,xmu2;

	if (x <= 0.0 || xnu < 0.0) nrerror("bad arguments in bessik");
	nl=(int)(xnu+0.5);
	xmu=xnu-nl;
	xmu2=xmu*xmu;
	xi=1.0/x;
	xi2=2.0*xi;
	h=xnu*xi;
	if (h < FPMIN) h=FPMIN;
	b=xi2*xnu;
	d=0.0;
	c=h;
	for (i=1;i<=MAXIT;i++) {
		b += xi2;
		d=1.0/(b+d);
		c=b+1.0/c;
		del=c*d;
		h=del*h;
		if (fabs(del-1.0) < EPS) break;
	}
	if (i > MAXIT) nrerror("x too large in bessik; try asymptotic expansion");
	ril=FPMIN;
	ripl=h*ril;
	ril1=ril;
	rip1=ripl;
	fact=xnu*xi;
	for (l=nl;l>=1;l--) {
		ritemp=fact*ril+ripl;
		fact -= xi;
		ripl=fact*ritemp+ril;
		ril=ritemp;
	}
	f=ripl/ril;
	if (x < XMIN) {
		x2=0.5*x;
		pimu=PI*xmu;
		fact = (fabs(pimu) < EPS ? 1.0 : pimu/sin(pimu));
		d = -log(x2);
		e=xmu*d;
		fact2 = (fabs(e) < EPS ? 1.0 : sinh(e)/e);
		beschb(xmu,&gam1,&gam2,&gampl,&gammi);
		ff=fact*(gam1*cosh(e)+gam2*fact2*d);
		sum=ff;
		e=exp(e);
		p=0.5*e/gampl;
		q=0.5/(e*gammi);
		c=1.0;
		d=x2*x2;
		sum1=p;
		for (i=1;i<=MAXIT;i++) {
			ff=(i*ff+p+q)/(i*i-xmu2);
			c *= (d/i);
			p /= (i-xmu);
			q /= (i+xmu);
			del=c*ff;
			sum += del;
			del1=c*(p-i*ff);
			sum1 += del1;
			if (fabs(del) < fabs(sum)*EPS) break;
		}
		if (i > MAXIT) nrerror("bessk series failed to converge");
		rkmu=sum;
		rk1=sum1*xi2;
	} else {
		b=2.0*(1.0+x);
		d=1.0/b;
		h=delh=d;
		q1=0.0;
		q2=1.0;
		a1=0.25-xmu2;
		q=c=a1;
		a = -a1;
		s=1.0+q*delh;
		for (i=2;i<=MAXIT;i++) {
			a -= 2*(i-1);
			c = -a*c/i;
			qnew=(q1-b*q2)/a;
			q1=q2;
			q2=qnew;
			q += c*qnew;
			b += 2.0;
			d=1.0/(b+a*d);
			delh=(b*d-1.0)*delh;
			h += delh;
			dels=q*delh;
			s += dels;
			if (fabs(dels/s) < EPS) break;
		}
		if (i > MAXIT) nrerror("bessik: failure to converge in cf2");
		h=a1*h;
		rkmu=sqrt(PI/(2.0*x))*exp(-x)/s;
		rk1=rkmu*(xmu+x+0.5-h)*xi;
	}
	rkmup=xmu*xi*rkmu-rk1;
	rimu=xi/(f*rkmu-rkmup);
	*ri=(rimu*ril1)/ril;
	*rip=(rimu*rip1)/ril;
	for (i=1;i<=nl;i++) {
		rktemp=(xmu+i)*xi2*rk1+rkmu;
		rkmu=rk1;
		rk1=rktemp;
	}
	*rk=rkmu;
	*rkp=xnu*xi*rkmu-rk1;
}




/*
Returns spherical Bessel functions jn(x), yn(x), and their derivatives j'
n(x), y' n(x) for integer n
*/

void sphbes(int n, double x, double *sj, double *sy, double *sjp, double *syp)
{
	void bessjy(double x, double xnu, double *rj, double *ry, double *rjp,
		double *ryp);
	void nrerror(char error_text[]);
	double factor,order,rj,rjp,ry,ryp;

	if (n < 0 || x <= 0.0) nrerror("bad arguments in sphbes");
	order=n+0.5;
	bessjy(x,order,&rj,&ry,&rjp,&ryp);
	factor=RTPIO2/sqrt(x);
	*sj=factor*rj;
	*sy=factor*ry;
	*sjp=factor*rjp-(*sj)/(2.0*x);
	*syp=factor*ryp-(*sy)/(2.0*x);
}


/*
Returns mod spherical Bessel functions jn(x), yn(x), and their derivatives j'
n(x), y' n(x) for integer n
*/

void sphbes_mod(int n, double x, double *sj, double *sy, double *sjp, double *syp)
{

	void nrerror(char error_text[]);
	double factor,order,rj,rjp,ry,ryp;

	if (n < 0 || x <= 0.0) nrerror("bad arguments in sphbes");
	order=n+0.5;
	// bessjy(x,order,&rj,&ry,&rjp,&ryp);
	bessik(x,order,&rj,&ry,&rjp,&ryp);

	factor=RTPIO2/sqrt(x);
	*sj=factor*rj;
	*sy=factor*ry;
	*sjp=factor*rjp-(*sj)/(2.0*x);
	*syp=factor*ryp-(*sy)/(2.0*x);
}
 //TODO: Remove hardcoding here! Implement general function to find the roots
void roots_sph_bessel(int order, double *roots){

    double roots_temp1[30] = {3.141592653589793,6.283185307179586,9.424777960769379,12.566370614359172,15.707963267948966,18.849555921538759,21.991148575128552,25.132741228718345,28.274333882308138,31.415926535897931,34.557519189487728,37.699111843077517,40.840704496667314,43.982297150257104,47.123889803846900,50.265482457436690,53.407075111026487,56.548667764616276,59.690260418206073,62.831853071795862,65.973445725385659,69.115038378975456,72.256631032565238,75.398223686155035,78.539816339744831,81.681408993334628,84.823001646924411,87.964594300514207,91.106186954104004,94.247779607693801};
    double roots_temp2[30] = {4.493409457909064,7.725251836937707,10.904121659428899,14.066193912831473,17.220755271930766,20.371302959287561,23.519452498689006,26.666054258812675,29.811598790892958,32.956389039822476,36.100622244375607,39.244432361164193,42.387913568131921,45.531134013991277,48.674144231954386,51.816982487279667,54.959678287888934,58.102254754495590,61.244730260374403,64.387119590557418,67.529434777144118,70.671685711619503,73.813880600680648,76.956026310331183,80.098128628945119,83.240192470723400,86.382222034728713,89.524220930417187,92.666192277622841,95.808138786861704};
    double roots_temp3[30] = {5.763459196894550,9.095011330476355,12.322940970566583,15.514603010886749,18.689036355362823,21.853874222709766,25.012803202289611,28.167829707993622,31.320141707447174,34.470488331284990,37.619365753588426,40.767115821406804,43.913981811364650,47.060141612760532,50.205728336738034,53.350843585293212,56.495566261811980,59.639958579558154,62.784070256180136,65.927941502958646,69.071605194609603,72.215088470407267,75.358413933321401,78.501600560239112,81.644664401382343,84.787619123785475,87.930476437957068,91.073246436016348,94.215937862023580,97.358558329859648};
    double roots_temp4[30] = {6.987932000500520,10.417118547379365,13.698023153249249,16.923621285213841,20.121806174453820,23.304246988939653,26.476763664539128,29.642604540315808,32.803732385196106,35.961405804709031,39.116470190271535,42.269514977781164,45.420963972256210,48.571129851631781,51.720248430387876,54.868500957500778,58.016029064100536,61.162945044814052,64.309339090670491,67.455284479802813,70.600841369236534,73.746059609192642,76.890980862079118,80.035640218852123,83.180067446675707,86.324287962487318,89.468323600292322,92.612193221471856,95.755913204366124,98.899497840121853};
    double roots_temp5[30] = {8.182561452571242,11.704907154570391,15.039664707616520,18.301255959541990,21.525417733399944,24.727565547835034,27.915576199421359,31.093933214079307,34.265390086101583,37.431736768201496,40.594189653421182,43.753605431119361,46.910605490089281,50.065651834734567,53.219095289737737,56.371207153137988,59.522200587399929,62.672245440662799,65.821478743015831,68.970012285027948,72.117938184726214,75.265333040663847,78.412261073721368,81.558776534164963,84.704925567196909,87.850747674178194,90.996276868325054,94.141542596987293,97.286570483779556,100.431382930366766};
    double roots_temp6[30] = {9.355812111042747,12.966530172774345,16.354709639350464,19.653152101821185,22.904550647903722,26.127750137225505,29.332562578584820,32.524661288578841,35.707576953061412,38.883630955463055,42.054416412826825,45.221065015923905,48.384403860550357,51.545052042588388,54.703482507686815,57.860062972845107,61.015083772306085,64.168777272967048,67.321331703748953,70.472901193808866,73.623613182517531,76.773573972534749,79.922872948410173,83.071585821281644,86.219777152809016,89.367502338833759,92.514809183291618,95.661739158006341,98.808328419269344,101.954608634361534};
    double roots_temp7[30] = {10.512835408093997,14.207392458842461,17.647974870165896,20.983463068944769,24.262768042397006,27.507868364904251,30.730380731646651,33.937108302641299,37.132331724860144,40.318892509226401,43.498757141347504,46.673332924951666,49.843655188816108,53.010503481655284,56.174476496546099,59.336041963479246,62.495570801177443,65.653361059393958,68.809655058770531,71.964651890115917,75.118516681081644,78.271387568661623,81.423381016029651,84.574595916305853,87.725116795232083,90.875016336052170,94.024357388662906,97.173194582176606,100.321575629517497,103.469542390696105};
    double roots_temp8[30] = {11.657032192516372,15.431289210268378,18.922999198546151,22.295348019130767,25.602855953810646,28.870373347042658,32.111196239682599,35.333194182716461,38.541364851678260,41.739052867128748,44.928589676680957,48.111654554975154,51.289490080283102,54.463036742900385,57.633020262474730,60.800010054846467,63.964459458167475,67.126734067748998,70.287132110166539,73.445899362327680,76.603240254653159,79.759326255454141,82.914302285927633,86.068291687185379,89.221400108163579,92.373718579299961,95.525325964844214,98.676290936014453,101.826673571089302,104.976526662433514};
    double roots_temp9[30] = {12.790781711972119,16.641002881512190,20.182470764949169,23.591274817982963,26.927040778818018,30.217262709361400,33.476800819501470,36.714529127244710,39.936127810867674,43.145425017603145,46.345106065321282,49.537116074546908,52.722901709733286,55.903563024163084,59.079952462193440,62.252741466539042,65.422466509266272,68.589561650489941,71.754382040803648,74.917221193885155,78.078323885262122,81.237895923936023,84.396111651418266,87.553119764685817,90.709047886338766,93.864006186842005,97.018090281466030,100.171383566519268,103.323959117972791,106.475881245531454};
    double roots_temp10[30] = {13.915822610504897,17.838643199205325,21.428486972115358,24.873213923875145,28.237134359968099,31.550188381831848,34.828696537685708,38.082479087327663,41.317864690244541,44.539144633409485,47.749345734442599,50.950671453544501,54.144768056254748,57.332892581420865,60.516022838023197,63.694931715858736,66.870238740087771,70.042446670344248,73.211968010754234,76.379144556138840,79.544262033322283,82.707561224887087,85.869246529185077,89.029492624303899,92.188449711091778,95.346247678346089,98.502999441321833,101.658803639701375,104.813746834565421,107.967905310078848};
    double roots_temp11[30] = {15.033469303743439,19.025853536127759,22.662720658136056,26.142767643379102,29.534634107843925,32.870534597687538,36.168157135911230,39.438214480008057,42.687651284661086,45.921201763835590,49.142221424746147,52.353163870811386,55.555869720665704,58.751749671409463,61.941904838461291,65.127208347459586,68.308362133832034,71.485937395829225,74.660403985764731,77.832152143381634,81.001508820172290,84.168750114081789,87.334110861953846,90.497792124757524,93.659967089880240,96.820785769995695,99.980378776932270,103.138860377339668,106.296330985499964,109.452879211206437};
    double roots_temp12[30] = {16.144742942301342,20.203942632811728,23.886530755968387,27.401259258866336,30.820794086451013,34.179474666483245,37.496273635785812,40.782747098125121,44.046425210943802,47.292465605269427,50.524539725571231,53.745342865793063,56.956904352701798,60.160784767975159,63.358206027105815,66.550139854152377,69.737369558625446,72.920534159705980,76.100160532320629,79.276687239300429,82.450482476366162,85.621857773482049,88.791078588095829,91.958372588947483,95.123936201353629,98.287939828081221,101.450532050233647,104.611843034693791,107.771987318621768,110.931066100665802};
    double roots_temp13[30] = {17.250454784125964,21.373972181162749,25.101038521008061,28.649796261728866,32.096676725386359,35.478013175177900,38.813988881213312,42.116958532549603,45.395009439860289,48.653704111808395,51.897017524457880,55.127878224399765,58.348498444312668,61.560584636349631,64.765476735839030,67.964243148098504,71.157747249819835,74.346695009290102,77.531669764238828,80.713158065321437,83.891569178982621,87.067250010010554,90.240496662443789,93.411563497683261,96.580670304900124,99.748008030729736,102.913743397433535,106.078022654924339,109.240974651651214,112.402713365257640};
    double roots_temp14[30] = {18.351261495947274,22.536817071119842,26.307181561217121,29.889316321005627,33.363191239018633,36.767017939239800,40.122124128645687,43.441622375282122,46.734130921754101,50.005599701085565,53.260295322285501,56.501371328065474,59.731217053744651,62.951680707121348,66.164217311104736,69.369989860231072,72.569940310333621,75.764840529095522,78.955329587829766,82.141941531448722,85.325126377483642,88.505266214528277,91.682687697290817,94.857671854045677,98.030461863473192,101.201269279116815,104.370279054214038,107.537653630282861,110.703536288321786,113.868053914320100};
    double roots_temp15[30] = {19.447703108094736,23.693208037471379,27.505753080845263,31.120621413603502,34.621122700007234,38.047244588610198,41.421399812460606,44.757421788631056,48.064435500173296,51.348761964937680,54.614948114048964,57.866364510160608,61.105571887371426,64.334556087882888,67.554884223051346,70.767811684917916,73.974357404892842,77.175357979301751,80.371507363470059,83.563386486963793,86.751485689037153,89.936221948625260,93.117952280352569,96.296984266305046,99.473584420323760,102.647984892754053,105.820388890862148,108.990975095463597,112.159901285872820,115.327307335195854};
    double roots_temp16[30] = {20.540229825048208,24.843762597586348,28.697430993615029,32.344403636628243,35.871154400649139,39.319355760770762,42.712451971517311,46.064963569380566,49.386499978827231,52.683738054305586,55.961494357765687,59.223348831146595,62.472028065878838,65.709651473901260,68.937895248233744,72.158104940924176,75.371374787978453,78.578604865029476,81.780543078719276,84.977816550138684,88.170955429510755,91.360411214881466,94.546571016704902,97.729768789284705,100.910294263630391,104.088400117937567,107.264307782310453,110.438212174656300,113.610285592532676,116.780680932856455};



	switch(order) {
        case 0: 
            for (int i = 0; i < 30; i++){
            	roots[i] = roots_temp1[i];
            }
            break;
        case 1: 
            for (int i = 0; i < 30; i++){
            	roots[i] = roots_temp2[i];
            }
            break;
        case 2:
            for (int i = 0; i < 30; i++){
            	roots[i] = roots_temp3[i];
            }
            break;
        case 3:
            for (int i = 0; i < 30; i++){
            	roots[i] = roots_temp4[i];
            }
            break;
        case 4:
            for (int i = 0; i < 30; i++){
            	roots[i] = roots_temp5[i];
            }
            break;
        case 5:
            for (int i = 0; i < 30; i++){
            	roots[i] = roots_temp6[i];
            }
            break;
        case 6:
            for (int i = 0; i < 30; i++){
            	roots[i] = roots_temp7[i];
            }
            break;
        case 7:
            for (int i = 0; i < 30; i++){
            	roots[i] = roots_temp8[i];
            }
            break;
        case 8:
            for (int i = 0; i < 30; i++){
            	roots[i] = roots_temp9[i];
            }
            break;
        case 9:
            for (int i = 0; i < 30; i++){
            	roots[i] = roots_temp10[i];
            }
            break;
        case 10:
            for (int i = 0; i < 30; i++){
            	roots[i] = roots_temp11[i];
            }
            break;
        case 11:
            for (int i = 0; i < 30; i++){
            	roots[i] = roots_temp12[i];
            }
            break;
        case 12:
            for (int i = 0; i < 30; i++){
            	roots[i] = roots_temp13[i];
            }
            break;
        case 13:
            for (int i = 0; i < 30; i++){
            	roots[i] = roots_temp14[i];
            }
            break;
        case 14:
            for (int i = 0; i < 30; i++){
            	roots[i] = roots_temp15[i];
            }
            break;
        case 15:
            for (int i = 0; i < 30; i++){
            	roots[i] = roots_temp16[i];
            }
            break;
        default:
            fprintf(stderr, "Order %d not supported. Please use an order between 0 and 15.\n", order);
            exit(EXIT_FAILURE);
        
	}
}