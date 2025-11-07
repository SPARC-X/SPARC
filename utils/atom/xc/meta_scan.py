"""
Meta-GGA SCAN Family Functionals

Implements meta-GGA functionals that require:
- ρ(r): electron density
- |∇ρ(r)|: density gradient
- τ(r): kinetic energy density

Two-Stage Implementation:
=========================
Stage 1 (Generic): SCAN functional in its original form
  - Uses ρ, σ = |∇ρ|², and τ in generic form
  - Returns v_generic, e_generic, de_dsigma, and de_dtau
  
Stage 2 (Spherical): Transform potential to radial atomic form
  - Apply gradient correction (like GGA)
  - Apply kinetic energy density corrections
  - Energy density ε remains unchanged

Functionals:
- SCAN: Strongly Constrained and Appropriately Normed
- RSCAN: Regularized SCAN (matches SPARC naming)
- R2SCAN: Revised rSCAN (matches SPARC naming)
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

from .evaluator import XCEvaluator, XCParameters, GenericXCResult, DensityData


@dataclass
class SCANParameters(XCParameters):
    """
    Parameters for SCAN meta-GGA functional.
    
    SCAN has many internal parameters. Most are fixed by design.
    Listed here are the key tunable ones (if needed for delta learning).
    
    Attributes
    ----------
    (Most SCAN parameters are hardcoded in the functional for now)
    """
    functional_name: str = 'SCAN'  # Fixed for this functional


@dataclass
class rSCANParameters(XCParameters):
    """
    Parameters for rSCAN meta-GGA functional.
    
    Regularized SCAN variant with improved numerical behavior.
    """
    functional_name: str = 'RSCAN'  # Fixed for this functional (matches SPARC naming)


@dataclass
class r2SCANParameters(XCParameters):
    """
    Parameters for r²SCAN meta-GGA functional.
    
    Revised regularized SCAN with further improvements.
    """
    functional_name: str = 'R2SCAN'  # Fixed for this functional (matches SPARC naming)



def _get_rho_tau_and_sigma(density_data: DensityData) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rho      = density_data.rho
    grad_rho = density_data.grad_rho
    tau      = density_data.tau
    sigma    = grad_rho**2       # σ = |∇ρ|²

    if grad_rho is None or tau is None:
        raise ValueError("rSCAN requires grad_rho and tau")
    
    # Avoid division by zero
    rho[rho<1e-15] = 1e-15
    sigma[sigma<1e-15] = 1e-15

    return rho, tau, sigma
    



class SCAN(XCEvaluator):
    """
    SCAN meta-GGA functional.
    
    Requires:
    - ρ(r): electron density
    - |∇ρ(r)|: density gradient magnitude
    - τ(r): kinetic energy density
    - derivative_matrix: for spherical coordinate transformation
    
    Two-stage calculation:
    1. Generic form: SCAN formulas with ρ, |∇ρ|, τ
    2. Spherical transform: Convert to radial form with proper derivatives
    
    References
    ----------
    Sun, Ruzsinszky, Perdew, Phys. Rev. Lett. 115, 036402 (2015)
    """
    
    def __init__(
        self, 
        derivative_matrix: Optional[np.ndarray] = None,
        r_quad: Optional[np.ndarray] = None,
        params: Optional[XCParameters] = None
    ):
        """Initialize SCAN evaluator."""
        if derivative_matrix is None:
            raise ValueError(
                "SCAN requires derivative_matrix for gradient/tau transformations"
            )
        super().__init__(derivative_matrix=derivative_matrix, r_quad=r_quad, params=params)
    
    def _default_params(self) -> SCANParameters:
        """Return default SCAN parameters."""
        return SCANParameters()
    
    def compute_exchange_generic(
        self,
        density_data: DensityData
    ) -> GenericXCResult:
        """
        Compute SCAN exchange in GENERIC form (Stage 1).
        
        Implements SCAN exchange as published, using ρ, σ = |∇ρ|², and τ.
        
        Parameters
        ----------
        rho : np.ndarray
            Electron density ρ(r)
        grad_rho : np.ndarray
            Density gradient magnitude |∇ρ|
        tau : np.ndarray
            Kinetic energy density τ
        
        Returns
        -------
        result : GenericXCResult
            Exchange results with:
            - v_generic: ∂εₓ/∂ρ
            - e_generic: εₓ
            - de_dsigma: ∂εₓ/∂σ
            - de_dtau: ∂εₓ/∂τ (meta-GGA specific!)
        
        Raises
        ------
        ValueError
            If grad_rho or tau is None
        
        Notes
        -----
        SCAN exchange depends on the iso-orbital indicator α:
            α = (τ - τ_W) / τ_unif
        where τ_W = |∇ρ|²/(8ρ) is the von Weizsäcker kinetic energy density
        """

        rho, tau, sigma = _get_rho_tau_and_sigma(density_data)

        N_q = len(rho)
        normDrho = sigma**(0.5)
        def compute_basic_variables_metaGGA(rho,normDrho,tau):
            s = normDrho/(2*((3*np.pi**2)**(1/3))*rho**(4/3))
            tauw = (normDrho**2)/(8*rho)
            tauUnif = 3/10*((3*np.pi**2)**(2/3))*rho**(5/3)
            alpha = (tau-tauw)/tauUnif
            DsDn = -2*normDrho/(3*((3*np.pi**2)**(1/3))*rho**(7/3))
            DsDDn = 1/(2*((3*np.pi**2)**(1/3))*rho**(4/3))
            DtauwDn = -(normDrho**2)/(8*rho**2)
            DtauwDDn = normDrho/(4*rho)
            DtauUnifDn = ((3*np.pi**2)**(2/3))/2*rho**(2/3)
            DalphaDn = (-DtauwDn*tauUnif - (tau-tauw)*DtauUnifDn)/(tauUnif**2)
            DalphaDDn = (-DtauwDDn)/tauUnif
            DalphaDtau = 1/tauUnif
            return [s,alpha,DsDn,DsDDn,DalphaDn,DalphaDDn,DalphaDtau]

        s, alpha, DsDn, DsDDn, DalphaDn, DalphaDDn, DalphaDtau = compute_basic_variables_metaGGA(rho, normDrho, tau)
        
        epsixUnif = -3/(4*np.pi)*(3*np.pi**2*rho)**(1/3)
        k1 = 0.065
        muak = 10/81
        b2 = (5913/405000)**(1/2)
        b1 = 511/13500/(2*b2)
        b3 = 0.5
        b4 = muak*muak/k1 -1606/18225 - b1**2
        
        x = muak*(s**2)*(1+b4*(s**2)/muak*np.exp(-np.abs(b4)*(s**2)/muak))+(b1*(s**2) + b2*(1-alpha)*np.exp(-b3*(1-alpha)**2))**2
        hx1 = 1+k1 - k1/(1+x/k1)
        hx0 = 1.174
        c1x = 0.667
        c2x = 0.8
        dx = 1.24
        alphaG1 = (alpha>1)
        alphaE1 = (alpha==1)
        alphaL1 = (alpha<1)

        fx= np.zeros((N_q))
        fx[alphaG1] = -dx*np.exp(c2x/(1-alpha[alphaG1]))
        fx[alphaE1] = 0
        fx[alphaL1] = np.exp(-c1x*(alpha[alphaL1])/(1-alpha[alphaL1]))
        a1 = 4.9479
        gx = 1-np.exp(-a1*(s**(-0.5)))
        
        Fx = (hx1+fx*(hx0-hx1))*gx
        ex = epsixUnif*Fx
        
        s2 = s*s
        term1 = 1+(b4*s2)/muak*np.exp(-np.abs(b4)*s2/muak)
        term2 = s2*(b4/muak*np.exp(-np.abs(b4)*s2/muak) + b4*s2/muak*np.exp(-np.abs(b4)*s2/muak)*(-np.abs(b4)/muak))
        term3 = 2*(b1*s2+b2*(1-alpha)*np.exp(-b3*(1-alpha)**2))
        term4 = b2*(-np.exp(-b3*(1-alpha)**2)+(1-alpha)*np.exp(-b3*(1-alpha)**2)*(2*b3*(1-alpha)))
        DxDs = 2*s*(muak*(term1+term2)+b1*term3)
        DxDalpha = term3*term4
        DxDn = DsDn*DxDs+ DalphaDn*DxDalpha
        DxDDn = DsDDn*DxDs + DalphaDDn*DxDalpha
        DxDtau = DalphaDtau*DxDalpha

        DgxDn = -np.exp(-a1*s**(-0.5))*(a1/2*s**(-1.5))*DsDn
        DgxDDn = -np.exp(-a1*s**(-0.5))*(a1/2*s**(-1.5))*DsDDn
        Dhx1Dx = 1/(1+x/k1)**2
        Dhx1Dn = DxDn*Dhx1Dx
        Dhx1DDn = DxDDn*Dhx1Dx
        Dhx1Dtau = DxDtau*Dhx1Dx
        DfxDalpha = np.zeros((N_q))
        DfxDalpha[alphaG1] = -dx*np.exp(c2x/(1-alpha[alphaG1]))*(c2x/(1-alpha[alphaG1])**2)
        DfxDalpha[alphaE1] = 0
        DfxDalpha[alphaL1] = np.exp(-c1x*alpha[alphaL1]/(1-alpha[alphaL1]))*(-c1x/(1-alpha[alphaL1])**2)
        DfxDn = DfxDalpha*DalphaDn
        DfxDDn = DfxDalpha*DalphaDDn
        DfxDtau = DfxDalpha*DalphaDtau

        DFxDn = (hx1+fx*(hx0-hx1))*DgxDn + gx*(1-fx)*Dhx1Dn + gx*(hx0-hx1)*DfxDn
        DFxDDn = (hx1+fx*(hx0-hx1))*DgxDDn + gx*(1-fx)*Dhx1DDn + gx*(hx0-hx1)*DfxDDn
        DFxDtau = gx*(1-fx)*Dhx1Dtau + gx*(hx0-hx1)*DfxDtau

        DepsixUnifDn = -((3*np.pi**2)**(1/3))/(4*np.pi)*(rho**(-2/3))
        v1x = (epsixUnif+rho*DepsixUnifDn)*Fx + rho*epsixUnif*DFxDn
        v2x = rho*epsixUnif*DFxDDn/normDrho
        v3x = rho*epsixUnif*DFxDtau


        
        return GenericXCResult(
            v_generic=v1x,
            e_generic=ex,
            de_dsigma=v2x,
            de_dtau=v3x
        )
    
    
    def compute_correlation_generic(
        self,
        density_data: DensityData
        ) -> GenericXCResult:
        """
        Compute SCAN correlation in GENERIC form (Stage 1).
        
        Parameters
        ----------
        density_data : DensityData
            Container with rho, grad_rho, and tau
        
        Returns
        -------
        result : GenericXCResult
            Correlation results with all derivatives
        
        Raises
        ------
        ValueError
            If grad_rho or tau is None
        """
        
        rho, tau, sigma = _get_rho_tau_and_sigma(density_data)
        N_q = len(rho)
        normDrho = sigma**(0.5)
        def compute_basic_variables_metaGGA(rho,normDrho,tau):
            s = normDrho/(2*((3*np.pi**2)**(1/3))*rho**(4/3)) 
            tauw = (normDrho**2)/(8*rho)
            tauUnif = 3/10*((3*np.pi**2)**(2/3))*rho**(5/3)
            alpha = (tau-tauw)/tauUnif
            DsDn = -2*normDrho/(3*((3*np.pi**2)**(1/3))*rho**(7/3))
            DsDDn = 1/(2*((3*np.pi**2)**(1/3))*rho**(4/3))
            DtauwDn = -(normDrho**2)/(8*rho**2)
            DtauwDDn = normDrho/(4*rho)
            DtauUnifDn = ((3*np.pi**2)**(2/3))/2*rho**(2/3)
            DalphaDn = (-DtauwDn*tauUnif - (tau-tauw)*DtauUnifDn)/(tauUnif**2)
            DalphaDDn = (-DtauwDDn)/tauUnif
            DalphaDtau = 1/tauUnif
            return [s,alpha,DsDn,DsDDn,DalphaDn,DalphaDDn,DalphaDtau]

        s, alpha, DsDn, DsDDn, DalphaDn, DalphaDDn, DalphaDtau = compute_basic_variables_metaGGA(rho, normDrho, tau)
        
        zeta = 0
        phi = ((1+zeta)**(2/3)+(1-zeta)**(2/3))/2
        rs = (0.75/(np.pi*rho))**(1/3)

        b1c = 0.0285764
        b2c = 0.0889
        b3c = 0.125541
        ecLDA0 = -b1c/(1+b2c*(rs**(0.5))+b3c*(rs))
        dx = 0.5*((1+zeta)**(4/3)+(1-zeta)**(4/3))
        cx0 = -3/(4*np.pi)*(9*np.pi/4)**(1/3)
        Gc = (1-2.3631*(dx-1))*(1-zeta**12)
        w0 = np.exp(-ecLDA0/b1c)-1
        betaConst =  0.06672455060314922
        betaRsInf = betaConst*0.1/0.1778
        f0 = -0.9
        xiInf0 = ((3*(np.pi**2)/16)**(2/3))*(betaRsInf*1/(cx0-f0))
        gInf0s = (1+4*xiInf0*s**2)**(-0.25)
        H0 = b1c*np.log(1+w0*(1-gInf0s))
        ec0 = (ecLDA0+H0)*Gc
        
        sqr_rs = rs**(0.5)
        rsm1_2 = 1/sqr_rs
        beta = betaConst*(1+0.1*rs)/(1+0.1778*rs)
        p=1
        AA = 0.0310907
        alpha1 = 0.21370
        beta1 = 7.5957
        beta2 = 3.5876
        beta3 = 1.6382
        beta4 = 0.49294
        
        ec0_q0 = -2*AA*(1+alpha1*rs)
        ec0_q1 = 2*(AA)*(beta1*sqr_rs+beta2*rs+beta3*rs*sqr_rs+beta4*rs*rs)
        ec0_q1p = AA*(beta1*rsm1_2+2*beta2+3*beta3*sqr_rs+4*beta4*rs)
        ec0_den = 1/(ec0_q1*ec0_q1+ec0_q1)
        ec0_log = np.zeros((N_q))
        
        f1 = 1/ec0_q1
        y1 = np.argwhere(f1<1)[:,0]
        f1_less_than_1 = f1[y1] 
        y2 = np.argwhere(f1>=1)[:,0]
        f1_greater_than_1 = f1[y2]
        ec0_log_for_f1_greater_than_1 = np.log(1+f1_greater_than_1) 
        ec0_log_for_f1_less_than_1 = np.log1p(f1_less_than_1) 
        ec0_log[y1] = ec0_log_for_f1_less_than_1
        ec0_log[y2] = ec0_log_for_f1_greater_than_1
        if np.any(ec0_log == 0): 
            y = np.argwhere(ec0_log==0)[:,0]
            ec0_log[y] =  1e-15
        
        ecrs0 = ec0_q0*ec0_log
        ec_lsda1 = ecrs0
        Dec_lsda1Drs = -2*AA*alpha1*ec0_log-ec0_q0*ec0_q1p*ec0_den
        
        r = 0.031091
        w1 = np.exp(-ec_lsda1/(r*phi**3))-1

        A = beta/(r*w1)
        t = (((3*np.pi**2)/16)**(1/3))*s/(phi*sqr_rs)
        g = (1+4*A*t*t)**(-0.25)
        H1 = r*(phi**3)*np.log(1+w1*(1-g))
        ec1 = ec_lsda1+H1
        
        c1c = 0.64
        c2c = 1.5            
        dc = 0.7
        alphaG1 = (alpha>1)
        alphaE1 = (alpha==1)
        alphaL1 = (alpha<1)
        fc = np.zeros((N_q))
        fc[alphaG1] = -dc*np.exp(c2c/(1-alpha[alphaG1]))
        fc[alphaE1] = 0
        fc[alphaL1] = np.exp(-c1c*alpha[alphaL1]/(1-alpha[alphaL1]))
        ec = ec1+fc*(ec0-ec1)

        DzetaDn = 0
        DrsDn = -4*np.pi/9*(4*np.pi/3*rho)**(-4/3)
        DdxDn = (4/3*(1+zeta)**(1/3) - 4/3*(1-zeta)**(1/3))*DzetaDn
        DGcDn = -2.3631*DdxDn*(1-zeta**12) + (1-2.3631*(dx-1))*((12*zeta**11)*(DzetaDn))
        DgInf0sDs = -0.25*((1+4*xiInf0*s*s)**(-1.25))*(4*xiInf0*2*s)
        DgInf0sDn = DgInf0sDs*DsDn
        DgInf0sDDn = DgInf0sDs*DsDDn
        DecLDA0Dn = b1c*(0.5*b2c*rs**(-0.5)+b3c)/((1+b2c*rs**(0.5)+b3c*rs)**2)*DrsDn
        Dw0Dn = (w0+1)*(-DecLDA0Dn/b1c)
        DH0Dn = b1c*(Dw0Dn*(1-gInf0s)-w0*DgInf0sDn)/(1+w0*(1-gInf0s))
        DH0DDn = b1c*(-w0*DgInf0sDDn)/(1+w0*(1-gInf0s))
    
        Dec0Dn = (DecLDA0Dn+DH0Dn)*Gc + (ecLDA0+H0)*DGcDn
        Dec0DDn = DH0DDn*Gc
        
        Dec_lsda1Dn = -(rs/rho/3)*(-2*AA*alpha1*np.log(1+1/(2*AA*(beta1*sqr_rs+beta2*rs+beta3*(rs**1.5)+beta4*(rs**(p+1)))))-((-2*AA*(1+alpha1*rs))*(AA*(beta1*(rs**(-0.5))+2*beta2+3*beta3*(rs**(0.5))+2*(p+1)*beta4*(rs**p))))/((2*AA*(beta1*sqr_rs+beta2*rs+beta3*(rs**1.5)+beta4*(rs**(p+1))))*(2*AA*(beta1*sqr_rs+beta2*rs+beta3*(rs**1.5)+beta4*(rs**(p+1))))+(2*AA*(beta1*(rs**(0.5))+beta2*rs+beta3*(rs**1.5)+beta4*(rs**(p+1))))))
        DbetaDn = 0.066725*(0.1*(1+0.1778*rs)-0.1778*(1+0.1*rs))/((1+0.1778*rs)**2)*DrsDn
        DphiDn = 0.5*(2/3*(1+zeta)**(-1/3) - 2/3*(1-zeta)**(-1/3))*DzetaDn
        DtDn = ((3*(np.pi**2)/16)**(1/3))*(phi*sqr_rs*DsDn-s*(DphiDn*sqr_rs+phi*DrsDn/(2*sqr_rs)))/((phi**2)*rs)
        DtDDn = t*DsDDn/s
        Dw1Dn = (w1+1)*(-((r*phi**3)*Dec_lsda1Dn-r*ec_lsda1*(3*(phi**2)*DphiDn))/((r*phi**3)**2))
        DADn = (w1*DbetaDn - beta*Dw1Dn)/(r*w1**2)
        DgDn = (-0.25*(1+4*A*t*t)**(-1.25))*(4*(DADn*t*t+2*A*t*DtDn))
        DgDDn = (-0.25*(1+4*A*t*t)**(-1.25))*(4*2*A*t*DtDDn)
        DH1Dn = r*((3*phi**2)*DphiDn*np.log(1+w1*(1-g)) + (phi**3)*(Dw1Dn*(1-g)-w1*DgDn)/(1+w1*(1-g)))
        DH1DDn = r*((phi**3)*(-w1*DgDDn)/(1+w1*(1-g)))
        Dec1Dn = Dec_lsda1Dn + DH1Dn
        Dec1DDn = DH1DDn
        
        DfcDalpha = np.zeros((N_q))
        DfcDalpha[alphaG1] = fc[alphaG1]*(c2c/(1-alpha[alphaG1])**2)
        DfcDalpha[alphaE1] = 0
        DfcDalpha[alphaL1] = fc[alphaL1]*(-c1c/(1-alpha[alphaL1])**2)
        DfcDn = DfcDalpha*DalphaDn
        DfcDDn = DfcDalpha*DalphaDDn
        DfcDtau  = DfcDalpha*DalphaDtau
        DepsiloncDn = Dec1Dn + fc*(Dec0Dn-Dec1Dn) + DfcDn*(ec0-ec1)
        DepsiloncDDn = Dec1DDn + fc*(Dec0DDn - Dec1DDn) + DfcDDn*(ec0-ec1)
        DepsiloncDtau = DfcDtau*(ec0-ec1)
        v1c = ec + rho*DepsiloncDn
        v2c = rho*(DepsiloncDDn)/normDrho
        v3c = rho*DepsiloncDtau



        return GenericXCResult(
            v_generic = v1c,
            e_generic = ec,
            de_dsigma = v2c,
            de_dtau = v3c
        )


class rSCAN(XCEvaluator):
    """
    rSCAN (regularized SCAN) meta-GGA functional.
    
    Regularized version of SCAN with improved numerical stability.
    
    References
    ----------
    Bartók, Yates, Phys. Rev. B 99, 235103 (2019)
    """
    
    def __init__(
        self,
        derivative_matrix: Optional[np.ndarray] = None,
        r_quad: Optional[np.ndarray] = None,
        params: Optional[XCParameters] = None
    ):
        """Initialize rSCAN evaluator."""
        if derivative_matrix is None:
            raise ValueError("rSCAN requires derivative_matrix")
        super().__init__(derivative_matrix=derivative_matrix, r_quad=r_quad, params=params)
    
    def _default_params(self) -> rSCANParameters:
        """Return default rSCAN parameters."""
        return rSCANParameters()
    


    
    def compute_exchange_generic(
        self,
        density_data: DensityData
        ) -> GenericXCResult:
        """
        Compute rSCAN exchange in generic form.
        
        Similar to SCAN but with regularized enhancement factor.
        """

        rho, tau, sigma = _get_rho_tau_and_sigma(density_data)

        normDrho = sigma**(0.5)

        s, alpha, DsDn, DsDDn, DalphaDn, DalphaDDn, DalphaDtau = \
            self._compute_basic_rscan_variables(rho, normDrho, tau)

        ex, v_x_generic, de_x_dsigma, de_x_dtau = \
            self._rSCAN_exchange(rho, s, alpha, DsDn, DsDDn, DalphaDn, DalphaDDn, DalphaDtau)

        de_x_dsigma = de_x_dsigma / normDrho

        return GenericXCResult(
            v_generic = v_x_generic,
            e_generic = ex,
            de_dsigma = de_x_dsigma,
            de_dtau   = de_x_dtau
        )
    
    
    def compute_correlation_generic(
        self,
        density_data: DensityData
        ) -> GenericXCResult:
        """Compute rSCAN correlation in generic form."""
        rho, tau, sigma = _get_rho_tau_and_sigma(density_data)

        # original code, without modification
        normDrho = sigma**(0.5)          

        s, alpha, DsDn, DsDDn, DalphaDn, DalphaDDn, DalphaDtau = \
            self._compute_basic_rscan_variables(rho, normDrho, tau)

        ec, v_c_generic, de_c_dsigma, de_c_dtau = \
            self._rSCAN_correlation(rho, s, alpha, DsDn, DsDDn, DalphaDn, DalphaDDn, DalphaDtau)

        de_c_dsigma = de_c_dsigma / normDrho

        # np.savetxt("v_c_generic.txt", v_c_generic)
        # np.savetxt("ec.txt", ec)
        # np.savetxt("de_c_dsigma.txt", de_c_dsigma)
        # np.savetxt("de_c_dtau.txt", de_c_dtau)
        # raise ValueError("Stop here")

        return GenericXCResult(
            v_generic = v_c_generic,
            e_generic = ec,
            de_dsigma = de_c_dsigma,
            de_dtau   = de_c_dtau
        )
    
    
    @staticmethod
    def _compute_basic_rscan_variables(rho, normDrho, tau):   
        s = normDrho/((2*(3*np.pi**2)**(1/3))*rho**(4/3))
        tauw = (normDrho**2)/(8*rho)         
        tauUnif = 3/10*((3*np.pi**2)**(2/3))*rho**(5/3)
        alpha = (tau-tauw)/(tauUnif+1e-4)
        alphaP = (alpha**3)/(alpha**2+1e-3)
        
        DsDn = -2*normDrho/((3*(3*np.pi**2)**(1/3))*rho**(7/3))
        DsDDn = 1/((2*(3*np.pi**2)**(1/3))*rho**(4/3))
        DtauwDn = -(normDrho**2)/(8*rho**2)
        DtauwDDn = normDrho/(4*rho)
        
        DtauUnifDn = ((3*np.pi**2)**(2/3))/2 * rho**(2/3)
        DalphaDn = (-DtauwDn*(tauUnif+1e-4)-(tau-tauw)*DtauUnifDn)/(tauUnif+1e-4)**2
        DalphaDDn = -DtauwDDn/(tauUnif+1e-4)
        DalphaDtau = 1/(tauUnif+1e-4)
        DalphaDalpha = (3*alpha**2*(alpha**2+1e-3)-alpha**3*(2*alpha))/(alpha**2+1e-3)**2
        DalphaPDn = DalphaDalpha*DalphaDn
        DalphaPDDn = DalphaDalpha*DalphaDDn
        DalphaPDtau = DalphaDalpha*DalphaDtau
        
        return [s, alphaP ,DsDn, DsDDn, DalphaPDn, DalphaPDDn, DalphaPDtau]


    @staticmethod
    def _rSCAN_correlation(rho, s, alpha, DsDn, DsDDn, DalphaDn, DalphaDDn, DalphaDtau):
        N_q = len(rho)

        zeta = 0
        phi = ((1+zeta)**(2/3)+(1-zeta)**(2/3))/2
        rs = (0.75/(np.pi*rho))**(1/3)
        b1c = 0.0285764
        b2c = 0.0889
        b3c = 0.125541
        ecLDA0 = -b1c/(1+b2c*(rs**(0.5))+b3c*rs)
        dx = 0.5*((1+zeta)**(4/3)+(1-zeta)**(4/3))
        cx0 = -3/(4*np.pi)*(9*np.pi/4)**(1/3)
        Gc = (1-2.3631*(dx-1))*(1-zeta**12)
        w0 = np.exp(-ecLDA0/b1c)-1
        betaConst = 0.06672455060314922
        betaRsInf = betaConst*0.1/0.1778
        f0 = -0.9                
        xiInf0 = (((3*np.pi**2)/16)**(2/3))*(betaRsInf*1/(cx0-f0))
        gInf0s = (1+4*xiInf0*s**2)**(-0.25)
        H0 = b1c*np.log(1+w0*(1-gInf0s))
        ec0 = (ecLDA0+H0)*Gc
        sqr_rs = rs**(0.5)
        rsm1_2 = 1/sqr_rs
        beta = betaConst*(1+0.1*rs)/(1+0.1778*rs)
        p=1
        AA = 0.0310907
        alpha1 = 0.21370
        beta1 = 7.5957
        beta2 = 3.5876
        beta3 = 1.6382
        beta4 = 0.49294
        
        ec0_q0 = -2*AA*(1+alpha1*rs)
        ec0_q1 = 2*AA*(beta1*sqr_rs+beta2*rs+beta3*rs*sqr_rs+beta4*rs*rs)
        ec0_q1p = AA*(beta1*rsm1_2+2*beta2+3*beta3*sqr_rs+4*beta4*rs)
        ec0_den = 1/(ec0_q1*ec0_q1+ec0_q1)
        ec0_log = np.zeros((N_q))
        
        f1 = 1/ec0_q1
        y1 = np.argwhere(f1<1)[:,0]
        f1_less_than_1 = f1[y1] 
        y2 = np.argwhere(f1>=1)[:,0]
        f1_greater_than_1 = f1[y2]
        ec0_log_for_f1_greater_than_1 = np.log(1+f1_greater_than_1) 
        ec0_log_for_f1_less_than_1 = np.log1p(f1_less_than_1) 
        ec0_log[y1] = ec0_log_for_f1_less_than_1
        ec0_log[y2] = ec0_log_for_f1_greater_than_1
        if np.any(ec0_log == 0): 
              y = np.argwhere(ec0_log==0)[:,0]
              ec0_log[y] =  1e-15
        
        ecrs0 = ec0_q0*ec0_log
        ec_lsda1 = ecrs0
        Dec_lsda1Drs = -2*AA*alpha1*ec0_log - ec0_q0*ec0_q1p*ec0_den
        
        r = 0.031091
        w1 = np.exp(-ec_lsda1/(r*phi**3))-1 + 1e-15
        A = beta/(r*w1)
        t = (((3*np.pi**2)/16)**(1/3))*s/(phi*sqr_rs)
        g = (1+4*A*t*t)**(-0.25)
        H1 = r*(phi**3)*np.log(1+w1*(1-g))
        ec1 = ec_lsda1 + H1
        
        c1c = 0.64
        c2c = 1.5
        dc = 0.7
        
        alphaG25 = (alpha>2.5)
        alpha0To25 = ((alpha>=0) & (alpha<=2.5))
        alphaL0 = (alpha<0)
        fc = np.zeros((N_q))
        fc[alphaG25] = -dc*np.exp(c2c/(1-alpha[alphaG25]))
        fc[alpha0To25] = 1 + (-0.64)*alpha[alpha0To25] + (-0.4352)*alpha[alpha0To25]**2 + (-1.535685604549)*alpha[alpha0To25]**3
        + (3.061560252175)*alpha[alpha0To25]**4 + (-1.915710236206)*alpha[alpha0To25]**5 + 0.516884468372*alpha[alpha0To25]**6
        + (-0.051848879792)*alpha[alpha0To25]**7
        
        fc[alphaL0] = np.exp(-c1c*alpha[alphaL0]/(1-alpha[alphaL0]))
        ec = ec1 + fc*(ec0-ec1)
        
        DzetaDn = 0
        DrsDn = -4*np.pi/9*(4*np.pi/3*rho)**(-4/3)
        DdxDn = (4/3*(1+zeta)**(1/3)-4/3*(1-zeta)**(1/3))*DzetaDn
        DGcDn = -2.3631*DdxDn*(1-zeta**12) + (1-2.3631*(dx-1))*((12*zeta**11)*DzetaDn)
        DgInf0sDs = (-0.25*(1+4*xiInf0*s*s)**(-1.25))*(4*xiInf0*2*s)
        DgInf0sDn = DgInf0sDs*DsDn
        DgInf0sDDn = DgInf0sDs*DsDDn
        DecLDA0Dn = b1c*(0.5*b2c*rs**(-0.5)+b3c)/((1+b2c*rs**(0.5)+b3c*rs)**(2))*DrsDn
        Dw0Dn = (w0+1)*(-DecLDA0Dn/b1c)
        DH0Dn = b1c*(Dw0Dn*(1-gInf0s)-w0*DgInf0sDn)/(1+w0*(1-gInf0s))
        DH0DDn = b1c*(-w0*DgInf0sDDn)/(1+w0*(1-gInf0s))
        Dec0Dn = (DecLDA0Dn + DH0Dn)*Gc + (ecLDA0+H0)*DGcDn
        Dec0DDn = DH0DDn*Gc
        
        Dec_lsda1Dn = -(rs/rho/3)*(-2*AA*alpha1*np.log(1+1/(2*AA*(beta1*sqr_rs+beta2*rs + beta3*(rs**1.5)+beta4*(rs**(p+1)))))
                        -((-2*AA*(1+alpha1*rs))*(AA*(beta1*(rs**(-0.5))+2*beta2 + 3*beta3*(rs**(0.5))+2*(p+1)*beta4*(rs**p))))
                        /((2*AA*(beta1*sqr_rs+beta2*rs+beta3*(rs**1.5)+beta4*(rs**(p+1))))
                        *(2*AA*(beta1*sqr_rs + beta2*rs + beta3*(rs**1.5)+beta4*(rs**(p+1)))) + (2*AA*(beta1*(rs**(0.5))+beta2*rs+beta3*(rs**1.5)+beta4*(rs**(p+1))))))
        
        DbetaDn = 0.066725*(0.1*(1+0.1778*rs)-0.1778*(1+0.1*rs))/((1+0.1778*rs)**2)*DrsDn
        DphiDn = 0.5*(2/3*(1+zeta)**(-1/3) - 2/3*(1-zeta)**(-1/3))*DzetaDn
        DtDn = ((3*np.pi**2)/16)**(1/3)*(phi*sqr_rs*DsDn-s*(DphiDn*sqr_rs+phi*DrsDn/(2*sqr_rs)))/((phi**2)*rs)
        DtDDn = t*DsDDn/s
        Dw1Dn = (1+w1)*(-(r*phi**3*Dec_lsda1Dn-r*ec_lsda1*(3*phi**2*DphiDn)))/((r*phi**3)**2)
        DADn = (w1*DbetaDn-beta*Dw1Dn)/(r*w1**2)
        DgDn = -0.25*(1+4*A*t*t)**(-1.25)*(4*(DADn*t*t+2*A*t*DtDn))
        DgDDn = -0.25*(1+4*A*t*t)**(-1.25)*(4*2*A*t*DtDDn)
        DH1Dn = r*(3*phi**2*DphiDn*np.log(1+w1*(1-g))+phi**3*(Dw1Dn*(1-g)-w1*DgDn)/(1+w1*(1-g)))
        DH1DDn = r*(phi**3*(-w1*DgDDn)/(1+w1*(1-g)))
        Dec1Dn = Dec_lsda1Dn + DH1Dn
        Dec1DDn = DH1DDn
        
        DfcDalpha = np.zeros((N_q))
        DfcDalpha[alphaG25] = fc[alphaG25]*(c2c/(1-alpha[alphaG25])**2)
        DfcDalpha[alpha0To25] = (-0.64) + (-0.4352)*alpha[alpha0To25]*2 + (-1.535685604549)*alpha[alpha0To25]**2*3
        +  3.061560252175*alpha[alpha0To25]**3*4 + (-1.915710236206)*alpha[alpha0To25]**4*5 + 0.516884468372*alpha[alpha0To25]**5*6
        + (-0.051848879792)*alpha[alpha0To25]**6*7

        DfcDalpha[alphaL0] = fc[alphaL0]*(-c1c/(1-alpha[alphaL0])**2)
        DfcDn = DfcDalpha*DalphaDn
        DfcDDn = DfcDalpha*DalphaDDn
        DfcDtau = DfcDalpha*DalphaDtau
        DepsiloncDn = Dec1Dn + fc*(Dec0Dn-Dec1Dn) + DfcDn*(ec0-ec1)
        DepsiloncDDn = Dec1DDn + fc*(Dec0DDn-Dec1DDn)+ DfcDDn*(ec0-ec1)
        DepsiloncDtau = DfcDtau*(ec0-ec1)
        v1c = ec + rho*DepsiloncDn
        v2c = rho*DepsiloncDDn
        v3c = rho*DepsiloncDtau

        return [ec, v1c, v2c, v3c]


    @staticmethod
    def _rSCAN_exchange(rho, s, alpha, DsDn, DsDDn, DalphaDn, DalphaDDn, DalphaDtau):
        N_q = len(rho)
        epsixUnif = -3/(4*np.pi)*(3*(np.pi**2)*rho)**(1/3)
        
        k1 = 0.065
        muak = 10/81
        b2 = np.sqrt(5913/405000)
        b1 = 511/13500/(2*b2)
        b3 = 0.5
        b4 = muak*muak/k1 -1606/18225 - b1*b1
        
        x = (muak*s**2)*(1+b4*(s**2)/muak*np.exp(-np.abs(b4)*(s**2)/muak)) + (b1*s**2+b2*(1-alpha)*np.exp(-b3*(1-alpha)**2))**2
        hx1 = 1+k1 - k1/(1+x/k1)
        hx0 = 1.174
        
        c1x = 0.667
        c2x = 0.8
        dx = 1.24
        alphaG25 = (alpha>2.5)
        alpha0To25 = ((alpha>=0) & (alpha<=2.5))
        alphaL0 =(alpha<0)
        fx = np.zeros((N_q))
        fx[alphaG25] = -dx*np.exp(c2x/(1-alpha[alphaG25]))
        fx[alpha0To25] = 1 + (-0.667)*alpha[alpha0To25] + (-0.4445555)*alpha[alpha0To25]**2 + (-0.663086601049)*alpha[alpha0To25]**3
        + 1.451297044490*alpha[alpha0To25]**4 + (-0.887998041597)*alpha[alpha0To25]**5 + 0.234528941479*alpha[alpha0To25]**6 
        + (-0.023185843322)*alpha[alpha0To25]**7
        fx[alphaL0] = np.exp(-c1x*alpha[alphaL0]/(1-alpha[alphaL0]))
        a1 = 4.9479
        gx = 1-np.exp(-a1*s**(-0.5))
        Fx = (hx1 + fx*(hx0-hx1))*gx
        
        ex = epsixUnif*Fx
        s2 = s*s
        term1 = 1 + (b4*s2)/muak*np.exp(-np.abs(b4)*s2/muak)
        term2 = s2*(b4/muak*np.exp(-np.abs(b4)*s2/muak) + b4*s2/muak*np.exp(-np.abs(b4)*s2/muak)*(-np.abs(b4)/muak))
        term3 = 2*(b1*s2 + b2*(1-alpha)*np.exp(-b3*(1-alpha)**2))
        term4 = b2*(-np.exp(-b3*(1-alpha)**2) + (1-alpha)*(np.exp(-b3*(1-alpha)**2))*(2*b3*(1-alpha)))
        DxDs = 2*s*(muak*(term1+term2) + b1*term3)
        DxDalpha = term3*term4
        DxDn = DsDn*DxDs + DalphaDn*DxDalpha
        DxDDn = DsDDn*DxDs + DalphaDDn*DxDalpha
        DxDtau = DalphaDtau*DxDalpha
        
        DgxDn = -np.exp(-a1*s**(-0.5))*(a1/2*s**(-1.5))*DsDn
        DgxDDn = -np.exp(-a1*s**(-0.5))*(a1/2*s**(-1.5))*DsDDn
        Dhx1Dx = 1/(1+x/k1)**2
        Dhx1Dn = DxDn*Dhx1Dx
        Dhx1DDn = DxDDn*Dhx1Dx
        Dhx1Dtau = DxDtau*Dhx1Dx
        
        DfxDalpha = np.zeros((N_q))
        DfxDalpha[alphaG25] = -dx*np.exp(c2x/(1-alpha[alphaG25]))*(c2x/(1-alpha[alphaG25])**2)
        DfxDalpha[alpha0To25] = -0.667 + (-0.4445555)*alpha[alpha0To25]*2 + (-0.663086601049)*(alpha[alpha0To25]**2)*3
        +  1.451297044490*(alpha[alpha0To25]**3)*4 + (-0.887998041597)*(alpha[alpha0To25]**4)*5 + 0.234528941479*(alpha[alpha0To25]**5)*6
        + (-0.023185843322)*(alpha[alpha0To25]**6)*7
        DfxDalpha[alphaL0] = np.exp(-c1x*alpha[alphaL0]/(1-alpha[alphaL0]))*(-c1x/(1-alpha[alphaL0])**2)

        DfxDn = DfxDalpha*DalphaDn
        DfxDDn = DfxDalpha*DalphaDDn
        DfxDtau =DfxDalpha*DalphaDtau
        
        DFxDn = (hx1 + fx*(hx0-hx1))*DgxDn + gx*(1-fx)*Dhx1Dn + gx*(hx0-hx1)*DfxDn
        DFxDDn = (hx1 + fx*(hx0-hx1))*DgxDDn + gx*(1-fx)*Dhx1DDn + gx*(hx0-hx1)*DfxDDn
        DFxDtau = gx*(1-fx)*Dhx1Dtau + gx*(hx0-hx1)*DfxDtau
        
        DepsixUnifDn = -((3*np.pi**2)**(1/3))/(4*np.pi)*rho**(-2/3)
        
        v1x = (epsixUnif + rho*DepsixUnifDn)*Fx + rho*epsixUnif*DFxDn
        v2x = rho*epsixUnif*DFxDDn
        v3x = rho*epsixUnif*DFxDtau
        return [ex,v1x,v2x,v3x]


class r2SCAN(XCEvaluator):
    """
    r²SCAN (revised regularized SCAN) meta-GGA functional.
    
    Further improved version with better performance.
    
    References
    ----------
    Furness et al., J. Phys. Chem. Lett. 11, 8208 (2020)
    """
    
    def __init__(
        self,
        derivative_matrix: Optional[np.ndarray] = None,
        r_quad: Optional[np.ndarray] = None,
        params: Optional[XCParameters] = None
    ):
        """Initialize r2SCAN evaluator."""
        if derivative_matrix is None:
            raise ValueError("r2SCAN requires derivative_matrix")
        super().__init__(derivative_matrix=derivative_matrix, r_quad=r_quad, params=params)
    
    def _default_params(self) -> r2SCANParameters:
        """Return default r2SCAN parameters."""
        return r2SCANParameters()
    
    
    def compute_exchange_generic(
        self,
        density_data: DensityData
    ) -> GenericXCResult:
        """Compute r2SCAN exchange in generic form."""
        rho, tau, sigma = _get_rho_tau_and_sigma(density_data)
        
        N_q = len(rho)
        normDrho = sigma**(0.5)
        def ex_basicr2SCANvariables(rho,normDrho,tau):
            s = normDrho/(2*(3*np.pi**2)**(1/3)*rho**(4/3))
            p = s**2
            tauw = (normDrho**2)/(8*rho)
            tauUnif = 3/10*(3*np.pi**2)**(2/3)*(rho**(5/3))
            eta = 0.001
            alpha = (tau-tauw)/(tauUnif+eta*tauw)
            #    print("alpahxmax",np.max(np.abs(alpha)))
            DsDn = -2*normDrho/(3*(3*np.pi**2)**(1/3)*rho**(7/3))
            DpDn = 2*s*DsDn
            DsDDn = 1/(2*(3*np.pi**2)**(1/3)*rho**(4/3))
            DpDDn = 2*s*DsDDn
            DtauwDn = -normDrho**2/(8*rho**2)
            DtauwDDn = normDrho/(4*rho)
            DtauUnifDn = (3*np.pi**2)**(2/3)/2*rho**(2/3)
            DalphaDn = (-DtauwDn*(tauUnif+eta*tauw) - (tau-tauw)*(DtauUnifDn+eta*DtauwDn))/((tauUnif+eta*tauw)**2)
            
            DalphaDDn = (-DtauwDDn*(tauUnif+eta*tauw) - (tau-tauw)*eta*DtauwDDn)/((tauUnif+eta*tauw)**2)
            DalphaDtau = 1/(tauUnif+eta*tauw)
            
            return [p, alpha,DpDn, DpDDn, DalphaDn, DalphaDDn, DalphaDtau]
                    
            
        p, alpha, DpDn, DpDDn, DalphaDn, DalphaDDn, DalphaDtau = ex_basicr2SCANvariables(rho, normDrho, tau)
        
        def exchanger2SCAN(rho,p,alpha,DpDn,DpDDn,DalphaDn, DalphaDDn, DalphaDtau):
            epsixUnif = -3/(4*np.pi)*(3*np.pi**2*rho)**(1/3)
            
            k0 = 0.174
            k1 = 0.065
            muak = 10/81
            
            eta = 0.001
            Ceta = 20/27 + eta*5/3
            C2 = -0.162742
            dp2 = 0.361
            x = (Ceta*C2*np.exp(-p**2/dp2**4)+muak)*p
            hx0 = 1+k0
            hx1 = 1+k1 - k1/(1+x/k1)
            
            c1x = 0.667
            c2x = 0.8
            dx = 1.24
            alphaG25 = (alpha>2.5)
            alpha0To25 = ((alpha>=0) & (alpha<=2.5))
            alphaL0 = (alpha<0)
            
            fx = np.zeros((N_q))
            fx[alphaG25] = -dx*np.exp(c2x/(1-alpha[alphaG25]))
            fx[alpha0To25] = 1 + (-0.667)*alpha[alpha0To25] + (-0.4445555)*alpha[alpha0To25]**2 + (-0.663086601049)*alpha[alpha0To25]**3 
            + 1.451297044490*alpha[alpha0To25]**4 + (-0.887998041597)*alpha[alpha0To25]**5 + 0.234528941479*alpha[alpha0To25]**6
            + (-0.023185843322)*alpha[alpha0To25]**7
            
            fx[alphaL0] = np.exp(-c1x*alpha[alphaL0]/(1-alpha[alphaL0]))
            a1 = 4.9479
            gx = 1-np.exp(-a1*p**(-0.25))
            Fx = (hx1 + fx*(hx0-hx1))*gx
            
            ex = epsixUnif*Fx
            
            DxDp = (Ceta*C2*np.exp(-p**2/dp2**4)+muak) + Ceta*C2*np.exp(-p**2/dp2**4)*(-2*p/dp2**4)*p
            DxDn = DxDp*DpDn
            DxDDn = DpDDn*DxDp
            
            DgxDn = -np.exp(-a1*p**(-0.25))*(a1/4*p**(-1.25))*DpDn
            DgxDDn = -np.exp(-a1*p**(-0.25))*(a1/4*p**(-1.25))*DpDDn
            Dhx1Dx = 1/(1+x/k1)**2
            Dhx1Dn = DxDn*Dhx1Dx
            Dhx1DDn = DxDDn*Dhx1Dx

            DfxDalpha = np.zeros((N_q))
            DfxDalpha[alphaG25] = -dx*np.exp(c2x/(1-alpha[alphaG25]))*(c2x/(1-alpha[alphaG25])**2)
            DfxDalpha[alpha0To25] = (-0.667) + (-0.4445555)*alpha[alpha0To25]*2 + (-0.663086601049)*alpha[alpha0To25]**2*3
            + 1.451297044490*alpha[alpha0To25]**3*4 + (-0.887998041597)*alpha[alpha0To25]**4*5 + 0.234528941479*alpha[alpha0To25]**5*6
            + (-0.023185843322)*alpha[alpha0To25]**6*7
            
            DfxDalpha[alphaL0] = np.exp(-c1x*alpha[alphaL0]/(1-alpha[alphaL0]))*(-c1x/(1-alpha[alphaL0])**2)
            DfxDn = DfxDalpha*DalphaDn
            DfxDDn = DfxDalpha*DalphaDDn
            DfxDtau = DfxDalpha*DalphaDtau
            
            DFxDn = (hx1 + fx*(hx0-hx1))*DgxDn + gx*(1-fx)*Dhx1Dn + gx*(hx0-hx1)*DfxDn
            DFxDDn = (hx1 + fx*(hx0-hx1))*DgxDDn + gx*(1-fx)*Dhx1DDn + gx*(hx0-hx1)*DfxDDn
            
            DFxDtau = gx*(hx0-hx1)*DfxDtau
            
            DepsixUnifDn = -(3*np.pi**2)**(1/3)/(4*np.pi)*rho**(-2/3)
            v1x = (epsixUnif + rho*DepsixUnifDn)*Fx + rho*epsixUnif*DFxDn
            v2x = rho*epsixUnif*DFxDDn
            v3x = rho*epsixUnif*DFxDtau
            
            return [ex, v1x, v2x, v3x]
        ex, v1x, v2x, v3x = exchanger2SCAN(rho, p, alpha, DpDn, DpDDn, DalphaDn, DalphaDDn, DalphaDtau)
        v2x = v2x/normDrho

        return GenericXCResult(
            v_generic = v1x,
            e_generic = ex,
            de_dsigma = v2x,
            de_dtau   = v3x
        )
    
    
    def compute_correlation_generic(
        self,
        density_data: DensityData
        ) -> GenericXCResult:
        """Compute r2SCAN correlation in generic form."""
        rho, tau, sigma = _get_rho_tau_and_sigma(density_data)

        N_q = len(rho)
        normDrho = sigma**(0.5)
        def co_basicr2SCANvariables(rho, normDrho, tau):
            s = normDrho/(2*(3*np.pi**2)**(1/3)*rho**(4/3))
            p = s**2
            zeta = 0
            tauw = (normDrho**2)/(8*rho)
            ds = 1
            tauUnif= 3/10*(3*np.pi**2)**(2/3)*(rho**(5/3))
            eta = 0.001 
            alpha = (tau-tauw)/(tauUnif+eta*tauw)
            # print("alpahcmac",np.max(np.abs(alpha)))
            DsDn = -2*normDrho/(3*(3*np.pi**2)**(1/3)*rho**(7/3))
            DpDn = 2*s*DsDn
            DsDDn = 1/(2*(3*np.pi**2)**(1/3)*rho**(4/3))
            DpDDn = 2*s*DsDDn
            DtauwDn = -normDrho**2/(8*rho**2)
            DtauwDDn = normDrho/(4*rho)
            
            DtauUnifDn = (3*np.pi**2)**(2/3)/2*rho**(2/3)
            
            DalphaDn = (-DtauwDn*(tauUnif+eta*tauw) - (tau-tauw)*(DtauUnifDn+eta*DtauwDn))/((tauUnif+eta*tauw)**2)
            DalphaDDn = (-DtauwDDn*(tauUnif+eta*tauw) - (tau-tauw)*eta*DtauwDDn)/((tauUnif+eta*tauw)**2)
            DalphaDtau = 1/(tauUnif+eta*tauw)
            
            return [s, p, alpha, DsDn, DsDDn, DpDn, DpDDn, DalphaDn, DalphaDDn, DalphaDtau]
        
        s, p, alpha, DsDn, DsDDn, DpDn, DpDDn, DalphaDn, DalphaDDn, DalphaDtau = co_basicr2SCANvariables(rho, normDrho, tau)
        
        def correlationr2SCAN(rho, s, p, alpha, DsDn, DsDDn, DpDn, DpDDn, DalphaDn, DalphaDDn, DalphaDtau):  
            phi = 1
            rs = (0.75/(np.pi*rho))**(1/3)
            
            b1c = 0.0285764
            b2c = 0.0889
            b3c = 0.125541
            ecLDA0 = -b1c/(1+b2c*(rs**(0.5))+b3c*rs)
            dx = 1
            cx0 = -3/(4*np.pi)*(9*np.pi/4)**(1/3)
            Gc = 1
            w0 = np.exp(-ecLDA0/b1c)-1
            betaConst = 0.06672455060314922
            betaRsInf = betaConst*0.1/0.1778
            f0 = -0.9
            chiInf = (3*np.pi**2/16)**(2/3)*(betaRsInf*1/(cx0-f0))
            
            gInf0s = (1+4*chiInf*(s**2))**(-0.25)
            H0 = b1c*np.log(1+w0*(1-gInf0s))
            ec0 = ecLDA0 + H0
            
            sqr_rs = rs**(0.5)
            rsm1_2 = 1/sqr_rs
            beta = betaConst*(1+0.1*rs)/(1+0.1778*rs)
            
            AAec0 = 0.0310907
            alpha1ec0 = 0.21370
            beta1ec0 = 7.5957
            beta2ec0 = 3.5876
            beta3ec0 = 1.6382
            beta4ec0 = 0.49294
            
            ec0_q0 = -2*AAec0*(1+alpha1ec0*rs)
            ec0_q1 = 2*AAec0*(beta1ec0*sqr_rs + beta2ec0*rs + beta3ec0*rs*sqr_rs + beta4ec0*rs*rs)
            ec0_q1p = AAec0*(beta1ec0*rsm1_2 + 2*beta2ec0 + 3*beta3ec0*sqr_rs + 4*beta4ec0*rs)
            ec0_den = 1/(ec0_q1*ec0_q1 + ec0_q1)
            ec0_log = np.zeros((N_q))
            
            f1 = 1/ec0_q1
            y1 = np.argwhere(f1<1)[:,0]
            f1_less_than_1 = f1[y1] 
            y2 = np.argwhere(f1>=1)[:,0]
            f1_greater_than_1 = f1[y2]
            ec0_log_for_f1_greater_than_1 = np.log(1+f1_greater_than_1) 
            ec0_log_for_f1_less_than_1 = np.log1p(f1_less_than_1) 
            ec0_log[y1] = ec0_log_for_f1_less_than_1
            ec0_log[y2] = ec0_log_for_f1_greater_than_1
            if np.any(ec0_log == 0): 
                y = np.argwhere(ec0_log==0)[:,0]
                ec0_log[y] =  1e-15
            
            ecrs0 = ec0_q0*ec0_log
            decrs0_drs = -2*AAec0*alpha1ec0*ec0_log - ec0_q0*ec0_q1p*ec0_den
            
            f_zeta = 0
            fp_zeta = 0
            zeta4 = 0
            
            ec_lsda1 = ecrs0
            declsda1_drs = decrs0_drs
            
            r = 0.031091
            w1 = np.exp(-ec_lsda1/r)-1 
            t = ((3*np.pi**2)/16)**(1/3)*s/sqr_rs
            
            y = beta/(r*w1)*(t**2)
            
            deltafc2 = 1*(-0.64) + 2*(-0.4352) + 3*(-1.535685604549) + 4*3.061560252175 + 5*(-1.915710236206) + 6*0.516884468372 + 7*(-0.051848879792)
            
            ds = 1
            eta = 0.001
            dp2 = 0.361
            ec_lsda0 = ecLDA0
            declsda0_drs = b1c*(0.5*b2c*rs**(-0.5)+b3c)/((1+b2c*rs**(0.5) + b3c*rs)**2)
            
            deltayPart1 = deltafc2/(27*r*w1)
            deltayPart2 = 20*rs*(declsda0_drs-declsda1_drs) - 45*eta*(ec_lsda0-ec_lsda1)
            deltayPart3 = p*np.exp(-p**2/dp2**4)
            
            deltay = deltayPart1*deltayPart2*deltayPart3
            
            g = (1+4*(y-deltay))**(-0.25)   
            H1 = r*phi**3*np.log(1+w1*(1-g))
            ec1 = ec_lsda1+H1
            
            c1c = 0.64
            c2c = 1.5
            dc = 0.7
            alphaG25 = (alpha>2.5)
            alpha0To25 = ((alpha>=0) & (alpha<=2.5))
            alphaL0 = (alpha<0)

            fc = np.zeros((N_q))
            fc[alphaG25] = -dc*np.exp(c2c/(1-alpha[alphaG25]))
            fc[alpha0To25] = 1 + (-0.64)*alpha[alpha0To25] + (-0.4352)*alpha[alpha0To25]**2 + (-1.535685604549)*alpha[alpha0To25]**3 
            + 3.061560252175*alpha[alpha0To25]**4 + (-1.915710236206)*alpha[alpha0To25]**5 + 0.516884468372*alpha[alpha0To25]**6 
            + (-0.051848879792)*alpha[alpha0To25]**7
            fc[alphaL0] = np.exp(-c1c*alpha[alphaL0]/(1-alpha[alphaL0]))
            ec = ec1 + fc*(ec0-ec1)
            
            DrsDn = -4*np.pi/9*(4*np.pi/3*rho)**(-4/3)
            DgInf0sDs = -0.25*(1+4*chiInf*s*s)**(-1.25)*(4*chiInf*2*s)
            DgInf0sDn = DgInf0sDs*DsDn
            DgInf0sDDn = DgInf0sDs*DsDDn
            DecLDA0Dn = b1c*(0.5*b2c*rs**(-0.5)+b3c)/((1+b2c*rs**(0.5)+b3c*rs)**2)*DrsDn
            Dw0Dn = (w0+1)*(-DecLDA0Dn/b1c)
            DH0Dn = b1c*(Dw0Dn*(1-gInf0s)-w0*DgInf0sDn)/(1+w0*(1-gInf0s))
            DH0DDn = b1c*(-w0*DgInf0sDDn)/(1+w0*(1-gInf0s))
            Dec0Dn = DecLDA0Dn + DH0Dn
            Dec0DDn = DH0DDn
            
            dec_lsda1_drs = decrs0_drs
            
            dec0log_drs = -ec0_q1p*ec0_den
            dec0q0_drs = -2*AAec0*alpha1ec0
            dec0q1p_drs = AAec0*((-0.5*beta1ec0*rsm1_2/rs + 3*0.5*beta3ec0*rsm1_2+4*beta4ec0))
            dec0den_drs = -(2*ec0_q1*ec0_q1p+ec0_q1p)/(ec0_q1*ec0_q1+ec0_q1)**2
            d2ecrs0_drs2 = -2*AAec0*alpha1ec0*dec0log_drs - (dec0q0_drs*ec0_q1p*ec0_den+ec0_q0*dec0q1p_drs*ec0_den+ec0_q0*ec0_q1p*dec0den_drs)
            d2eclsda1_drs2 = d2ecrs0_drs2
            
            Ddeclsda1_drsDn = d2eclsda1_drs2*DrsDn
            
            Dec_lsda1Dn = (-rs/3*dec_lsda1_drs)/rho
            DbetaDn = 0.066725*(0.1*(1+0.1778*rs) - 0.1778*(1+0.1*rs))/((1+0.1778*rs)**2)*DrsDn
            
            DtDn = (3*np.pi**2/16)**(1/3)*(sqr_rs*DsDn - s*(DrsDn/(2*sqr_rs)))/rs
            DtDDn = t*DsDDn/s

            Dw1Dn = (w1+1)*(-(r*Dec_lsda1Dn)/(r**2))
            DyDn = (w1*DbetaDn - beta*Dw1Dn)/(r*w1**2)*(t**2) + beta/(r*w1)*(2*t)*DtDn
            DyDDn = beta/(r*w1) * (2*t)*DtDDn
            
            Declsda0Dn = declsda0_drs*DrsDn
            
            d2eclsda0_drs2 = b1c*((0.5*b2c*(-0.5)*rs**(-1.5))*((1+b2c*rs**(0.5)+b3c*rs)**2)-(0.5*b2c*rs**(-0.5)+b3c)*2*(1+b2c*rs**(0.5)+b3c*rs)*(0.5*b2c*rs**(-0.5)+b3c))/(((1+b2c*rs**(0.5)+b3c*rs)**2)**2)
            Ddeclsda0_drsDn = d2eclsda0_drs2*DrsDn
            
            d_deltayPart1_dn = 0 + 0 + deltafc2/(27*r)*(-1)*w1**(-2)*Dw1Dn
            d_deltayPart2_dn = 20*(declsda0_drs-declsda1_drs)*DrsDn + 20*rs*(Ddeclsda0_drsDn - Ddeclsda1_drsDn) - 45*eta*(Declsda0Dn - Dec_lsda1Dn)
            d_deltayPart3_dp = np.exp(-p**2/dp2**4) + p*np.exp(-p**2/dp2**4)*(-2*p/dp2**4)
            
            DdeltayDn = d_deltayPart1_dn*deltayPart2*deltayPart3 + deltayPart1*d_deltayPart2_dn*deltayPart3 + deltayPart1*deltayPart2*d_deltayPart3_dp*DpDn
            
            DdeltayDDn = deltayPart1*deltayPart2*d_deltayPart3_dp*DpDDn

            DgDn = -0.25*(1+4*(y-deltay))**(-1.25)*(4*(DyDn-DdeltayDn))
            DgDDn = -0.25*(1+4*(y-deltay))**(-1.25)*(4*(DyDDn-DdeltayDDn))
            
            DH1Dn = r*((Dw1Dn*(1-g)-w1*DgDn)/(1+w1*(1-g)))
            DH1DDn = r*((-w1*DgDDn)/(1+w1*(1-g)))

            Dec1Dn = Dec_lsda1Dn + DH1Dn
            Dec1DDn = DH1DDn
            
            DfcDalpha = np.zeros((N_q))
            DfcDalpha[alphaG25] = fc[alphaG25]*(c2c/(1-alpha[alphaG25])**2)
            DfcDalpha[alpha0To25] = (-0.64) + (-0.4352)*alpha[alpha0To25]*2 + (-1.535685604549)*alpha[alpha0To25]**2*3 
            + 3.061560252175*alpha[alpha0To25]**3*4 + (-1.915710236206)*alpha[alpha0To25]**4*5 + 0.516884468372*alpha[alpha0To25]**5*6
            + (-0.051848879792)*alpha[alpha0To25]**6*7
            DfcDalpha[alphaL0] = fc[alphaL0]*(-c1c/(1-alpha[alphaL0])**2)
            DfcDn = DfcDalpha*DalphaDn
            DfcDDn = DfcDalpha*DalphaDDn
            DfcDtau = DfcDalpha*DalphaDtau
            DepsiloncDn = Dec1Dn + fc*(Dec0Dn-Dec1Dn) + DfcDn*(ec0-ec1)
            DepsiloncDDn = Dec1DDn + fc*(Dec0DDn-Dec1DDn) + DfcDDn*(ec0-ec1)
            DepsiloncDtau = DfcDtau*(ec0-ec1)
            
            v1c = ec + rho*DepsiloncDn
            v2c = rho*DepsiloncDDn
            v3c = rho*DepsiloncDtau
            
            return [ec, v1c, v2c,v3c]  
        ec, v1c, v2c, v3c = correlationr2SCAN(rho, s, p, alpha, DsDn, DsDDn, DpDn, DpDDn, DalphaDn, DalphaDDn, DalphaDtau)
        v2c = v2c/normDrho

        return GenericXCResult(
            v_generic = v1c,
            e_generic = ec,
            de_dsigma = v2c,
            de_dtau   = v3c
        )
    
    



if __name__ == "__main__":
    pass