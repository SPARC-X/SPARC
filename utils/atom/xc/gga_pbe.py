"""
GGA PBE (Generalized Gradient Approximation - Perdew-Burke-Ernzerhof) Functional

Implements the PBE GGA functional for exchange and correlation.

Two-Stage Implementation:
=========================
Stage 1 (Generic): PBE functional in its original form
  - Uses ρ and σ = |∇ρ|² in generic form
  - Returns v_generic (∂ε/∂ρ), e_generic (ε), and de_dsigma (∂ε/∂σ)
  
Stage 2 (Spherical): Transform potential to radial atomic form
  - Apply gradient correction: V = ∂ε/∂ρ - (2/r²)·d/dr[r²·∂ε/∂σ·dρ/dr]
  - Energy density ε remains unchanged (scalar, coordinate-independent)

Reference:
    Perdew, Burke, and Ernzerhof, Phys. Rev. Lett. 77, 3865 (1996)
"""

from __future__ import annotations
import numpy as np
from typing import Optional
from dataclasses import dataclass

from .evaluator import XCEvaluator, XCParameters, GenericXCResult, DensityData


@dataclass
class PBEParameters(XCParameters):
    """
    Parameters for PBE GGA functional.
    
    All parameters can be optimized using autodiff for delta learning.
    
    Attributes
    ----------
    mu : float
        PBE gradient enhancement parameter
        Standard: 0.2195149727645171
    kappa : float
        PBE parameter κ
        Standard: 0.804
    """
    functional_name: str = 'GGA_PBE'  # Fixed for this functional
    mu: float = 0.2195149727645171  # Default: standard PBE value
    kappa: float = 0.804  # Default: standard PBE value


class GGA_PBE(XCEvaluator):
    """
    GGA-PBE exchange-correlation functional.
    
    Requires:
    - ρ(r): electron density
    - |∇ρ(r)|: density gradient magnitude
    - derivative_matrix: for spherical coordinate transformation
    
    Two-stage calculation:
    1. Generic form: PBE formulas with ρ and |∇ρ|
    2. Spherical transform: Convert gradient terms to radial form
    
    Examples
    --------
    >>> from atomic_dft.mesh import MeshBuilder
    >>> from atomic_dft.xc import GGA_PBE
    >>> 
    >>> mesh = MeshBuilder(...).build()
    >>> D = mesh.derivative_matrix
    >>> 
    >>> evaluator = GGA_PBE(derivative_matrix=D)
    >>> density_data = DensityData(rho=rho, grad_rho=grad_rho)
    >>> potential_data = evaluator.compute_xc(density_data)
    """
    
    def __init__(
        self, 
        derivative_matrix: Optional[np.ndarray] = None,
        r_quad: Optional[np.ndarray] = None,
        params: Optional[XCParameters] = None
    ):
        """
        Initialize GGA-PBE evaluator.
        
        Parameters
        ----------
        derivative_matrix : np.ndarray
            FEM derivative matrix for gradient transformations
            REQUIRED for GGA functionals
        r_quad : np.ndarray, optional
            Radial quadrature nodes (coordinates)
            Required for spherical coordinate transformations
        params : XCParameters, optional
            Functional parameters
            
        Raises
        ------
        ValueError
            If derivative_matrix is None (required for GGA)
        """
        if derivative_matrix is None:
            raise ValueError(
                "GGA_PBE requires derivative_matrix for gradient transformations"
            )
        if r_quad is None:
            raise ValueError(
                "GGA_PBE requires r_quad for spherical coordinate transformations"
            )
        super().__init__(derivative_matrix=derivative_matrix, r_quad=r_quad, params=params)
    
    def _default_params(self) -> PBEParameters:
        """
        Return default PBE parameters.
        
        Returns
        -------
        params : PBEParameters
            Standard PBE parameters (Perdew, Burke, Ernzerhof 1996)
        """
        return PBEParameters()  # All defaults are in the class definition
    


    
    def compute_exchange_generic(
        self,
        density_data: DensityData
        ) -> GenericXCResult:
        """
        Compute PBE exchange in GENERIC form (Stage 1).
        
        Implements PBE exchange as published, using ρ and σ = |∇ρ|².
        
        Parameters
        ----------
        density_data : DensityData
            Container with rho and grad_rho
        
        Returns
        -------
        result : GenericXCResult
            Exchange results with:
            - v_generic: ∂εₓ/∂ρ (V_1_X_term in reference)
            - e_generic: εₓ (ex in reference)
            - de_dsigma: ∂εₓ/∂σ (V_2_X_term/rho in reference)
            - de_dtau: None (GGA doesn't use tau)
        
        Raises
        ------
        ValueError
            If grad_rho is None (required for GGA)
        
        Notes
        -----
        PBE exchange enhancement factor:
            F_x(s) = 1 + κ - κ/(1 + μs²/κ)
        where s = |∇ρ|/(2k_F ρ) is the reduced gradient
        
        The de_dsigma term is essential for the spherical transformation:
            V_x^spherical = v_generic - (2/r²)·d/dr[r²·de_dsigma·dρ/dr]
        """
        rho = density_data.rho
        grad_rho = density_data.grad_rho
        
        if grad_rho is None:
            raise ValueError("GGA_PBE requires grad_rho for exchange calculation")
        
        # PBE parameters
        mu = self.params.mu
        kappa = self.params.kappa
        
        # Constants
        threefourth_divpi = 3.0 / 4.0 / np.pi
        sixpi2_1_3 = (6.0 * (np.pi**2))**(1.0/3.0)
        sixpi2m1_3 = 1.0 / sixpi2_1_3
        mu_divkappa = mu / kappa
        
        # Spin-unpolarized: rho_up = rho_down = rho/2
        rho_updn = rho / 2.0
        rho_updnm1_3 = rho_updn**(-1.0/3.0)
        rhomot = rho_updnm1_3
        
        # LDA exchange energy density
        ex_lsd = -threefourth_divpi * sixpi2_1_3 * (rhomot * rhomot * rho_updn)
        
        # Reduced gradient variable
        rho_inv = rhomot * rhomot * rhomot
        coeffss = (1.0/4.0) * sixpi2m1_3 * sixpi2m1_3 * (rho_inv * rho_inv * rhomot * rhomot)
        
        # sigma = |∇ρ|²
        sigma = grad_rho**2
        # Avoid division by zero
        sigma = np.where(sigma < 1e-20, 1e-20, sigma)
        
        # s² term
        ss = (sigma / 4.0) * coeffss
        divss = 1.0 / (1.0 + mu_divkappa * ss)
        
        # Enhancement factor and derivatives
        dfxdss = mu * (divss**2)
        fx = 1.0 + kappa * (1.0 - divss)
        
        # Derivatives w.r.t. density and gradient
        dssdn = (-8.0/3.0) * (ss * rho_inv)
        dfxdn = dfxdss * dssdn
        dssdg = 2.0 * coeffss
        dfxdg = dfxdss * dssdg
        
        # Energy density
        ex = ex_lsd * fx
        
        # Potential: ∂(ρ·εx)/∂ρ = εx + ρ·∂εx/∂ρ
        v_x_generic = ex_lsd * ((4.0/3.0) * fx + rho_updn * dfxdn)
        
        # Derivative w.r.t. sigma (needed for spherical transform)
        # de_x_dsigma = ∂εx/∂σ = V_2_X_term from reference code
        # V_2_X_term = 0.5*ex_lsd*rho_updn*dfxdg
        de_x_dsigma = 0.5 * ex_lsd * rho_updn * dfxdg
        
        return GenericXCResult(
            v_generic=v_x_generic,
            e_generic=ex,
            de_dsigma=de_x_dsigma,
            de_dtau=None
        )
    
    
    def compute_correlation_generic(
        self,
        density_data: DensityData
        ) -> GenericXCResult:
        """
        Compute PBE correlation in GENERIC form (Stage 1).
        
        Implements PBE correlation as published, using ρ and σ = |∇ρ|².
        
        Parameters
        ----------
        density_data : DensityData
            Container with rho and grad_rho
        
        Returns
        -------
        result : GenericXCResult
            Correlation results with:
            - v_generic: ∂εc/∂ρ (V_1_C_term in reference)
            - e_generic: εc (ec in reference)
            - de_dsigma: ∂εc/∂σ (V_2_C_term/rho in reference)
            - de_dtau: None
        
        Raises
        ------
        ValueError
            If grad_rho is None (required for GGA)
        
        Notes
        -----
        PBE correlation is based on the LDA correlation plus gradient corrections.
        """
        rho = density_data.rho
        grad_rho = density_data.grad_rho
        
        if grad_rho is None:
            raise ValueError("GGA_PBE requires grad_rho for correlation calculation")
        
        # PBE correlation parameter
        beta = 0.066725
        
        # Constants
        rsfac = 0.6203504908994000
        sq_rsfac = rsfac**(0.5)
        sq_rsfac_inverse = 1.0 / sq_rsfac
        third = 1.0 / 3.0
        twom1_3 = 2.0**(-1.0/3.0)
        
        # Perdew-Wang LDA correlation parameters
        ec0_aa = 0.031091
        ec0_a1 = 0.21370
        ec0_b1 = 7.5957
        ec0_b2 = 3.5876
        ec0_b3 = 1.6382
        ec0_b4 = 0.49294
        
        gamma = (1.0 - np.log(2.0)) / ((np.pi)**2)
        gamma_inv = 1.0 / gamma
        coeff_tt = 1.0 / ((4.0*4.0/np.pi) * ((3.0*(np.pi)**2)**(third)))
        
        # Spin-unpolarized: rho_up = rho_down = rho/2
        rho_u_d_1_3 = (rho / 2.0)**(-1.0/3.0)
        rho_m_1_3 = twom1_3 * rho_u_d_1_3
        rho_tot_inverse = (rho_m_1_3**3)
        rhotmo6 = (rho_m_1_3)**(0.5)
        rhoto6 = rho * (rho_m_1_3**2) * rhotmo6
        
        # Seitz radius rs
        rs = rsfac * rho_m_1_3
        sqr_rs = sq_rsfac * rhotmo6
        rsm1_2 = sq_rsfac_inverse * rhoto6
        
        # LDA correlation (Perdew-Wang parametrization)
        ec0_q0 = -2.0 * ec0_aa * (1.0 + ec0_a1 * rs)
        ec0_q1 = 2.0 * ec0_aa * (ec0_b1 * sqr_rs + ec0_b2 * rs + 
                                 ec0_b3 * rs * sqr_rs + ec0_b4 * rs * rs)
        ec0_q1p = ec0_aa * (ec0_b1 * rsm1_2 + 2.0 * ec0_b2 + 
                            3.0 * ec0_b3 * sqr_rs + 4.0 * ec0_b4 * rs)
        ec0_den = 1.0 / (ec0_q1 * ec0_q1 + ec0_q1)
        
        # Compute logarithm carefully to avoid numerical issues
        ec0_log = np.zeros(len(rho))
        f1 = 1.0 / ec0_q1
        y1 = np.argwhere(f1 < 1)[:, 0]
        f1_less_than_1 = f1[y1]
        y2 = np.argwhere(f1 >= 1)[:, 0]
        f1_greater_than_1 = f1[y2]
        ec0_log[y1] = np.log1p(f1_less_than_1)
        ec0_log[y2] = np.log(1 + f1_greater_than_1)
        # Avoid exact zeros
        ec0_log = np.where(ec0_log == 0, 1e-15, ec0_log)
        
        # LDA correlation energy density and potential
        ecrs0 = ec0_q0 * ec0_log
        ec = ecrs0
        decrs0_drs = -2.0 * ec0_aa * ec0_a1 * ec0_log - ec0_q0 * ec0_q1p * ec0_den
        v_c_lda = ecrs0 - (rs / 3.0) * decrs0_drs
        
        # PBE gradient correction
        bb = ecrs0 * gamma_inv
        exp_pbe = np.exp(-bb)
        dbb_drs = decrs0_drs * gamma_inv
        cc = 1.0 / (exp_pbe - 1.0)
        dcc_dbb = cc * cc * exp_pbe
        dcc_drs = dcc_dbb * dbb_drs
        coeff_aa = beta * gamma_inv
        aa = coeff_aa * cc
        daa_drs = coeff_aa * dcc_drs
        
        # Gradient term t
        dtt_dg = 2.0 * (rho_tot_inverse * rho_tot_inverse) * rho_m_1_3 * coeff_tt
        sigma = grad_rho**2
        tt = 0.5 * sigma * dtt_dg
        
        # A(rs, t) and derivatives
        xx = aa * tt
        dxx_drs = daa_drs * tt
        dxx_dtt = aa
        
        # Padé approximant
        pade_den = 1.0 / (1.0 + xx * (1.0 + xx))
        pade = (1.0 + xx) * pade_den
        dpade_dxx = -xx * (2.0 + xx) * (pade_den**2)
        dpade_drs = dpade_dxx * dxx_drs
        dpade_dtt = dpade_dxx * dxx_dtt
        
        # H(rs, t)
        qq = tt * pade
        dqq_drs = tt * dpade_drs
        dqq_dtt = pade + tt * dpade_dtt
        
        # Gradient correction to correlation energy
        arg_rr = 1.0 + beta * gamma_inv * qq
        div_rr = 1.0 / arg_rr
        rr = gamma * np.log(arg_rr)
        drr_dqq = beta * div_rr
        drr_drs = drr_dqq * dqq_drs
        drr_dtt = drr_dqq * dqq_dtt
        
        # Full correlation potential and energy
        drohh_drho = rr - third * rs * drr_drs - (7.0/3.0) * tt * drr_dtt
        ec = ec + rr
        v_c_generic = v_c_lda + drohh_drho
        
        # Derivative w.r.t. sigma (needed for spherical transform)
        # de_c_dsigma = ∂εc/∂σ = V_2_C_term from reference code
        # V_2_C_term = rho * dtt_dg * drr_dtt
        de_c_dsigma = rho * dtt_dg * drr_dtt
        
        return GenericXCResult(
            v_generic=v_c_generic,
            e_generic=ec,
            de_dsigma=de_c_dsigma,
            de_dtau=None
        )
    
