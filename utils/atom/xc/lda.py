"""
LDA (Local Density Approximation) Functionals

Implements LDA exchange-correlation functionals:
- LDA_PZ: Slater exchange + Perdew-Zunger correlation (uses VWN in current implementation)
- LDA_PW: Slater exchange + Perdew-Wang correlation

LDA functionals only depend on local density ρ(r), not on gradients or tau.
Therefore, no spherical transformation is needed (already in radial form).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .evaluator import XCEvaluator, XCParameters, GenericXCResult, DensityData


@dataclass
class LDASVWNParameters(XCParameters):
    """
    Parameters for LDA_PZ functional (Slater + VWN correlation, matching SPARC's LDA_PZ name).
    
    All parameters can be optimized using autodiff for delta learning.
    
    Attributes
    ----------
    C_x : float
        Exchange multiplier (scales the exchange contribution)
        Standard: 1.0 (no scaling)
        - C_x = 1.0: standard Slater exchange
        - C_x > 1.0: enhanced exchange
        - C_x < 1.0: reduced exchange
    
    # VWN Correlation Parameters (Vosko-Wilk-Nusair 1980)
    A : float
        VWN parameter A (controls correlation strength)
        Standard: 0.0621814
    b : float
        VWN parameter b
        Standard: 3.72744
    c : float
        VWN parameter c
        Standard: 12.9352
    y0 : float
        VWN parameter y
        Standard: -0.10498
    """
    functional_name: str = 'LDA_PZ'  # Fixed for this functional (matches SPARC naming)
    
    # Slater exchange multiplier
    C_x: float = 1.0  # Default: 1.0 (standard Slater)
    
    # VWN correlation (standard values from VWN 1980)
    A : float = 0.0621814
    b : float = 3.72744
    c : float = 12.9352
    y0: float = -0.10498

@dataclass
class LDASPWParameters(XCParameters):
    """
    Parameters for LDA_PW functional (Slater + Perdew-Wang).
    
    All parameters can be optimized using autodiff for delta learning.
    
    Attributes
    ----------
    C_x : float
        Exchange multiplier (scales the exchange contribution)
        Standard: 1.0 (no scaling)
    
    # Perdew-Wang Correlation Parameters (Perdew-Wang 1992)
    A : float
        PW parameter A
        Standard: 0.031091
    alpha1 : float
        PW parameter α₁
        Standard: 0.21370
    beta1 : float
        PW parameter β₁
        Standard: 7.5957
    beta2 : float
        PW parameter β₂
        Standard: 3.5876
    beta3 : float
        PW parameter β₃
        Standard: 1.6382
    beta4 : float
        PW parameter β₄
        Standard: 0.49294
    """
    functional_name: str = 'LDA_PW'  # Fixed for this functional (matches SPARC naming)
    
    # Slater exchange multiplier
    C_x: float = 1.0  # Default: 1.0 (standard Slater)
    
    # Perdew-Wang correlation parameters (standard values from PW 1992)
    A: float = 0.031091
    alpha1: float = 0.21370
    beta1 : float = 7.5957
    beta2 : float = 3.5876
    beta3 : float = 1.6382
    beta4 : float = 0.49294


class LDA_SVWN(XCEvaluator):
    """
    LDA with Slater exchange and Vosko-Wilk-Nusair correlation.
    
    Note: This is mapped to 'LDA_PZ' to match SPARC naming convention,
    though SPARC's LDA_PZ uses Perdew-Zunger correlation instead.
    
    Exchange: Slater (1951)
    Correlation: Vosko, Wilk, Nusair (1980)
    
    Only requires electron density ρ(r).
    No gradient transformation needed (LDA is already local).
    """
    
    def _default_params(self) -> LDASVWNParameters:
        """
        Return default LDA-SVWN parameters.
        
        Returns
        -------
        params : LDASVWNParameters
            Standard Slater exchange (C_x=1.0) + VWN correlation parameters
        """
        return LDASVWNParameters()  # All defaults are in the class definition
    
    
    def compute_exchange_generic(
        self,
        density_data: DensityData
        ) -> GenericXCResult:
        """
        Compute Slater exchange in generic form.
        
        For LDA, the generic form IS the spherical form (no gradients involved).
        
        Parameters
        ----------
        density_data : DensityData
            Container with electron density ρ(r)
        
        Returns
        -------
        result : GenericXCResult
            Exchange results with:
            - v_generic: Vₓ(ρ) (exchange potential)
            - e_generic: εₓ(ρ) (exchange energy density)
            - de_dsigma: None (LDA has no gradient dependence)
            - de_dtau: None (LDA has no tau dependence)
        
        Notes
        -----
        Slater exchange (1951):
            εₓ(ρ) = C_x * [-3/(4π) * (3π²)^(1/3) * ρ^(1/3)]
            Vₓ(ρ) = C_x * [-0.9847450218427 * ρ^(1/3)]
        
        where C_x is a multiplier (default: 1.0 for standard Slater).
        """
        rho      = density_data.rho
        rho_cbrt = rho**(1/3)  # Cube root of density
        
        # Standard Slater exchange (before multiplier)
        e_x_standard = -0.7385587663820224 * rho_cbrt   # (3/4) * (3π²)^(1/3) / π
        v_x_standard = -0.9847450218426966 * rho_cbrt   # (3π²)^(1/3) / π
        
        # Apply exchange multiplier (for delta learning optimization)
        C_x = self.params.C_x
        e_x = C_x * e_x_standard
        v_x = C_x * v_x_standard
        
        return GenericXCResult(
            v_generic=v_x,
            e_generic=e_x,
            de_dsigma=None,  # LDA: no gradient dependence
            de_dtau=None     # LDA: no tau dependence
        )
    
    
    def compute_correlation_generic(
        self,
        density_data: DensityData
        ) -> GenericXCResult:
        """
        Compute VWN (Vosko-Wilk-Nusair) correlation in generic form.
        
        For LDA, the generic form IS the spherical form (no gradients involved).
        
        Parameters
        ----------
        density_data : DensityData
            Container with electron density ρ(r)
        
        Returns
        -------
        result : GenericXCResult
            Correlation results with:
            - v_generic: Vc(ρ) (correlation potential)
            - e_generic: εc(ρ) (correlation energy density)
            - de_dsigma: None (LDA has no gradient dependence)
            - de_dtau: None (LDA has no tau dependence)
        
        Notes
        -----
        VWN correlation (1980):
        A parametrization of the correlation energy for the uniform electron gas.
        
        Reference:
            Vosko, Wilk, Nusair, Can. J. Phys. 58, 1200 (1980)
        """
        rho = density_data.rho
        rho_cbrt = rho**(1/3)  # Cube root of density
        
        # Get VWN parameters from self.params (can be optimized via autodiff!)
        A = self.params.A
        b = self.params.b
        c = self.params.c
        y0 = self.params.y0
        
        # Compute rs (Wigner-Seitz radius) and its square root
        rs_wigner_seitz = (3 / (4 * np.pi))**(1/3) / rho_cbrt
        rs_sqrt = rs_wigner_seitz**(0.5)
        
        # Auxiliary quantities for VWN parametrization
        Q_vwn = (4*c - b**2)**(0.5)
        poly_y0 = y0**2 + b*y0 + c  # Polynomial at y0
        poly_y  = rs_sqrt**2 + b*rs_sqrt + c  # Polynomial at y = sqrt(rs)
        
        # VWN correlation energy density
        log_term1 = np.log(rs_sqrt**2 / poly_y)
        arctan_term = 2*b/Q_vwn * np.arctan(Q_vwn / (2*rs_sqrt + b))
        log_term2 = np.log((rs_sqrt - y0)**2 / poly_y)
        arctan_term2 = 2*(b + 2*y0)/Q_vwn * np.arctan(Q_vwn / (2*rs_sqrt + b))
        
        e_c = A/2 * (
            log_term1 
            + arctan_term
            - b*y0/poly_y0 * (log_term2 + arctan_term2)
        )
        
        # VWN correlation potential (functional derivative)
        potential_correction = A/6 * (c*(rs_sqrt - y0) - b*y0*rs_sqrt) / ((rs_sqrt - y0) * poly_y)
        v_c = e_c - potential_correction
        
        return GenericXCResult(
            v_generic=v_c,
            e_generic=e_c,
            de_dsigma=None,  # LDA: no gradient dependence
            de_dtau=None     # LDA: no tau dependence
        )


class LDA_SPW(XCEvaluator):
    """
    LDA with Slater exchange and Perdew-Wang correlation.
    
    Exchange: Slater (1951)
    Correlation: Perdew-Wang (1992)
    
    Only requires electron density ρ(r).
    No gradient transformation needed (LDA is already local).
    """
    
    def _default_params(self) -> LDASPWParameters:
        """
        Return default LDA-SPW parameters.
        
        Returns
        -------
        params : LDASPWParameters
            Standard Slater exchange (C_x=1.0) + Perdew-Wang correlation parameters
        """
        return LDASPWParameters()  # All defaults are in the class definition
    
    def compute_exchange_generic(
        self,
        density_data: DensityData
        ) -> GenericXCResult:
        """
        Compute Slater exchange (same as LDA_PZ).
        
        Parameters
        ----------
        density_data : DensityData
            Container with electron density
        
        Returns
        -------
        result : GenericXCResult
            Exchange results
        
        Notes
        -----
        Uses the same Slater exchange as LDA_PZ.
        Only the correlation part differs (Perdew-Wang vs VWN).
        """
        rho = density_data.rho
        rho_cbrt = rho**(1/3)  # Cube root of density
        
        # Standard Slater exchange (before multiplier)
        e_x_standard = -0.7385587663820224 * rho_cbrt   # (3/4) * (3π²)^(1/3) / π
        v_x_standard = -0.9847450218426966 * rho_cbrt   # (3π²)^(1/3) / π
        
        # Apply exchange multiplier
        C_x = self.params.C_x
        e_x = C_x * e_x_standard
        v_x = C_x * v_x_standard
        
        return GenericXCResult(
            v_generic=v_x,
            e_generic=e_x,
            de_dsigma=None,
            de_dtau=None
        )
    
    
    def compute_correlation_generic(
        self,
        density_data: DensityData
        ) -> GenericXCResult:
        """
        Compute Perdew-Wang correlation.
        
        Parameters
        ----------
        density_data : DensityData
            Container with electron density
        
        Returns
        -------
        result : GenericXCResult
            Correlation results
        
        Notes
        -----
        Perdew-Wang 1992 correlation parametrization.
        See: Perdew & Wang, Phys. Rev. B 45, 13244 (1992)
        """
        rho = density_data.rho
        rho_cbrt = rho**(1/3)  # Cube root of density
        
        # Get Perdew-Wang parameters from self.params (can be optimized via autodiff!)
        A = self.params.A
        alpha1 = self.params.alpha1
        beta1 = self.params.beta1
        beta2 = self.params.beta2
        beta3 = self.params.beta3
        beta4 = self.params.beta4
        
        # Compute rs (Wigner-Seitz radius) and its powers
        rs_wigner_seitz = ((0.75 / np.pi)**(1/3)) * (1 / rho_cbrt)
        rs_sqrt = np.sqrt(rs_wigner_seitz)      # rs^(1/2)
        rs_inv_sqrt = 1 / rs_sqrt               # rs^(-1/2)
        rs_3_2 = rs_wigner_seitz * rs_sqrt      # rs^(3/2)
        rs_squared = rs_wigner_seitz**2         # rs^2
        
        # Compute omega function and its rs-derivative (PW parametrization)
        omega = 2*A * (beta1*rs_sqrt + beta2*rs_wigner_seitz + beta3*rs_3_2 + beta4*rs_squared)
        d_omega_d_rs = A * (beta1*rs_inv_sqrt + 2*beta2 + 3*beta3*rs_sqrt + 4*beta4*rs_wigner_seitz)
        
        # Logarithmic term
        log_term = np.log(1 + 1/omega)
        
        # PW correlation energy density
        prefactor = -2*A * (1 + alpha1*rs_wigner_seitz)
        e_c = prefactor * log_term
        
        # PW correlation potential (functional derivative)
        rs_derivative_term = (rs_wigner_seitz/3) * (
            -2*A*alpha1*log_term 
            - (prefactor*d_omega_d_rs) / (omega + omega**2)
        )
        v_c = e_c - rs_derivative_term
        
        return GenericXCResult(
            v_generic=v_c,
            e_generic=e_c,
            de_dsigma=None,
            de_dtau=None
        )