"""
Exchange-Correlation Evaluator Base Class

This module provides the abstract base class for all XC functional implementations.

Design Philosophy:
==================
All XC functionals are implemented in TWO stages:

Stage 1: Generic Form (compute_exchange_generic / compute_correlation_generic)
  - Implements the XC functional in its original form
  - Usually defined in 3D Cartesian coordinates
  - Returns generic potentials and energy densities

Stage 2: Spherical Transform (transform_to_spherical)
  - Transforms generic form to spherical coordinates
  - Applies rotational symmetry simplifications
  - Handles gradient transformations: ∇ρ in Cartesian → ∂ρ/∂r in spherical
  - Returns final spherical potentials ready for radial Schrödinger equation

This separation ensures:
  - Clear distinction between functional definition and coordinate system
  - Reusable generic implementations
  - Proper handling of spherical symmetry in atomic calculations
"""

from __future__ import annotations
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np

# Import DensityData from scf module
from ..scf.density import DensityData


@dataclass
class XCParameters:
    """
    Base class for XC functional parameters.
    
    This is a simple container for functional parameters.
    Each functional should subclass this to define its specific parameters.
    
    Attributes
    ----------
    functional_name : str
        Name of the functional (e.g., 'LDA_PZ', 'GGA_PBE', 'SCAN')
    
    Examples
    --------
    >>> # Subclass for specific functional
    >>> @dataclass
    >>> class LDAParameters(XCParameters):
    ...     C_x: float = -(3/4) * (3/np.pi)**(1/3)  # Slater constant
    >>> 
    >>> params = LDAParameters(functional_name='LDA_PZ')
    >>> print(params.C_x)
    
    Notes
    -----
    For JAX compatibility, subclasses can be registered as pytrees.
    All numeric fields will be differentiable.
    """
    functional_name: str
    
    def __repr__(self) -> str:
        """String representation"""
        params_str = ', '.join(
            f'{k}={v:.6f}' if isinstance(v, float) else f'{k}={v}'
            for k, v in self.__dict__.items()
            if k != 'functional_name'
        )
        return f"{self.__class__.__name__}({params_str})"


@dataclass(frozen=True)
class GenericXCResult:
    """
    Container for exchange or correlation results in GENERIC form (Stage 1).
    
    This contains the raw outputs from the XC functional evaluation before
    spherical coordinate transformation.
    
    Attributes
    ----------
    v_generic : np.ndarray
        Potential in generic form: ∂ε/∂ρ
    e_generic : np.ndarray
        Energy density in generic form: ε
    de_dsigma : np.ndarray, optional
        Derivative ∂ε/∂σ where σ = |∇ρ|² (for GGA and meta-GGA)
    de_dtau : np.ndarray, optional
        Derivative ∂ε/∂τ (for meta-GGA)
    """
    v_generic: np.ndarray      # ∂ε/∂ρ
    e_generic: np.ndarray      # ε
    de_dsigma: Optional[np.ndarray] = None  # ∂ε/∂σ (GGA, meta-GGA)
    de_dtau  : Optional[np.ndarray] = None  # ∂ε/∂τ (meta-GGA)


@dataclass(frozen=True)
class XCPotentialData:
    """
    Container for exchange-correlation potentials and energy densities.
    
    This dataclass is immutable (frozen=True) to prevent accidental modification.
    All XC functional evaluations must return this type.
    
    These are the FINAL potentials in spherical coordinates, ready to be used
    in the radial Kohn-Sham equation.
    
    Attributes
    ----------
    v_x : np.ndarray
        Exchange potential V_x(r) at quadrature points (spherical form)
    v_c : np.ndarray
        Correlation potential V_c(r) at quadrature points (spherical form)
    e_x : np.ndarray
        Exchange energy density ε_x(r) at quadrature points (spherical form)
    e_c : np.ndarray
        Correlation energy density ε_c(r) at quadrature points (spherical form)
    de_x_dtau : np.ndarray, optional
        Derivative ∂ε_x/∂τ of exchange energy density w.r.t. kinetic energy density (for meta-GGA)
    de_c_dtau : np.ndarray, optional
        Derivative ∂ε_c/∂τ of correlation energy density w.r.t. kinetic energy density (for meta-GGA)
    
    Examples
    --------
    >>> pot_data = XCPotentialData(
    ...     v_x=np.array([...]),
    ...     v_c=np.array([...]),
    ...     e_x=np.array([...]),
    ...     e_c=np.array([...])
    ... )
    >>> v_total = pot_data.v_xc  # V_x + V_c
    """
    v_x: np.ndarray  # Exchange potential (spherical)
    v_c: np.ndarray  # Correlation potential (spherical)
    e_x: np.ndarray  # Exchange energy density (spherical)
    e_c: np.ndarray  # Correlation energy density (spherical)
    de_x_dtau: Optional[np.ndarray] = None  # ∂ε_x/∂τ (meta-GGA)
    de_c_dtau: Optional[np.ndarray] = None  # ∂ε_c/∂τ (meta-GGA)
    
    def __post_init__(self):
        """Validate shapes - all arrays must have the same length"""
        n_points = self.v_x.shape[0]
        assert self.v_c.shape[0] == n_points, "v_c shape mismatch"
        assert self.e_x.shape[0] == n_points, "e_x shape mismatch"
        assert self.e_c.shape[0] == n_points, "e_c shape mismatch"
        if self.de_x_dtau is not None:
            assert self.de_x_dtau.shape[0] == n_points, "de_x_dtau shape mismatch"
        if self.de_c_dtau is not None:
            assert self.de_c_dtau.shape[0] == n_points, "de_c_dtau shape mismatch"
    
    @property
    def v_xc(self) -> np.ndarray:
        """Total XC potential: V_xc = V_x + V_c"""
        return self.v_x + self.v_c
    
    @property
    def e_xc(self) -> np.ndarray:
        """Total XC energy density: ε_xc = ε_x + ε_c"""
        return self.e_x + self.e_c
    
    @property
    def de_xc_dtau(self) -> np.ndarray:
        """Total XC energy density derivative ∂ε_xc/∂τ"""
        if self.de_x_dtau is None or self.de_c_dtau is None:
            assert (self.de_x_dtau is None and self.de_c_dtau is None), \
                "de_x_dtau and de_c_dtau must both be None or both be not None"
            return None
        return self.de_x_dtau + self.de_c_dtau


class XCEvaluator(ABC):
    """
    Abstract base class for exchange-correlation functional evaluators.
    
    All XC functionals must inherit from this class and implement:
    
    Required Abstract Methods:
    -------------------------
    1. compute_exchange_generic(rho, grad_rho, tau, ...) → (v_x_generic, e_x_generic)
       - Implements exchange in its original/generic form
       - Static method: pure function, no side effects
       
    2. compute_correlation_generic(rho, grad_rho, tau, ...) → (v_c_generic, e_c_generic)
       - Implements correlation in its original/generic form
       - Static method: pure function, no side effects
    
    Provided Methods:
    ----------------
    - compute_xc(density_data, derivative_matrix) → PotentialData
      Orchestrates the two-stage calculation:
      Stage 1: compute generic form
      Stage 2: transform to spherical coordinates
    
    - transform_to_spherical(...) → transformed potentials
      Default implementation for spherical coordinate transformation
      Can be overridden for functionals with special requirements
    
    Design Philosophy:
    -----------------
    Stage 1 (Generic): Functional definition (universal)
      Input: ρ, ∇ρ, τ in generic form
      Output: V_x, V_c, ε_x, ε_c in generic form
      
    Stage 2 (Spherical): Coordinate transformation (atom-specific)
      Input: Generic potentials + derivative matrix
      Output: Spherical potentials for radial equation
      Transformation: ∇ρ(x,y,z) → ∂ρ/∂r (radial derivative)
    
    Examples
    --------
    >>> # Define parameter class for your functional
    >>> @dataclass
    >>> class MyLDAParameters(XCParameters):
    ...     C_x: float = -(3/4) * (3/np.pi)**(1/3)  # Slater constant
    >>> 
    >>> class MyLDA(XCEvaluator):
    ...     def _default_params(self):
    ...         return MyLDAParameters(functional_name='MyLDA')
    ...     
    ...     def compute_exchange_generic(self, density_data):
    ...         rho = density_data.rho
    ...         C_x = self.params.C_x  # Direct access
    ...         
    ...         # LDA exchange
    ...         e_x = C_x * rho**(4/3)
    ...         v_x = (4/3) * C_x * rho**(1/3)
    ...         
    ...         return GenericXCResult(
    ...             v_generic=v_x, 
    ...             e_generic=e_x,
    ...             de_dsigma=None,
    ...             de_dtau=None
    ...         )
    ...     
    ...     def compute_correlation_generic(self, density_data):
    ...         # Similar structure
    ...         pass
    >>> 
    >>> # GGA example
    >>> @dataclass
    >>> class MyGGAParameters(XCParameters):
    ...     mu: float = 0.2195
    ...     kappa: float = 0.804
    >>> 
    >>> class MyGGA(XCEvaluator):
    ...     def _default_params(self):
    ...         return MyGGAParameters(functional_name='MyGGA')
    ...     
    ...     def compute_exchange_generic(self, density_data):
    ...         rho = density_data.rho
    ...         grad_rho = density_data.grad_rho
    ...         mu = self.params.mu  # Direct access
    ...         
    ...         # GGA exchange
    ...         e_x = ...
    ...         v_x = ...
    ...         de_x_dsigma = ...
    ...         
    ...         return GenericXCResult(
    ...             v_generic=v_x, 
    ...             e_generic=e_x,
    ...             de_dsigma=de_x_dsigma,
    ...             de_dtau=None
    ...         )
    """
    
    def __init__(
        self, 
        derivative_matrix: Optional[np.ndarray] = None,
        r_quad: Optional[np.ndarray] = None,
        params: Optional[XCParameters] = None
        ):
        """
        Initialize XC evaluator.
        
        Parameters
        ----------
        derivative_matrix : np.ndarray, optional
            Finite element derivative matrix for gradient transformations.
            Required for GGA and meta-GGA functionals.
            Shape: (n_elements, n_quad_per_element, n_quad_per_element)
        r_quad : np.ndarray, optional
            Radial quadrature nodes (coordinates).
            Required for spherical coordinate transformations in GGA and meta-GGA.
            Shape: (n_quad_points,)
        params : XCParameters, optional
            Functional parameters (for delta learning).
            If None, uses default parameters.
        """
        self.derivative_matrix = derivative_matrix
        self.r_quad = r_quad
        self.params = params if params is not None else self._default_params()
    
    def _default_params(self) -> XCParameters:
        """
        Return default parameters for this functional.
        
        Subclasses should override this to provide functional-specific parameters.
        
        Returns
        -------
        params : XCParameters (or subclass)
            Default parameters for this functional
        
        Examples
        --------
        >>> def _default_params(self):
        ...     return LDAParameters(functional_name='LDA_PZ', C_x=-0.738559)
        """
        return XCParameters(functional_name=self.__class__.__name__)
    
    
    @abstractmethod
    def compute_exchange_generic(
        self,
        density_data: DensityData
        ) -> GenericXCResult:
        """
        Compute exchange in GENERIC form (Stage 1).
        
        This method implements the XC functional in its original/published form,
        typically in 3D Cartesian coordinates or coordinate-independent form.
        
        Parameters
        ----------
        density_data : DensityData
            Container with:
            - rho: electron density ρ(r)
            - grad_rho: density gradient magnitude |∇ρ| (for GGA and meta-GGA)
            - tau: kinetic energy density τ (for meta-GGA)
        
        Returns
        -------
        result : GenericXCResult
            Container with:
            - v_generic: ∂εₓ/∂ρ (exchange potential in generic form)
            - e_generic: εₓ (exchange energy density)
            - de_dsigma: ∂εₓ/∂σ (for GGA, meta-GGA), where σ = |∇ρ|²
            - de_dtau: ∂εₓ/∂τ (for meta-GGA)
        
        Notes
        -----
        For LDA: only v_generic and e_generic are needed
        For GGA: v_generic, e_generic, and de_dsigma are needed
        For meta-GGA: all four components are needed
        
        The derivatives de_dsigma and de_dtau are essential for computing
        the correct potential in spherical coordinates:
            V = ∂ε/∂ρ - (2/r²)·d/dr[r²·∂ε/∂σ·dρ/dr] - ∇·(∂ε/∂τ·∇τ)
        
        Examples
        --------
        >>> density_data = DensityData(rho=rho, grad_rho=grad_rho)
        >>> result = evaluator.compute_exchange_generic(density_data)
        >>> v_x = result.v_generic
        """
        pass
    
    
    @abstractmethod
    def compute_correlation_generic(
        self,
        density_data: DensityData
        ) -> GenericXCResult:
        """
        Compute correlation in GENERIC form (Stage 1).
        
        This method implements the XC functional in its original/published form,
        typically in 3D Cartesian coordinates or coordinate-independent form.
        
        Parameters
        ----------
        density_data : DensityData
            Container with:
            - rho: electron density ρ(r)
            - grad_rho: density gradient magnitude |∇ρ| (for GGA and meta-GGA)
            - tau: kinetic energy density τ (for meta-GGA)
        
        Returns
        -------
        result : GenericXCResult
            Container with:
            - v_generic: ∂εc/∂ρ (correlation potential in generic form)
            - e_generic: εc (correlation energy density)
            - de_dsigma: ∂εc/∂σ (for GGA, meta-GGA), where σ = |∇ρ|²
            - de_dtau: ∂εc/∂τ (for meta-GGA)
        
        Notes
        -----
        For LDA: only v_generic and e_generic are needed
        For GGA: v_generic, e_generic, and de_dsigma are needed
        For meta-GGA: all four components are needed
        
        The derivatives de_dsigma and de_dtau are essential for computing
        the correct potential in spherical coordinates:
            V = ∂ε/∂ρ - (2/r²)·d/dr[r²·∂ε/∂σ·dρ/dr] - ∇·(∂ε/∂τ·∇τ)
        
        Examples
        --------
        >>> density_data = DensityData(rho=rho, grad_rho=grad_rho)
        >>> result = evaluator.compute_correlation_generic(density_data)
        >>> v_c = result.v_generic
        """
        pass
    
    
    @staticmethod
    def _transform_potential_to_spherical(
        v_generic         : np.ndarray,
        de_dsigma         : Optional[np.ndarray],
        density_data      : DensityData,
        derivative_matrix : Optional[np.ndarray],
        r_quad            : Optional[np.ndarray]
        ) -> np.ndarray:
        """
        Transform a SINGLE XC potential (exchange or correlation) to spherical form.
        
        This is a static utility method that transforms one potential component at a time.
        It ensures that v_x and v_c are processed using the SAME transformation logic.
        
        IMPORTANT: Only the potential V needs transformation, NOT the energy density ε.
        Energy density ε is a scalar field and is invariant under coordinate transformations.
        
        For spherically symmetric atomic systems, the potential transformation is:
        
        LDA:
            V = V1 = ∂ε/∂ρ  (no transformation needed)
        
        GGA:
            V = V1 - D@(V2·dρ/dr) - 2·V2·(dρ/dr)/r
            where V1 = ∂ε/∂ρ, V2 = ∂ε/∂σ
        
        meta-GGA:
            V = V1 - D@(V2·dρ/dr) - 2·V2·(dρ/dr)/r + (terms involving ∂ε/∂τ)
        
        Parameters
        ----------
        v_generic : np.ndarray
            Generic potential V1 = ∂ε/∂ρ (coordinate-independent part)
        de_dsigma : np.ndarray, optional
            Derivative V2 = ∂ε/∂σ where σ = |∇ρ|² (None for LDA)
        density_data : DensityData
            Density information (must contain grad_rho for GGA/meta-GGA)
        derivative_matrix : np.ndarray, optional
            FEM derivative matrix (required for GGA and meta-GGA)
        r_quad : np.ndarray, optional
            Radial quadrature nodes (required for spherical transformation)
        
        Returns
        -------
        v_spherical : np.ndarray
            Potential in spherical coordinates, ready for radial Kohn-Sham equation
        
        Notes
        -----
        Reference implementation:
            V = V1 - D@(V2·grad_rho) - 2·V2·grad_rho/r
        
        This base class method handles LDA, GGA, and meta-GGA automatically
        by checking if de_dsigma is provided.
        """
        # LDA case: no gradient correction
        if de_dsigma is None:
            return v_generic
        
        # GGA/meta-GGA case: add gradient correction
        # V = V1 - D@(V2·dρ/dr) - 2·V2·(dρ/dr)/r
        grad_rho = density_data.grad_rho
        
        if grad_rho is None:
            raise ValueError("grad_rho required for GGA/meta-GGA transformation")
        if r_quad is None:
            raise ValueError("r_quad required for spherical transformation")
        if derivative_matrix is None:
            raise ValueError("derivative_matrix required for GGA/meta-GGA transformation")
        
        # Get problem dimensions
        n_elem = derivative_matrix.shape[0]  # number of elements
        n_quad = derivative_matrix.shape[1]  # quadrature points per element
        
        # Compute f = V2 · dρ/dr
        f = de_dsigma * grad_rho
        
        # Apply derivative operator: D @ f
        f_reshaped = f.reshape(n_elem, n_quad, 1)
        df_dr = np.matmul(derivative_matrix, f_reshaped).reshape(-1)
        
        # Final potential in spherical coordinates
        v_spherical = v_generic - (df_dr + 2.0 * f / r_quad)
        
        return v_spherical
    
    
    def transform_to_spherical(
        self,
        xc_result: GenericXCResult,
        density_data: DensityData
        ) -> np.ndarray:
        """
        Transform a generic XC potential to spherical coordinates (Stage 2).
        
        This is a convenience wrapper that calls _transform_potential_to_spherical.
        Subclasses typically override _transform_potential_to_spherical instead.
        
        NOTE: Only the potential V is transformed. The energy density ε is a scalar
        and remains unchanged (it's coordinate-independent).
        
        Parameters
        ----------
        xc_result : GenericXCResult
            Generic XC calculation result (contains v, e, and derivatives)
        density_data : DensityData
            Density information (needed for gradient transformations)
        
        Returns
        -------
        v_spherical : np.ndarray
            Potential in spherical coordinates, ready for radial Kohn-Sham equation
        """
        return self._transform_potential_to_spherical(
            v_generic         = xc_result.v_generic,
            de_dsigma         = xc_result.de_dsigma,
            density_data      = density_data,
            derivative_matrix = self.derivative_matrix,
            r_quad            = self.r_quad
        )
    
    
    def compute_xc(self, density_data: DensityData) -> XCPotentialData:
        """
        Compute exchange-correlation potentials and energy densities (TWO STAGES).
        
        This is the main interface that orchestrates the two-stage calculation:
        
        Stage 1: Generic Form
        ---------------------
        Call compute_exchange_generic() and compute_correlation_generic()
        to get potentials in their original functional form, along with
        necessary derivatives (∂ε/∂σ, ∂ε/∂τ) for GGA and meta-GGA.
        
        Stage 2: Spherical Transform (Potentials ONLY)
        ----------------------------------------------
        Call transform_to_spherical() separately for exchange and correlation
        to convert generic potentials to spherical coordinates suitable for
        radial atomic calculations. v_x and v_c use the SAME transformation logic.
        
        IMPORTANT: Only potentials V are transformed. Energy densities ε are scalars
        and remain unchanged (coordinate-independent).
        
        Parameters
        ----------
        density_data : DensityData
            Container with electron density ρ(r) and optionally:
            - grad_rho: |∇ρ(r)| for GGA functionals
            - tau: kinetic energy density for meta-GGA functionals
        
        Returns
        -------
        potential_data : XCPotentialData
            Container with:
            - V_x, V_c: potentials in SPHERICAL coordinates
            - ε_x, ε_c: energy densities (coordinate-independent scalars)
            This is an immutable (frozen) dataclass
        
        Raises
        ------
        ValueError
            If required density quantities are missing
            (e.g., GGA needs grad_rho but it's None)
        
        Examples
        --------
        >>> density_data = DensityData(rho=rho, grad_rho=grad_rho)
        >>> evaluator = GGA_PBE(derivative_matrix=D)
        >>> pot_data = evaluator.compute_xc(density_data)
        >>> v_xc_spherical = pot_data.v_xc  # Ready for radial equation
        """
        # ===== Stage 1: Compute generic form =====
        # Returns GenericXCResult with v, e, and derivatives (de_dsigma, de_dtau)
        x_result = self.compute_exchange_generic(density_data)
        c_result = self.compute_correlation_generic(density_data)

        
        # ===== Stage 2: Transform potentials to spherical =====
        # Process exchange and correlation separately using the SAME transformation
        # Energy densities are scalars and don't need transformation
        v_x = self.transform_to_spherical(x_result, density_data)
        v_c = self.transform_to_spherical(c_result, density_data)
        
        # Energy densities are coordinate-independent (scalars)
        e_x = x_result.e_generic
        e_c = c_result.e_generic

        # Derivatives of energy densities
        de_x_dtau = x_result.de_dtau
        de_c_dtau = c_result.de_dtau
        
        return XCPotentialData(v_x=v_x, v_c=v_c, e_x=e_x, e_c=e_c, de_x_dtau=de_x_dtau, de_c_dtau=de_c_dtau)
    
    
    def __repr__(self) -> str:
        """String representation of the evaluator"""
        return f"{self.__class__.__name__}()"


def create_xc_evaluator(
    functional_name: str, 
    derivative_matrix: Optional[np.ndarray] = None,
    r_quad: Optional[np.ndarray] = None
    ) -> XCEvaluator:
    """
    Factory function to create the appropriate XC evaluator.
    
    This is a simple factory that maps functional names to their implementations.
    
    Parameters
    ----------
    functional_name : str
        Name of the XC functional (e.g., 'LDA_PZ', 'GGA_PBE', 'SCAN')
    derivative_matrix : np.ndarray, optional
        Finite element derivative matrix for gradient-dependent functionals.
        Required for GGA and meta-GGA.
    r_quad : np.ndarray, optional
        Radial quadrature nodes (coordinates).
        Required for spherical coordinate transformations in GGA and meta-GGA.
    
    Returns
    -------
    evaluator : XCEvaluator
        Instance of the appropriate XC evaluator subclass
    
    Raises
    ------
    ValueError
        If functional_name is not recognized
    
    Examples
    --------
    >>> # LDA: no derivative matrix or r_quad needed
    >>> evaluator = create_xc_evaluator('LDA_PZ')
    >>> 
    >>> # GGA: needs derivative matrix and r_quad for transformation
    >>> evaluator = create_xc_evaluator('GGA_PBE', derivative_matrix=D, r_quad=r)
    >>> 
    >>> density_data = DensityData(rho=rho, grad_rho=grad_rho)
    >>> pot_data = evaluator.compute_xc(density_data)
    """
    # Import functional implementations
    from .lda import LDA_SVWN, LDA_SPW
    from .gga_pbe import GGA_PBE
    from .meta_scan import SCAN, rSCAN, r2SCAN
    
    # Simple mapping: functional name → class
    FUNCTIONAL_MAP = {
        # LDA functionals
        'LDA_PZ': LDA_SVWN,  # Note: LDA_PZ uses VWN correlation in current implementation
        'LDA_PW': LDA_SPW,
        
        # GGA functionals
        'GGA_PBE': GGA_PBE,
        
        # meta-GGA functionals
        'SCAN': SCAN,
        'RSCAN': rSCAN,
        'R2SCAN': r2SCAN,
        
        # hybrid functionals
        'PBE0': GGA_PBE,
        # 'HF': HF,
        
        # TODO: Add OEP and RPA
        # 'OEPx': OEPx,
        # 'RPA': RPA,
    }
    
    if functional_name not in FUNCTIONAL_MAP:
        available = ', '.join(FUNCTIONAL_MAP.keys())
        raise ValueError(
            f"Unknown XC functional: '{functional_name}'\n"
            f"Available functionals: {available}"
        )
    
    # Create and return instance with derivative matrix and r_quad
    functional_class = FUNCTIONAL_MAP[functional_name]
    return functional_class(derivative_matrix=derivative_matrix, r_quad=r_quad)


# =============================================================================
# JAX Compatibility (Optional)
# =============================================================================
# The dataclasses GenericXCResult and PotentialData are automatically
# compatible with JAX autodiff if registered as pytrees.
# This registration is optional and only activated if JAX is installed.

try:
    import jax
    from jax import tree_util
    
    # Note: XCParameters subclasses can be registered individually
    # Each functional can register its specific parameter class as needed
    # Example:
    #   tree_util.register_pytree_node(
    #       LDAParameters,
    #       lambda p: ((p.C_x,), {'functional_name': p.functional_name}),
    #       lambda aux, ch: LDAParameters(functional_name=aux['functional_name'], C_x=ch[0])
    #   )
    
    # Register GenericXCResult as JAX pytree
    def _generic_xc_result_flatten(result):
        """Flatten GenericXCResult for JAX transformations"""
        children = (result.v_generic, result.e_generic, 
                   result.de_dsigma, result.de_dtau)
        aux_data = None
        return children, aux_data
    
    def _generic_xc_result_unflatten(aux_data, children):
        """Reconstruct GenericXCResult from flattened form"""
        return GenericXCResult(*children)
    
    tree_util.register_pytree_node(
        GenericXCResult,
        _generic_xc_result_flatten,
        _generic_xc_result_unflatten
    )
    
    # Register XCPotentialData as JAX pytree
    def _xc_potential_data_flatten(potential):
        """Flatten XCPotentialData for JAX transformations"""
        children = (potential.v_x, potential.v_c, 
                   potential.e_x, potential.e_c)
        aux_data = None
        return children, aux_data
    
    def _xc_potential_data_unflatten(aux_data, children):
        """Reconstruct XCPotentialData from flattened form"""
        return XCPotentialData(*children)
    
    tree_util.register_pytree_node(
        XCPotentialData,
        _xc_potential_data_flatten,
        _xc_potential_data_unflatten
    )
    
    # Note: DensityData should also be registered (done in scf/density.py)
    
    _JAX_AVAILABLE = True

except ImportError:
    # JAX not installed - this is fine, NumPy operations work as usual
    _JAX_AVAILABLE = False
