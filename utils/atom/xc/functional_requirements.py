"""
Functional requirements for XC functionals

This module defines computational requirements for different XC functionals:
- LDA: only needs density ρ
- GGA: needs density ρ and gradient |∇ρ|
- meta-GGA: needs density ρ, gradient |∇ρ|, and kinetic energy density τ
"""

from typing import Dict, Literal
from dataclasses import dataclass


# Functional type definitions
FunctionalType = Literal['LDA', 'GGA', 'meta-GGA', 'hybrid-GGA', 'hybrid-meta-GGA', 'OEP', 'RPA', 'None']


@dataclass
class FunctionalRequirements:
    """
    Computational requirements for an XC functional
    
    Attributes
    ----------
    needs_gradient : bool
        Whether the functional requires density gradient |∇ρ|
    needs_tau : bool
        Whether the functional requires kinetic energy density τ
    functional_type : str
        Type of functional (LDA, GGA, meta-GGA, etc.)
    needs_orbitals : bool
        Whether the functional needs explicit orbital information (HF, OEP, RPA)
    """
    needs_gradient  : bool
    needs_tau       : bool
    functional_type : FunctionalType
    needs_orbitals  : bool = False
    
    @property
    def is_lda(self) -> bool:
        """Check if functional is LDA type"""
        return self.functional_type == 'LDA'
    
    @property
    def is_gga(self) -> bool:
        """Check if functional is GGA type (including hybrids)"""
        return self.functional_type in ['GGA', 'hybrid-GGA']
    
    @property
    def is_meta_gga(self) -> bool:
        """Check if functional is meta-GGA type (including hybrids)"""
        return self.functional_type in ['meta-GGA', 'hybrid-meta-GGA']
    
    @property
    def is_hybrid(self) -> bool:
        """Check if functional is a hybrid"""
        return 'hybrid' in self.functional_type


# Registry of functional requirements
_FUNCTIONAL_REQUIREMENTS: Dict[str, FunctionalRequirements] = {
    # LDA functionals
    'LDA_PZ': FunctionalRequirements(
        needs_gradient=False,
        needs_tau=False,
        functional_type='LDA'
    ),
    'LDA_PW': FunctionalRequirements(
        needs_gradient=False,
        needs_tau=False,
        functional_type='LDA'
    ),
    
    # GGA functionals
    'GGA_PBE': FunctionalRequirements(
        needs_gradient=True,
        needs_tau=False,
        functional_type='GGA'
    ),
    
    # Hybrid GGA functionals
    'PBE0': FunctionalRequirements(
        needs_gradient=True,
        needs_tau=False,
        functional_type='hybrid-GGA',
        needs_orbitals=True  # Needs orbitals for exact exchange
    ),
    'HF': FunctionalRequirements(
        needs_gradient=False,
        needs_tau=False,
        functional_type='hybrid-GGA',
        needs_orbitals=True
    ),
    
    # meta-GGA functionals
    'SCAN': FunctionalRequirements(
        needs_gradient=True,
        needs_tau=True,
        functional_type='meta-GGA'
    ),
    'RSCAN': FunctionalRequirements(
        needs_gradient=True,
        needs_tau=True,
        functional_type='meta-GGA'
    ),
    'R2SCAN': FunctionalRequirements(
        needs_gradient=True,
        needs_tau=True,
        functional_type='meta-GGA'
    ),
    
    # Optimized Effective Potential
    'OEPx': FunctionalRequirements(
        needs_gradient=True,
        needs_tau=False,
        functional_type='OEP',
        needs_orbitals=True
    ),
    
    # Random Phase Approximation
    'RPA': FunctionalRequirements(
        needs_gradient=True,
        needs_tau=False,
        functional_type='RPA',
        needs_orbitals=True
    ),
    
    # Pure exact exchange
    'None': FunctionalRequirements(
        needs_gradient=False,
        needs_tau=False,
        functional_type='None'
    ),
}


def get_functional_requirements(xc_functional: str) -> FunctionalRequirements:
    """
    Get computational requirements for a given XC functional
    
    This function returns what computational quantities are needed
    for a specific exchange-correlation functional:
    - LDA: only density ρ
    - GGA: density ρ + gradient |∇ρ|
    - meta-GGA: density ρ + gradient |∇ρ| + kinetic energy density τ
    
    Parameters
    ----------
    xc_functional : str
        Name of the XC functional (e.g., 'LDA_PZ', 'GGA_PBE', 'SCAN')
    
    Returns
    -------
    requirements : FunctionalRequirements
        Object containing:
        - needs_gradient: bool
        - needs_tau: bool
        - functional_type: str
        - needs_orbitals: bool
    
    Raises
    ------
    ValueError
        If the functional is not recognized
    
    Examples
    --------
    >>> req = get_functional_requirements('LDA_PZ')
    >>> req.needs_gradient
    False
    >>> req.needs_tau
    False
    
    >>> req = get_functional_requirements('GGA_PBE')
    >>> req.needs_gradient
    True
    >>> req.needs_tau
    False
    
    >>> req = get_functional_requirements('SCAN')
    >>> req.needs_gradient
    True
    >>> req.needs_tau
    True
    """
    if xc_functional not in _FUNCTIONAL_REQUIREMENTS:
        available = ', '.join(_FUNCTIONAL_REQUIREMENTS.keys())
        raise ValueError(
            f"Unknown XC functional: '{xc_functional}'\n"
            f"Available functionals: {available}"
        )
    
    return _FUNCTIONAL_REQUIREMENTS[xc_functional]


def register_functional(
    name: str, 
    needs_gradient: bool,
    needs_tau: bool,
    functional_type: FunctionalType,
    needs_orbitals: bool = False
    ) -> None:
    """
    Register a new functional with its requirements
    
    This allows users to add custom functionals to the registry.
    
    Parameters
    ----------
    name : str
        Name of the functional
    needs_gradient : bool
        Whether gradient is needed
    needs_tau : bool
        Whether kinetic energy density is needed
    functional_type : str
        Type of functional
    needs_orbitals : bool, optional
        Whether orbitals are needed (for hybrid/OEP/RPA)
    
    Examples
    --------
    >>> register_functional(
    ...     name='MY_CUSTOM_GGA',
    ...     needs_gradient=True,
    ...     needs_tau=False,
    ...     functional_type='GGA'
    ... )
    """
    _FUNCTIONAL_REQUIREMENTS[name] = FunctionalRequirements(
        needs_gradient=needs_gradient,
        needs_tau=needs_tau,
        functional_type=functional_type,
        needs_orbitals=needs_orbitals
    )


def list_available_functionals() -> list[str]:
    """
    List all registered functionals
    
    Returns
    -------
    functionals : list[str]
        List of functional names
    """
    return list(_FUNCTIONAL_REQUIREMENTS.keys())


def get_functionals_by_type(functional_type: FunctionalType) -> list[str]:
    """
    Get all functionals of a specific type
    
    Parameters
    ----------
    functional_type : str
        Type to filter by ('LDA', 'GGA', 'meta-GGA', etc.)
    
    Returns
    -------
    functionals : list[str]
        List of functional names matching the type
    """
    return [
        name for name, req in _FUNCTIONAL_REQUIREMENTS.items()
        if req.functional_type == functional_type
    ]



if __name__ == "__main__":
    print(list_available_functionals())
    print(get_functional_requirements('GGA_PBE'))