from __future__ import annotations
import numpy as np
from typing import Dict, Any
from scipy.interpolate import make_interp_spline


class NonLocalPseudopotential:
    """
    Non-local pseudopotential components using Kleinman-Bylander projectors.
    
    The non-local pseudopotential is represented as:
        V_nl = Σ_{ljm} |χ_{lj}⟩ γ_{lj} ⟨χ_{lj}|
    
    where:
        χ_{lj}(r) - Projector functions (from pseudopotential file)
        γ_{lj}    - Energy coefficients (from pseudopotential file)
        l, j, m   - Angular momentum quantum numbers
    """
    
    def __init__(
        self,
        pseudo,  # LocalPseudopotential instance
        ops_builder  # RadialOperatorsBuilder instance
        ):
        """
        Initialize non-local pseudopotential calculator.
        
        Parameters
        ----------
        pseudo : LocalPseudopotential
            Pseudopotential data (contains nonlocal_projectors, etc.)
        ops_builder : RadialOperatorsBuilder
            Finite element operators for matrix assembly
        """
        self.pseudo = pseudo
        self.ops = ops_builder
        
        # Extract data from pseudopotential
        self.nonlocal_projectors = pseudo.nonlocal_projectors
        self.n_projectors_per_l = pseudo.n_projectors_per_l
        self.r_cutoff_max_per_l = pseudo.r_cutoff_max_per_l
        self.r_grid_local = pseudo.r_grid_local
        
        # Extract grid data from operators
        self.r_quad = ops_builder.quadrature_nodes
        self.w_quad = ops_builder.quadrature_weights
        self.lagrange_basis = ops_builder.lagrange_basis
        self.n_elem = ops_builder.number_of_finite_elements
        self.n_quad = ops_builder.quadrature_node_number
        self.n_global_dofs = self.n_elem * (ops_builder.physical_node_number - 1) + 1
    
    
    def compute_nonlocal_matrix(self, l_channel: int) -> np.ndarray:
        """
        Compute non-local pseudopotential matrix for angular momentum l.
        
        This implements the Kleinman-Bylander form:
            H_nl = Σⱼ |χⱼ⟩ γⱼ ⟨χⱼ|
        
        Parameters
        ----------
        l_channel : int
            Angular momentum quantum number
        
        Returns
        -------
        np.ndarray
            Non-local potential matrix, shape (n_global_dofs, n_global_dofs)
            
        Notes
        -----
        The matrix is assembled from projector overlaps:
            V_nl[i,j] = Σₖ γₖ ∫ φᵢ(r) χₖ(r) r dr ∫ χₖ(r') φⱼ(r') r' dr'
        """
        if l_channel >= len(self.nonlocal_projectors):
            # No projectors for this l
            return np.zeros((self.n_global_dofs, self.n_global_dofs))
        
        # Get projector data for this l channel
        nl_data = self.nonlocal_projectors[l_channel]
        gamma_coefficients = nl_data['gamma_Jl']  # Energy coefficients
        projector_functions = nl_data['proj']      # Projector radial functions
        r_cutoff = self.r_cutoff_max_per_l[l_channel]
        n_projectors = self.n_projectors_per_l[l_channel]
        
        # Initialize non-local matrix
        V_nonlocal = np.zeros((self.n_global_dofs, self.n_global_dofs))
        
        # Loop over projectors for this l
        for proj_idx in range(n_projectors):
            # Step 1: Interpolate projector to quadrature grid
            chi_at_quad = self._interpolate_projector(
                projector_functions[:, proj_idx],
                r_cutoff
            )
            
            # Step 2: Compute projection integrals ⟨χ|φᵢ⟩ for all basis functions
            chi_r = chi_at_quad * self.r_quad  # r*χ(r)
            projection_integrals = self._compute_projection_integrals(chi_r)
            
            # Step 3: Assemble matrix: V += γ |⟨χ|φᵢ⟩⟨χ|φⱼ⟩|
            gamma = gamma_coefficients[proj_idx]
            V_nonlocal += gamma * np.outer(projection_integrals, projection_integrals)
        
        return V_nonlocal
    
    
    def _interpolate_projector(
        self,
        projector_values: np.ndarray,
        r_cutoff: float
        ) -> np.ndarray:
        """
        Interpolate projector function to quadrature grid.
        
        Parameters
        ----------
        projector_values : np.ndarray
            Projector values on pseudopotential grid
        r_cutoff : float
            Cutoff radius for this projector
        
        Returns
        -------
        np.ndarray
            Projector values at quadrature points (zero beyond r_cutoff)
        """
        # Find quadrature points within cutoff
        mask_within_cutoff = self.r_quad <= r_cutoff
        r_within_cutoff = self.r_quad[mask_within_cutoff]
        
        # Interpolate projector using cubic spline
        chi_interpolator = make_interp_spline(
            self.r_grid_local,
            projector_values,
            k=3
        )
        
        # Evaluate at quadrature points
        chi_at_quad = np.zeros_like(self.r_quad)
        chi_at_quad[mask_within_cutoff] = chi_interpolator(r_within_cutoff)
        
        return chi_at_quad
    
    
    def _compute_projection_integrals(
        self,
        chi_r: np.ndarray
        ) -> np.ndarray:
        """
        Compute projection integrals ⟨χ|φᵢ⟩ = ∫ χ(r) φᵢ(r) r dr.
        
        Parameters
        ----------
        chi_r : np.ndarray
            r * χ(r) at quadrature points, shape (n_quad_total,)
        
        Returns
        -------
        np.ndarray
            Projection integrals for all global basis functions
            Shape: (n_global_dofs,)
        """
        # Reshape chi_r to element structure
        from atom.mesh.builder import Mesh1D
        
        chi_r_reshaped = Mesh1D.fe_flat_to_block2d(
            chi_r,
            self.n_elem,
            endpoints_shared=False
        )  # Shape: (n_elem, n_quad)
        
        # Compute local integrals: ∫ χ(r) φᵢ(r) r w dr
        # lagrange_basis: (n_elem, n_quad, n_basis)
        # chi_r_reshaped: (n_elem, n_quad)
        # w_quad_reshaped: (n_elem, n_quad)
        
        w_quad_reshaped = Mesh1D.fe_flat_to_block2d(
            self.w_quad,
            self.n_elem,
            endpoints_shared=False
        )
        
        # Local integrals for each element
        # Shape: (n_elem, n_basis)
        local_integrals = np.einsum(
            'emk,em,em->ek',
            self.lagrange_basis,         # (n_elem, n_quad, n_basis)
            chi_r_reshaped,               # (n_elem, n_quad)
            w_quad_reshaped,              # (n_elem, n_quad)
            optimize=True
        )
        
        # Assemble to global DOFs (handle shared nodes)
        global_integrals = self._assemble_local_to_global_vector(local_integrals)
        
        return global_integrals
    
    
    def _assemble_local_to_global_vector(
        self,
        local_vector: np.ndarray
        ) -> np.ndarray:
        """
        Assemble local element vectors to global vector.
        
        Parameters
        ----------
        local_vector : np.ndarray
            Shape: (n_elem, n_local_dofs)
        
        Returns
        -------
        np.ndarray
            Global vector, shape: (n_global_dofs,)
        """
        n_elem = local_vector.shape[0]
        n_local = local_vector.shape[1]
        stride = n_local - 1
        
        # Initialize global vector
        global_vector = np.zeros(self.n_global_dofs)
        
        # Assemble: add contributions from each element
        for elem_idx in range(n_elem):
            global_start = elem_idx * stride
            global_end = global_start + n_local
            global_vector[global_start:global_end] += local_vector[elem_idx]
        
        return global_vector
    
    
    def compute_all_nonlocal_matrices(
        self,
        l_channels: np.ndarray
        ) -> Dict[int, np.ndarray]:
        """
        Compute non-local matrices for all required l channels.
        
        Parameters
        ----------
        l_channels : np.ndarray
            Array of unique l values to compute
        
        Returns
        -------
        dict
            Dictionary mapping l → V_nl matrix
            Each matrix has shape (n_global_dofs, n_global_dofs)
        """
        nonlocal_matrices = {}
        
        for l in l_channels:
            # Remove boundary DOFs (keep only interior points)
            V_nl_full = self.compute_nonlocal_matrix(l)
            # V_nl_interior = V_nl_full[1:-1, 1:-1]
            nonlocal_matrices[l] = V_nl_full
        
        return nonlocal_matrices


    def compute_nonlocal_energy(
        self,
        orbitals        : np.ndarray,
        occupations     : np.ndarray,
        l_values        : np.ndarray,
        unique_l_values : np.ndarray
        ) -> float:
        """
        Compute non-local pseudopotential energy contribution.
        
        This implements the Kleinman-Bylander form:
            E_nl = Σ_l Σ_j γ_{lj} ⟨φ|χ_{lj}⟩⟨χ_{lj}|φ⟩
        
        where:
            γ_{lj} = energy coefficients
            χ_{lj} = projector functions
            φ = occupied orbitals
        
        Parameters
        ----------
        orbitals : np.ndarray
            Kohn-Sham orbitals (radial wavefunctions)
            Shape: (n_grid, n_orbitals)
        occupations : np.ndarray
            Occupation numbers for each orbital
            Shape: (n_orbitals,)
        l_values : np.ndarray
            Angular momentum quantum numbers for each orbital
            Shape: (n_orbitals,)
        unique_l_values : np.ndarray
            Unique angular momentum values to loop over
        
        Returns
        -------
        float
            Total non-local pseudopotential energy contribution
        """
        E_nonlocal_total = 0.0
        
        # Loop over unique l channels
        for l in unique_l_values:
            # Skip if no projectors for this l
            if l >= len(self.nonlocal_projectors) or self.n_projectors_per_l[l] == 0:
                continue
            
            # Get projector data for this l channel
            nl_data = self.nonlocal_projectors[l]
            gamma_coefficients = nl_data['gamma_Jl']  # Energy coefficients
            projector_functions = nl_data['proj']      # Projector functions
            r_cutoff = self.r_cutoff_max_per_l[l]
            n_projectors = self.n_projectors_per_l[l]
            
            # Get orbitals for this l channel
            mask_l = (l_values == l)
            orbitals_l = orbitals[:, mask_l]  # Shape: (n_grid, n_orbitals_l)
            occupations_l = occupations[mask_l]  # Shape: (n_orbitals_l,)
            
            # Loop over projectors for this l channel
            for proj_idx in range(n_projectors):
                # Interpolate projector to quadrature grid
                chi_at_quad = self._interpolate_projector(
                    projector_functions[:, proj_idx],
                    r_cutoff
                )
                
                # Compute r * chi(r)
                r_chi_l = chi_at_quad * self.r_quad
                
                # Compute energy contribution using einsum
                # Following reference code: gamma_Jl[i]*np.einsum('ij,jk,ik->', Occ*orb*w, r_chi*r_chi, orb*w)
                # The einsum computes: Σ_i,j occ_i * orb_i(j) * w(j) * r_chi(j) * r_chi(j) * orb_i(j) * w(j)
                # = Σ_i occ_i * ⟨φ_i|χ⟩⟨χ|φ_i⟩
                gamma = gamma_coefficients[proj_idx]
                
                # einsum('ij,jk,ik->', Occ*orb*w, r_chi*r_chi, orb*w)
                # i = orbital index, j,k = grid indices
                energy_contribution = gamma * np.einsum(
                    'ij,jk,ik->',
                    occupations_l[:, np.newaxis] * orbitals_l.T * self.w_quad[np.newaxis, :],
                    r_chi_l[:, np.newaxis] * r_chi_l[np.newaxis, :],
                    orbitals_l.T * self.w_quad[np.newaxis, :],
                    optimize=True
                )
                
                E_nonlocal_total += energy_contribution
        
        return E_nonlocal_total
