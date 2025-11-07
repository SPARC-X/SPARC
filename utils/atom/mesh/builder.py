from __future__ import annotations
import numpy as np
from typing import Tuple

# error messages for class Quadrature1D
QUADRATURE_POINT_NUMBER_NOT_INTEGER_ERROR = \
    "quadrature_point_number must be an integer, get {} instead"
QUADRATURE_POINT_NUMBER_NOT_GREATER_THAN_0_ERROR = \
    "quadrature_point_number must be greater than 0, get {} instead"

# error messages for class Mesh1D
DOMAIN_RADIUS_NOT_FLOAT_ERROR = \
    "domain_radius must be a float, get {} instead"
DOMAIN_RADIUS_NOT_GREATER_THAN_0_ERROR = \
    "domain_radius must be greater than 0, get {} instead"
NUMBER_OF_FINITE_ELEMENTS_NOT_INTEGER_ERROR = \
    "number_of_finite_elements must be an integer, get {} instead"
NUMBER_OF_FINITE_ELEMENTS_NOT_GREATER_THAN_0_ERROR = \
    "number_of_finite_elements must be greater than 0, get {} instead"
MESH_TYPE_NOT_STRING_ERROR = \
    "mesh_type must be a string, get {} instead"
MESH_TYPE_NOT_IN_VALID_LIST_ERROR = \
    "mesh_type must be in {}, get {} instead"
CLUSTERING_PARAMETER_NOT_FLOAT_ERROR = \
    "clustering_parameter must be a float, get {} instead"
CLUSTERING_PARAMETER_NOT_GREATER_THAN_0_ERROR = \
    "clustering_parameter must be greater than 0, get {} instead"
CLUSTERING_PARAMETER_NOT_NONE_FOR_UNIFORM_MESH_TYPE_WARNING = \
    "Warning: clustering_parameter must be None for uniform mesh type"
EXP_SHIFT_NOT_FLOAT_ERROR = \
    "exp_shift must be a float, get {} instead"
EXP_SHIFT_NEGATIVE_ERROR = \
    "exp_shift must be greater than or equal to 0, get {} instead"
EXP_SHIFT_NOT_NONE_FOR_UNIFORM_AND_POLYNOMIAL_MESH_TYPE_WARNING = \
    "Warning: exp_shift must be None for uniform and polynomial mesh type"

MESH_NODES_NOT_EQUAL_TO_NUMBER_OF_FINITE_ELEMENTS_PLUS_1_ERROR = \
    "Number of nodes should be equal to number of finite elements + 1, get {} instead"
MESH_WIDTH_NOT_EQUAL_TO_NUMBER_OF_FINITE_ELEMENTS_ERROR = \
    "Number of widths should be equal to number of finite elements, get {} instead"

BOUNDARIES_NODES_NOT_NUMPY_ARRAY_ERROR = \
    "boundaries_nodes must be a numpy array"
INTERP_NODES_NOT_NUMPY_ARRAY_ERROR = \
    "interp_nodes must be a numpy array"


BASE_NODES_NOT_NUMPY_ARRAY_ERROR = \
    "base_nodes must be a numpy array"
BASE_NODES_NOT_1D_ARRAY_ERROR = \
    "base_nodes must be a 1D numpy array"
BASE_NODES_NOT_MONOTONICALLY_INCREASING_OR_DECREASING_ERROR = \
    "base_nodes must be monotonically increasing or decreasing"

BOUNDARIES_NODES_NOT_MONOTONICALLY_INCREASING_ERROR = \
    "boundaries_nodes must be monotonically increasing"
INTERP_NODES_NOT_MONOTONICALLY_INCREASING_ERROR = \
    "interp_nodes must be monotonically increasing"
INTERP_WEIGHTS_NOT_NUMPY_ARRAY_ERROR = \
    "interp_weights must be a numpy array"
    
BOUNDARIES_NODES_NOT_AT_LEAST_2_ERROR = \
    "boundaries_nodes must have length >= 2"
INTERP_NODES_AND_WEIGHTS_NOT_THE_SAME_LENGTH_ERROR = \
    "interp_nodes and interp_weights must have the same length"
INTERP_NODES_NOT_AT_LEAST_1_ERROR = \
    "interp_nodes must have length >= 1"
BOUNDARIES_NODES_NOT_NONDECREASING_ERROR = \
    "boundaries_nodes must be nondecreasing"
INTERP_NODES_NOT_INCLUDE_MINUS_1_AT_LEFT_BOUNDARY_ERROR = \
    "interp_nodes must include -1 at the left boundary"
INTERP_NODES_NOT_INCLUDE_1_AT_RIGHT_BOUNDARY_ERROR = \
    "interp_nodes must include 1 at the right boundary"
GLOBAL_NODES_NOT_EQUAL_TO_EXPECTED_LENGTH_ERROR = \
    "Expected {} nodes, got {} instead"

FLAT_NOT_NUMPY_ARRAY_ERROR = \
    "flat must be a numpy array, get {} instead"
FLAT_NOT_1D_ARRAY_ERROR = \
    "flat must be a 1D numpy array, get {} instead"
N_ELEM_NOT_INT_ERROR = \
    "n_elem must be an integer, get {} instead"
ENDPOINTS_SHARED_NOT_BOOL_ERROR = \
    "endpoints_shared must be a boolean, get {} instead"
POINTS_PER_ELEM_NOT_INT_ERROR = \
    "points_per_elem must be an integer, get {} instead"
FLAT_NOT_EXPECTED_LENGTH_ENDPOINTS_SHARED_TRUE_ERROR = \
    "flat must have length = n_elem * (m - 1) + m if endpoints_shared = True, get length={}, n_elem={} instead"
FLAT_NOT_EXPECTED_LENGTH_ENDPOINTS_SHARED_FALSE_ERROR = \
    "flat must have length = n_elem * m if endpoints_shared = False, get length={}, n_elem={} instead"

# error messages for class LagrangeShapeFunctions
X_NODE_NOT_NUMPY_ARRAY_ERROR = \
    "x_node must be a numpy array, get {} instead"
X_NODE_NOT_AT_LEAST_2_ERROR = \
    "x_node must have length >= 2, get {} instead"
X_NODE_NOT_1D_OR_2D_ARRAY_ERROR = \
    "x_node must be a 1D or 2D numpy array, get dimension {} instead"
X_EVAL_NOT_NUMPY_ARRAY_ERROR = \
    "x_eval must be a numpy array, get {} instead"
X_EVAL_NOT_1D_OR_2D_ARRAY_ERROR = \
    "x_eval must be a 1D or 2D numpy array, get dimension {} instead"
X_EVAL_NOT_AT_LEAST_1_ERROR = \
    "x_eval must have length >= 1, get {} instead"
X_NODE_AND_X_EVAL_NOT_THE_SAME_SHAPE_ERROR = \
    "x_node and x_eval must have the same shape, get dimension {} and {} instead"




class Quadrature1D:
    """Factory for Gauss-Legendre / Lobatto nodes and weights on [-1, 1]."""

    @staticmethod
    def gauss_legendre(n: int) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Gauss-Legendre quadrature with Golub-Welsch algorithm
            \int_{-1}^1 f(x) dx ≈ \sum_{i=1}^n w_i f(x_i)
        Return nodes (x_i) and weights (w_i) for Gauss-Legendre on [-1,1].

        Ref: 
        - https://en.wikipedia.org/wiki/Gauss-Legendre_quadrature
        - https://en.wikipedia.org/wiki/Gaussian_quadrature#The_Golub-Welsch_algorithm

        Note: 
        - Exact for polynomials up to degree 2n-1
        - Does not include the boundaries
        """
        assert isinstance(n, int), \
            QUADRATURE_POINT_NUMBER_NOT_INTEGER_ERROR.format(type(n))
        assert n > 0, \
            QUADRATURE_POINT_NUMBER_NOT_GREATER_THAN_0_ERROR.format(n)

        beta = 0.5/np.sqrt(1 - 1 / (2*(np.arange(1,n)))**(2))
        T = np.diag(beta,k=1) + np.diag(beta,k=-1)
        nodes, eigvecs = np.linalg.eigh(T,UPLO='L')
        index = np.arange(nodes.size)
        legendre_weights = 2*(eigvecs[0,index])**2
        return nodes, legendre_weights


    @staticmethod
    def lobatto(n: int, tol: float = 1e-14) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Legendre-Gauss-lobatto points with Chebyshev-Lobatto initial guess
            \int_{-1}^1 f(x) dx ≈ \sum_{i=1}^n w_i f(x_i)
        Return nodes (x_i) and weights (w_i) for Legendre-Gauss-Lobatto on [-1,1].
        
        Ref: 
        - Greg von Winckel (2024). Legende-Gauss-Lobatto nodes and weights
        - https://www.mathworks.com/matlabcentral/fileexchange/4775-legende-gauss-lobatto-nodes-and-weights

        Note:
        - Exact for polynomials up to degree 2n-3
        - Includes the boundaries
        """

        assert isinstance(n, int), \
            QUADRATURE_POINT_NUMBER_NOT_INTEGER_ERROR.format(type(n))
        assert n > 0, \
            QUADRATURE_POINT_NUMBER_NOT_GREATER_THAN_0_ERROR.format(n)

        # initial guess: Chebyshev–Lobatto
        nodes = np.cos(np.pi * np.arange(n+1) / n)
        nodes_old = 2*np.ones_like(nodes)
        Pvals = np.zeros((n+1, n+1))
        while np.max(np.abs(nodes - nodes_old)) > tol:
            nodes_old = nodes
            Pvals[0, :] = 1.0
            Pvals[1, :] = nodes
            for k in range(2, n+1):
                Pvals[k, :] = ((2*k-1)*nodes*Pvals[k-1,:] - (k-1)*Pvals[k-2,:]) / k
            nodes = nodes_old - (nodes*Pvals[n,:] - Pvals[n-1,:]) / ((n+1)*Pvals[n,:])
        legendre_weights = 2.0 / (n*(n+1) * (Pvals[n,:]**2))
        return nodes, legendre_weights



class Mesh1D:
    """1D radial mesh on [0, 2R] with uniform / polynomial / exponential spacing."""

    def __init__(
        self,
        domain_radius       : float,            # extent of radial domain, in Bohr
        finite_elements_num : int,              # number of finite elements
        mesh_type           : str,              # 'uniform', 'polynomial', 'exponential'
        clustering_param    : float | None,     # concentration parameter for node distribution
        exp_shift           : float | None,     # shift parameter for exponential distribution
        ):

        self.R = float(domain_radius)
        self.n_elem = finite_elements_num
        self.mesh_type = mesh_type
        self.concentration = clustering_param
        self.exp_shift = exp_shift

        self.check_initial_parameters()


    def check_initial_parameters(self):
        # domain radius
        assert isinstance(self.R, float), \
            DOMAIN_RADIUS_NOT_FLOAT_ERROR.format(type(self.R))
        assert self.R > 0., \
            DOMAIN_RADIUS_NOT_GREATER_THAN_0_ERROR.format(self.R)
        
        # number of finite elements
        assert isinstance(self.n_elem, int), \
            NUMBER_OF_FINITE_ELEMENTS_NOT_INTEGER_ERROR.format(type(self.n_elem))
        assert self.n_elem > 0, \
            NUMBER_OF_FINITE_ELEMENTS_NOT_GREATER_THAN_0_ERROR.format(self.n_elem)
        
        # mesh type
        assert isinstance(self.mesh_type, str), \
            MESH_TYPE_NOT_STRING_ERROR.format(type(self.mesh_type))
        assert self.mesh_type in ['uniform', 'polynomial', 'exponential'], \
            MESH_TYPE_NOT_IN_VALID_LIST_ERROR.format(['uniform', 'polynomial', 'exponential'], self.mesh_type)
        
        # clustering parameter
        if self.mesh_type in ['polynomial', 'exponential']:
            assert isinstance(self.concentration, float), \
                CLUSTERING_PARAMETER_NOT_FLOAT_ERROR.format(type(self.concentration))
            assert self.concentration > 0., \
                CLUSTERING_PARAMETER_NOT_GREATER_THAN_0_ERROR.format(self.concentration)
            if self.mesh_type == 'exponential' and self.concentration == 1.0:
                raise NotImplementedError("Exponential mesh with concentration 1.0 is not implemented yet, should degrade to uniform mesh")
        elif self.mesh_type == 'uniform':
            if self.concentration is not None:
                print(CLUSTERING_PARAMETER_NOT_NONE_FOR_UNIFORM_MESH_TYPE_WARNING)
        else:
            raise ValueError("This error should never be raised")

        # exp_shift parameter
        if self.mesh_type == 'exponential':
            if self.exp_shift is None:
                self.exp_shift = 0.0  # default value
            assert isinstance(self.exp_shift, float), \
                EXP_SHIFT_NOT_FLOAT_ERROR.format(type(self.exp_shift))
            assert self.exp_shift >= 0., \
                EXP_SHIFT_NEGATIVE_ERROR.format(self.exp_shift)
        elif self.mesh_type in ['uniform', 'polynomial']:
            if self.exp_shift is not None:
                print(EXP_SHIFT_NOT_NONE_FOR_UNIFORM_AND_POLYNOMIAL_MESH_TYPE_WARNING)
        else:
            raise ValueError("This error should never be raised")


    def generate_mesh_nodes_and_width(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate mesh nodes in [0, 2R] for uniform, polynomial, and exponential distributions.
        For polynomial and exponential distributions, the nodes are distributed according to the clustering parameter (s or a) and exp_shift (c).

        Mesh types:
        - Uniform
            r_i = i * (2R) / n_elem
        - Polynomial:
            r_i = 2R * (i**s) / (n_elem ** s)
        - Exponential:
            r_i = c + (2R-c) * (a**(i/(n_elem-1)) - 1) / (a**(n_elem/(n_elem-1)) - 1)
        
        Returns:
            mesh_nodes: array of mesh nodes r in [0, 2R]
            mesh_width: array of mesh widths
        """
        if self.mesh_type == "uniform":
            mesh_nodes = np.linspace(0.0, 2.0 * self.R, self.n_elem + 1)
        elif self.mesh_type == "polynomial":
            s = float(self.concentration)
            mesh_nodes = 2.0 * self.R * (np.arange(self.n_elem+1)**s) / (self.n_elem ** s)
        elif self.mesh_type == "exponential":
            a = float(self.concentration)
            c = float(self.exp_shift)
            beta = np.log(a) / (self.n_elem - 1)
            alpha = (2.0 * self.R - c) / (np.exp(beta*self.n_elem) - 1.0)
            mesh_nodes = c + alpha * (np.exp(beta*np.arange(self.n_elem+1)) - 1.0)
            mesh_nodes[0]  = 0.0 
            mesh_nodes[-1] = 2.0 * self.R
        else:
            raise ValueError(f"Unknown mesh_type={self.mesh_type}")
        
        mesh_width = mesh_nodes[1:] - mesh_nodes[:-1]

        assert len(mesh_nodes) == self.n_elem + 1, \
            MESH_NODES_NOT_EQUAL_TO_NUMBER_OF_FINITE_ELEMENTS_PLUS_1_ERROR.format(len(mesh_nodes))
        assert len(mesh_width) == self.n_elem, \
            MESH_WIDTH_NOT_EQUAL_TO_NUMBER_OF_FINITE_ELEMENTS_ERROR.format(len(mesh_width))
        return mesh_nodes, mesh_width


    @staticmethod
    def generate_fe_nodes(boundaries_nodes: np.ndarray,
                          interp_nodes: np.ndarray) -> np.ndarray:
        """
        Map reference nodes from [-1,1] to each physical element [r_i, r_{i+1}],
        remove duplicate interface nodes, and return the global FE nodes.

        Parameters
        ----------
        boundaries_nodes : np.ndarray, shape (num_elements+1,)
            Coordinates of boundaries of finite elements (monotonically increasing or decreasing).
        interp_nodes : np.ndarray, shape (poly_order+1,)
            Interpolation nodes in [-1,1], e.g. Lobatto nodes.

        Returns
        -------
        global_nodes : np.ndarray, shape (num_elements * poly_order + 1,)
            Concatenated FE nodes over [0,2R] without duplicate interface nodes.
        """
        assert isinstance(boundaries_nodes, np.ndarray), \
            BOUNDARIES_NODES_NOT_NUMPY_ARRAY_ERROR
        assert isinstance(interp_nodes, np.ndarray), \
            INTERP_NODES_NOT_NUMPY_ARRAY_ERROR

        # reverse the nodes if the nodes are decreasing
        if np.all(np.diff(boundaries_nodes) < 0.0):
            boundaries_nodes = boundaries_nodes[::-1]
        if np.all(np.diff(interp_nodes) < 0.0):
            interp_nodes = interp_nodes[::-1]

        # check if the nodes are monotonically increasing
        assert np.all(np.diff(boundaries_nodes) > 0.0), \
            BOUNDARIES_NODES_NOT_MONOTONICALLY_INCREASING_ERROR
        assert np.all(np.diff(interp_nodes) > 0.0), \
            INTERP_NODES_NOT_MONOTONICALLY_INCREASING_ERROR


        # check if interp_nodes includes -1 and 1 at both boundaries
        assert np.isclose(interp_nodes[0], -1.0), \
            INTERP_NODES_NOT_INCLUDE_MINUS_1_AT_LEFT_BOUNDARY_ERROR
        assert np.isclose(interp_nodes[-1], 1.0), \
            INTERP_NODES_NOT_INCLUDE_1_AT_RIGHT_BOUNDARY_ERROR

        num_elements = boundaries_nodes.size - 1
        poly_order = interp_nodes.size - 1

        left = boundaries_nodes[:-1]
        right = boundaries_nodes[1:]

        # affine map
        mapped = 0.5*(right - left)[:, None] * interp_nodes[None, :] \
                 + 0.5*(right + left)[:, None]

        global_nodes = mapped.ravel()

        # remove duplicates: last node of each element except the last
        remove_idx = np.arange(poly_order,
                               (poly_order+1)*(num_elements-1)+poly_order,
                               poly_order+1)
        global_nodes = np.delete(global_nodes, remove_idx)

        # sanity check
        expected_len = num_elements * poly_order + 1
        assert global_nodes.size == expected_len, \
            GLOBAL_NODES_NOT_EQUAL_TO_EXPECTED_LENGTH_ERROR.format(expected_len, global_nodes.size)

        return global_nodes


    @staticmethod
    def refine_interpolation_nodes(base_nodes: np.ndarray) -> np.ndarray:
        """
        Refine interpolation nodes by inserting midpoints between existing nodes
        and adding extra midpoints near the two boundaries.

        Parameters
        ----------
        base_nodes : np.ndarray
            Original interpolation nodes (e.g., Lobatto nodes on [-1,1]).

        Returns
        -------
        refined_nodes : np.ndarray
            Refined nodes, containing original nodes, all midpoints,
            and extra midpoints near the left and right boundaries.
            The output is returned in descending order (consistent with some FE codes).
        """
        assert isinstance(base_nodes, np.ndarray), \
            BASE_NODES_NOT_NUMPY_ARRAY_ERROR
        assert base_nodes.ndim == 1, \
            BASE_NODES_NOT_1D_ARRAY_ERROR
        
        # reverse the nodes if the nodes are decreasing
        monotonically_increasing_flag = True
        if np.all(np.diff(base_nodes) < 0.0):
            base_nodes = base_nodes[::-1]
            monotonically_increasing_flag = False
        assert np.all(np.diff(base_nodes) > 0.0), \
            BASE_NODES_NOT_MONOTONICALLY_INCREASING_OR_DECREASING_ERROR


        # insert midpoints between consecutive nodes
        midpoints = base_nodes[:-1] + 0.5*np.diff(base_nodes)
        refined_nodes = np.sort(np.concatenate((base_nodes, midpoints)))

        # extra midpoint near left boundary
        refined_nodes = np.insert(refined_nodes, 1, 0.5*(refined_nodes[0] + refined_nodes[1]))

        # extra midpoint near right boundary
        refined_nodes = np.insert(refined_nodes, len(refined_nodes)-1, 0.5*(refined_nodes[-1] + refined_nodes[-2]))

        # flip to descending order (optional, to match legacy FE codes)
        if monotonically_increasing_flag:
            refined_nodes = np.flip(refined_nodes)

        return refined_nodes


    @staticmethod
    def map_quadrature_to_physical_elements(
        boundaries_nodes : np.ndarray,
        interp_nodes     : np.ndarray,
        interp_weights   : np.ndarray,
        flatten          : bool = True,
        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Affine-map Gauss-Legendre (or any reference) quadrature nodes/weights from [-1,1]
        to each physical element [x_i, x_{i+1}] and return global points/weights.

        Parameters
        ----------
        boundaries_nodes : (N_e+1,) array_like
            Monotonically nondecreasing element boundaries [x_0, ..., x_{N_e}].
        interp_nodes : (q,) array_like
            Quadrature nodes in [-1,1] on the reference interval.
        interp_weights : (q,) array_like
            Corresponding quadrature weights on [-1,1].
        flatten : bool, default True
            If True, return 1D arrays of length N_e*q (global concatenation).
            If False, return 2D arrays of shape (N_e, q) per element.

        Returns
        -------
        quad_x : ndarray
            Mapped quadrature nodes. Shape (N_e*q,) if flatten else (N_e, q).
        quad_w : ndarray
            Mapped quadrature weights. Shape (N_e*q,) if flatten else (N_e, q).

        Notes
        -----
        Mapping formula for each element [xL, xR]:
            x = 0.5*(xR - xL)*xi + 0.5*(xR + xL)
            w = 0.5*(xR - xL)*w_ref
        """
        assert isinstance(boundaries_nodes, np.ndarray), \
            BOUNDARIES_NODES_NOT_NUMPY_ARRAY_ERROR
        assert isinstance(interp_nodes, np.ndarray), \
            INTERP_NODES_NOT_NUMPY_ARRAY_ERROR
        assert isinstance(interp_weights, np.ndarray), \
            INTERP_WEIGHTS_NOT_NUMPY_ARRAY_ERROR

        xb = np.asarray(boundaries_nodes, dtype=float).reshape(-1)
        xi = np.asarray(interp_nodes, dtype=float).reshape(-1)
        wi = np.asarray(interp_weights, dtype=float).reshape(-1)

        # basic checks
        assert xb.size >= 2, \
            BOUNDARIES_NODES_NOT_AT_LEAST_2_ERROR
        assert xi.size == wi.size, \
            INTERP_NODES_AND_WEIGHTS_NOT_THE_SAME_LENGTH_ERROR
        assert xi.size >= 1, \
            INTERP_NODES_NOT_AT_LEAST_1_ERROR
        assert np.all(np.diff(xb) >= 0.0), \
            BOUNDARIES_NODES_NOT_NONDECREASING_ERROR


        Ne = xb.size - 1
        q = xi.size

        xL = xb[:-1]                    # (Ne,)
        xR = xb[1:]                     # (Ne,)
        h  = (xR - xL)[:, None]         # (Ne,1)

        # map nodes & weights: (Ne,q)
        quad_x = 0.5 * h * xi[None, :] + 0.5 * (xR + xL)[:, None]
        quad_w = 0.5 * h * wi[None, :]

        if flatten:
            return quad_x.reshape(-1), quad_w.reshape(-1)
        return quad_x, quad_w


    @staticmethod
    def fe_flat_to_block2d(
        flat: np.ndarray,
        n_elem: int,
        endpoints_shared: bool,
        ) -> np.ndarray:

        r"""
        Convert a 1D concatenated FE grid into a 2D array of shape (n_elem, m),
        where m is the number of points per element.

            Two supported layouts for 'flat':
            1) endpoints_shared=True:
                Global array stores unique boundary nodes, so consecutive elements share 1 node.
                Expected length: len(flat) = n_elem * (m - 1) + m.
            2) endpoints_shared=False:
                Elements are simply stacked; no overlapping boundary nodes.
                Expected length: len(flat) = n_elem * m.

            Parameters
            ----------
            flat : (L,) 1D ndarray
                Concatenated points on the reference line.
            n_elem : int
                Number of finite elements.
            points_per_elem : Optional[int]
                Points per element (m). If None, it will be inferred from len(flat), n_elem,
                and endpoints_shared (if provided) or by trying both layouts.
            endpoints_shared : Optional[bool]
                If True/False, use the specified layout. If None, auto-detect by matching lengths.

            Returns
            -------
            blocks : (n_elem, m) ndarray
                The 2D array where each row corresponds to one element's local grid.

            Examples
            --------
            # Example 1: endpoints shared (unique global nodes)
            
                Suppose each element has m=4 nodes and n_elem=3:
                element 0: [0, 1, 2, 3]
                element 1: [3, 4, 5, 6]
                element 2: [6, 7, 8, 9]
                flat1: 
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  (length = 3*4 - 2 = 10)
                blocks1:
                    [[0, 1, 2, 3],
                     [3, 4, 5, 6],
                     [6, 7, 8, 9]]

            # Example 2: endpoints NOT shared (stacked)
                Suppose each element has m=4 nodes and n_elem=3:
                element 0: [0, 1, 2, 3]
                element 1: [4, 5, 6, 7]
                element 2: [8, 9,10,11]
                flat2:
                    [0,1,2,3,  4,5,6,7,  8,9,10,11] (length = 3*4 = 12)
                blocks2:
                    [[0, 1, 2, 3],
                     [4, 5, 6, 7],
                     [8, 9,10,11]]
        """

        assert isinstance(flat, np.ndarray), \
            FLAT_NOT_NUMPY_ARRAY_ERROR
        assert flat.ndim == 1, \
            FLAT_NOT_1D_ARRAY_ERROR
        assert isinstance(n_elem, int), \
            N_ELEM_NOT_INT_ERROR
        assert isinstance(endpoints_shared, bool), \
            ENDPOINTS_SHARED_NOT_BOOL_ERROR


        # ---- Construct the (n_elem, m) view/array ----
        if endpoints_shared:
            assert (flat.size - 1) % n_elem == 0, \
                FLAT_NOT_EXPECTED_LENGTH_ENDPOINTS_SHARED_TRUE_ERROR.format(flat.size, n_elem)
            # Overlapping windows with stride (m-1)
            m = (flat.size - 1) // n_elem + 1
            out = np.empty((n_elem, m), dtype=flat.dtype)
            for i_elem in range(n_elem):
                start = i_elem * (m - 1)
                out[i_elem, :] = flat[start : start + m]
            return out

        else:
            assert flat.size % n_elem == 0, \
                FLAT_NOT_EXPECTED_LENGTH_ENDPOINTS_SHARED_FALSE_ERROR.format(flat.size, n_elem)
            # Simple reshape when elements are stacked without overlap
            return flat.reshape(n_elem, -1)



class LagrangeShapeFunctions:
    """
    Lagrange shape functions (and their derivatives) built from a given set of nodes.
    This class replaces the ad-hoc broadcasting logic in
    - lagrange_polynomial(...)
    - lagrange_polynomial_t_obtain_quantities_uniform_grid(...)

    It supports:
      1) Reference-element evaluation: given reference nodes xj in [-1,1], evaluate Phi,dPhi at xi.
      2) Physical-element evaluation: given element-wise physical nodes rj (Ne,p+1) and eval points re (Ne,n_eval),
         evaluate Phi(re), dPhi(re) w.r.t. the physical coordinate r (chain rule handled).
      3) Interpolation to a global uniform grid with per-element selection (returns lag_poly and raw_indexes).
    """



    # @classmethod
    # def lagrange_basis_and_derivatives(cls, x_node: np.ndarray, x_eval: np.ndarray, atol: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    #     return cls.lagrange_basis_and_derivatives_old(x_node, x_eval, atol)


    @staticmethod
    def lagrange_basis_and_derivatives(x_node: np.ndarray, x_eval: np.ndarray, atol: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Compute Lagrange basis values and their first derivatives on each FE element,
        fully vectorized over elements, evaluation points, and basis indices.

        Parameters
        ----------
        x_node : ndarray, shape (n_elem, n_node)
            Element-local nodal coordinates x_j for each element (one row per element).
            For an element with polynomial degree p, m = p+1.
        x_eval : ndarray, shape (n_elem, n_eval)
            Evaluation points on each element (one row per element). 
            These can be quadrature points or the nodes themselves.
        atol : float
            Absolute tolerance to decide if an evaluation point exactly coincides
            with a node (used to avoid 0/0 and to apply the exact nodal limits).


        Returns
        -------
        L : ndarray, shape (n_elem, n_eval, n_node)
            Lagrange basis values. L[i, j, k] = L_k(x_eval_ij) on element i.
        dLdx : ndarray, shape (n_elem, n_eval, n_node)
            First derivatives of the Lagrange basis. dLdx[i, j, k] = d/dx L_k at x_eval_ij.


        Mathematics (per element)
        -------------------------
        Let nodes be {x_k}_{k=0}^{m-1}, m = n_node.
        Barycentric weights:
            ω_k = 1 / prod_{t≠k} (x_k - x_t)

        For a non-nodal evaluation point x:
            L_k(x) = (ω_k / (x - x_k)) / ( sum_t(ω_t / (x - x_t)) )

        Derivative at a non-nodal x:
            L'_k(x) = L_k(x) * ( sum_t(1/(x - x_t)) - 1/(x - x_k) )

        For nodal x = x_a, the derivative row equals the a-th row of the
        nodal differentiation matrix D, with off-diagonal entries
            D[a,b] = (c_a / c_b) / (x_a - x_b),  a ≠ b,
        where c_k = prod_{t≠k} (x_k - x_t), and diagonal fixed by
            D[a,a] = - sum_{b≠a}( D[a,b] ).


        Notes
        -----
        * Uses absolute matching tolerance `atol` with rtol=0 to detect x exactly at nodes.
        * Avoids forming large/small products at evaluation points (stable vs. "full-product then divide").
        """

        # basic checks
        assert isinstance(x_node, np.ndarray), \
            X_NODE_NOT_NUMPY_ARRAY_ERROR.format(type(x_node))
        assert isinstance(x_eval, np.ndarray), \
            X_EVAL_NOT_NUMPY_ARRAY_ERROR.format(type(x_eval))
        if x_node.ndim == 1:
            x_node = x_node[None, :]
        if x_eval.ndim == 1:
            x_eval = x_eval[None, :]
        assert x_node.ndim == 2, \
            X_NODE_NOT_1D_OR_2D_ARRAY_ERROR.format(x_node.ndim)
        assert x_eval.ndim == 2, \
            X_EVAL_NOT_1D_OR_2D_ARRAY_ERROR.format(x_eval.ndim)
        assert x_node.shape[0] == x_eval.shape[0], \
            X_NODE_AND_X_EVAL_NOT_THE_SAME_SHAPE_ERROR.format(x_node.shape[0], x_eval.shape[0])
        assert x_node.shape[1] >= 2, \
            X_NODE_NOT_AT_LEAST_2_ERROR.format(x_node.shape[1])
        assert x_eval.shape[1] >= 1, \
            X_EVAL_NOT_AT_LEAST_1_ERROR.format(x_eval.shape[1])

        # shapes
        n_elem, n_node = x_node.shape
        n_elem, n_eval = x_eval.shape

        # 1) Precompute barycentric weights: omega = 1 / c, where
        #    c_k = prod_{t≠k} (x_k - x_t), done per element.

        # diffs_nodes[i, k, t] = x_k - x_t (on element i)
        diffs_nodes = x_node[:, :, None] - x_node[:, None, :]                 # (n_elem, n_node, n_node)
        mask_offdiag = ~np.eye(n_node, dtype=bool)                            # (n_node, n_node) True for a≠b

        # c[i, k] = prod_{t≠k} (x_k - x_t): set diagonal to 1 so the product ignores it
        c = np.where(mask_offdiag[None, :, :], diffs_nodes, 1.0).prod(axis=2) # (n_elem, n_node)
        omega = 1.0 / c                                                       # (n_elem, n_node)

        # 2) Distances from evaluation points to nodes: dx = x - x_k
        dx = x_eval[:, :, None] - x_node[:, None, :]                          # (n_elem, n_eval, n_node)

        # 3) Identify nodal evaluations (x == x_k) using absolute tolerance
        is_nodal = np.isclose(x_eval[:, :, None], x_node[:, None, :], atol=atol)  # (n_elem, n_eval, n_node)
        has_nodal_row = is_nodal.any(axis=2)                                      # (n_elem, n_eval)

        # 4) Basis values at non-nodal points via barycentric formula
        #    L_k(x) = (omega_k /(x - x_k)) / sum_t (omega_t /(x - x_t))
        with np.errstate(divide='ignore', invalid='ignore'):
            inv_dx = 1.0 / dx                                               # (n_elem, n_point, n_node)
            numer = omega[:, None, :] * inv_dx                              # (n_elem, n_point, n_node)
            denom = np.sum(numer, axis=2, keepdims=True)                    # (n_elem, n_point, 1)
            L = numer / denom                                               # (n_elem, n_point, n_node)

        # Enforce nodal rows to be exact one-hot (interpolation property)
        # First zero out rows that contain nodal matches; then set matching entries to 1.
        L = np.where(np.isfinite(L), L, 0.0)
        L = np.where(has_nodal_row[:, :, None], 0.0, L)
        L = np.where(is_nodal, 1.0, L)

        # 5) Derivatives at non-nodal points:
        #    L'_k(x) = L_k(x) * ( sum_t 1/(x - x_t) - 1/(x - x_k) )
        with np.errstate(divide='ignore', invalid='ignore'):
            harmonic_sum = np.sum(inv_dx, axis=2, keepdims=True)            # (n_elem, n_point, 1)
            dLdx = L * (harmonic_sum - inv_dx)                              # (n_elem, n_point, n_node)

        # 6) Nodal differentiation matrix D on the nodes {x_k} (per element)
        #    Off-diagonal: D[a,b] = (c_a / c_b) / (x_a - x_b)
        #    Diagonal:     D[a,a] = -sum_{b≠a} D[a,b]
        with np.errstate(divide='ignore', invalid='ignore'):
            D = (c[:, :, None] / c[:, None, :]) / diffs_nodes               # (n_elem, n_node, n_node)
        D = np.where(mask_offdiag[None, :, :], D, 0.0)
        D[..., np.arange(n_node), np.arange(n_node)] = -np.sum(D, axis=2)

        # Overwrite derivative rows where x coincides with a node:
        # For each (i, j) with a nodal match, find which node index k* and set dLdx[i, j, :] = D[i, k*, :]
        if has_nodal_row.any():
            nodal_index = np.argmax(is_nodal, axis=2)                       # (n_elem, n_point), argmax is safe since rows with no nodal are masked by has_nodal_row
            i_idx, j_idx = np.nonzero(has_nodal_row)                        # indices of rows to overwrite
            k_idx = nodal_index[i_idx, j_idx]
            dLdx[i_idx, j_idx, :] = D[i_idx, k_idx, :]

        # 7) Cleanup any residual non-finite entries (should be rare)
        L = np.where(np.isfinite(L), L, 0.0)
        dLdx = np.where(np.isfinite(dLdx), dLdx, 0.0)

        return L, dLdx



    @staticmethod
    def lagrange_basis_and_derivatives_old(
        x_node: np.ndarray,
        x_eval: np.ndarray,
        atol: float = 1e-12
        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Old-style algorithm that mirrors the legacy implementation's math:
        - build (x - x_k) slabs and take products
        - make denominator by (x_k - x_t) with diagonal shifted to 1
        - handle nodal points by replacing invalid numerators with denominators
        - assemble derivative via "leave-one-factor-out" product sums
        Returns L, dLdx with the same shapes/semantics as the optimized version.
        """

        raise NotImplementedError("This function is now deprecated and will be removed in the future.")

        # --- shape normalization (match new API) ---
        # basic checks
        assert isinstance(x_node, np.ndarray), \
            X_NODE_NOT_NUMPY_ARRAY_ERROR.format(type(x_node))
        assert isinstance(x_eval, np.ndarray), \
            X_EVAL_NOT_NUMPY_ARRAY_ERROR.format(type(x_eval))
        if x_node.ndim == 1:
            x_node = x_node[None, :]
        if x_eval.ndim == 1:
            x_eval = x_eval[None, :]
        assert x_node.ndim == 2, \
            X_NODE_NOT_1D_OR_2D_ARRAY_ERROR.format(x_node.ndim)
        assert x_eval.ndim == 2, \
            X_EVAL_NOT_1D_OR_2D_ARRAY_ERROR.format(x_eval.ndim)
        assert x_node.shape[0] == x_eval.shape[0], \
            X_NODE_AND_X_EVAL_NOT_THE_SAME_SHAPE_ERROR.format(x_node.shape[0], x_eval.shape[0])
        assert x_node.shape[1] >= 2, \
            X_NODE_NOT_AT_LEAST_2_ERROR.format(x_node.shape[1])
        assert x_eval.shape[1] >= 1, \
            X_EVAL_NOT_AT_LEAST_1_ERROR.format(x_eval.shape[1])

        # Normalize shapes
        if x_node.ndim == 1:
            x_node = x_node[None, :]
        if x_eval.ndim == 1:
            x_eval = x_eval[None, :]
        n_elem, n_node = x_node.shape
        _, n_eval = x_eval.shape
        assert n_node >= 2 and n_eval >= 1, "Need at least 2 nodes and 1 evaluation point."

        # Legacy variable names
        init_FE    = x_node[:, None, :]     # (n_elem, 1, n_node)
        int_points = x_eval[:, :,  None]    # (n_elem, n_eval, 1)

        # (x - x_k)
        lp   = lambda r, x: r - x
        lagr = lp(int_points, init_FE)      # (n_elem, n_eval, n_node)

        # Numerator: ∏_t (x - x_t) / (x - x_k)
        prod_lagr    = np.prod(lagr, axis=2)                   # (n_elem, n_eval)
        one_by_lagr  = 1.0 / lagr                              # (n_elem, n_eval, n_node)
        lag_poly_num = prod_lagr[:, :, None] * one_by_lagr     # (n_elem, n_eval, n_node)

        # Denominator: ∏_{t≠k} (x_k - x_t), with diagonal set to 1
        diffs_nodes    = (np.transpose(init_FE, (0, 2, 1)) - init_FE)  # (n_elem, n_node, n_node)
        eye            = np.eye(n_node)[None, :, :]
        lag_poly_d_w_z = diffs_nodes + eye                            # diagonal -> 1
        lag_poly_d     = np.prod(lag_poly_d_w_z, axis=2)              # (n_elem, n_node)
        lag_poly_den   = lag_poly_d[:, None, :]                       # (n_elem, 1, n_node)

        # Nodal coincidence detection: x ≈ x_k
        is_nodal = np.isclose(x_eval[:, :, None], x_node[:, None, :], atol=atol, rtol=0.0)

        # Replace non-finite numerator with denominator (nodal limit => 1)
        bad = ~np.isfinite(lag_poly_num)
        if np.any(bad):
            ii, jj, kk = np.where(bad)
            lag_poly_num[ii, jj, kk] = lag_poly_den[ii, 0, kk]

        # Basis values
        L = lag_poly_num / lag_poly_den
        if np.any(is_nodal):
            L = np.where(is_nodal.any(axis=2, keepdims=True), 0.0, L)
            L = np.where(is_nodal, 1.0, L)
        L = np.where(np.isfinite(L), L, 0.0)

        # ===== Derivative via "leave-one-factor-out" =====
        # lagr_k_t[e, j, k, t] = (x_eval - x_t), replicated along k
        lagr_k_t = np.broadcast_to(lagr[:, :, None, :], (n_elem, n_eval, n_node, n_node))

        # Build per-row indices t ≠ k in a vectorized way
        # off_idx[k] = [0,1,...,k-1,k+1,...,n_node-1]
        all_idx = np.tile(np.arange(n_node), (n_node, 1))      # (n_node, n_node)
        mask    = ~np.eye(n_node, dtype=bool)                  # (n_node, n_node)
        off_idx = all_idx[mask].reshape(n_node, n_node - 1)    # (n_node, n_node-1)

        # Gather (x - x_t) for all t ≠ k -> shape (n_elem, n_eval, n_node, n_node-1)
        lagr_broad_d = np.take_along_axis(lagr_k_t, off_idx[None, None, :, :], axis=3)

        # Replicate numerator along the excluded-factor axis
        diff_lag = np.broadcast_to(lag_poly_num[:, :, :, None],
                                (n_elem, n_eval, n_node, n_node - 1))

        # Divide and handle nodal-limit fixes
        diff_lag_poly_num = diff_lag / lagr_broad_d

        bad_d = ~np.isfinite(diff_lag_poly_num)
        if np.any(bad_d):
            ii, jj, kk, mm = np.where(bad_d)
            tmp = lagr_broad_d.copy()
            # Temporarily set the problematic factor to NaN, then multiply the rest
            tmp[ii, jj, kk, mm] = np.nan
            fill = np.nancumprod(tmp[ii, jj, kk, :], axis=1)[:, -1]  # product of remaining factors
            # Restore the slot and write back the limit
            tmp[ii, jj, kk, mm] = 0.0
            diff_lag_poly_num[ii, jj, kk, mm] = fill

        # Sum over exclusions and divide by denominator
        diff_lag_poly_num_sum = np.sum(diff_lag_poly_num, axis=3)  # (n_elem, n_eval, n_node)
        dLdx = diff_lag_poly_num_sum / lag_poly_den

        # Overwrite nodal rows with the exact nodal differentiation matrix
        if np.any(is_nodal):
            c = np.prod(np.where(eye.astype(bool), 1.0, diffs_nodes), axis=2)   # (n_elem, n_node)
            with np.errstate(divide='ignore', invalid='ignore'):
                D = (c[:, :, None] / c[:, None, :]) / diffs_nodes
            D = np.where(~np.eye(n_node, dtype=bool)[None, :, :], D, 0.0)
            D[..., np.arange(n_node), np.arange(n_node)] = -np.sum(D, axis=2)

            nodal_rows = is_nodal.any(axis=2)  # (n_elem, n_eval)
            if np.any(nodal_rows):
                nodal_index = np.argmax(is_nodal, axis=2)
                ie, je = np.nonzero(nodal_rows)
                ke = nodal_index[ie, je]
                dLdx[ie, je, :] = D[ie, ke, :]

        dLdx = np.where(np.isfinite(dLdx), dLdx, 0.0)

        return L, dLdx



    @staticmethod
    def _barycentric_weights(xj: np.ndarray) -> np.ndarray:
        """Compute barycentric weights w_j = 1 / ∏_{m≠j} (x_j - x_m)."""
        xj = np.asarray(xj, dtype=float)
        n = xj.size
        w = np.ones(n, dtype=float)
        for j in range(n):
            diff = xj[j] - xj
            diff[j] = 1.0  # avoid zero
            w[j] = 1.0 / np.prod(diff)
        return w




    @staticmethod
    def _derivative_matrix_at_nodes(xj: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        Derivatives of Lagrange basis evaluated at the nodes:
            D[k,j] = φ_j'(x_k)
        with φ_j the cardinal basis through xj. Closed form:
          for j≠k: D[k,j] = w[j] / ( w[k] * (x_k - x_j) )
          for j=k: D[k,k] = -∑_{m≠k} D[k,m]
        """
        xj = np.asarray(xj, dtype=float)
        n = xj.size
        D = np.empty((n, n), dtype=float)
        for k in range(n):
            for j in range(n):
                if j == k:
                    continue
                D[k, j] = w[j] / (w[k] * (xj[k] - xj[j]))
            D[k, k] = -np.sum(D[k, :]) + D[k, k] if np.isfinite(D[k, k]) else -np.sum(D[k, :])
        return D



class RPAFrequencyGrid:
    """
    Frequency integration grid for RPA correlation energy calculations.
    
    Transforms Gauss-Legendre quadrature from [-1,1] to [0, \infty) for imaginary
    frequency integration required in RPA:
        E_c^RPA = \int_0^\infty f(ω) dω
    
    Reference: https://journals.aps.org/prl/supplemental/10.1103/PhysRevLett.134.016402/scrpa4_SM.pdf
    """
    
    def __init__(
        self,
        n_frequency_points: int,
        frequency_scale: float = 2.5
        ):
        """
        Initialize RPA frequency grid generator.
        
        Parameters
        ----------
        n_frequency_points : int
            Number of frequency integration points (q_omega)
        frequency_scale : float
            Transformation scale parameter (default 2.5)
        """
        self.n_frequency_points = n_frequency_points
        self.frequency_scale = frequency_scale
        
        # Generate main frequency grid
        self.omega_points, self.omega_weights = self._generate_main_grid()
    
    
    def _generate_main_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate main frequency grid [0, ∞).
        
        Transformation: ω = scale * (1+ξ)/(1-ξ), ξ ∈ [-1,1]
        Jacobian: dω/dξ = 2*scale/(1-ξ)²
        """
        # Get Gauss-Legendre nodes on reference interval [-1, 1]
        reference_nodes, reference_weights = Quadrature1D.gauss_legendre(self.n_frequency_points)
        
        # Transform to semi-infinite interval [0, ∞)
        omega_points = self.frequency_scale * (1 + reference_nodes) / (1 - reference_nodes)
        
        # Apply Jacobian transformation to weights
        jacobian = 2 * self.frequency_scale / (1 - reference_nodes)**2
        omega_weights = reference_weights * jacobian
        
        return omega_points, omega_weights
    
    
    def generate_smoothing_grid(self,
        n_smoothing_points: int,
        smoothing_cutoff: float
        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate low-frequency smoothing grid [0, cutoff].
        
        Parameters
        ----------
        n_smoothing_points : int
            Number of smoothing points (q_smoothening)
        smoothing_cutoff : float
            Cutoff frequency (f_c_smoothening)
        
        Returns
        -------
        r_quad_smoothening : np.ndarray
            Smoothing frequency points
        w_smoothening : np.ndarray
            Smoothing frequency weights
        """
        # Get Gauss-Legendre nodes on reference interval [-1, 1]
        reference_nodes, reference_weights = Quadrature1D.gauss_legendre(n_smoothing_points)
        
        # Linear transformation: [-1,1] → [0, cutoff]
        # f = cutoff * (ξ + 1) / 2
        smoothing_frequency_points = smoothing_cutoff * (reference_nodes + 1) / 2
        
        # Scale weights by interval length
        smoothing_frequency_weights = reference_weights * (smoothing_cutoff / 2)
        
        return smoothing_frequency_points, smoothing_frequency_weights
    
    
    def get_frequency_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get main frequency integration grid.
        
        Returns
        -------
        r_quad_omega : np.ndarray
            Frequency points
        w_omega : np.ndarray
            Frequency weights
        """
        return self.omega_points, self.omega_weights



if __name__ == "__main__":
    # nodes, legendre_weights = Quadrature1D.gauss_legendre(95)
    # print("legendre_weights: \n", legendre_weights)
    # print("nodes: \n", nodes)
    
    
    
    print(Mesh1D.map_quadrature_to_physical_elements(
        boundaries_nodes=np.array([0.0, 1.0, 2.0, 3.0]), 
        interp_nodes=np.array([-1.0, 0.0, 1.0]), 
        interp_weights=np.array([1.0, 1.0, 1.0])))
    pass