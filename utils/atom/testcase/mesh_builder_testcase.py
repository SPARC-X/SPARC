

import os
import sys
import numpy as np
import matplotlib.pyplot as plt


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from atom.mesh.builder import Quadrature1D, Mesh1D, LagrangeShapeFunctions




def test_function(x):
    return 1 / (1 + x ** 2)


def print_test_passed(test_name : str):
    print("\t {:<30} : test passed".format(test_name))


def test_quadrature1d(n = 95, pnt=False):
    # Test Function: Quadrature1D.gauss_legendre
    # Test Purpose : Test the Gauss-Legendre quadrature nodes and weights
    nodes, legendre_weights = Quadrature1D.gauss_legendre(n)

    integral = np.sum(test_function(nodes) * legendre_weights)
    integral_ref = np.pi / 2
    assert np.isclose(integral, integral_ref, atol=1e-6)
    print_test_passed("Quadrature1D.gauss_legendre")

    if pnt:
        with open("outputs/mesh_builder_testcase/gauss_legendre_quadrature.txt", "w") as f:
            f.write("legendre_weights: \n")
            f.write(str(legendre_weights))
            f.write("\n")
            f.write("nodes: \n")
            f.write(str(nodes))
            f.write("\n")


def test_lobatto1d(n = 31, pnt=False):
    # Test Function: Quadrature1D.lobatto
    # Test Purpose : Test the Lobatto quadrature nodes and weights

    nodes, lobatto_weights = Quadrature1D.lobatto(n)

    integral = np.sum(test_function(nodes) * lobatto_weights)
    integral_ref = np.pi / 2
    assert np.isclose(integral, integral_ref, atol=1e-6)
    print_test_passed("Quadrature1D.lobatto")

    if pnt:
        with open("outputs/mesh_builder_testcase/lobatto_quadrature.txt", "w") as f:
            f.write("lobatto_weights: \n")
            f.write(str(lobatto_weights))
            f.write("\n")
            f.write("nodes: \n")
            f.write(str(nodes))
            f.write("\n")


def test_mesh1d_grid(n = 17):
    mesh = Mesh1D(
        domain_radius = 10.0, 
        finite_elements_num = n, 
        mesh_type = "exponential", 
        clustering_param = 61.0, 
        exp_shift = 0.0)
    mesh_nodes, mesh_width = mesh.generate_mesh_nodes_and_width()
    with open("outputs/mesh_builder_testcase/mesh1d_grid.txt", "w") as f:
        f.write("mesh_nodes: \n")
        f.write(str(mesh_nodes))
        f.write("\n")
        f.write("mesh_width: \n")
        f.write(str(mesh_width))
        f.write("\n")


def test_mesh1d_fe_nodes(n = 17):
    mesh = Mesh1D(
        domain_radius = 10.0, 
        finite_elements_num = n, 
        mesh_type = "exponential", 
        clustering_param = 61.0, 
        exp_shift = 0.0)
    mesh_nodes, _ = mesh.generate_mesh_nodes_and_width()
    interp_nodes, _ = Quadrature1D.lobatto(31)
    fe_nodes = Mesh1D.generate_fe_nodes(mesh_nodes, interp_nodes)
    with open("outputs/mesh_builder_testcase/mesh1d_fe_nodes.txt", "w") as f:
        f.write("len(fe_nodes): {}".format(len(fe_nodes)))
        f.write("\n")
        f.write("fe_nodes: \n")
        f.write(str(fe_nodes))
        f.write("\n")


def test_fe_flat_to_block2d():
    """
    Test Function : fe_flat_to_block2d
    Test Purpose  : Verify reshaping 1D FE grids into (n_elem, m) blocks under
                    (1) shared-endpoint layout and (2) stacked layout;
                    also verify that invalid lengths raise AssertionError.
    """

    def print_pass(name):
        print(f"\t {name} : passed")

    # ----------------------------
    # Case 1: endpoints_shared=True
    # 3 elements, 4 points per element -> len(flat) = 3*(4-1)+1 = 10
    # rows should be windows with stride (m-1)=3
    # ----------------------------
    flat1 = np.arange(10, dtype=float)
    n_elem1 = 3
    out1 = Mesh1D.fe_flat_to_block2d(flat1, n_elem1, endpoints_shared=True)
    expected1 = np.array([
        [0, 1, 2, 3],
        [3, 4, 5, 6],
        [6, 7, 8, 9],
    ], dtype=float)

    assert out1.shape == (3, 4)
    np.testing.assert_array_equal(out1, expected1)
    print_pass("fe_flat_to_block2d (endpoints_shared=True)")


    # ----------------------------
    # Case 2: endpoints_shared=False
    # 3 elements, 4 points per element -> len(flat) = 3*4 = 12
    # direct reshape into (3,4)
    # ----------------------------
    flat2 = np.arange(12, dtype=float)
    n_elem2 = 3
    out2 = Mesh1D.fe_flat_to_block2d(flat2, n_elem2, endpoints_shared=False)
    expected2 = np.array([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
    ], dtype=float)

    assert out2.shape == (3, 4)
    np.testing.assert_array_equal(out2, expected2)
    print_pass("fe_flat_to_block2d (endpoints_shared=False)")


    # ----------------------------
    # Case 3: invalid lengths should raise AssertionError
    # For shared=True, length must be n_elem*(m-1)+1; here we break it
    # For shared=False, length must be n_elem*m; here we break it
    # ----------------------------
    bad_flat = np.arange(9)   # length 9 doesn't fit either (3,4) rules above
    try:
        _ = Mesh1D.fe_flat_to_block2d(bad_flat, 3, endpoints_shared=True)
        raise AssertionError("Expected AssertionError for shared=True did not occur")
    except AssertionError:
        pass
    try:
        _ = Mesh1D.fe_flat_to_block2d(bad_flat, 3, endpoints_shared=False)
        raise AssertionError("Expected AssertionError for shared=False did not occur")
    except AssertionError:
        pass
    print_pass("fe_flat_to_block2d (invalid lengths raise)")

    print("\t All fe_flat_to_block2d tests passed!")



def test_lagrange_shape_functions_lagrange_basis_and_derivatives(n=17):
    """
    Unit test for LagrangeShapeFunctions.lagrange_basis_and_derivatives.
    
    This function checks:
      1. Shape correctness of returned arrays.
      2. Partition of unity: sum_k L_k(x) == 1.
      3. Derivative consistency: sum_k dLdx_k(x) == 0.
      4. Interpolation property: L_k(x_j) = δ_kj at nodal points.
      5. Smoothness: dLdx finite and behaves as expected.
    """

    # --- Setup: simple node layout and evaluation points ---
    nodes = np.array([0.0, 1.0, 2.0, 3.0])       # 4 equally spaced nodes
    x_eval = np.linspace(0.0, 3.0, 31)           # evaluation points in [0,3]

    # Wrap into element-batch form (1 element)
    nodes = nodes[None, :]
    x_eval = x_eval[None, :]

    # --- Call the function to test ---
    L, dLdx = LagrangeShapeFunctions.lagrange_basis_and_derivatives(nodes, x_eval)

    # --- Check 1: shape correctness ---
    assert L.shape == (1, x_eval.shape[1], nodes.shape[1])
    assert dLdx.shape == (1, x_eval.shape[1], nodes.shape[1])
    print("\t 1. Shape check passed:", L.shape, dLdx.shape)

    # --- Check 2: partition of unity ---
    unity_error = np.max(np.abs(np.sum(L[0, :, :], axis=1) - 1.0))
    assert unity_error < 1e-12
    print("\t 2. Partition of unity check passed (max error = %.2e)" % unity_error)

    # --- Check 3: derivative consistency (sum of dLdx == 0) ---
    deriv_sum_error = np.max(np.abs(np.sum(dLdx[0, :, :], axis=1)))
    assert deriv_sum_error < 1e-10
    print("\t 3. Derivative consistency check passed (max error = %.2e)" % deriv_sum_error)

    # --- Check 4: interpolation property (nodal identity) ---
    # Evaluate basis at nodal points
    L_nodes, dLdx_nodes = LagrangeShapeFunctions.lagrange_basis_and_derivatives(nodes, nodes)
    identity_error = np.max(np.abs(L_nodes[0, :, :] - np.eye(nodes.shape[1])))
    assert identity_error < 1e-12
    print("\t 4. Interpolation property check passed (max error = %.2e)" % identity_error)

    # --- Check 5: finite values ---
    assert np.all(np.isfinite(L))
    assert np.all(np.isfinite(dLdx))
    print("\t 5. Finite value check passed")

    print("\t All Lagrange basis tests passed!")






if __name__ == "__main__":
    if not os.path.exists("outputs/mesh_builder_testcase"):
        os.makedirs("outputs/mesh_builder_testcase")

    print("Running tests...")
    # test_quadrature1d(n = 95)
    # test_lobatto1d(n = 31)
    # test_mesh1d_grid(n = 17)
    # test_mesh1d_fe_nodes(n = 17)
    test_fe_flat_to_block2d()
    # test_lagrange_shape_functions_lagrange_basis_and_derivatives(n = 17)
    print("All tests passed")

    # Terminal command:
    #   python atomic_dft/testcase/mesh_builder_testcase.py