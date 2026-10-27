#ifndef YUKAWA_H
#define YUKAWA_H

#include "isddft.h"

// Struct for a 3D vector
typedef struct _Vector3D {
    double x, y, z;
} Vector3D;

// Struct to store energy, force, stress for each structure
typedef struct _DFT_FIT_OBJ {
    double energy;
    double *atom_pos;
    double *forces;
    double stress[6];
} DFT_FIT_OBJ;

// Struct to cache geometry (r and r_vecs) for all training structures
// This prevents re-calculating MIC and neighbors inside the optimization loop
typedef struct _GeomCache {
    int n_pairs;        // Total pairs including images within cutoff
    double *r_vals;     // Distances [n_pairs]
    double *r_vecs;     // Vectors [3 * n_pairs] (flat: x, y, z, x, y, z...)
    double *weights;    // 0.5 for self-images, 1.0 otherwise [n_pairs]
    int *idx_i;         // Atom index i [n_pairs]
    int *idx_j;         // Atom index j [n_pairs]
    double vol;         // Volume of the cell
    int Nat;            // Number of atoms in this snapshot
} GeomCache;

/**
Function to solve Yukawa and broadcast the solution
*/
void solve_Yukawa_broadcast(SPARC_OBJ *pSPARC);

/**
Function to calculate the displacement vector with PBC and MIC
For now only orthogonal cells implemented
Input:  pos_a, pos_b as 3D Vectors
        range_x, range_y, range_z are CELL dimensions in x, y, and z directions
Output: r_ij = pos_a - pos_b as 3D Vector     
*/
Vector3D get_displacement(Vector3D pos_a, Vector3D pos_b, 
                        double range_x, double range_y, double range_z);

/**
 * Function to calculate Yukawa energy, force, and stress (virial)
 */
void calculate_Yukawa_energy_force_stress(SPARC_OBJ *pSPARC, double *energy, double *forces, double *stress);

/**
 * Optimizes the yukawa parameters on rank 0 and broadcasts the parameters
 */
void optimize_Yukawa_broadcast(SPARC_OBJ *pSPARC, int n_structs);

/**
 * Main Optimization routine
 * 1. Pre-computes geometry for all reference structures.
 * 2. Uses Golden Section Search to find optimal mu.
 * 3. Solves for A and C at optimal mu.
 * 4. Updates pSPARC with new parameters.
 */
void optimize_yukawa_parameters(SPARC_OBJ *pSPARC, int n_structs);

/**
 * Objective Function: J(mu)
 * Calculates the best A and C for a given mu via Linear Least Squares
 * Returns the sum of squared residuals.
 */
double projected_objective(double mu, SPARC_OBJ *pSPARC, GeomCache *geom, int n_structs, double *opt_A, double *opt_C);

/**
 * Pre-computes neighbor lists (images) for all structures.
 */
void precompute_geometry_optimization(SPARC_OBJ *pSPARC, GeomCache *geom, int n_structs);

/**
 * Free geometric info cache
 */
void free_geom_cache(GeomCache *geom, int n_structs);

/**
 * Printing for debug
 */
void print_output_yukawa(SPARC_OBJ *pSPARC);

/**
 * @brief Symmetrize any force array so components sum to zero.
 */
void Symmetrize_force_array(double *forces, int n_atom);

#endif