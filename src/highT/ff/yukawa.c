#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#ifdef USE_MKL
    #define MKL_Complex16 double _Complex
    #include <mkl.h>
#else
    #include <cblas.h>
    #include <lapacke.h>
#endif

#include "isddft.h"
#include "yukawa.h"

#define RCUT_YUK 15.0 
#define r_sq_tol 1e-12
// optimization weights
#define W_ENERGY 1.0
#define W_FORCE  1.0
#define W_STRESS 1.0
#define MU_MIN   0.05
#define MU_MAX   10.0
#define TOL_X    1e-8

/**
Function to solve Yukawa and broadcast the solution
*/
void solve_Yukawa_broadcast(SPARC_OBJ *pSPARC) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // Initialize if not 
    if (pSPARC->F_base_model == NULL) {
        pSPARC->F_base_model = (double *)malloc(3 * pSPARC->n_atom * sizeof(double));
        for (int k = 0; k < 6; k++) pSPARC->stress_base_model[k] = 0.0;
        pSPARC->pres_base_model = 0.0;
    }
    int buffsize = 1 + 3*pSPARC->n_atom + 6 + 1; // energy, forces, stress, pressure
    double *tempbuff = (double *)malloc(buffsize*sizeof(double));

    int count,i;

#ifdef DEBUG
    double t1, t2;
#endif

    // Solve Yukawa on rank 0 and then broadcast solution
    if (rank == 0){
#ifdef DEBUG
    t1 = MPI_Wtime();
#endif
        // Energy and force
        calculate_Yukawa_energy_force_stress(pSPARC, &pSPARC->E_base_model, pSPARC->F_base_model, pSPARC->stress_base_model);

        // Symmetrize forces to make it drift free
        Symmetrize_force_array(pSPARC->F_base_model, pSPARC->n_atom);

        // Pressure
        pSPARC->pres_base_model = -1.0*(pSPARC->stress_base_model[0] + pSPARC->stress_base_model[3] + pSPARC->stress_base_model[5])/3.0;

#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("Solving for Yukawa took %0.3f ms\n",(t2-t1)*1000);
#endif

        // Pack tempbuff to broadcast once
        tempbuff[0] = pSPARC->E_base_model; // energy
        count = 1;

        // force
        for(i = 0; i < pSPARC->n_atom; i++) {
            tempbuff[count++] = pSPARC->F_base_model[3*i];
            tempbuff[count++] = pSPARC->F_base_model[3*i+1];
            tempbuff[count++] = pSPARC->F_base_model[3*i+2];
        }

        // stress
        for (i = 0; i < 6; i++) {
            tempbuff[count++] = pSPARC->stress_base_model[i];
        }

        // pressure
        tempbuff[count++] = pSPARC->pres_base_model;
#ifdef DEBUG
    t1 = MPI_Wtime();
#endif
        // Broadcast tempbuff
        MPI_Bcast(tempbuff, buffsize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("Broadcasting Yukawa solution took %0.3f ms\n",(t2-t1)*1000);
#endif

    } else {
        // Broadcast tempbuff
        MPI_Bcast(tempbuff, buffsize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Unpack tempbuff
        pSPARC->E_base_model = tempbuff[0]; // Energy
        count = 1;

        // force
        for(i = 0; i < pSPARC->n_atom; i++) {
            pSPARC->F_base_model[3*i] = tempbuff[count++];
            pSPARC->F_base_model[3*i+1] = tempbuff[count++];
            pSPARC->F_base_model[3*i+2] = tempbuff[count++];
        }

        // stress
        for (i = 0; i < 6; i++) {
            pSPARC->stress_base_model[i] = tempbuff[count++];
        }

        // pressure
        pSPARC->pres_base_model = tempbuff[count++];
    }

    // deallocate memory
    free(tempbuff);
}

/**
Function to calculate the displacement vector with PBC and MIC
For now only orthogonal cells implemented
Input:  pos_a, pos_b as 3D Vectors
        range_x, range_y, range_z are CELL dimensions in x, y, and z directions
Output: r_ij = pos_a - pos_b as 3D Vector     
*/
Vector3D get_displacement(Vector3D pos_a, Vector3D pos_b, 
                        double range_x, double range_y, double range_z) {
    Vector3D r_ij;
    r_ij.x = pos_a.x - pos_b.x;
    r_ij.y = pos_a.y - pos_b.y;
    r_ij.z = pos_a.z - pos_b.z;

    // Minimum Image Convention
    // round helps find the nearest periodic image
    r_ij.x -= range_x * round(r_ij.x / range_x);
    r_ij.y -= range_y * round(r_ij.y / range_y);
    r_ij.z -= range_z * round(r_ij.z / range_z);

    return r_ij;
}

/**
 * @brief   Index of the species pair (ti, tj) in A_yukawa: pairs ordered 11 12 .. 1n 22 .. nn
 */
int Yukawa_pair_index(int ti, int tj, int Ntypes) {
    if (ti > tj) { int t = ti; ti = tj; tj = t; }
    return ti * Ntypes - ti * (ti - 1) / 2 + tj - ti;
}

/**
 * @brief   Species (type index) of every atom; the atoms are stored type by type
 */
static int *atom_species(SPARC_OBJ *pSPARC) {
    int *typ = (int *)malloc(pSPARC->n_atom * sizeof(int));
    int i = 0;
    for (int t = 0; t < pSPARC->Ntypes; t++)
        for (int k = 0; k < pSPARC->nAtomv[t]; k++) typ[i++] = t;
    return typ;
}

/**
Function to calculate Yukawa energy, force, and stress (virial)
For a given position in SPARC_OBJ: 
r_ij = r_j - r_i, r = sqrt(|r_ij|^2), A_ij = amplitude of the species pair of atoms i and j
E =  sum_{i<j} A_ij*exp(-mu*r)/r
|F_ij(r)| = A_ij*exp(-mu*r) * (1 + mu*r) / r^2
Force on atom i: F_ij = -(force_magnitude / r_ij) * r_ij
Force on atom j: -F_ij (Newton's 3rd Law)
Stress (Virial): sigma_ab = - (1/Vol) * sum_pairs [ f_a * r_b ]
*/
void calculate_Yukawa_energy_force_stress(SPARC_OBJ *pSPARC, double *energy, double *forces, double *stress){
    int n_atm = pSPARC->n_atom;
    int i,j, nx, ny, nz;
    Vector3D pos_i, pos_j, r_ij, r_vec, F_ij;
    double r, r_sq, force_magnitude;
    double mu_yukawa = pSPARC->mu_yukawa;
    double r_cut_sq = RCUT_YUK * RCUT_YUK;

    // energy, forces and stress of each species pair p with A = 1 (multiplied by A_yukawa[p] at the end)
    int n_pair = pSPARC->n_A_yukawa;
    int *typ = atom_species(pSPARC);
    double *energy_p = (double *)calloc(n_pair, sizeof(double));
    double *forces_p = (double *)calloc(n_pair * 3 * n_atm, sizeof(double));
    double *stress_p = (double *)calloc(n_pair * 6, sizeof(double));

    // Initialise energy, force, and stress to 0
    *energy = 0.0;
    memset(forces, 0, 3 * pSPARC->n_atom * sizeof(double));
    for(i=0; i<6; i++) stress[i] = 0.0;

    // Store box dimentions (orthogonal for now)
    double box_x, box_y, box_z;
    if (pSPARC->Flag_latvec_scale == 0){
        box_x = pSPARC->range_x; box_y = pSPARC->range_y; box_z = pSPARC->range_z;
    } else {
        box_x = pSPARC->latvec_scale_x*pSPARC->LatVec[0];
        box_y = pSPARC->latvec_scale_y*pSPARC->LatVec[4];
        box_z = pSPARC->latvec_scale_z*pSPARC->LatVec[8];
    }

    double volume = box_x * box_y * box_z;

    // Determine number of shells to check
    int nx_max = (int)ceil(RCUT_YUK / box_x);
    int ny_max = (int)ceil(RCUT_YUK / box_y);
    int nz_max = (int)ceil(RCUT_YUK / box_z);

    for (i = 0; i < n_atm; i++){
        pos_i.x = pSPARC->atom_pos[3*i]; 
        pos_i.y = pSPARC->atom_pos[3*i+1];
        pos_i.z = pSPARC->atom_pos[3*i+2];

        // Loop j starting at i to capture self-images (and avoid double counting pairs)
        for (j = i; j < n_atm; j++){
            pos_j.x = pSPARC->atom_pos[3*j]; 
            pos_j.y = pSPARC->atom_pos[3*j+1];
            pos_j.z = pSPARC->atom_pos[3*j+2];

            // Get Base MIC displacement
            r_ij = get_displacement(pos_i, pos_j, box_x, box_y, box_z);

            int p = Yukawa_pair_index(typ[i], typ[j], pSPARC->Ntypes);
            double *f = forces_p + 3 * n_atm * p, *sig = stress_p + 6 * p;

            // Loop over periodic images
            for (nx = -nx_max; nx <= nx_max; nx++) {
                for (ny = -ny_max; ny <= ny_max; ny++) {
                    for (nz = -nz_max; nz <= nz_max; nz++) {
                        
                        // SKIP pure self-interaction (Atom i with itself in central box)
                        if (i == j && nx == 0 && ny == 0 && nz == 0) continue;

                        r_vec.x = r_ij.x + nx * box_x;
                        r_vec.y = r_ij.y + ny * box_y;
                        r_vec.z = r_ij.z + nz * box_z;

                        r_sq = r_vec.x * r_vec.x + r_vec.y * r_vec.y + r_vec.z * r_vec.z;

                        // Check Cutoff
                        if (r_sq > r_cut_sq) continue;

                        // Avoid division by zero
                        if (r_sq < r_sq_tol) r_sq = r_sq_tol;

                        r = sqrt(r_sq);
                        double exp_term = exp(-mu_yukawa * r);

                        // --- Energy Calculation ---
                        // Calculate Yukawa energy term
                        double term_energy = exp_term / r;

                        // Apply 0.5 factor for self-image interactions to avoid double counting
                        // (Because loop visits +L and -L, which are the same bond)
                        if (i == j) {
                            term_energy *= 0.5;
                        }

                        energy_p[p] += term_energy;

                        // --- Force Calculation ---
                        // Calculate Yukawa force magnitude (before A multiplication): 
                        // F(r)/A = exp(-mu*r) * (1 + mu*r) / r^2
                        force_magnitude = (exp_term / r_sq) * (1.0 + mu_yukawa * r);

                        // Calculate force vector
                        // F_vec = (F_mag / r) * r_vec
                        F_ij.x = force_magnitude * r_vec.x / r;
                        F_ij.y = force_magnitude * r_vec.y / r;
                        F_ij.z = force_magnitude * r_vec.z / r;

                        // Note: multiplied by A later
                        f[3*i] += F_ij.x;
                        f[3*i+1] += F_ij.y;
                        f[3*i+2] += F_ij.z;

                        // Newton's 3rd Law only if i != j. 
                        // If i == j, the reaction force is on the ghost image (ignored).
                        if (i != j) {
                            f[3*j] -= F_ij.x;
                            f[3*j+1] -= F_ij.y;
                            f[3*j+2] -= F_ij.z;
                        }

                        // --- Stress Calculation ---
                        /** This is the scalar S_ij * r_ij.x * r_ij.x
                        force_magnitude is (A/(r^2))*(1 + mu*r)*exp(-mu*r) (Here A is factored out)
                        vir_scalar = force_magnitude / r */
                        
                        double virial_scalar = force_magnitude / r; // Effectively (1/r^3)*(1+mu*r)*exp

                        // Multiply self-image interactions by 0.5
                        if (i == j) {
                            virial_scalar *= 0.5;
                        }

                        // NOTE: the stress here accumulates values assuming A=1. 
                        // We will multiply by A at the end.
                        
                        /** Add the pair contribution to each component of tensor
                        Contribution: r_ij,alpha * F_ij,beta = (r_ij,alpha * r_ij,beta) * (virial_scalar)
                        Note: F_ij,beta = virial_scalar * r_ij,beta */
                        // Subtract the contribution because F and r are in opposite directions
                        // Repulsive force F ~ -r. So r*F ~ -r*r.
                        
                        sig[0] -= virial_scalar * r_vec.x * r_vec.x;
                        sig[1] -= virial_scalar * r_vec.x * r_vec.y;
                        sig[2] -= virial_scalar * r_vec.x * r_vec.z;
                        sig[3] -= virial_scalar * r_vec.y * r_vec.y;
                        sig[4] -= virial_scalar * r_vec.y * r_vec.z;
                        sig[5] -= virial_scalar * r_vec.z * r_vec.z;
                    }
                }
            }
        }
    }

    // Finalize: multiply each species pair by its A (stress also divided by volume)
    for (int p = 0; p < n_pair; p++) {
        double A = pSPARC->A_yukawa[p];
        *energy += A * energy_p[p];
        for (i = 0; i < 3 * n_atm; i++) forces[i] += A * forces_p[3 * n_atm * p + i];
        double stress_pref = A / volume;
        for (i = 0; i < 6; i++) stress[i] += stress_p[6 * p + i] * stress_pref;
    }
    *energy += pSPARC->Econst_yukawa * pSPARC->n_atom; // add Yukawa energy constant shift

    free(typ); free(energy_p); free(forces_p); free(stress_p);
}

/**
 * Optimizes the yukawa parameters on rank 0 and broadcasts the parameters
 */
void optimize_Yukawa_broadcast(SPARC_OBJ *pSPARC, int n_structs) {
    int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int n_pair = pSPARC->n_A_yukawa;
    int buffsize = n_pair + 2; // A of each species pair, mu, E_const
    double *tempbuff = (double *)malloc(buffsize*sizeof(double));

// #ifdef DEBUG
    double t1, t2;
// #endif

    // Solve Yukawa on rank 0 and then broadcast solution
    if (rank == 0){
// #ifdef DEBUG
    t1 = MPI_Wtime();
// #endif

        // Call optimization routine
        optimize_yukawa_parameters(pSPARC, n_structs);

// #ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("Optimizing Yukawa parameters took %0.3f ms\n",(t2-t1)*1000);
// #endif

        // Pack tempbuff to broadcast once
        for (int p = 0; p < n_pair; p++) tempbuff[p] = pSPARC->A_yukawa[p];
        tempbuff[n_pair] = pSPARC->mu_yukawa;
        tempbuff[n_pair+1] = pSPARC->Econst_yukawa;

// #ifdef DEBUG
    t1 = MPI_Wtime();
// #endif
        // Broadcast tempbuff
        MPI_Bcast(tempbuff, buffsize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

// #ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("Broadcasting Yukawa parameters took %0.3f ms\n",(t2-t1)*1000);
// #endif

    } else {
        // Broadcast tempbuff
        MPI_Bcast(tempbuff, buffsize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Unpack tempbuff
        for (int p = 0; p < n_pair; p++) pSPARC->A_yukawa[p] = tempbuff[p];
        pSPARC->mu_yukawa = tempbuff[n_pair];
        pSPARC->Econst_yukawa = tempbuff[n_pair+1];
    }

    // deallocate memory
    free(tempbuff);
}

/**
 * Main Optimization routine
 * 1. Pre-computes geometry for all reference structures.
 * 2. Uses Golden Section Search to find optimal mu.
 * 3. Solves for A (one per species pair) and C at optimal mu.
 * 4. Updates pSPARC with new parameters.
 */
void optimize_yukawa_parameters(SPARC_OBJ *pSPARC, int n_structs) {
    printf("Starting Yukawa Optimization ...\n");

    // Pre-compute Geometry (Neighbor lists, distances)
    GeomCache *geom = (GeomCache *)malloc(n_structs * sizeof(GeomCache));
    precompute_geometry_optimization(pSPARC, geom, n_structs);

    // Least-squares column of each species pair; a pair with no atoms within RCUT_YUK in the
    // fitting structures cannot be fitted and keeps A = 0
    int n_pair = pSPARC->n_A_yukawa, n_fit = 0;
    int *fit_col = (int *)malloc(n_pair * sizeof(int));
    for (int p = 0; p < n_pair; p++) fit_col[p] = -1;
    for (int s = 0; s < n_structs; s++)
        for (int k = 0; k < geom[s].n_pairs; k++) fit_col[geom[s].pair_type[k]] = 0;
    for (int p = 0; p < n_pair; p++)
        if (fit_col[p] == 0) fit_col[p] = n_fit++;

    // Golden section search for mu
    double a = MU_MIN;
    double b = MU_MAX;
    double gr = (sqrt(5.0) - 1.0) / 2.0; // Golden ratio
    double c = b - gr * (b - a);
    double d = a + gr * (b - a);

    double *A_opt = (double *)malloc(n_pair * sizeof(double)), C_opt = 0.0; // also placeholders for the search
    int rank; // rank of the least-squares system
    double fc = projected_objective(c, pSPARC, geom, n_structs, fit_col, n_fit, A_opt, &C_opt, &rank);
    double fd = projected_objective(d, pSPARC, geom, n_structs, fit_col, n_fit, A_opt, &C_opt, &rank);

    while (fabs(b - a) > TOL_X) {
        if (fc < fd) {
            b = d;
            d = c;
            fd = fc;
            c = b - gr * (b - a);
            fc = projected_objective(c, pSPARC, geom, n_structs, fit_col, n_fit, A_opt, &C_opt, &rank);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + gr * (b - a);
            fd = projected_objective(d, pSPARC, geom, n_structs, fit_col, n_fit, A_opt, &C_opt, &rank);
        }
    }

    double mu_opt = (a + b) / 2.0;

    // Final Solve for A and C using optimal mu
    double final_resid = projected_objective(mu_opt, pSPARC, geom, n_structs, fit_col, n_fit, A_opt, &C_opt, &rank);

    // Update SPARC Object
    pSPARC->mu_yukawa = mu_opt;
    for (int p = 0; p < n_pair; p++) pSPARC->A_yukawa[p] = A_opt[p];
    pSPARC->Econst_yukawa = C_opt / pSPARC->n_atom;

    printf("\n--- Optimization Results ---\n");
    printf("Residual: %g\n", final_resid);
    printf("Mu (bohr^{-1}):       %0.14f\n", pSPARC->mu_yukawa);
    if (rank < n_fit + 1)
        printf("Amplitudes not independent in the fitting structures (rank %d of %d): minimum-norm solution\n",
               rank, n_fit + 1);
    for (int ti = 0; ti < pSPARC->Ntypes; ti++) {
        for (int tj = ti; tj < pSPARC->Ntypes; tj++) {
            int p = Yukawa_pair_index(ti, tj, pSPARC->Ntypes);
            printf("A %s-%s (ha-bohr):  %0.14f%s\n", &pSPARC->atomType[L_ATMTYPE*ti], &pSPARC->atomType[L_ATMTYPE*tj],
                   pSPARC->A_yukawa[p], fit_col[p] < 0 ? "  (no pairs within the cutoff: not fitted)" : "");
        }
    }
    printf("C/atom (ha/atom):   %0.14f\n", pSPARC->Econst_yukawa);
    printf("----------------------------\n");

    // Free memory
    free_geom_cache(geom, n_structs);
    free(geom);
    free(fit_col);
    free(A_opt);
}

/**
 * Objective Function: J(mu)
 * Calculates the best A (one per species pair) and C for a given mu via Linear Least Squares
 * Returns the sum of squared residuals.
 */
double projected_objective(double mu, SPARC_OBJ *pSPARC, GeomCache *geom, int n_structs, const int *fit_col, int n_fit,
                           double *opt_A, double *opt_C, int *rank) {
    // Solve the linear system: M * theta = V
    // theta = [A of each fitted species pair (column fit_col[p]), C]^T
    
    // Calculate Total Matrix Size (Rows)
    // Rows = \Sum_{structures} (1 [Energy] + 3*Nat [Force] + 6 [Stress, only if stress was computed])
    int n_stress_rows = (pSPARC->Calc_stress == 1) ? 6 : 0;
    int total_rows = 0;
    for (int s = 0; s < n_structs; s++) {
        if (geom[s].Nat > 0) total_rows += 1 + 3 * geom[s].Nat + n_stress_rows;
    }

    int n_pair = pSPARC->n_A_yukawa;
    int n_params = n_fit + 1; // A of the fitted species pairs, and C

    // A_mat is [Cols 0..n_fit-1 (coeff for each A) | Col n_fit (coeff for C)]
    double *A_mat = (double *)calloc(total_rows * n_params, sizeof(double));
    double *B_vec = (double *)calloc(total_rows, sizeof(double));

    // Variables needed
    int Nat, i, j, p, row_idx = 0;
    double *phi_E, *phi_F, *phi_S, r, rx, ry, rz, w, exp_term, 
            r_inv, r2_inv, r3_inv, scalar_F, fx, fy, fz, s_term, w_E_node, w_F_node, w_S_node;

    DFT_FIT_OBJ *dft; GeomCache *g;

    // Construct the Linear System
    for (int s = 0; s < n_structs; s++) {
        dft = &pSPARC->Ref_struct_DFT[s];
        g = &geom[s];

        Nat = g->Nat;
        if (Nat == 0) continue;

        // Calculate Basis for this structure, for each species pair p
        phi_E = (double *)calloc(n_pair, sizeof(double));
        phi_F = (double *)calloc(n_pair*3*Nat, sizeof(double));
        phi_S = (double *)calloc(n_pair*6, sizeof(double));

        for (int k = 0; k < g->n_pairs; k++) {
            r = g->r_vals[k]; 
            rx = g->r_vecs[3*k]; ry = g->r_vecs[3*k+1]; rz = g->r_vecs[3*k+2];
            w = g->weights[k]; i = g->idx_i[k]; j = g->idx_j[k]; p = g->pair_type[k];
            double *pF = phi_F + p*3*Nat, *pS = phi_S + p*6;

            exp_term = exp(-mu*r);
            r_inv = 1.0/r; r2_inv = r_inv*r_inv; r3_inv = r2_inv * r_inv;

            // Energy basis: sum (w * exp(-mu*r)/r)
            phi_E[p] += w * exp_term * r_inv;

            // Force basis
            scalar_F = (1.0 + mu * r) * exp_term * r3_inv;

            fx = scalar_F * rx; fy = scalar_F * ry; fz = scalar_F * rz;

            // Accumulate forces
            pF[3*i] += fx; pF[3*i+1] += fy; pF[3*i+2] += fz;

            // Newton's 3rd law
            if (i != j) {
                pF[3*j]   -= fx;
                pF[3*j+1] -= fy;
                pF[3*j+2] -= fz;
            }

            // Accumulate Stress
            s_term = w * scalar_F;
            pS[0] -= s_term * rx * rx; 
            pS[1] -= s_term * rx * ry; 
            pS[2] -= s_term * rx * rz; 
            pS[3] -= s_term * ry * ry; 
            pS[4] -= s_term * ry * rz; 
            pS[5] -= s_term * rz * rz;
        }

        // Normalize Stress by Volume
        for(int k=0; k<6*n_pair; k++) phi_S[k] /= g->vol;

        // Weights:
        w_E_node = sqrt(W_ENERGY / (double)Nat);
        w_F_node = sqrt(W_FORCE / (3.0 * Nat));
        w_S_node = sqrt(W_STRESS);

        // NOTE: LAPACK is Column-Major!
        // A_mat[row + col * total_rows]

        // Energy Row
        // sum_p A_p*phi_E[p] + C*1 = E_dft
        for (p = 0; p < n_pair; p++)
            if (fit_col[p] >= 0) A_mat[row_idx + fit_col[p] * total_rows] = phi_E[p] * w_E_node; // Col of A_p
        A_mat[row_idx + n_fit * total_rows] = 1.0 * w_E_node;   // Col n_fit (C)
        B_vec[row_idx] = dft->energy * w_E_node;                 // RHS
        row_idx++;

        // Force Rows (C column stays 0)
        // sum_p A_p*phi_F[p] = F_dft
        for(int k=0; k<3*Nat; k++) {
            for (p = 0; p < n_pair; p++)
                if (fit_col[p] >= 0) A_mat[row_idx + fit_col[p] * total_rows] = phi_F[p*3*Nat + k] * w_F_node;
            B_vec[row_idx] = dft->forces[k] * w_F_node;
            row_idx++;
        }

        // Stress Rows (skipped when the DFT stress was not computed; C column stays 0)
        // sum_p A_p*phi_S[p] = S_dft
        for(int k=0; k<n_stress_rows; k++) {
            for (p = 0; p < n_pair; p++)
                if (fit_col[p] >= 0) A_mat[row_idx + fit_col[p] * total_rows] = phi_S[p*6 + k] * w_S_node;
            B_vec[row_idx] = dft->stress[k] * w_S_node;
            row_idx++;
        }

        free(phi_E); free(phi_F); free(phi_S);
    }

    // Solve using LAPACK dgels: Minimizes || A * X - B ||_2
    // Keep the system and the column norms for the rank check below
    double *A_copy = (double *)malloc(total_rows * n_params * sizeof(double));
    double *B_copy = (double *)malloc(total_rows * sizeof(double));
    double *col_norm = (double *)malloc(n_params * sizeof(double));
    memcpy(A_copy, A_mat, total_rows * n_params * sizeof(double));
    memcpy(B_copy, B_vec, total_rows * sizeof(double));
    for (int k = 0; k < n_params; k++) col_norm[k] = cblas_dnrm2(total_rows, A_mat + k * total_rows, 1);

    int info = LAPACKE_dgels(LAPACK_COL_MAJOR, 'N', total_rows, n_params, 1,
                             A_mat, total_rows, B_vec, total_rows);

    // dgels assumes independent columns. Column k depends on columns 0..k-1 if R_kk (QR factor, now in A_mat) is ~0
    // relative to the column: e.g. two species pairs seen only at the same distances, such as the periodic images of
    // two single atoms. Then take the minimum-norm solution (SVD, dgelsd), which shares the amplitude equally.
    *rank = n_params;
    for (int k = 0; k < n_params && info == 0; k++)
        if (fabs(A_mat[k + k * total_rows]) <= 1e-10 * col_norm[k]) *rank = -1;
    if (*rank < 0) {
        double *sing_val = (double *)malloc(n_params * sizeof(double));
        memcpy(A_mat, A_copy, total_rows * n_params * sizeof(double));
        memcpy(B_vec, B_copy, total_rows * sizeof(double));
        info = LAPACKE_dgelsd(LAPACK_COL_MAJOR, total_rows, n_params, 1, A_mat, total_rows, B_vec, total_rows,
                              sing_val, 1e-10, rank);
        free(sing_val);
    }

    if (info != 0) {
        printf("Error: LAPACK least squares failed with info %d\n", info);
        free(A_mat); free(B_vec); free(A_copy); free(B_copy); free(col_norm);
        for (p = 0; p < n_pair; p++) opt_A[p] = 0;
        *opt_C = 0;
        return 1e20;
    }

    // Extract Results
    // The solution (Theta) is stored in the first N elements of B_vec
    for (p = 0; p < n_pair; p++) opt_A[p] = (fit_col[p] >= 0) ? B_vec[fit_col[p]] : 0.0;
    *opt_C = B_vec[n_fit];

    // Calculate Residual J = || b - Ax ||_2^2
    double residual_sum_sq = 0.0;
    if (*rank == n_params) {
        // dgels stores the residual components in B_vec[N...M]
        for (int i = n_params; i < total_rows; i++) {
            residual_sum_sq += B_vec[i] * B_vec[i];
        }
    } else {
        for (int i = 0; i < total_rows; i++) {
            double res = -B_copy[i];
            for (int k = 0; k < n_params; k++) res += A_copy[i + k * total_rows] * B_vec[k];
            residual_sum_sq += res * res;
        }
    }

    free(A_mat);
    free(B_vec);
    free(A_copy); free(B_copy); free(col_norm);
    
    return residual_sum_sq;
}

/**
 * Pre-computes neighbor lists (images) for all structures.
 */
void precompute_geometry_optimization(SPARC_OBJ *pSPARC, GeomCache *geom, int n_structs) {
    double box_x, box_y, box_z;
    if (pSPARC->Flag_latvec_scale == 0){
        box_x = pSPARC->range_x; box_y = pSPARC->range_y; box_z = pSPARC->range_z;
    } else {
        box_x = pSPARC->latvec_scale_x*pSPARC->LatVec[0];
        box_y = pSPARC->latvec_scale_y*pSPARC->LatVec[4];
        box_z = pSPARC->latvec_scale_z*pSPARC->LatVec[8];
    }
    
    double r_cut_sq = RCUT_YUK * RCUT_YUK;
    int nx_max = (int)ceil(RCUT_YUK / box_x);
    int ny_max = (int)ceil(RCUT_YUK / box_y);
    int nz_max = (int)ceil(RCUT_YUK / box_z);
    int *typ = atom_species(pSPARC);

    for (int s = 0; s < n_structs; s++) {
        DFT_FIT_OBJ *dft = &pSPARC->Ref_struct_DFT[s];

        int Nat = pSPARC->n_atom; // Assuming n_atom is constant
        geom[s].Nat = Nat;
        geom[s].vol = box_x * box_y * box_z;

        // Temporary dynamic arrays to store pairs before we know exact count
        // Max pairs is hard to predict, so we use a resizable approach or conservative guess.
        // For simplicity let's alloc a large buffer and realloc down.
        int cap = Nat * 1000; 
        int count = 0;

        double *temp_r = (double *)malloc(cap * sizeof(double));
        double *temp_vec = (double *)malloc(cap * 3 * sizeof(double));
        double *temp_w = (double *)malloc(cap * sizeof(double));
        int *temp_i = (int *)malloc(cap * sizeof(int));
        int *temp_j = (int *)malloc(cap * sizeof(int));
        int *temp_p = (int *)malloc(cap * sizeof(int));

        for (int i = 0; i < Nat; i++) {
            Vector3D pos_i;
            pos_i.x = dft->atom_pos[3*i];
            pos_i.y = dft->atom_pos[3*i+1];
            pos_i.z = dft->atom_pos[3*i+2];

            for (int j = i; j < Nat; j++) {
                Vector3D pos_j;
                pos_j.x = dft->atom_pos[3*j];
                pos_j.y = dft->atom_pos[3*j+1];
                pos_j.z = dft->atom_pos[3*j+2];

                Vector3D r_base = get_displacement(pos_i, pos_j, box_x, box_y, box_z);

                for (int nx = -nx_max; nx <= nx_max; nx++) {
                    for (int ny = -ny_max; ny <= ny_max; ny++) {
                        for (int nz = -nz_max; nz <= nz_max; nz++) {
                            if (i == j && nx == 0 && ny == 0 && nz == 0) continue;

                            Vector3D r_vec;
                            r_vec.x = r_base.x + nx * box_x;
                            r_vec.y = r_base.y + ny * box_y;
                            r_vec.z = r_base.z + nz * box_z;

                            double r_sq = r_vec.x*r_vec.x + r_vec.y*r_vec.y + r_vec.z*r_vec.z;

                            if (r_sq > r_cut_sq) continue;
                            if (r_sq < r_sq_tol) r_sq = r_sq_tol;

                            // Resize if needed
                            if (count >= cap) {
                                cap *= 2;
                                temp_r = realloc(temp_r, cap * sizeof(double));
                                temp_vec = realloc(temp_vec, cap * 3 * sizeof(double));
                                temp_w = realloc(temp_w, cap * sizeof(double));
                                temp_i = realloc(temp_i, cap * sizeof(int));
                                temp_j = realloc(temp_j, cap * sizeof(int));
                                temp_p = realloc(temp_p, cap * sizeof(int));
                            }

                            temp_r[count] = sqrt(r_sq);
                            temp_vec[3*count] = r_vec.x;
                            temp_vec[3*count+1] = r_vec.y;
                            temp_vec[3*count+2] = r_vec.z;
                            temp_i[count] = i;
                            temp_j[count] = j;
                            temp_p[count] = Yukawa_pair_index(typ[i], typ[j], pSPARC->Ntypes);

                            // Weight: 0.5 for self-images, 1.0 for others
                            if (i == j) temp_w[count] = 0.5;
                            else temp_w[count] = 1.0;

                            count++;
                        }
                    }
                }

            }

        }

        // Save to struct (shrink and fit)
        geom[s].n_pairs = count;
        geom[s].r_vals = (double *)realloc(temp_r, count * sizeof(double));
        geom[s].r_vecs = (double *)realloc(temp_vec, count * 3 * sizeof(double));
        geom[s].weights = (double *)realloc(temp_w, count * sizeof(double));
        geom[s].idx_i = (int *)realloc(temp_i, count * sizeof(int));
        geom[s].idx_j = (int *)realloc(temp_j, count * sizeof(int));
        geom[s].pair_type = (int *)realloc(temp_p, count * sizeof(int));
    }
    free(typ);
}

/**
 * Free geometric info cache
 */
void free_geom_cache(GeomCache *geom, int n_structs) {
    for (int i = 0; i < n_structs; i++) {
        free(geom[i].r_vals);
        free(geom[i].r_vecs);
        free(geom[i].weights);
        free(geom[i].idx_i);
        free(geom[i].idx_j);
        free(geom[i].pair_type);
    }
}

/**
 * Printing for debug
 */
void print_output_yukawa(SPARC_OBJ *pSPARC){

    if(pSPARC->MDFlag == 1) {
        printf("                     (YUKAWA #%d)                     \n",
                pSPARC->MDCount + pSPARC->restartCount + (pSPARC->RestartFlag == 0));
    
    } else if (pSPARC->RelaxFlag >= 1) {
        printf("                     (YUKAWA #%d)                     \n",
                pSPARC->RelaxCount + pSPARC->restartCount + (pSPARC->RestartFlag == 0));
    }

    // Print energy
    printf("Yukawa Energy                      :%18.10f (Ha)\n", pSPARC->E_base_model); 

    // Calculate forces statistics
    int i; 
    double avgF = 0.0, maxF = 0.0, temp;
    for (i = 0; i< pSPARC->n_atom; i++){
        temp = fabs(sqrt(pow(pSPARC->F_base_model[3*i  ],2.0) 
                        + pow(pSPARC->F_base_model[3*i+1],2.0) 
                        + pow(pSPARC->F_base_model[3*i+2],2.0)));
        avgF += temp;
        if (temp > maxF) maxF = temp;	
    }
    avgF /= pSPARC->n_atom;

    printf("RMS Yukawa force                   :%18.10f (Ha/Bohr)\n", avgF);
    printf("Maximum Yukawa force               :%18.10f (Ha/Bohr)\n", maxF);

    // Calculate Stress statistics
    int stress_len = 6; 
    double maxS = 0.0; 
    
    for (i = 0; i < stress_len; i++){
        double temp_s = fabs(pSPARC->stress_base_model[i]); 
        if(temp_s > maxS)
            maxS = temp_s;
    }
    printf("Maximum Yukawa stress              :%18.10f (GPa)\n", maxS * CONST_HA_BOHR3_GPA);

    // Print pressure
    printf("Pressure                           :%18.3f (GPa)\n", CONST_HA_BOHR3_GPA * pSPARC->pres_base_model);

// Print high verbosity only if DEBUG mode is on
#ifdef DEBUG
    // Print forces
    printf("\nAtomic force (ha/bohr): \n");
    for (i = 0; i < pSPARC->n_atom; i++){
        printf("%18.10f      %18.10f      %18.10f\n", pSPARC->F_base_model[3*i],   
                                                      pSPARC->F_base_model[3*i+1],
                                                      pSPARC->F_base_model[3*i+2]);
    }

    // Print stress tensor (ha/bohr^3)
    printf("\nStress (ha/bohr^3): \n");
    printf("%18.10f   %18.10f   %18.10f\n"
           "%18.10f   %18.10f   %18.10f\n"
           "%18.10f   %18.10f   %18.10f\n", 
           pSPARC->stress_base_model[0], pSPARC->stress_base_model[1], pSPARC->stress_base_model[2],
           pSPARC->stress_base_model[1], pSPARC->stress_base_model[3], pSPARC->stress_base_model[4],
           pSPARC->stress_base_model[2], pSPARC->stress_base_model[4], pSPARC->stress_base_model[5]);

    printf("Pressure (ha/bohr^3): %18.10f\n", pSPARC->pres_base_model);

    // Print stress tensor (GPa)
    printf("\nStress (GPa): \n");
    printf("%18.3f   %18.3f   %18.3f\n"
           "%18.3f   %18.3f   %18.3f\n"
           "%18.3f   %18.3f   %18.3f\n", 
           CONST_HA_BOHR3_GPA * pSPARC->stress_base_model[0], CONST_HA_BOHR3_GPA * pSPARC->stress_base_model[1], CONST_HA_BOHR3_GPA * pSPARC->stress_base_model[2],
           CONST_HA_BOHR3_GPA * pSPARC->stress_base_model[1], CONST_HA_BOHR3_GPA * pSPARC->stress_base_model[3], CONST_HA_BOHR3_GPA * pSPARC->stress_base_model[4],
           CONST_HA_BOHR3_GPA * pSPARC->stress_base_model[2], CONST_HA_BOHR3_GPA * pSPARC->stress_base_model[4], CONST_HA_BOHR3_GPA * pSPARC->stress_base_model[5]);

#endif

    printf("====================================================================\n");
}


/**
 * @brief Symmetrize any force array so components sum to zero.
 */
void Symmetrize_force_array(double *forces, int n_atom) {
    double shift[3] = {0.0, 0.0, 0.0};
    int i;

    // Calculate average drift
    for (i = 0; i < n_atom; i++) {
        shift[0] += forces[3*i];
        shift[1] += forces[3*i+1];
        shift[2] += forces[3*i+2];
    }
    shift[0] /= n_atom;
    shift[1] /= n_atom;
    shift[2] /= n_atom;

    // Subtract drift
    for (i = 0; i < n_atom; i++) {
        forces[3*i]   -= shift[0];
        forces[3*i+1] -= shift[1];
        forces[3*i+2] -= shift[2];
    }
}