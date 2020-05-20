#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <mkl.h> 
#include <mpi.h>


#include "isddft.h"

/**
 * @brief   CG method for solving minimum eigenvalues of the Hamiltonian.
 *           
 *          This is the code used in SQDFT, however it's slightly different
 *          from Dr. Pask's implementation. It has some issues when initial
 *          guess is too far away (sometimes doesn't converge)          
 */
void CG_min_eigensolver(const SPARC_OBJ *pSPARC, int *DMVertices, double *eigmin, double TOL, 
                        int MAXIT, double *x0, int kpt, MPI_Comm comm, MPI_Request *req_veff_loc)
{
    if (comm == MPI_COMM_NULL) return;    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int i, DMnd, count;
    double *x, *Ax, *p, *Ap, *g, x_norm, g_norm, lambda, vscal;
        
    DMnd = (1 - DMVertices[0] + DMVertices[1]) * 
           (1 - DMVertices[2] + DMVertices[3]) * 
           (1 - DMVertices[4] + DMVertices[5]);

    x = (double*)malloc(DMnd * sizeof(double));
    Ax = (double*)malloc(DMnd * sizeof(double));
    Ap = (double*)malloc(DMnd * sizeof(double));
    p = (double*)malloc(DMnd * sizeof(double));
    g = (double*)malloc(DMnd * sizeof(double));
    
    // TODO: user have to make sure x0 has unit norm!
    for (i = 0; i < DMnd; i++) x[i] = x0[i];
    
    MPI_Wait(req_veff_loc, MPI_STATUS_IGNORE);
    // find Ax = A * x, where A is the Hamiltonian
    Hamiltonian_vec_mult(pSPARC, DMnd, DMVertices, 1, x, Ax, comm);
    // lambda = <x, Ax>
    VectorDotProduct(x, Ax, DMnd, &lambda, comm);
    // find p = g = Ax - lambda * x
    for (i = 0; i < DMnd; i++) p[i] = g[i] = Ax[i] - lambda * x[i];
    // find g_norm = ||g||
    Vector2Norm(g, DMnd, &g_norm, comm);
//    if (g_norm < TOL) { *eigmin = lambda; return; }
    
    double a, b, c, d, A, B, C, delta, alpha, beta, gAp, pAp;
    count = 1;
    while (g_norm > TOL && count <= MAXIT) {
        // compute Ap
        Hamiltonian_vec_mult(pSPARC, DMnd, DMVertices, 1, p, Ap, comm);
        
        // compute alpha
        VectorDotProduct(x, Ap, DMnd, &a, comm); // a = <x,Ap>
        VectorDotProduct(p, Ap, DMnd, &b, comm); // b = <p,Ap>
        VectorDotProduct(x, p, DMnd, &c, comm);  // c = <x,p>
        VectorDotProduct(p, p, DMnd, &d, comm);  // d = <p,p>
        A = b*c-a*d;
        B = b - lambda * d;
        C = a - lambda * c;
        delta = B*B-4.0*A*C;
        alpha = -(B - sqrt(delta)) / (2.0*A);
        
        // update and normalize x
        for (i = 0; i < DMnd; i++) x[i] += alpha * p[i];
        Vector2Norm(x, DMnd, &x_norm, comm);
        vscal = 1.0 / x_norm;
        for (i = 0; i < DMnd; i++) x[i] *= vscal;
        
        // update lambda
        // find Ax = A * x, where A is the Hamiltonian
        Hamiltonian_vec_mult(pSPARC, DMnd, DMVertices, 1, x, Ax, comm);
        VectorDotProduct(x, Ax, DMnd, &lambda, comm); // lambda = <x, Ax>
        
        // update search direction
        for (i = 0; i < DMnd; i++) g[i] = Ax[i] - lambda * x[i]; // g = Ax - lambda * x
        Vector2Norm(g, DMnd, &g_norm, comm); // g_norm = ||g||
        VectorDotProduct(g, Ap, DMnd, &gAp, comm); // gAp = <g,Ap>
        VectorDotProduct(p, Ap, DMnd, &pAp, comm); // pAp = <p,Ap>
        beta = -gAp / pAp;
        for (i = 0; i < DMnd; i++) p[i] = g[i] + beta * p[i];
        
        if (rank == 0) {
            printf(GRN"    CG(W) iter %d, eigmin  = %.9f, err_eigmin = %.3e\n"RESET,count, lambda, g_norm);
        }
        
        count++;
    }
    
    *eigmin = lambda;
    free(x); free(Ax); free(p); free(Ap); free(g);
}




/**
 * @brief   Calculate minimum eigenvalue of the Hamiltonian using CG method.
 *
 * @reference Dr. John Pask's (Physics Division, Lawrence Livermore National Laboratory) Mathematica test code.
 */
void mininum_eig_solver(const SPARC_OBJ *pSPARC, int *DMVertices, double *Veff_loc, 
                        ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc, NLOC_PROJ_OBJ *nlocProj, 
                        double *eigmin, double TOL, int MAXIT, double *x0, int kpt, 
                        MPI_Comm comm, MPI_Request *req_veff_loc) 
{
    if (comm == MPI_COMM_NULL) return;    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int i, j, DMnd, count;
    double *x, *x_old, *Ax, *g, g_norm, x_n2, x_old_n2, lambda;
    
    DMnd = (1 - DMVertices[0] + DMVertices[1]) * 
           (1 - DMVertices[2] + DMVertices[3]) * 
           (1 - DMVertices[4] + DMVertices[5]);

    x = (double*)malloc(DMnd * sizeof(double));
    x_old = (double*)malloc(DMnd * sizeof(double));
    Ax = (double*)malloc(DMnd * sizeof(double));
    g = (double*)malloc(DMnd * sizeof(double));
    if (x == NULL || x_old == NULL) {
        printf("\nMemory allocation failed!\n");
        exit(EXIT_FAILURE);
    }
    
    // TODO: user have to make sure x0 has unit norm!
    for (i = 0; i < DMnd; i++) x[i] = x0[i];
    
    MPI_Wait(req_veff_loc, MPI_STATUS_IGNORE);
    // find Ax = A * x, where A is the Hamiltonian
    Hamiltonian_vec_mult(pSPARC, DMnd, DMVertices, Veff_loc, Atom_Influence_nloc, 
                         nlocProj, 1, 0.0, x, Ax, comm);
    // lambda = <x, Ax>
    VectorDotProduct(x, Ax, DMnd, &lambda, comm);
    // find g = Ax - lambda * x
    for (i = 0; i < DMnd; i++) g[i] = Ax[i] - lambda * x[i];
    // find g_norm = ||g||
    Vector2Norm(g, DMnd, &g_norm, comm);
    if (g_norm < TOL) { *eigmin = lambda; return; }
    
    double *S, *AS, SAS[9], vscal;
    S  = (double*)malloc(3 * DMnd * sizeof(double));
    AS = (double*)malloc(3 * DMnd * sizeof(double));
    double d[3], e[2], tau[3]; // for LAPACKE eigensolver
    unsigned icol;
    
    // Note: the array S stores  x, g and x_old
    //   |----------------|----------------|----------------| 
    // S |        x       |       g        |      x_old     |
    //   |----------------|----------------|----------------| 
    //   |<--   DMnd   -->|<--   DMnd   -->|<--   DMnd   -->| 
    
    ////////////////////
    double t1, t2, t;
    ////////////////////  
    
    /*************************
     * start first iteration *
     *************************/
    count = 1;
    
    /* S = [x, g] and S = orth(S) */
    // NOTE: since g is orthogonal to x by construction,
    // we just have to normalize g
    for (i = 0; i < DMnd; i++) S[i] = x[i];
    vscal = 1.0 / g_norm;
    for (i = 0; i < DMnd; i++) S[DMnd+i] = g[i] * vscal;
    
    /* find A*S, TODO: CHECK if mult 2 vectors at a time is working! */
    Hamiltonian_vec_mult(pSPARC, DMnd, DMVertices, Veff_loc, Atom_Influence_nloc, 
                         nlocProj, 2, 0.0, S, AS, comm);
    /* find S' * AS */
    t1 = MPI_Wtime();
    //TODO: use cblas_dgemmt instead if available
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 2, 2, DMnd, 
                1.0, S, DMnd, AS, DMnd, 0.0, SAS, 2);
    t2 = MPI_Wtime();
    t = t2 - t1;
    t1 = MPI_Wtime();
    MPI_Allreduce(MPI_IN_PLACE, SAS, 4, MPI_DOUBLE, MPI_SUM, comm);
    t2 = MPI_Wtime();
    
    t1 = MPI_Wtime();
    
    /* find eigvals and eigvecs of SAS */
    // in this case T = SAS, i.e. SAS is already tridiagonal (since it's 2x2)
    d[0] = SAS[0]; d[1] = SAS[3]; // diagonal of T
    e[0] = SAS[2]; // off diagonal of T
    // first reduce SAS to tridiag form
    //LAPACKE_dsytrd(LAPACK_COL_MAJOR, 'U', 2, SAS, 2, d, e, tau); 
    // then solve for eigval and eigvec of T
    int m, isuppz[2], tryrac;
    double eigvec[3];
    LAPACKE_dstemr(LAPACK_COL_MAJOR, 'V', 'I', 2, d, e, -1.0, -1.0, 1, 1, 
                   &m, &lambda, eigvec, DMnd, 1, isuppz, &tryrac );

    t2 = MPI_Wtime();
    
    /* update x_old*/ // TODO: see if we can use AS(:,2) to store x_old and save some memory
    for (i = 0; i < DMnd; i++) x_old[i] = x[i];
    // update x = S * eigvec
    cblas_dgemv(CblasColMajor, CblasNoTrans, DMnd, 2, 1.0, S, DMnd, eigvec, 1, 0.0, x, 1);
    // calculate Ax
    Hamiltonian_vec_mult(pSPARC, DMnd, DMVertices, Veff_loc, Atom_Influence_nloc, 
                         nlocProj, 1, 0.0, x, Ax, comm);
    // find g = Ax - lambda * x
    for (i = 0; i < DMnd; i++) g[i] = Ax[i] - lambda * x[i];
    // find g_norm = ||g||
    Vector2Norm(g, DMnd, &g_norm, comm);
    if (g_norm < TOL) { *eigmin = lambda; return; }
    
    /*******************************************
     *  start the loop (start with count = 2)  *
     *******************************************/
    double StS[9];
    count = 2;
    // keep track of which col to store x_old
    icol = 2 * (count % 2); 
    while (g_norm > TOL && count <= MAXIT) {
        /* create S = [x, g, x_old] */
        for (i = 0; i < DMnd; i++) S[i] = x[i];
        for (i = 0; i < DMnd; i++) S[DMnd+i] = g[i];
        for (i = 0; i < DMnd; i++) S[2*DMnd+i] = x_old[i];
        
        /* S = orth(S)*/ // Procedure: chol(S'S) => U, orth(S) = S * inv(U)
        // Find StS = S' * S
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 3, 3, 
                    DMnd, 1.0, S, DMnd, S, DMnd, 0.0, StS, 3);
        MPI_Allreduce(MPI_IN_PLACE, StS, 9, MPI_DOUBLE, MPI_SUM, comm);
        
        //VectorDotProduct(x, x_old, DMnd, &vscal, comm);
        //StS[0] = 1.0;   StS[3] = 0.0;            StS[6] = vscal;
        //StS[1] = 0.0;   StS[4] = g_norm*g_norm;  StS[7] = 0.0;
        //StS[2] = vscal; StS[5] = 0.0;            StS[8] = 1.0;

        // Find Cholesky decomposition of StS = U'*U
        // note: StS is overwritten by U
        LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'U', 3, StS, 3);
        // Find inv(U), note that only upper triangular part is overwritten
        LAPACKE_dtrtri(LAPACK_COL_MAJOR, 'U', 'N', 3, StS, 3);
        // multiply S and triangular matrix inv(U) 
        //cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, 
        //            CblasNonUnit, DMnd, 3, 1.0, StS, 3, S, DMnd);       

        t1 = MPI_Wtime();  
   
        /* find A*S */ //TODO: CHECK if mult 3 vectors at a time is working!
        //Hamiltonian_vec_mult(pSPARC, DMnd, DMVertices, 3, 0.0, S, AS, comm);
        
        //Hamiltonian_vec_mult(pSPARC, DMnd, DMVertices, 1, 0.0, x, AS, comm);
        //Hamiltonian_vec_mult(pSPARC, DMnd, DMVertices, 1, 0.0, g, &AS[DMnd], comm);
        //Hamiltonian_vec_mult(pSPARC, DMnd, DMVertices, 1, 0.0, x_old, &AS[DMnd*2], comm);
        
        for (i = 0; i < DMnd; i++) AS[i] = Ax[i];
        Hamiltonian_vec_mult(pSPARC, DMnd, DMVertices, Veff_loc, Atom_Influence_nloc, 
                             nlocProj, 1, 0.0, g, &AS[DMnd], comm);
        //Hamiltonian_vec_mult(pSPARC, DMnd, DMVertices, 1, 0.0, x_old, &AS[DMnd*2], comm);

        t2 = MPI_Wtime();
        t = t2 - t1;
        
        /* find S' * AS */
        //TODO: use cblas_dgemmt instead if available
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 3, 3, DMnd, 
                    1.0, S, DMnd, AS, DMnd, 0.0, SAS, 3);
        MPI_Allreduce(MPI_IN_PLACE, SAS, 9, MPI_DOUBLE, MPI_SUM, comm);

        // multiply triangular matrix inv(U)^T and SAS
        cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, 
                    3, 3, 1.0, StS, 3, SAS, 3);
        // multiply SAS and triangular matrix inv(U) 
        cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, 
                    3, 3, 1.0, StS, 3, SAS, 3);

        /* find eigvals and eigvecs of SAS */
        // first reduce SAS to tridiag form, it seems SAS is already triangular!
        //LAPACKE_dsytrd(LAPACK_COL_MAJOR, 'U', 3, SAS, 3, d, e, tau); 
        d[0] = SAS[0]; d[1] = SAS[4]; d[2] = SAS[8];
        e[0] = SAS[3]; e[1] = SAS[7];
        // then calculate eigenvalue and eigenvector of T
        LAPACKE_dstemr(LAPACK_COL_MAJOR, 'V', 'I', 3, d, e, -1.0, -1.0, 1, 1, 
                       &m, &lambda, eigvec, DMnd, 1, isuppz, &tryrac );

        /* update x_old*/ 
        for (i = 0; i < DMnd; i++) x_old[i] = x[i];
        
        cblas_dtrmv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 3, StS, 3, eigvec, 1);
        
        /* update x = S * eigvec */
        cblas_dgemv(CblasColMajor, CblasNoTrans, DMnd, 3, 1.0, S, DMnd, eigvec, 1, 0.0, x, 1);

        /* before updating Ax, store old Ax into AS(:,3) */
        for (i = 0; i < DMnd; i++) AS[2*DMnd+i] = Ax[i];

        t1 = MPI_Wtime();
        /* calculate Ax */
        Hamiltonian_vec_mult(pSPARC, DMnd, DMVertices, Veff_loc, Atom_Influence_nloc, 
                             nlocProj, 1, 0.0, x, Ax, comm);
        t2 = MPI_Wtime();       
        /* find g = Ax - lambda * x */
        for (i = 0; i < DMnd; i++) g[i] = Ax[i] - lambda * x[i];
        
        /* find g_norm = ||g|| */
        Vector2Norm(g, DMnd, &g_norm, comm);
        //for (i = 0; i < DMnd; i++) g[i] /= g_norm;
        //if (rank == 0) {
        //    printf(GRN"CG iter %d, eigmin  = %.9f, err_eigmin = %.3e, H_S_mult took %.3f ms,"
        //           " Hx took %.3f ms\n"RESET,count, lambda, g_norm, t*1e3, (t2-t1) *1e3);
        //}
        count++;
    }
#ifdef DEBUG    
    if (rank == 0) {
        printf("CG iter %d, eigmin  = %.9f, err_eigmin = %.3e, H_S_mult took %.3f ms, Hx took %.3f ms\n",
                count, lambda, g_norm, t*1e3, (t2-t1) *1e3);
    }
#endif    
    *eigmin = lambda;
    free(x); free(x_old); free(Ax); free(g); free(S); free(AS);
}





/**
 * @brief   Look for a 2D decomposition of number of processors according to application size.
 *          This function is adapted to optimizing 2D block cyclic distribution of matrices, 
 *          where the number of processors are kept fixed and number of nodes/direction is not 
 *          substantial regarding performance.
 *
 * @param   NPSTRIDE    Maximum deviation of searching nodes in each direction, if set to -1,
 *                      the program will choose it automatically. The larger this number is, 
 *                      the more possiblilities of decomposition of the number of processors
 *                      it will search.
 */

void SPARC_Dims_create_2D_BLCYC(int nproc, int *gridsizes, int minsize, int *dims, int *NPSTRIDE, int *ierr) 
{
    *ierr = 0;
    double r, *rnormi,  min_rnorm, tmp1, tmp2;
    int i, j, count, best_ind, max_npi, tmpint1, tmpint2, N_tot, *dimx, *dimy, *npi;
    r = sqrt((double)nproc) / sqrt( (double)gridsizes[0] * (double)gridsizes[1] );
    minsize = max(1, minsize);
    r = min(r, 1/(double)minsize);
    double r2 = r * r; 
    // initial estimate
    dims[0] = round(gridsizes[0] * r);
    dims[1] = round(gridsizes[1] * r);
    
    for (i = 0; i < 2; i++) {
        // if the input NPSTRIDE is negative, choose NPSTRIDE automatically
        if (NPSTRIDE[i] < 0) {
            int stride; double stride_ratio = 0.5;
            stride = round(dims[i]*stride_ratio);
            if (stride < 10) {
                stride = dims[i]-1;
            }
            NPSTRIDE[i] = stride;
        }
    }
    
    N_tot = (NPSTRIDE[0] * 2 + 1) + (NPSTRIDE[1] * 2 + 1);
    rnormi = (double *)malloc( N_tot  * sizeof(double));
    dimx = (int *)malloc( N_tot * sizeof(int));
    dimy = (int *)malloc( N_tot * sizeof(int));
    npi  = (int *)malloc( N_tot * sizeof(int));

    
    count = 0;
    for (i = 0; i < (NPSTRIDE[0]*2+1); i++) {
        tmpint1 = dims[0] + i - NPSTRIDE[0];
        dimx[count] = (tmpint1 > 0) ? (tmpint1 <= nproc ? tmpint1 : (tmpint1-(NPSTRIDE[0]*2+1))) : (tmpint1+(NPSTRIDE[0]*2+1));
        // dimx[count] > 0 ? (dimx[count] <= nproc ? : (dimx[count] = 1)) : (dimx[count] = 1);
        if (dimx[count] > 0 && dimx[count] <= nproc) {
            SPARC_Dims_create(max(1,nproc/dimx[count]), 1, &gridsizes[1], minsize, &dimy[count], ierr);
            npi[count] = dimx[count] * dimy[count];
            tmp1 = ((double) gridsizes[0]) / dimx[count];
            tmp2 = ((double) gridsizes[1]) / dimy[count];
            rnormi[count] = (tmp1 - tmp2) * (tmp1 - tmp2);  
            rnormi[count] *= r2;
        } else {
            dimx[count] = 0; dimy[count] = 0; npi[count] = 0;
            rnormi[count] = 0;
        }
        count++;
    }          
    for (j = 0; j < (NPSTRIDE[1]*2+1); j++) {
        tmpint2 = dims[1] + j - NPSTRIDE[1];
        dimy[count] = (tmpint2 > 0) ? (tmpint2 <= nproc ? tmpint2 : (tmpint2-(NPSTRIDE[1]*2+1))) : (tmpint2+(NPSTRIDE[1]*2+1));
        //dimy[count] > 0 ? (dimy[count] <= nproc ? : (dimy[count] = 1)) : (dimy[count] = 1);
        if (dimy[count] > 0 && dimy[count] <= nproc) {
            SPARC_Dims_create(max(1,nproc/dimy[count]), 1, &gridsizes[0], minsize, &dimx[count], ierr);
            npi[count] = dimx[count] * dimy[count];
            tmp1 = ((double) gridsizes[0]) / dimx[count];
            tmp2 = ((double) gridsizes[1]) / dimy[count];
            rnormi[count] = (tmp1 - tmp2) * (tmp1 - tmp2);
            rnormi[count] *= r2;
        } else {
            dimx[count] = 0; dimy[count] = 0; npi[count] = 0;
            rnormi[count] = 0;
        }  
        count++;
    } 


    // check which one uses largest number of processes provided
    max_npi = 0;
    count = 0; best_ind = -1; min_rnorm = 1e4;    
    for (i = 0; i < N_tot; i++) {
        if (npi[count] < max_npi || npi[count] > nproc || npi[count] <= 0) {
            count++;
            continue;
        }
        if (npi[count] > max_npi && gridsizes[0]/dimx[count] >= minsize && gridsizes[1]/dimy[count] >= minsize) {
            best_ind = count;
            max_npi = npi[count];
            min_rnorm = rnormi[count];                 
        } else if (npi[count] == max_npi && rnormi[count] < min_rnorm) {
            best_ind = count;
            min_rnorm = rnormi[count];
        }
        count++;
    }  
    //printf("npmax    npi    dimx    dimy           rnormi\n");
    //for (i = 0; i < (NPSTRIDE*2+1)*2; i++) {
    // printf("%-4d    %-4d    %-3d      %-3d          %-.13e\n",nproc,npi[i],dimx[i],dimy[i], rnormi[i]);
    //}        
    //printf("best_ind for 2d = %d\n", best_ind);

    
    // TODO: after first scan, perhaps we can allow np to be up to 3% smaller, and choose the one with smaller rnormi,
    // the idea is that by reducing total number of process we lose speed by 3% or less, but we might gain speed up in
    // communication by more than 3% 
    

    if (best_ind != -1) {
        dims[0] = dimx[best_ind]; dims[1] = dimy[best_ind];
        if (dims[0] * dims[1] != nproc) {
            dims[0] = nproc; dims[1] = 1;;
            *ierr = 1; // cannot find any admissable distribution
        }
    } else {
        dims[0] = nproc; dims[1] = 1;;
        *ierr = 1; // cannot find any admissable distribution
    }  

    free(rnormi);
    free(dimx);
    free(dimy);
    free(npi);
}



/**
 * @brief   Calculate (Gradient + c * I) times a bunch of vectors in a matrix-free way.
 *          
 *          This function simply calls the Gradient_vec_mult (or the sequential version) 
 *          multiple times. For some reason it is more efficient than calling it ones and do  
 *          the multiplication together. TODO: think of a more efficient way!
 */
void Gradient_vectors(const SPARC_OBJ *pSPARC, int DMnd, int *DMVertices, int ncol, double c, double *x, double *Dx, double *Dy, double *Dz, MPI_Comm comm)
{
    double t1, t2;
    unsigned i;
    int nproc;
    //t1 = MPI_Wtime();
    MPI_Comm_size(comm, &nproc);
    //t2 = MPI_Wtime();
    //printf("getting size of comm in H_vectors_mult took %.3f ms\n", (t2-t1) * 1e3);
    if ((unsigned)nproc != 1) {
        for (i = 0; i < ncol; i++)
            Gradient_vec(pSPARC, DMnd, DMVertices, 1, c, x+i*(unsigned)DMnd, Dx+i*(unsigned)DMnd, Dy+i*(unsigned)DMnd, Dz+i*(unsigned)DMnd, comm);
    } else {
        //printf(YEL"Calling sequential alg for DX!\n"RESET);
        for (i = 0; i < ncol; i++)
            Gradient_vec_seq(pSPARC, DMnd, DMVertices, 1, c, x+i*(unsigned)DMnd, Dx+i*(unsigned)DMnd, Dy+i*(unsigned)DMnd, Dz+i*(unsigned)DMnd);
        //Gradient_vec_mult_seq(pSPARC, DMnd, DMVertices, Atom_Influence_nloc, nlocProj, ncol, c, x, Hx);
    }
}



/**
 * @brief   Calculate (Gradient + c * I) times vectors in a matrix-free way.
 */
void Gradient_vec(const SPARC_OBJ *pSPARC, int DMnd, int *DMVertices, int ncol, double c, double *x, double *Dx, double *Dy, double *Dz, MPI_Comm comm)
{

}



/**
 * @brief   Calculate (Gradient + c * I) times vectors in a matrix-free way ON A SINGLE PROCESS.
 *          
 *          Note: this function is to take advantage of the band parallelization scheme when a single
 *                process contains the whole vector and therefore no communication is needed.
 */
void Gradient_vec_seq(const SPARC_OBJ *pSPARC, int DMnd, int *DMVertices, int ncol, double c, double *x, double *Dx, double *Dy, double *Dz)
{
#define INDEX(n,i,j,k) ((n)*DMnd+(k)*DMnx*DMny+(j)*DMnx+(i))
#define INDEX_EX(n,i,j,k) ((n)*DMnd_ex+(k)*DMnx_ex*DMny_ex+(j)*DMnx_ex+(i))
#define X(n,i,j,k) x[(n)*DMnd+(k)*DMnx*DMny+(j)*DMnx+(i)]
#define Dx(n,i,j,k) Dx[(n)*DMnd+(k)*DMnx*DMny+(j)*DMnx+(i)]
#define x_ex(n,i,j,k) x_ex[(n)*DMnd_ex+(k)*DMnx_ex*DMny_ex+(j)*DMnx_ex+(i)]
//double t1, t2;
//t1 = MPI_Wtime();
    int n, i, j, k, p, FDn, ip, jp, kp, DMnx, DMny, DMnz, 
        DMnx_ex, DMny_ex, DMnz_ex, DMnd_ex, Nx, Ny, nd_in, nd_out,
        nbrcount, count, ind, ind_ex, pshift, *pshifty, *pshiftz, 
        *pshifty_ex, *pshiftz_ex, i_global, j_global, k_global;
    double w1_diag, dx, dy, dz;
    int periods[3];
    
    periods[0] = 1 - pSPARC->BCx;
    periods[1] = 1 - pSPARC->BCy;
    periods[2] = 1 - pSPARC->BCz;
    FDn = pSPARC->order / 2;
    
    // The user has to make sure DMnd = DMnx * DMny * DMnz
    DMnx = 1 - DMVertices[0] + DMVertices[1];
    DMny = 1 - DMVertices[2] + DMVertices[3];
    DMnz = 1 - DMVertices[4] + DMVertices[5];
    
    DMnx_ex = DMnx + pSPARC->order;
    DMny_ex = DMny + pSPARC->order;
    DMnz_ex = DMnz + pSPARC->order;
    DMnd_ex = DMnx_ex * DMny_ex * DMnz_ex;
    
    Nx = pSPARC->Nx;
    Ny = pSPARC->Ny;
    
    // shift the gradient by c
    w1_diag = c;

    pshifty = (int *)malloc( (FDn+1) * sizeof(int));
    pshiftz = (int *)malloc( (FDn+1) * sizeof(int));
    pshifty_ex = (int *)malloc( (FDn+1) * sizeof(int));
    pshiftz_ex = (int *)malloc( (FDn+1) * sizeof(int));
    double *x_ex = (double *)malloc(ncol * DMnd_ex * sizeof(double));
    
    if (pshifty == NULL || pshiftz == NULL || pshifty_ex == NULL || pshiftz_ex == NULL) {
        printf("\nMemory allocation failed!\n");
        exit(EXIT_FAILURE);
    }
    
    pshifty[0] = pshiftz[0] = pshifty_ex[0] = pshiftz_ex[0] = 0;
    for (p = 1; p <= FDn; p++) {
        // for x
        pshifty[p] = p * DMnx;
        pshiftz[p] = pshifty[p] * DMny;
        // for x_ex
        pshifty_ex[p] = p * DMnx_ex;
        pshiftz_ex[p] = pshifty_ex[p] * DMny_ex;
    }
    
    // copy x into x_ex
    count = 0;
    for (n = 0; n < ncol; n++) {    
        for (kp = FDn; kp < DMnz+FDn; kp++) {
            for (jp = FDn; jp < DMny+FDn; jp++) {
                for (ip = FDn; ip < DMnx+FDn; ip++) {
                    x_ex(n,ip,jp,kp) = x[count++]; // this saves index calculation time
                }
            }
        }
    }  

    // set up copy indices for sending edge nodes in x
    int istart[6] = {0,    DMnx-FDn, 0,    0,        0,    0}, 
          iend[6] = {FDn,  DMnx,     DMnx, DMnx,     DMnx, DMnx}, 
        jstart[6] = {0,    0,        0,    DMny-FDn, 0,    0},  
          jend[6] = {DMny, DMny,     FDn,  DMny,     DMny, DMny}, 
        kstart[6] = {0,    0,        0,    0,        0,    DMnz-FDn}, 
          kend[6] = {DMnz, DMnz,     DMnz, DMnz,     FDn,  DMnz};
    // set up start and end indices for copying edge nodes in x_ex
    int istart_in[6], iend_in[6], jstart_in[6], jend_in[6], kstart_in[6], kend_in[6];
    istart_in[0] = 0;        iend_in[0] = FDn;        jstart_in[0] = FDn;      jend_in[0] = DMny+FDn;   kstart_in[0] = FDn;      kend_in[0] = DMnz+FDn;
    istart_in[1] = DMnx+FDn; iend_in[1] = DMnx+2*FDn; jstart_in[1] = FDn;      jend_in[1] = DMny+FDn;   kstart_in[1] = FDn;      kend_in[1] = DMnz+FDn; 
    istart_in[2] = FDn;      iend_in[2] = DMnx+FDn;   jstart_in[2] = 0;        jend_in[2] = FDn;        kstart_in[2] = FDn;      kend_in[2] = DMnz+FDn;
    istart_in[3] = FDn;      iend_in[3] = DMnx+FDn;   jstart_in[3] = DMny+FDn; jend_in[3] = DMny+2*FDn; kstart_in[3] = FDn;      kend_in[3] = DMnz+FDn;
    istart_in[4] = FDn;      iend_in[4] = DMnx+FDn;   jstart_in[4] = FDn;      jend_in[4] = DMny+FDn;   kstart_in[4] = 0;        kend_in[4] = FDn;
    istart_in[5] = FDn;      iend_in[5] = DMnx+FDn;   jstart_in[5] = FDn;      jend_in[5] = DMny+FDn;   kstart_in[5] = DMnz+FDn; kend_in[5] = DMnz+2*FDn;

    int nbr_i, bc;

    // copy the extended part from x into x_ex
    for (nbr_i = 0; nbr_i < 6; nbr_i++) {
        // if dims[i] < 3 and periods[i] == 1, switch send buffer for left and right neighbors
        nbrcount = nbr_i + (1 - 2 * (nbr_i % 2)); // * (int)(dims[nbr_i / 2] < 3 && periods[nbr_i / 2]);
        //bc = periods[nbr_i / 2];
        //for (n = 0; n < ncol; n++) {
        //    for (k = kstart[nbrcount], kp = kstart_in[nbr_i]; k < kend[nbrcount]; k++, kp++) {
        //        for (j = jstart[nbrcount], jp = jstart_in[nbr_i]; j < jend[nbrcount]; j++, jp++) {
        //            for (i = istart[nbrcount], ip = istart_in[nbr_i]; i < iend[nbrcount]; i++, ip++) {
        //                x_ex(n,ip,jp,kp) = X(n,i,j,k) * bc;
        //            }
        //        }
        //    }
        //}
        if (periods[nbr_i / 2])
            for (n = 0; n < ncol; n++) {
                for (k = kstart[nbrcount], kp = kstart_in[nbr_i]; k < kend[nbrcount]; k++, kp++) {
                    for (j = jstart[nbrcount], jp = jstart_in[nbr_i]; j < jend[nbrcount]; j++, jp++) {
                        for (i = istart[nbrcount], ip = istart_in[nbr_i]; i < iend[nbrcount]; i++, ip++) {
                            x_ex(n,ip,jp,kp) = X(n,i,j,k);
                        }
                    }
                }
            }
        else
            for (n = 0; n < ncol; n++) {
                for (kp = kstart_in[nbr_i]; kp < kend_in[nbr_i]; kp++) {
                    for (jp = jstart_in[nbr_i]; jp < jend_in[nbr_i]; jp++) {
                        for (ip = istart_in[nbr_i]; ip < iend_in[nbr_i]; ip++) {
                            x_ex(n,ip,jp,kp) = 0.0;
                        }
                    }
                }
            }
    }

    count = 0;
    for (n = 0; n < ncol; n++) {
        for (k = 0; k < DMnz; k++) {
            kp = k + FDn; 
            k_global = k + DMVertices[4];
            for (j = 0; j < DMny; j++) {
                jp = j + FDn; 
                j_global = j + DMVertices[2];
                for (i = 0; i < DMnx; i++) {
                    ip = i + FDn; 
                    i_global = i + DMVertices[0];
                    //dx = x_ex(n,ip,jp,kp) * w1_diag;
                    ind_ex = INDEX_EX(n,ip,jp,kp);
                    dx = x_ex[ind_ex] * w1_diag;
                    for (p = 1; p <= FDn; p++) {
                        //dx += (x_ex(n,ip+p,jp,kp) - x_ex(n,ip-p,jp,kp)) * pSPARC->D2_stencil_coeffs_x[p];
                        //dy += (x_ex(n,ip,jp+p,kp) - x_ex(n,ip,jp-p,kp)) * pSPARC->D2_stencil_coeffs_y[p];
                        //dz += (x_ex(n,ip,jp,kp+p) - x_ex(n,ip,jp,kp-p)) * pSPARC->D2_stencil_coeffs_z[p];
                        //pshifty = p * DMnx_ex;
                        //pshiftz = pshifty * DMny_ex;
                        dx += (x_ex[ind_ex+p]             - x_ex[ind_ex-p]            ) * pSPARC->D1_stencil_coeffs_x[p];
                        dy += (x_ex[ind_ex+pshifty_ex[p]] - x_ex[ind_ex-pshifty_ex[p]]) * pSPARC->D1_stencil_coeffs_y[p];
                        dz += (x_ex[ind_ex+pshiftz_ex[p]] - x_ex[ind_ex-pshiftz_ex[p]]) * pSPARC->D1_stencil_coeffs_z[p];
                    }
                    Dx[count] = dx; Dy[count] = dy; Dz[count] = dz;
                    count++;
                }
            }
        }   
    }
    
    free(x_ex);
    free(pshifty);
    free(pshiftz);
   
#undef INDEX
#undef INDEX_EX
#undef X
#undef Dx
#undef x_ex
}


/**
 * @brief   Perform truncated Kerker preconditioner.
 *
 *          Apply truncated Kerker preconditioner in real space. For given 
 *          function f, this function returns 
 *          Pf := sum_i a(i) * (L - lambda_sqr(i))^-1 * Lf + k*f, 
 *          where L is the discrete Laplacian operator. The result 
 *          is written in Pf.
 */
void TruncatedKerker_precond(
    SPARC_OBJ *pSPARC, double *f, const double tol, const int DMnd, 
    const int *DMVertices, double *Pf, MPI_Comm comm
)
{
#ifdef DEBUG
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (!rank) {
        printf(BLU "Start applying truncated Kerker preconditioner ...\n" RESET);
        int n;
        printf(BLU "precondcoeff_k = %f\n" RESET, pSPARC->precondcoeff_k);
        for (n = 0; n < pSPARC->precondcoeff_n; n++) {
            printf(BLU "precondcoeff_a[%d/%d] = %f + %f i\n" RESET, n, pSPARC->precondcoeff_n, 
                creal(pSPARC->precondcoeff_a[n]), cimag(pSPARC->precondcoeff_a[n]));
            printf(BLU "precondcoeff_lambda_sqr[%d/%d] = %f + %f i\n" RESET, n, pSPARC->precondcoeff_n, 
                creal(pSPARC->precondcoeff_lambda_sqr[n]), cimag(pSPARC->precondcoeff_lambda_sqr[n]));
        }
    }
#endif

    // call RSfit_Precond with the appropriate coeffs
    RSfit_Precond(pSPARC, f, pSPARC->precondcoeff_n, pSPARC->precondcoeff_a,
        pSPARC->precondcoeff_lambda_sqr, pSPARC->precondcoeff_k, tol, DMnd,
        DMVertices, Pf, comm);
    
#ifdef DEBUG
    if (!rank) printf(BLU "Finished\n" RESET);
#endif
}





/** 
 * @brief   RSFIT_PRECOND applies real-space preconditioner with any rational fit 
 *          coefficients.
 *
 *          RSFIT_PRECOND effectively applies sum_i (a_i*(Lap - k_TF2_i)^-1 * Lap + k*I) 
 *          to f by solving the linear systems a_i*(Lap - k_TF2_i) s = Lap*f and 
 *          summing the sols. To apply any preconditioner, simply perform a
 *          rational curve fit to the preconditioner in fourier space and provide
 *          the fit coeffs a(i), lambda_TF(i) and a constant k.
 */
void RSfit_Precond(
    SPARC_OBJ *pSPARC, double *f, const int npow, 
    const double complex *a, 
    const double complex *lambda_sqr, 
    const double k,  // k should always be real
    const double tol, const int DMnd, const int *DMVertices, 
    double *Pf, MPI_Comm comm
)
{
    if (comm == MPI_COMM_NULL) return;
    double *Lf;
    Lf = (double *)malloc(DMnd * sizeof(double));
    // find Lf 
    Lap_vec_mult(pSPARC, DMnd, DMVertices, 1, 0.0, f, Lf, comm);

    double complex *Lf_complex, *Pf_complex;
    Lf_complex = (double complex *)malloc(DMnd * sizeof(double complex));
    Pf_complex = (double complex *)calloc(DMnd , sizeof(double complex));
    
    int i;
    for (i = 0; i < DMnd; i++) {
        Lf_complex[i] = Lf[i] + 0.0 * I;
        //Pf_complex[i] = 1E-6 + 0.0 * I;
    }

    //** solve -(L - lambda_TF^2) * Pf = Lf for Pf **//
    double omega = 0.6, beta = 0.6; 
    int    m = 7, p = 6;

    // declare residual function
    void helmholz_res(
        SPARC_OBJ* pSPARC, int N, double complex c, double complex *x, 
        double complex *b, double complex *r, MPI_Comm comm, double *time_info
    );
    // declare precod function
    void Jacobi_precond_complex(
        SPARC_OBJ *pSPARC, int N, double complex c, 
        double complex *r, double complex *f, MPI_Comm comm
    );

    void (*precond_fun) (
        SPARC_OBJ *pSPARC, int N, double complex c, 
        double complex *r, double complex *f, MPI_Comm comm
    ) = Jacobi_precond_complex;

    void (*res_fun) (
        SPARC_OBJ* pSPARC, int N, double complex c, double complex *x, 
        double complex *b, double complex *r, MPI_Comm comm, double *time_info
    ) = helmholz_res;

    for (i = 0; i < DMnd; i++) {
        Pf[i] = k * f[i];
    }

    int n;
    for (n = 0; n < npow; n++) {
        AAR_complex(pSPARC, res_fun, precond_fun, -lambda_sqr[n], DMnd, 
            Pf_complex, Lf_complex, omega, beta, m, p, tol, 1000, comm);
        for (i = 0; i < DMnd; i++) {
            Pf[i] -= creal(a[n] * Pf_complex[i]); // note the '-' sign
        }
    }

    free(Lf);
    free(Lf_complex);
    free(Pf_complex);
}



/**
 * @brief   Perform Resta preconditioner.
 *
 *          Apply Resta preconditioner in real space. For given 
 *          function f, this function returns 
 *          Pf := sum_i a(i) * (L - lambda_sqr(i))^-1 * Lf + k*f, 
 *          where L is the discrete Laplacian operator. The result 
 *          is written in Pf.
 */
void Resta_precond(
    SPARC_OBJ *pSPARC, double *f, const double tol, const int DMnd, 
    const int *DMVertices, double *Pf, MPI_Comm comm
)
{
#ifdef DEBUG
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if (!rank) {
        printf(BLU "Start applying Resta preconditioner ...\n" RESET);
        int n;
        printf(BLU "precondcoeff_k = %f\n" RESET, pSPARC->precondcoeff_k);
        for (n = 0; n < pSPARC->precondcoeff_n; n++) {
            printf(BLU "precondcoeff_a[%d/%d] = %f + %f i\n" RESET, n, pSPARC->precondcoeff_n, 
                creal(pSPARC->precondcoeff_a[n]), cimag(pSPARC->precondcoeff_a[n]));
            printf(BLU "precondcoeff_lambda_sqr[%d/%d] = %f + %f i\n" RESET, n, pSPARC->precondcoeff_n, 
                creal(pSPARC->precondcoeff_lambda_sqr[n]), cimag(pSPARC->precondcoeff_lambda_sqr[n]));
        }
    }
#endif

    // call RSfit_Precond with the appropriate coeffs
    RSfit_Precond(pSPARC, f, pSPARC->precondcoeff_n, pSPARC->precondcoeff_a,
        pSPARC->precondcoeff_lambda_sqr, pSPARC->precondcoeff_k, tol, DMnd,
        DMVertices, Pf, comm);

#ifdef DEBUG
    if (!rank) printf(BLU "Finished\n" RESET);
#endif
}





/**
 * @brief   Calculate the residual of the Helmholtz type of equation:
 *                             -(Lap + c) * x = b.
 *          The residual is defined as r = b - (-(Lap+c) * x) = b + (Lap+c)*x.
 *
 *          The vector x is assumed to be stored domain-wisely among the processors. The
 *          structure pSPARC contains the description of the distribution info of x, and
 *          in this case the info of Laplacian operator such as boundary conditions, 
 *          finite-difference order and coefficients etc.
 */
void helmholz_res(
    SPARC_OBJ* pSPARC, int N, double complex c, double complex *x, 
    double complex *b, double complex *r, MPI_Comm comm, double *time_info
)
{
    // TODO: since we don't have a general routine to do Lap * complex vector
    // we currently will multiply the real and complex part seperately

    double t1 = MPI_Wtime(); // start timer for Lap * complex vector
    
    double *x_real  = (double *)malloc(N * sizeof(double));
    double *x_imag  = (double *)malloc(N * sizeof(double));
    double *Lx_real = (double *)malloc(N * sizeof(double));
    double *Lx_imag = (double *)malloc(N * sizeof(double));

    int i;
    for (i = 0; i < N; i++) {
        x_real[i] = creal(x[i]);
        x_imag[i] = cimag(x[i]);
    }

    // Calculate Lap * x (x is complex)
    // real(r) = Lap * real(x)
    Lap_vec_mult(pSPARC, N, pSPARC->DMVertices, 1, 0.0, x_real, Lx_real, comm);
    // imag(r) = Lap * imag(x)
    Lap_vec_mult(pSPARC, N, pSPARC->DMVertices, 1, 0.0, x_imag, Lx_imag, comm);

    double t2 = MPI_Wtime();
    *time_info = t2 - t1;

    for (i = 0; i < N; i++) {
        r[i] = b[i] + Lx_real[i] + Lx_imag[i] * I + c * x[i];
    }

    free(x_real);
    free(x_imag);
    free(Lx_real);
    free(Lx_imag);
}


void Jacobi_precond_complex(
    SPARC_OBJ *pSPARC, int N, double complex c, 
    double complex *r, double complex *f, MPI_Comm comm
) 
{
    int i;
    double complex m_inv;
    // TODO: m_inv can be calculated in advance and stored into pSPARC
    m_inv =  pSPARC->D2_stencil_coeffs_x[0] 
           + pSPARC->D2_stencil_coeffs_y[0] 
           + pSPARC->D2_stencil_coeffs_z[0] + c;
    if (fabs(creal(m_inv)) < 1e-14 && fabs(cimag(m_inv)) < 1e-14) {
        m_inv = 1.0;
    }
    m_inv = - 1.0 / m_inv;
    for (i = 0; i < N; i++)
        f[i] = m_inv * r[i];
}




void  AndersonExtrapolation_complex(
        const int N, const int m, double complex *x_kp1, const double complex *x_k, 
        const double complex *f_k, const double complex *X, const double complex *F, 
        const double beta, MPI_Comm comm
) 
{
    unsigned i;
    double complex *f_wavg = (double complex *)malloc( N * sizeof(double complex) );
    
    // find the weighted average vectors
    AndersonExtrapWtdAvg_complex(N, m, x_k, f_k, X, F, x_kp1, f_wavg, comm);
    
    // add beta * f to x_{k+1}
    for (i = 0; i < N; i++)
        x_kp1[i] += beta * f_wavg[i];
    
    free(f_wavg);
}




void  AndersonExtrapWtdAvg_complex(
        const int N, const int m, const double complex *x_k, const double complex *f_k, 
        const double complex *X, const double complex *F, double complex *x_wavg, double complex *f_wavg, 
        MPI_Comm comm
) 
{
    double complex *Gamma = (double complex *)calloc( m , sizeof(double complex) );
    assert(Gamma != NULL); 
    
    // find extrapolation weigths Gamma = inv(F^T * F) * F^T * f_k
    AndersonExtrapCoeff_complex(N, m, f_k, F, Gamma, comm); 
    
    unsigned i;
    
    // find weighted average x_{k+1} = x_k - X*Gamma
    for (i = 0; i < N; i++) x_wavg[i] = x_k[i];
    double complex alpha, beta;
    alpha = -1.0; beta = 1.0;
    cblas_zgemv(CblasColMajor, CblasNoTrans, N, m, &alpha, X, 
        N, Gamma, 1, &beta, x_wavg, 1);

    // find weighted average f_{k+1} = f_k - F*Gamma
    for (i = 0; i < N; i++) f_wavg[i] = f_k[i];

    alpha = -1.0; beta = 1.0;
    cblas_zgemv(CblasColMajor, CblasNoTrans, N, m, &alpha, F, 
        N, Gamma, 1, &beta, f_wavg, 1);

    free(Gamma);
}




void AndersonExtrapCoeff_complex(
    const int N, const int m, const double complex *f, const double complex *F, 
    double complex *Gamma, MPI_Comm comm
) 
{
// #define FtF(i,j) FtF[(j)*m+(i)]
    int matrank;
    double complex *FtF;
    double *s;

    FtF = (double complex *)malloc( m * m * sizeof(double complex) );
    s   = (double *)malloc( m * sizeof(double) );
    assert(FtF != NULL && s != NULL);  

    //# If mkl-11.3 or later version is available, one may use cblas_zgemmt #
    // calculate F^T * F, only update the LOWER part of the matrix
    //cblas_zgemmt(CblasColMajor, CblasLower, CblasTrans, CblasNoTrans, 
    //             m, N, 1.0, F, N, F, N, 0.0, FtF_Ftf, m);
    //// copy the lower half of the matrix to the upper half (LOCAL)
    //for (j = 0; j < m; j++)
    //    for (i = 0; i < j; i++)
    //        FtF_Ftf(i,j) = FtF_Ftf(j,i);
    
    //#   Otherwise use cblas_dgemm instead    #
    double complex alpha, beta;
    alpha = 1.0; beta = 0.0;
    cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, m, m, N, 
                &alpha, F, N, F, N, &beta, FtF, m);

    // calculate F^T * f using CBLAS  (LOCAL)
    alpha = 1.0; beta = 0.0;
    cblas_zgemv(CblasColMajor, CblasConjTrans, N, m, 
                &alpha, F, N, f, 1, &beta, Gamma, 1);

    // Sum the local results of F^T * F and F^T * f (GLOBAL)
    MPI_Allreduce(MPI_IN_PLACE, FtF, m*m, MPI_DOUBLE_COMPLEX, MPI_SUM, comm);
    MPI_Allreduce(MPI_IN_PLACE, Gamma, m, MPI_DOUBLE_COMPLEX, MPI_SUM, comm);

    // find inv(F^T * F) * (F^T * f) by solving (F^T * F) * x = F^T * f (LOCAL)
    LAPACKE_zgelsd(LAPACK_COL_MAJOR, m, m, 1, FtF, m, Gamma, m, s, -1.0, &matrank);

    free(s);
    free(FtF);
// #undef FtF 
}