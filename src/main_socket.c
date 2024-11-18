/**
 * @file    main_socket.c
 * @brief   This file contains the main function for SPARC
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Tian Tian <alchem0x2a@gmail.com, tian.tian@gatech.edu>
 *          Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
 
#include "initialization.h"
#include "md.h"
#include "relax.h" 
//#include "tools.h"
#include "finalization.h"
#include "isddft.h"
#include "electronicGroundState.h"

#ifdef USE_SOCKET
#include "driver.h"
#endif


int main(int argc, char *argv[]) {
    // set up MPI
    MPI_Init(&argc, &argv);
    // get communicator size and my rank
    MPI_Comm comm = MPI_COMM_WORLD;
    int nproc, rank;
    MPI_Comm_size(comm, &nproc);
    MPI_Comm_rank(comm, &rank);
    
    SPARC_OBJ SPARC;

    double t1, t2;
    
    MPI_Barrier(MPI_COMM_WORLD);
    // start timer
    t1 = MPI_Wtime(); SPARC.time_start = t1;
    
    // Read files and initialize
    Initialize(&SPARC, argc, argv);

    if (SPARC.MDFlag == 1)
        main_MD(&SPARC);
    else if (SPARC.RelaxFlag != 0)
        main_Relax(&SPARC);
    else
#ifdef USE_SOCKET
      if (SPARC.SocketFlag == 1)
        main_Socket(&SPARC);
      else
        Calculate_Properties(&SPARC);
        //Calculate_electronicGroundState(&SPARC);
#else
        Calculate_Properties(&SPARC);
        //Calculate_electronicGroundState(&SPARC);
#endif
    Finalize(&SPARC);

    MPI_Barrier(MPI_COMM_WORLD);
    // end timer
    t2 = MPI_Wtime();
    if (rank == 0) {
        printf("The program took %.3f s.\n", t2 - t1); 
    }
    // ensure stdout flushed to prevent Finalize hang
    fflush(stdout);

    // finalize MPI
    MPI_Finalize();
    return 0;
}
