/**
 * @file    readfiles.h
 * @brief   This file declares the functions for reading files.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#ifndef READFILES_H
#define READFILES_H

#include "isddft.h"

/**
 * @brief   find the type name of element from the input name
 *
 */
void find_element(char *element, char *atom_type);

/**
 * @brief   Read input file.
 *
 *          This function reads the input file and saves the data into the 
 *          structure pSPARC_Input and pSPARC. Note that only root process
 *          (rank = 0) will call this function to read data from the input file, 
 *          which will later be broadcasted to all processes.
 *
 * @param pSPARC_Input  The pointer that points to SPARC_Input_OBJ type structure.
 * @param pSPARC    The pointer that points to SPARC_OBJ type structure SPARC.
 */
void read_input(SPARC_INPUT_OBJ *pSPARC_Input, SPARC_OBJ *pSPARC);


/**
 * @brief   Read ion file.
 *
 *          This function reads the ion file and saves the data into the 
 *          structure pSPARC. Note that only root process (rank = 0) 
 *          will call this function.
 *          
 * @param pSPARC   The pointer that points to SPARC_OBJ type structure.
 */
void read_ion(SPARC_INPUT_OBJ *pSPARC_Input, SPARC_OBJ *pSPARC);


/**
 * @brief   Read pseudopotential files (PSP format).
 */
void read_pseudopotential_PSP(SPARC_INPUT_OBJ *pSPARC_Input, SPARC_OBJ *pSPARC);


#endif // READFILES_H


