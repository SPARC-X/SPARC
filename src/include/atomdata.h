/**
 * @file    atomdata.c
 * @brief   This file contains the delaration of atom data functions.
 *
 * @author  Qimen Xu <qimenxu@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#ifndef ATOMDATA_H
#define ATOMDATA_H 

/**
 * @brief   Find atomic mass info. for given element type.
 *
 * @ref     NIST (https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl?ele=&all=all&ascii=html)
 *
 * @param element   Element name. (OUTPUT)
 * @param atom_type User provided atom type name.
 */
void atomdata_mass(char *element, double *mass);

/**
 * @brief   Find atomic number info. for given element type.
 *
 *
 * @param element   Element name. (OUTPUT)
 * @param Z pointer of atomic number of the element
 */
void atomdata_number(char *element, int *Z);

#endif //ATOMDATA_H
