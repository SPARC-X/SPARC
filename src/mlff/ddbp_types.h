/**
 * @file    ddbp_types.h
 * @brief   This file contains the DDBP types definition.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 *
 * Copyright (c) 2021 Material Physics & Mechanics Group, Georgia Tech.
 */

#ifndef _DDBP_TYPES_H
#define _DDBP_TYPES_H

#include <mpi.h>

// declare datatypes (defined elsewhere)
typedef struct _SPARC_OBJ SPARC_OBJ;
typedef struct _ATOM_NLOC_INFLUENCE_OBJ ATOM_NLOC_INFLUENCE_OBJ;
typedef struct _NLOC_PROJ_OBJ NLOC_PROJ_OBJ;
typedef struct _PSD_OBJ PSD_OBJ;
typedef struct _DDBP_ELEM DDBP_ELEM;

#define INIT_CAPACITY 4
typedef int value_type;

/**
 * @brief  Data type for dynamic array.
 */
typedef struct {
    value_type *array;
    size_t len;  // length of the array (used)
    size_t capacity; // total capacity of memory available
} dyArray;


#endif // _DDBP_TYPES_H

