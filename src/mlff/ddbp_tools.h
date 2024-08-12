/**
 * @file    ddbp_tools.h
 * @brief   This file contains the function declarations for the Discrete
 *          Discontinuous Basis Projection (DDBP) method.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 *
 * Copyright (c) 2021 Material Physics & Mechanics Group, Georgia Tech.
 */

#ifndef _DDBP_TOOLS_H
#define _DDBP_TOOLS_H

#include "isddft.h"



/**
 * @brief Initialize the dynamic array, allocate initial
 *        memory and set size.
 *
 * @param initsize  Initial size of memory to be allocated.
 */
void init_dyarray(dyArray *a);

/**
 * @brief Realloc the dynamic array to the given new capacity.
 *
 *        Note that if the array is extended, all the previous data
 *        are still preserved. If the array is shrinked, all the
 *        previous data up to the new capacity is preserved.
 */
void realloc_dyarray(dyArray *a, size_t new_capacity);

/**
 * @brief Append an element to the dynamic array.
 *
 */
void append_dyarray(dyArray *a, value_type element);


/**
 * @brief Pop the last element from the dynamic array.
 *
 */
value_type pop_dyarray(dyArray *a);


/**
 * @brief Clear the dynamic array.
 *
 *        This function does not destroy the array, it simply
 *        resets the lenght of the dynamic array to 0, and resets
 *        the capacity.
 */
void clear_dyarray(dyArray *a);


/**
 * @brief Delete the dynamic array.
 *
 */
void delete_dyarray(dyArray *a);


//* for debugging purpose *//
void print_dyarray(const dyArray *a);
void show_dyarray(const dyArray *a);



#endif // _DDBP_TOOLS_H

