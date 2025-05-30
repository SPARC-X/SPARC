#ifndef SPARCATOM_H
#define SPARCATOM_H

void getAtomicStates(int Z, int **nOcc, int **lOcc, int **fupOcc, int **fdwOcc, int *sz);
void sparc_atom(SPARC_OBJ *pSPARC, int ityp, SPARC_INPUT_OBJ *pSPARC_Input);

#endif