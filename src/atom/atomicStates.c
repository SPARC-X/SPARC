#include <stdio.h>
#include <stdlib.h>

#include "atomicStates.h"

void getAtomicStates(int Z, int **nOcc, int **lOcc, int **fupOcc, int **fdwOcc, int *size_arr){
    // Sum fupOcc + Sum fdwOcc = Z
    int sz;
    switch(Z){
        case 1:{
            sz = 1;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = {1};
            int lOcc_val[] = {0};
            int fupOcc_val[] = {1};
            int fdwOcc_val[] = {0};

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 2:{
            sz = 1;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = {1};
            int lOcc_val[] = {0};
            int fupOcc_val[] = {1};
            int fdwOcc_val[] = {1};

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 3:{
            sz = 2;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = {1,2};
            int lOcc_val[] = {0,0};
            int fupOcc_val[] = {1,1};
            int fdwOcc_val[] = {1,0};

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 4:{
            sz = 2;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = {1,2};
            int lOcc_val[] = {0,0};
            int fupOcc_val[] = {1,1};
            int fdwOcc_val[] = {1,1};

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 5:{
            sz = 3;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = {1,2,2};
            int lOcc_val[] = {0,0,1};
            int fupOcc_val[] = {1,1,1};
            int fdwOcc_val[] = {1,1,0};

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 6:{
            sz = 3;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = {1,2,2};
            int lOcc_val[] = {0,0,1};
            int fupOcc_val[] = {1,1,2};
            int fdwOcc_val[] = {1,1,0};

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 7:{
            sz = 3;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = {1,2,2};
            int lOcc_val[] = {0,0,1};
            int fupOcc_val[] = {1,1,3};
            int fdwOcc_val[] = {1,1,0};

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 8:{
            sz = 3;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = {1,2,2};
            int lOcc_val[] = {0,0,1};
            int fupOcc_val[] = {1,1,3};
            int fdwOcc_val[] = {1,1,1};

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 9:{
            sz = 3;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = {1,2,2};
            int lOcc_val[] = {0,0,1};
            int fupOcc_val[] = {1,1,3};
            int fdwOcc_val[] = {1,1,2};

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 10:{
            sz = 3;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = {1,2,2};
            int lOcc_val[] = {0,0,1};
            int fupOcc_val[] = {1,1,3};
            int fdwOcc_val[] = {1,1,3};

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 11:{
            sz = 4;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = {1,2,2,3};
            int lOcc_val[] = {0,0,1,0};
            int fupOcc_val[] = {1,1,3,1};
            int fdwOcc_val[] = {1,1,3,0};

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
        break;}
        case 12:{
            sz = 4;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = {1,2,2,3};
            int lOcc_val[] = {0,0,1,0};
            int fupOcc_val[] = {1,1,3,1};
            int fdwOcc_val[] = {1,1,3,1};

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 13:{
            sz = 5;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = {1,2,2,3,3};
            int lOcc_val[] = {0,0,1,0,1};
            int fupOcc_val[] = {1,1,3,1,1};
            int fdwOcc_val[] = {1,1,3,1,0};

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 14:{
            sz = 5;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = {1,2,2,3,3};
            int lOcc_val[] = {0,0,1,0,1};
            int fupOcc_val[] = {1,1,3,1,2};
            int fdwOcc_val[] = {1,1,3,1,0};

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 15:{
            sz = 5;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = {1,2,2,3,3};
            int lOcc_val[] = {0,0,1,0,1};
            int fupOcc_val[] = {1,1,3,1,3};
            int fdwOcc_val[] = {1,1,3,1,0};

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 16:{
            sz = 5;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = {1,2,2,3,3};
            int lOcc_val[] = {0,0,1,0,1};
            int fupOcc_val[] = {1,1,3,1,3};
            int fdwOcc_val[] = {1,1,3,1,1};

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 17:{
            sz = 5;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = {1,2,2,3,3};
            int lOcc_val[] = {0,0,1,0,1};
            int fupOcc_val[] = {1,1,3,1,3};
            int fdwOcc_val[] = {1,1,3,1,2};

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 18:{
            sz = 5;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = {1,2,2,3,3};
            int lOcc_val[] = {0,0,1,0,1};
            int fupOcc_val[] = {1,1,3,1,3};
            int fdwOcc_val[] = {1,1,3,1,3};

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 19:{
            sz = 6;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = {1,2,2,3,3,4};
            int lOcc_val[] = {0,0,1,0,1,0};
            int fupOcc_val[] = {1,1,3,1,3,1};
            int fdwOcc_val[] = {1,1,3,1,3,0};

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 20:{
            sz = 6;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = {1,2,2,3,3,4};
            int lOcc_val[] = {0,0,1,0,1,0};
            int fupOcc_val[] = {1,1,3,1,3,1};
            int fdwOcc_val[] = {1,1,3,1,3,1};

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 21:{
            sz = 7;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = {1, 2, 2, 3, 3, 3, 4};
            int lOcc_val[] = {0, 0, 1, 0, 1, 2, 0};
            int fupOcc_val[] = {1, 1, 3, 1, 3, 1, 1};
            int fdwOcc_val[] = {1, 1, 3, 1, 3, 0, 1};

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 22:{
            sz = 7;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = {1, 2, 2, 3, 3, 3, 4};
            int lOcc_val[] = {0, 0, 1, 0, 1, 2, 0};
            int fupOcc_val[] = {1, 1, 3, 1, 3, 2, 1};
            int fdwOcc_val[] = {1, 1, 3, 1, 3, 0, 1};

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 23:{
            sz = 7;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 3, 1 }; 
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 0, 1 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 24:{
            sz = 7;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 0, 0 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 25:{
            sz = 7;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 0, 1 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 26:{
            sz = 7;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 1, 1 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 27:{
            sz = 7;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 2, 1 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 28:{
            sz = 7;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 3, 1 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 29:{
            sz = 7;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 0 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 30:{
            sz = 7;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 31:{
            sz = 8;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 1 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 0 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 32:{
            sz = 8;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 2 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 0 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 33:{
            sz = 8;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 0 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 34:{
            sz = 8;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 1 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 35:{
            sz = 8;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 2 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 36:{
            sz = 8;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 37:{
            sz = 9;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4, 5 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1, 0 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 1 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 0 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 38:{
            sz = 9;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4, 5 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1, 0 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 1 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 1 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 39:{
            sz = 10;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4, 4, 5 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1, 2, 0 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 1, 1 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 0, 1 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 40:{
            sz = 10;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4, 4, 5 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1, 2, 0 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 2, 1 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 0, 1 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 41:{
            sz = 10;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4, 4, 5 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1, 2, 0 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 4, 1 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 0, 0 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 42:{
            sz = 10;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4, 4, 5 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1, 2, 0 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 1 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 0, 0 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 43:{
            sz = 10;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4, 4, 5 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1, 2, 0 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 1 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 0, 1 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 44:{
            sz = 10;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4, 4, 5 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1, 2, 0 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 1 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 2, 0 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 45:{
            sz = 10;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4, 4, 5 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1, 2, 0 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 1 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 3, 0 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 46:{
            sz = 9;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4, 4 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1, 2 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 47:{
            sz = 10;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4, 4, 5 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1, 2, 0 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 1 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 0 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 48:{
            sz = 10;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4, 4, 5 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1, 2, 0 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 1 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 1 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 49:{
            sz = 11;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1, 2, 0, 1 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 1, 1 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 1, 0 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 50:{
            sz = 11;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1, 2, 0, 1 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 1, 2 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 1, 0 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 51:{
            sz = 11;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1, 2, 0, 1 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 1, 3 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 1, 0 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 52:{
            sz = 11;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1, 2, 0, 1 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 1, 3 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 1, 1 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 53:{
            sz = 11;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1, 2, 0, 1 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 1, 3 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 1, 2 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 54:{
            sz = 11;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1, 2, 0, 1 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 1, 3 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 1, 3 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 55:{
            sz = 12;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1, 2, 0, 1, 0 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 1, 3, 1 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 1, 3, 0 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 56:{
            sz = 12;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1, 2, 0, 1, 0 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 1, 3, 1 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 1, 3, 1 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 57:{
            sz = 13;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 1, 3, 1, 1 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 1, 3, 0, 1 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 58:{
            sz = 14;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 1, 1, 3, 1, 1 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 0, 1, 3, 0, 1 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 59:{
            sz = 13;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 0 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 3, 1, 3, 1 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 0, 1, 3, 1 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 60:{
            sz = 13;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 0 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 4, 1, 3, 1 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 0, 1, 3, 1 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 61:{
            sz = 13;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 0 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 5, 1, 3, 1 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 0, 1, 3, 1 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 62:{
            sz = 13;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 0 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 6, 1, 3, 1 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 0, 1, 3, 1 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 63:{
            sz = 13;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 0 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 1 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 0, 1, 3, 1 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 64:{
            sz = 13;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 1, 1 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 0, 1, 3, 0, 1 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 65:{
            sz = 13;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 0 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 1 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 2, 1, 3, 1 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 66:{
            sz = 13;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 0 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 1 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 3, 1, 3, 1 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 67:{
            sz = 13;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 0 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 1 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 4, 1, 3, 1 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 68:{
            sz = 13;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 0 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 1 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 5, 1, 3, 1 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 69:{
            sz = 13;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 0 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 1 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 6, 1, 3, 1 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 70:{
            sz = 13;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 0 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 1 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 1 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 71:{
            sz = 14;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 1, 1 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 0, 1 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 72:{
            sz = 14;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 2, 1 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 0, 1 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 73:{
            sz = 14;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 3, 1 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 0, 1 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 74:{
            sz = 14;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 4, 1 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 0, 1 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 75:{
            sz = 14;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 0, 1 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 76:{
            sz = 14;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 1, 1 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 77:{
            sz = 14;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 2, 1 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 78:{
            sz = 14;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 4, 0 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 79:{
            sz = 14;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 0 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 80:{
            sz = 14;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 81:{
            sz = 15;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0, 1 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1, 1 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1, 0 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 82:{
            sz = 15;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0, 1 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1, 2 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1, 0 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 83:{
            sz = 15;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0, 1 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1, 3 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1, 0 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 84:{
            sz = 15;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0, 1 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1, 3 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1, 1 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 85:{
            sz = 15;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0, 1 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1, 3 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1, 2 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 86:{
            sz = 15;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0, 1 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1, 3 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1, 3 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 87:{
            sz = 16;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 7 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0, 1, 0 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1, 3, 1 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1, 3, 0 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 88:{
            sz = 16;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 7 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0, 1, 0 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1, 3, 1 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1, 3, 1 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 89:{
            sz = 17;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 0 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1, 3, 1, 1 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1, 3, 0, 1 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 90:{
            sz = 17;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 0 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1, 3, 2, 1 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1, 3, 0, 1 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 91:{
            sz = 18;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 7 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 0 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 2, 1, 3, 1, 1 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 0, 1, 3, 0, 1 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        case 92:{
            sz = 18;
            *nOcc = (int *)malloc(sz * sizeof(int));
            *lOcc = (int *)malloc(sz * sizeof(int));
            *fupOcc = (int *)malloc(sz * sizeof(int));
            *fdwOcc = (int *)malloc(sz * sizeof(int));
            int nOcc_val[] = { 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 7 };
            int lOcc_val[] = { 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 0 };
            int fupOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 3, 1, 3, 1, 1 };
            int fdwOcc_val[] = { 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 0, 1, 3, 0, 1 };

            for (int i = 0; i < sz; i++) {
                (*nOcc)[i] = nOcc_val[i];
                (*lOcc)[i] = lOcc_val[i];
                (*fupOcc)[i] = fupOcc_val[i];
                (*fdwOcc)[i] = fdwOcc_val[i];
            }
            break;}
        default:
            fprintf(stderr, "Atomic number Z = %d not supported.\n", Z);
            exit(EXIT_FAILURE);
        

    }
    *size_arr = sz;
}