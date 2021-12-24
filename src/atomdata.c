/**
 * @file    atomdata.c
 * @brief   This file contains the atom data information.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * @Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
 
#include "atomdata.h"
#include "isddft.h"
#include "tools.h"

/**
 * @brief   Find atomic mass info. for given element type.
 *
 * @ref     NIST (https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl?ele=&all=all&ascii=html)
 *
 * @param element   Element name. (OUTPUT)
 * @param atom_type User provided atom type name.
 */
void atomdata_mass(char *element, double *mass) {
    if (strcmpi(element,"H") == 0) {
        *mass = 1.007975; 
    } else if (strcmpi(element,"He") == 0) {
        *mass = 4.002602; // avg min&max
    } else if (strcmpi(element,"Li") == 0) {
        *mass = 6.9675; // avg min&max
    } else if (strcmpi(element,"Be") == 0) {
        *mass = 9.0121831;
    } else if (strcmpi(element,"B") == 0) {
        *mass = 10.8135; // avg min&max
    } else if (strcmpi(element,"C") == 0) {
        *mass = 12.0106; // avg min&max
    } else if (strcmpi(element,"N") == 0) {
        *mass = 14.006855; // avg min&max
    } else if (strcmpi(element,"O") == 0) {
        *mass = 15.9994; // avg min&max
    } else if (strcmpi(element,"F") == 0) {
        *mass = 18.998403163;
    } else if (strcmpi(element,"Ne") == 0) {
        *mass = 20.1797;
    } else if (strcmpi(element,"Na") == 0) {
        *mass = 22.98976928;
    } else if (strcmpi(element,"Mg") == 0) {
        *mass = 24.3055; // avg min&max
    } else if (strcmpi(element,"Al") == 0) {
        *mass = 26.9815385;
    } else if (strcmpi(element,"Si") == 0) {
        *mass = 28.085; // avg min&max
    } else if (strcmpi(element,"P") == 0) {
        *mass = 30.973761998;
    } else if (strcmpi(element,"S") == 0) {
        *mass = 32.0675; // avg min&max
    } else if (strcmpi(element,"Cl") == 0) {
        *mass = 35.4515; // avg min&max
    } else if (strcmpi(element,"Ar") == 0) {
        *mass = 39.948;
    } else if (strcmpi(element,"K") == 0) {
        *mass = 39.0983;
    } else if (strcmpi(element,"Ca") == 0) {
        *mass = 40.078;
    } else if (strcmpi(element,"Sc") == 0) {
        *mass = 44.955908;
    } else if (strcmpi(element,"Ti") == 0) {
        *mass = 47.867;
    } else if (strcmpi(element,"V") == 0) {
        *mass = 50.9415;
    } else if (strcmpi(element,"Cr") == 0) {
        *mass = 51.9961;
    } else if (strcmpi(element,"Mn") == 0) {
        *mass = 54.938044;
    } else if (strcmpi(element,"Fe") == 0) {
        *mass = 55.845;
    } else if (strcmpi(element,"Co") == 0) {
        *mass = 58.933194;
    } else if (strcmpi(element,"Ni") == 0) {
        *mass = 58.6934;
    } else if (strcmpi(element,"Cu") == 0) {
        *mass =  63.546; 
    } else if (strcmpi(element,"Zn") == 0) {
        *mass =  65.38; 
    } else if (strcmpi(element,"Ga") == 0) {
        *mass =  69.723; 
    } else if (strcmpi(element,"Ge") == 0) {
        *mass = 72.630; 
    } else if (strcmpi(element,"As") == 0) {
        *mass = 74.921595; 
    } else if (strcmpi(element,"Se") == 0) {
        *mass = 78.971; 
    } else if (strcmpi(element,"Br") == 0) {
        *mass = 79.904; // avg min&max
    } else if (strcmpi(element,"Kr") == 0) {
        *mass = 83.798; 
    } else if (strcmpi(element,"Rb") == 0) {
        *mass = 85.4678; 
    } else if (strcmpi(element,"Sr") == 0) {
        *mass = 87.62; 
    } else if (strcmpi(element,"Y") == 0) {
        *mass = 88.90584; 
    } else if (strcmpi(element,"Zr") == 0) {
        *mass = 91.224; 
    } else if (strcmpi(element,"Nb") == 0) {
        *mass = 92.90637; 
    } else if (strcmpi(element,"Mo") == 0) {
        *mass = 95.95; 
    } else if (strcmpi(element,"Tc") == 0) {
        *mass = 98; 
    } else if (strcmpi(element,"Ru") == 0) {
        *mass = 101.07; 
    } else if (strcmpi(element,"Rh") == 0) {
        *mass = 102.90550; 
    } else if (strcmpi(element,"Pd") == 0) {
        *mass = 106.42; 
    } else if (strcmpi(element,"Ag") == 0) {
        *mass = 107.8682; 
    } else if (strcmpi(element,"Cd") == 0) {
        *mass = 112.414; 
    } else if (strcmpi(element,"In") == 0) {
        *mass = 114.818; 
    } else if (strcmpi(element,"Sn") == 0) {
        *mass = 118.710; 
    } else if (strcmpi(element,"Sb") == 0) {
        *mass = 121.760; 
    } else if (strcmpi(element,"Te") == 0) {
        *mass = 127.60; 
    } else if (strcmpi(element,"I") == 0) {
        *mass = 126.90447; 
    } else if (strcmpi(element,"Xe") == 0) {
        *mass = 131.293; 
    } else if (strcmpi(element,"Cs") == 0) {
        *mass = 132.90545196; 
    } else if (strcmpi(element,"Ba") == 0) {
        *mass = 137.327; 
    } else if (strcmpi(element,"La") == 0) {
        *mass = 138.90547; 
    } else if (strcmpi(element,"Ce") == 0) {
        *mass = 140.116; 
    } else if (strcmpi(element,"Pr") == 0) {
        *mass = 140.90766; 
    } else if (strcmpi(element,"Nd") == 0) {
        *mass = 144.242; 
    } else if (strcmpi(element,"Pm") == 0) {
        *mass = 145; 
    } else if (strcmpi(element,"Sm") == 0) {
        *mass = 150.36; 
    } else if (strcmpi(element,"Eu") == 0) {
        *mass = 151.964; 
    } else if (strcmpi(element,"Gd") == 0) {
        *mass = 157.25; 
    } else if (strcmpi(element,"Tb") == 0) {
        *mass = 158.92535; 
    } else if (strcmpi(element,"Dy") == 0) {
        *mass = 162.500; 
    } else if (strcmpi(element,"Ho") == 0) {
        *mass = 164.93033; 
    } else if (strcmpi(element,"Er") == 0) {
        *mass = 167.259; 
    } else if (strcmpi(element,"Tm") == 0) {
        *mass = 168.93422; 
    } else if (strcmpi(element,"Yb") == 0) {
        *mass = 173.054; 
    } else if (strcmpi(element,"Lu") == 0) {
        *mass = 174.9668; 
    } else if (strcmpi(element,"Hf") == 0) {
        *mass = 178.49; 
    } else if (strcmpi(element,"Ta") == 0) {
        *mass = 180.94788; 
    } else if (strcmpi(element,"W") == 0) {
        *mass = 183.84; 
    } else if (strcmpi(element,"Re") == 0) {
        *mass = 186.207; 
    } else if (strcmpi(element,"Os") == 0) {
        *mass = 190.23; 
    } else if (strcmpi(element,"Ir") == 0) {
        *mass = 192.217; 
    } else if (strcmpi(element,"Pt") == 0) {
        *mass = 195.084; 
    } else if (strcmpi(element,"Au") == 0) {
        *mass = 196.966569; 
    } else if (strcmpi(element,"Hg") == 0) {
        *mass = 200.592; 
    } else if (strcmpi(element,"Tl") == 0) {
        *mass = 204.3835; // avg min&max
    } else if (strcmpi(element,"Pb") == 0) {
        *mass = 207.2; 
    } else if (strcmpi(element,"Bi") == 0) {
        *mass = 208.98040; 
    } else if (strcmpi(element,"Po") == 0) {
        *mass = 209; 
    } else if (strcmpi(element,"At") == 0) {
        *mass = 210; 
    } else if (strcmpi(element,"Rn") == 0) {
        *mass = 222; 
    } else if (strcmpi(element,"Fr") == 0) {
        *mass = 223; 
    } else if (strcmpi(element,"Ra") == 0) {
        *mass = 226; 
    } else if (strcmpi(element,"Ac") == 0) {
        *mass = 227; 
    } else if (strcmpi(element,"Th") == 0) {
        *mass = 232.0377; 
    } else if (strcmpi(element,"Pa") == 0) {
        *mass = 231.03588; 
    } else if (strcmpi(element,"U") == 0) {
        *mass = 238.02891; 
    } else if (strcmpi(element,"Np") == 0) {
        *mass = 237; 
    } else if (strcmpi(element,"Pu") == 0) {
        *mass = 244; 
    } else if (strcmpi(element,"Am") == 0) {
        //*mass = ; 
        printf("No atomic data for element %s, please provide in input file!\n", element);
    } else if (strcmpi(element,"Cm") == 0) {
        //*mass = ; 
        printf("No atomic data for element %s, please provide in input file!\n", element);
    } else if (strcmpi(element,"Bk") == 0) {
        //*mass = ; 
        printf("No atomic data for element %s, please provide in input file!\n", element);
    } else if (strcmpi(element,"Cf") == 0) {
        //*mass = ; 
        printf("No atomic data for element %s, please provide in input file!\n", element);
    } else if (strcmpi(element,"Es") == 0) {
        //*mass = ; 
        printf("No atomic data for element %s, please provide in input file!\n", element);
    } else if (strcmpi(element,"Fm") == 0) {
        //*mass = ; 
        printf("No atomic data for element %s, please provide in input file!\n", element);
    } else if (strcmpi(element,"Md") == 0) {
        //*mass = ; 
        printf("No atomic data for element %s, please provide in input file!\n", element);
    } else if (strcmpi(element,"No") == 0) {
        //*mass = ; 
        printf("No atomic data for element %s, please provide in input file!\n", element);
    } else if (strcmpi(element,"Lw") == 0 || strcmpi(element,"Lr") == 0) {
        //*mass = ; 
        printf("No atomic data for element %s, please provide in input file!\n", element);
    } else {
        printf("No atomic data for element %s, please provide in input file!\n", element);
    }
}


/**
 * @brief   Find atomic number info. for given element type.
 *
 *
 * @param element   Element name. (OUTPUT)
 * @param atom_type User provided atom type name.
 */
void atomdata_number(char *element, int *Z) {
    if (strcmpi(element,"H") == 0) {
        *Z = 1; 
    } else if (strcmpi(element,"He") == 0) {
        *Z = 2; // avg min&max
    } else if (strcmpi(element,"Li") == 0) {
        *Z = 3; // avg min&max
    } else if (strcmpi(element,"Be") == 0) {
        *Z = 4;
    } else if (strcmpi(element,"B") == 0) {
        *Z = 5; // avg min&max
    } else if (strcmpi(element,"C") == 0) {
        *Z = 6; // avg min&max
    } else if (strcmpi(element,"N") == 0) {
        *Z = 7; // avg min&max
    } else if (strcmpi(element,"O") == 0) {
        *Z = 8; // avg min&max
    } else if (strcmpi(element,"F") == 0) {
        *Z = 9;
    } else if (strcmpi(element,"Ne") == 0) {
        *Z = 10;
    } else if (strcmpi(element,"Na") == 0) {
        *Z = 11;
    } else if (strcmpi(element,"Mg") == 0) {
        *Z = 12; // avg min&max
    } else if (strcmpi(element,"Al") == 0) {
        *Z = 13;
    } else if (strcmpi(element,"Si") == 0) {
        *Z = 14; // avg min&max
    } else if (strcmpi(element,"P") == 0) {
        *Z = 15;
    } else if (strcmpi(element,"S") == 0) {
        *Z = 16; // avg min&max
    } else if (strcmpi(element,"Cl") == 0) {
        *Z = 17; // avg min&max
    } else if (strcmpi(element,"Ar") == 0) {
        *Z = 18;
    } else if (strcmpi(element,"K") == 0) {
        *Z = 19;
    } else if (strcmpi(element,"Ca") == 0) {
        *Z = 20;
    } else if (strcmpi(element,"Sc") == 0) {
        *Z = 21;
    } else if (strcmpi(element,"Ti") == 0) {
        *Z = 22;
    } else if (strcmpi(element,"V") == 0) {
        *Z = 23;
    } else if (strcmpi(element,"Cr") == 0) {
        *Z = 24;
    } else if (strcmpi(element,"Mn") == 0) {
        *Z = 25;
    } else if (strcmpi(element,"Fe") == 0) {
        *Z = 26;
    } else if (strcmpi(element,"Co") == 0) {
        *Z = 27;
    } else if (strcmpi(element,"Ni") == 0) {
        *Z = 28;
    } else if (strcmpi(element,"Cu") == 0) {
        *Z =  29; 
    } else if (strcmpi(element,"Zn") == 0) {
        *Z =  30; 
    } else if (strcmpi(element,"Ga") == 0) {
        *Z =  31; 
    } else if (strcmpi(element,"Ge") == 0) {
        *Z = 32; 
    } else if (strcmpi(element,"As") == 0) {
        *Z = 33; 
    } else if (strcmpi(element,"Se") == 0) {
        *Z = 34; 
    } else if (strcmpi(element,"Br") == 0) {
        *Z = 35; // avg min&max
    } else if (strcmpi(element,"Kr") == 0) {
        *Z = 36; 
    } else if (strcmpi(element,"Rb") == 0) {
        *Z = 37; 
    } else if (strcmpi(element,"Sr") == 0) {
        *Z = 38; 
    } else if (strcmpi(element,"Y") == 0) {
        *Z = 39; 
    } else if (strcmpi(element,"Zr") == 0) {
        *Z = 40; 
    } else if (strcmpi(element,"Nb") == 0) {
        *Z = 41; 
    } else if (strcmpi(element,"Mo") == 0) {
        *Z = 42; 
    } else if (strcmpi(element,"Tc") == 0) {
        *Z = 43; 
    } else if (strcmpi(element,"Ru") == 0) {
        *Z = 44; 
    } else if (strcmpi(element,"Rh") == 0) {
        *Z = 45; 
    } else if (strcmpi(element,"Pd") == 0) {
        *Z = 46; 
    } else if (strcmpi(element,"Ag") == 0) {
        *Z = 47; 
    } else if (strcmpi(element,"Cd") == 0) {
        *Z = 48; 
    } else if (strcmpi(element,"In") == 0) {
        *Z = 49; 
    } else if (strcmpi(element,"Sn") == 0) {
        *Z = 50; 
    } else if (strcmpi(element,"Sb") == 0) {
        *Z = 51; 
    } else if (strcmpi(element,"Te") == 0) {
        *Z = 52; 
    } else if (strcmpi(element,"I") == 0) {
        *Z = 53; 
    } else if (strcmpi(element,"Xe") == 0) {
        *Z = 54; 
    } else if (strcmpi(element,"Cs") == 0) {
        *Z = 55; 
    } else if (strcmpi(element,"Ba") == 0) {
        *Z = 56; 
    } else if (strcmpi(element,"La") == 0) {
        *Z = 57; 
    } else if (strcmpi(element,"Ce") == 0) {
        *Z = 58; 
    } else if (strcmpi(element,"Pr") == 0) {
        *Z = 59; 
    } else if (strcmpi(element,"Nd") == 0) {
        *Z = 60; 
    } else if (strcmpi(element,"Pm") == 0) {
        *Z = 61; 
    } else if (strcmpi(element,"Sm") == 0) {
        *Z = 62; 
    } else if (strcmpi(element,"Eu") == 0) {
        *Z = 63; 
    } else if (strcmpi(element,"Gd") == 0) {
        *Z = 64; 
    } else if (strcmpi(element,"Tb") == 0) {
        *Z = 65; 
    } else if (strcmpi(element,"Dy") == 0) {
        *Z = 66; 
    } else if (strcmpi(element,"Ho") == 0) {
        *Z = 67; 
    } else if (strcmpi(element,"Er") == 0) {
        *Z = 68; 
    } else if (strcmpi(element,"Tm") == 0) {
        *Z = 69; 
    } else if (strcmpi(element,"Yb") == 0) {
        *Z = 70; 
    } else if (strcmpi(element,"Lu") == 0) {
        *Z = 71; 
    } else if (strcmpi(element,"Hf") == 0) {
        *Z = 72; 
    } else if (strcmpi(element,"Ta") == 0) {
        *Z = 73; 
    } else if (strcmpi(element,"W") == 0) {
        *Z = 74; 
    } else if (strcmpi(element,"Re") == 0) {
        *Z = 75; 
    } else if (strcmpi(element,"Os") == 0) {
        *Z = 76; 
    } else if (strcmpi(element,"Ir") == 0) {
        *Z = 77; 
    } else if (strcmpi(element,"Pt") == 0) {
        *Z = 78; 
    } else if (strcmpi(element,"Au") == 0) {
        *Z = 79; 
    } else if (strcmpi(element,"Hg") == 0) {
        *Z = 80; 
    } else if (strcmpi(element,"Tl") == 0) {
        *Z = 81; // avg min&max
    } else if (strcmpi(element,"Pb") == 0) {
        *Z = 82; 
    } else if (strcmpi(element,"Bi") == 0) {
        *Z = 83; 
    } else if (strcmpi(element,"Po") == 0) {
        *Z = 84; 
    } else if (strcmpi(element,"At") == 0) {
        *Z = 85; 
    } else if (strcmpi(element,"Rn") == 0) {
        *Z = 86; 
    } else if (strcmpi(element,"Fr") == 0) {
        *Z = 87; 
    } else if (strcmpi(element,"Ra") == 0) {
        *Z = 88; 
    } else if (strcmpi(element,"Ac") == 0) {
        *Z = 89; 
    } else if (strcmpi(element,"Th") == 0) {
        *Z = 90; 
    } else if (strcmpi(element,"Pa") == 0) {
        *Z = 91; 
    } else if (strcmpi(element,"U") == 0) {
        *Z = 92; 
    } else if (strcmpi(element,"Np") == 0) {
        *Z = 93; 
    } else if (strcmpi(element,"Pu") == 0) {
        *Z = 94; 
    } else if (strcmpi(element,"Am") == 0) {
        //*Z = ; 
        printf("No atomic data for element %s, please provide in input file!\n", element);
    } else if (strcmpi(element,"Cm") == 0) {
        //*Z = ; 
        printf("No atomic data for element %s, please provide in input file!\n", element);
    } else if (strcmpi(element,"Bk") == 0) {
        //*Z = ; 
        printf("No atomic data for element %s, please provide in input file!\n", element);
    } else if (strcmpi(element,"Cf") == 0) {
        //*Z = ; 
        printf("No atomic data for element %s, please provide in input file!\n", element);
    } else if (strcmpi(element,"Es") == 0) {
        //*Z = ; 
        printf("No atomic data for element %s, please provide in input file!\n", element);
    } else if (strcmpi(element,"Fm") == 0) {
        //*Z = ; 
        printf("No atomic data for element %s, please provide in input file!\n", element);
    } else if (strcmpi(element,"Md") == 0) {
        //*Z = ; 
        printf("No atomic data for element %s, please provide in input file!\n", element);
    } else if (strcmpi(element,"No") == 0) {
        //*Z = ; 
        printf("No atomic data for element %s, please provide in input file!\n", element);
    } else if (strcmpi(element,"Lw") == 0 || strcmpi(element,"Lr") == 0) {
        //*Z = ; 
        printf("No atomic data for element %s, please provide in input file!\n", element);
    } else {
        printf("No atomic data for element %s, please provide in input file!\n", element);
    }
}