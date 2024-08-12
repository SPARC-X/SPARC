#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <math.h>
#include <time.h>
#ifdef USE_MKL
    #define MKL_Complex16 double _Complex
    #include <mkl.h>
#else
    #include <cblas.h>
    #include <lapacke.h>
#endif

#include "tools_mlff.h"
#include "spherical_harmonics.h"
#include "soap_descriptor.h"
#include "mlff_types.h"
#include "covariance_matrix.h"
#include "sparsification.h"
#include "ddbp_tools.h"
#include "tools.h"
#include "regression.h"

#define au2GPa 29421.02648438959

void intialize_print_MLFF(MLFF_Obj *mlff_str, SPARC_OBJ *pSPARC){
    int nprocs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#ifdef DEBUG
    if (!rank)	
    	printf("intialize_print_MLFF called!\n");
#endif

	FILE *fptr;
    if (rank==0){
        fptr = fopen(mlff_str->ref_str_name,"w");
        if(fptr == NULL)
        {
            printf("Error opening the file!");   
            exit(1);             
        }
        fprintf(fptr, "SPARC MLFF on-the-fly trained\n");
        fprintf(fptr, "Structures:\n");
        fclose(fptr);
    }
}

void print_ref_atom_MLFF(MLFF_Obj *mlff_str){
    
#ifdef DEBUG
    printf("mlff_str->ref_atom_name: %s\n",mlff_str->ref_atom_name);
#endif
    FILE *fptr;
    fptr = fopen(mlff_str->ref_atom_name,"w");

    int *index_elem;
    int count;

    if(fptr == NULL)
    {
        printf("Error opening the file!");   
        exit(1);             
    }

    fprintf(fptr, "nelem: %d\n", mlff_str->nelem);
    for (int i=0; i < mlff_str->nelem; i++){
        fprintf(fptr,"el_ID: %d natom_ref: %d\n", i, mlff_str->natm_train_elemwise[i]);
    }


    for (int j=0; j < mlff_str->natm_train_total; j++){
        fprintf(fptr, "Atom_type: %d weight: %.15f\n", mlff_str->natm_typ_train[j], mlff_str->weights[j]);
        for (int k=0; k < mlff_str->size_X3; k++){
            fprintf(fptr,"%.15f ",mlff_str->X3_traindataset[j][k]);
        }
        if (mlff_str->size_X3 > 0) fprintf(fptr,"\n");
    }

   fclose(fptr);

}



void print_new_ref_structure_MLFF(MLFF_Obj *mlff_str, int nstr, NeighList *nlist, double *atompos, double Etot, double *force, double *stress){
    FILE *fptr;
	fptr = fopen(mlff_str->ref_str_name,"a");

    if(fptr == NULL)
    {
        printf("Error opening the file!");   
        exit(1);             
    }
    fprintf(fptr, "structure_no: %d\n", nstr);
    fprintf(fptr, "CELL: \n");
    fprintf(fptr, "%f %f %f\n", nlist->cell_len[0], nlist->cell_len[1], nlist->cell_len[2]);

    fprintf(fptr, "LatUVec: \n");
    fprintf(fptr, "%10.9f %10.9f %10.9f\n", nlist->LatUVec[0], nlist->LatUVec[1], nlist->LatUVec[2]);
    fprintf(fptr, "%10.9f %10.9f %10.9f\n", nlist->LatUVec[3], nlist->LatUVec[4], nlist->LatUVec[5]);
    fprintf(fptr, "%10.9f %10.9f %10.9f\n", nlist->LatUVec[6], nlist->LatUVec[7], nlist->LatUVec[8]);


    fprintf(fptr,"natom: \n");
    fprintf(fptr, "%d\n", nlist->natom);
    fprintf(fptr,"natom_elem:\n");
    for (int i=0; i < nlist->nelem; i++){
    	fprintf(fptr,"%d\n", nlist->natom_elem[i]);
    }

	fprintf(fptr,"Atom_positions: \n");
    for (int i=0; i < nlist->natom; i++){
    	fprintf(fptr, "%10.9f %10.9f %10.9f\n",atompos[3*i], atompos[3*i+1], atompos[3*i+2]);
    }

    fprintf(fptr, "Etot(Ha):\n");
    fprintf(fptr, "%10.9f\n", Etot);
    fprintf(fptr, "F(Ha/bohr):\n");
    for (int i=0; i < nlist->natom; i++){
    	fprintf(fptr, "%10.9f %10.9f %10.9f\n", force[3*i], force[3*i+1], force[3*i+2]);
    }

    fprintf(fptr, "Stress(GPa)\n");
    fprintf(fptr,"%10.9f %10.9f %10.9f\n", au2GPa*stress[0], au2GPa*stress[1], au2GPa*stress[2]);
    fprintf(fptr,"%10.9f %10.9f %10.9f\n", au2GPa*stress[1], au2GPa*stress[3], au2GPa*stress[4]);
    fprintf(fptr,"%10.9f %10.9f %10.9f\n", au2GPa*stress[2], au2GPa*stress[4], au2GPa*stress[5]);
    fclose(fptr);
}


void print_restart_MLFF(MLFF_Obj *mlff_str){
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int nrows_total;

    MPI_Reduce(&mlff_str->n_rows, &nrows_total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

	FILE *fptr;
    // char str1[512] = "MLFF_RESTART.txt"; 
    // strcpy(mlff_str->restart_name, str1);

    if (rank==0){
        fptr = fopen(mlff_str->restart_name,"w");
        if(fptr == NULL){
            printf("Error opening the file!");   
            exit(1);             
        }
        fprintf(fptr, "SPARC MLFF on-the-fly trained\n");
        fprintf(fptr, "nelem:\n");
        fprintf(fptr, "%d\n",mlff_str->nelem);
        fprintf(fptr, "Znucl:\n");
        for (int i=0; i < mlff_str->nelem; i++)
            fprintf(fptr, "%d\n", mlff_str->Znucl[i]);
        // fprintf(fptr, "beta_2:\n");
        // fprintf(fptr, "%10.9f\n", mlff_str->beta_2);
        // fprintf(fptr, "beta_3:\n");
        // fprintf(fptr, "%10.9f\n", mlff_str->beta_3);
        fprintf(fptr, "xi_3:\n");
        fprintf(fptr, "%10.9f\n", mlff_str->xi_3);
        fprintf(fptr, "N_max:\n");
        fprintf(fptr, "%d\n", mlff_str->Nmax);
        if (mlff_str->descriptor_typ == 2) {
            fprintf(fptr, "Sigmas:\n");
            for (int i = 0; i < mlff_str->Nmax; i++) {
                fprintf(fptr, "%10.9f ", mlff_str->sigmas[i]);
            }
            fprintf(fptr, "\n");
        }
        fprintf(fptr, "L_max:\n");
        fprintf(fptr, "%d\n", mlff_str->Lmax);
        fprintf(fptr, "Rcut:\n");
        fprintf(fptr, "%10.9f\n", mlff_str->rcut);
        fprintf(fptr, "size_X3:\n");
        fprintf(fptr, "%d\n", mlff_str->size_X3);
        fprintf(fptr, "stress_len:\n");
        fprintf(fptr, "%d\n", mlff_str->stress_len);
        fprintf(fptr, "mu_E:\n");
        fprintf(fptr, "%10.9f\n", mlff_str->mu_E);
        fprintf(fptr, "std_E:\n");
        fprintf(fptr, "%10.9f\n", mlff_str->std_E);
        fprintf(fptr, "std_F:\n");
        fprintf(fptr, "%10.9f\n", mlff_str->std_F);
        fprintf(fptr, "std_Stress:\n");
        for (int i = 0; i < mlff_str->stress_len; i++){
             fprintf(fptr, "%10.9f ",  mlff_str->std_stress[i]);
        }
        fprintf(fptr, "\n");

        fprintf(fptr, "E_scale:\n");
        fprintf(fptr, "%10.9f\n", mlff_str->E_scale);
        fprintf(fptr, "F_scale:\n");
        fprintf(fptr, "%10.9f\n", mlff_str->F_scale);

        fprintf(fptr, "stress_scale:\n");
        for (int i = 0; i < mlff_str->stress_len; i++){
             fprintf(fptr, "%10.9f ",  mlff_str->stress_scale[i]);
        }
        fprintf(fptr, "\n");

        fprintf(fptr, "N_ref_str: \n");
        fprintf(fptr, "%d\n", mlff_str->n_str);

        fprintf(fptr, "N_ref_atoms_elemwise: \n");
        for (int i=0; i < mlff_str->nelem; i++){
            fprintf(fptr, "%d\n", mlff_str->natm_train_elemwise[i]);
        }
        fprintf(fptr, "n_rows: \n");
        fprintf(fptr, "%d\n", nrows_total);
        fprintf(fptr, "n_cols: \n");
        fprintf(fptr, "%d\n", mlff_str->n_cols);

        fprintf(fptr, "weights_regression:\n");
        for (int i = 0; i < mlff_str->n_cols; i++){
            fprintf(fptr, "%10.9f\n", mlff_str->weights[i]);
        }
        fclose(fptr);
    }
	
   
  int print_K_train =0;
  if (print_K_train){
    if (!rank){
        fptr = fopen("K_train.txt","w");
        fclose(fptr);
      }
      MPI_Barrier(MPI_COMM_WORLD);  
      int temp_index;
      for (int s = 0; s < mlff_str->n_str; s++){
        for (int r= 0 ; r < nprocs; r++){
          if (rank==r){
                fptr = fopen("K_train.txt","a");
                if (!rank)
                  temp_index = 1+3*mlff_str->natom_domain;
                else
                  temp_index = 3*mlff_str->natom_domain;

                for (int i = 0; i < temp_index; i++){
                  for (int j=0; j< mlff_str->n_cols; j++){
                     fprintf(fptr, "%10.9f ",mlff_str->K_train[s*temp_index+i][j]);
                  }
                  fprintf(fptr, "\n");
                }
                fclose(fptr);
              }
              MPI_Barrier(MPI_COMM_WORLD);
        }
      }
  }

  int print_bvec_train =0;
  if (print_bvec_train){
    if (!rank){
        fptr = fopen("bvec.txt","w");
        fclose(fptr);
      }
      MPI_Barrier(MPI_COMM_WORLD);  
      int temp1;
      for (int s = 0; s < mlff_str->n_str; s++){
        for (int r= 0 ; r < nprocs; r++){
          if (rank==r){
                fptr = fopen("bvec.txt","a");
                if (!rank)
                  temp1 = 1+3*mlff_str->natom_domain;
                else
                  temp1 = 3*mlff_str->natom_domain;

                for (int i = 0; i < temp1; i++){
                     fprintf(fptr, "%10.9f\n",mlff_str->b_no_norm[s*temp1+i]);
                }
                fclose(fptr);
              }
              MPI_Barrier(MPI_COMM_WORLD);
        }
      }
  }
  

}



void read_MLFF_files(MLFF_Obj *mlff_str, SPARC_OBJ *pSPARC){
    FILE *fptr;
    int info;
    char a1[512], str[512];

    // char str1[512] = "MLFF_data_reference_atoms.txt"; 
    // strcpy(mlff_str->ref_atom_name, str1);

    // char str2[512] = "MLFF_data_reference_structures.txt"; 
    // strcpy(mlff_str->ref_str_name, str2);

    // char str3[512] = "MLFF_RESTART.txt"; 
    // strcpy(mlff_str->restart_name, str3);

  
    // mlff_str->restart_name = "MLFF_RESTART.txt";
    fptr = fopen(mlff_str->restart_name,"r");

    if(fptr == NULL)
    {
      printf("Error opening the file!");   
      exit(1);             
    }

    fgets(a1, sizeof (a1), fptr);

    int nelem;
    int Znucl[pSPARC->Ntypes];
    double K_elem;
   
    while (!feof(fptr)){
        fscanf(fptr,"%s",str);
        fscanf(fptr, "%*[^\n]\n");
        // enable commenting with '#'
        if (str[0] == '#' || str[0] == '\n'|| strcmpi(str,"undefined") == 0) {
        fscanf(fptr, "%*[^\n]\n"); // skip current line
        continue;
        }
        if (strcmpi(str,"nelem:") == 0) {
        fscanf(fptr,"%d", &nelem);
        if (nelem != pSPARC->Ntypes){
            printf("Number of element types in MLFF file and the input file for DFT are not the same!");
            exit(1); 
        }
        fscanf(fptr, "%*[^\n]\n");
        } else if (strcmpi(str,"Znucl:") == 0){
        for (int i=0; i < pSPARC->Ntypes; i++){
            fscanf(fptr,"%d", &Znucl[i]);
            if (Znucl[i] != pSPARC->Znucl[i]){
                printf("Atomic number of the element types in MLFF file and the input file for DFT are not the same!");
                //    exit(1);
            }
            fscanf(fptr, "%*[^\n]\n");
        }
        // } else if (strcmpi(str,"beta_2:") == 0){
        //     fscanf(fptr,"%lf", &mlff_str->beta_2);
        //     fscanf(fptr, "%*[^\n]\n");
        // } else if (strcmpi(str,"beta_3:") == 0){
        //     fscanf(fptr,"%lf", &mlff_str->beta_3);
        //     fscanf(fptr, "%*[^\n]\n");
        } else if (strcmpi(str,"xi_3:") == 0){
            fscanf(fptr,"%lf", &mlff_str->xi_3);
            fscanf(fptr, "%*[^\n]\n");
        } else if (strcmpi(str,"N_max:") == 0){
            fscanf(fptr,"%d", &mlff_str->Nmax);
            fscanf(fptr, "%*[^\n]\n");
        } else if (strcmpi(str,"Sigmas:") == 0){
            for (int i = 0; i < mlff_str->Nmax; i++){
                fscanf(fptr,"%lf", &mlff_str->sigmas[i]);
            }
            fscanf(fptr, "%*[^\n]\n");
        } else if (strcmpi(str,"L_max:") == 0){
            fscanf(fptr,"%d", &mlff_str->Lmax);
            fscanf(fptr, "%*[^\n]\n");
        } else if (strcmpi(str,"Rcut:") == 0){
            fscanf(fptr,"%lf", &mlff_str->rcut);
            fscanf(fptr, "%*[^\n]\n");
        } else if (strcmpi(str,"size_X3:") == 0){
            fscanf(fptr,"%d", &mlff_str->size_X3);
            fscanf(fptr, "%*[^\n]\n");
        } else if (strcmpi(str,"stress_len:") == 0){
            fscanf(fptr,"%d", &mlff_str->stress_len);
            fscanf(fptr, "%*[^\n]\n");
        } else if (strcmpi(str,"mu_E:") == 0){
            fscanf(fptr,"%lf", &mlff_str->mu_E);
            fscanf(fptr, "%*[^\n]\n");
        } else if (strcmpi(str,"std_E:") == 0){
            fscanf(fptr,"%lf", &mlff_str->std_E);
            fscanf(fptr, "%*[^\n]\n");
        } else if (strcmpi(str,"std_F:") == 0){
            fscanf(fptr,"%lf", &mlff_str->std_F);
            fscanf(fptr, "%*[^\n]\n"); 
        } else if (strcmpi(str,"std_Stress:") == 0){
            for (int i=0; i < mlff_str->stress_len; i++){
                fscanf(fptr,"%lf", &mlff_str->std_stress[i]);   
            }   
            fscanf(fptr, "%*[^\n]\n");
        } else if (strcmpi(str,"E_scale:") == 0){
            fscanf(fptr,"%lf", &mlff_str->E_scale);
            fscanf(fptr, "%*[^\n]\n");
        } else if (strcmpi(str,"F_scale:") == 0){
            fscanf(fptr,"%lf", &mlff_str->F_scale);
            fscanf(fptr, "%*[^\n]\n");
        } else if (strcmpi(str,"stress_scale:") == 0){
            for (int i=0; i < mlff_str->stress_len; i++){
            fscanf(fptr,"%lf", &mlff_str->stress_scale[i]);
            }
            fscanf(fptr, "%*[^\n]\n");
        } else if (strcmpi(str,"N_ref_str:") == 0){
            fscanf(fptr,"%d", &mlff_str->n_str);
            fscanf(fptr, "%*[^\n]\n");
        } else if (strcmpi(str,"N_ref_atoms_elemwise:") == 0){
            // mlff_str->natm_train_elemwise = (int *)malloc(sizeof(int)*pSPARC->Ntypes);
            int natm_train_total=0;
            mlff_str->natm_train_total = 0;
            for (int i = 0; i < pSPARC->Ntypes; i++){
                fscanf(fptr,"%d", &mlff_str->natm_train_elemwise[i]);
                natm_train_total += mlff_str->natm_train_elemwise[i];
                fscanf(fptr, "%*[^\n]\n");
            }
            mlff_str->natm_train_total = natm_train_total;    
        } else if (strcmpi(str,"n_rows:") == 0){
            int n_rows;
            fscanf(fptr,"%d", &n_rows);
            mlff_str->n_rows = n_rows;
            fscanf(fptr, "%*[^\n]\n");
        } else if (strcmpi(str,"n_cols:") == 0){
            fscanf(fptr,"%d", &mlff_str->n_cols);
            fscanf(fptr, "%*[^\n]\n");
        } else if (strcmpi(str,"weights_regression:") == 0){
            for (int i = 0; i < mlff_str->n_cols; i++){
                fscanf(fptr,"%lf", &K_elem);
                mlff_str->weights[i] = K_elem;
                fscanf(fptr, "%*[^\n]\n");
            }
        } else {
            printf("\nCannot recognize input variable identifier: \"%s\"\n",str);
            exit(EXIT_FAILURE);
        }
    }
    fclose(fptr);

    fptr = fopen(mlff_str->ref_atom_name,"r");
    if(fptr == NULL)
    {
        printf("Error opening the file!");   
        exit(1);             
    }
    fgets(a1, sizeof (a1), fptr);
    for (int i=0; i < pSPARC->Ntypes; i++){
        fgets(a1, sizeof (a1), fptr);
    }

    int atm_typ, nimg;
    int count=0;
    int img_no;
    double wt_temp;
    int temp_int;
    while (!feof(fptr)){
        fgets(a1, sizeof(a1), fptr);
        sscanf(a1, "Atom_type: %d weight: %lf", &mlff_str->natm_typ_train[count], &mlff_str->weights[count]);

        for (int i=0; i < mlff_str->size_X3; i++){
            fscanf(fptr,"%lf", &mlff_str->X3_traindataset[count][i]);
        }
        if (mlff_str->size_X3 > 0) fscanf(fptr, "%*[^\n]\n");
        count++;
    }
    fclose(fptr);

    mlff_str->relative_scale_F = pSPARC->F_rel_scale;

}


void read_structures_MLFF_data(
    char *fname,
    int nstr,
    int nelem, 
    double **cell_data, 
    double **LatUVec_data,
    double **apos_data, 
    double *Etot_data, 
    double **F_data, 
    double **stress_data,
    int *natom_data, 
    int **natom_elem_data)
{  
   FILE *fptr;
   fptr = fopen(fname,"r");
   if(fptr == NULL)
   {
      printf("Error opening the file!");   
      exit(1);             
   }

   char a1[512], str[512];

   fgets(a1, sizeof (a1), fptr);
   fgets(a1, sizeof (a1), fptr);
   int str_no;
    while (!feof(fptr)){
        fscanf(fptr,"%s",str);

        fscanf(fptr,"%d", &str_no);
        str_no = str_no - 1;
        

        fscanf(fptr, "%*[^\n]\n");
        fscanf(fptr,"%s",str);
        fscanf(fptr,"%lf", &cell_data[str_no][0]);
        fscanf(fptr,"%lf", &cell_data[str_no][1]);
        fscanf(fptr,"%lf", &cell_data[str_no][2]);
        fscanf(fptr, "%*[^\n]\n");
        fscanf(fptr,"%s",str);
        for (int i = 0; i < 9; i++){
            fscanf(fptr,"%lf", &LatUVec_data[str_no][i]);
        }


        fscanf(fptr, "%*[^\n]\n");
        fscanf(fptr,"%s",str);
        fscanf(fptr,"%d", &natom_data[str_no]);

        apos_data[str_no] = (double *) malloc(sizeof(double)*3*natom_data[str_no]);
        F_data[str_no] = (double *) malloc(sizeof(double)*3*natom_data[str_no]);

        fscanf(fptr,"%s",str);
        for (int i=0; i < nelem; i++){
            fscanf(fptr,"%d", &natom_elem_data[str_no][i]);
        }

        fscanf(fptr, "%*[^\n]\n");
        fscanf(fptr,"%s",str);
        for (int i=0; i < natom_data[str_no]; i++){
            fscanf(fptr,"%lf", &apos_data[str_no][3*i]);
            fscanf(fptr,"%lf", &apos_data[str_no][3*i+1]);
            fscanf(fptr,"%lf", &apos_data[str_no][3*i+2]);
            fscanf(fptr, "%*[^\n]\n");
        }

        fscanf(fptr,"%s",str);
        fscanf(fptr,"%lf", &Etot_data[str_no]);
        fscanf(fptr,"%s",str);

        for (int i=0; i < natom_data[str_no]; i++){
            fscanf(fptr,"%lf", &F_data[str_no][3*i]);
            fscanf(fptr,"%lf", &F_data[str_no][3*i+1]);
            fscanf(fptr,"%lf", &F_data[str_no][3*i+2]);
            fscanf(fptr, "%*[^\n]\n");
        }
        fscanf(fptr,"%s",str);
        fscanf(fptr,"%lf", &stress_data[str_no][0]);
        fscanf(fptr,"%lf", &stress_data[str_no][1]);
        fscanf(fptr,"%lf", &stress_data[str_no][2]);
        fscanf(fptr, "%*[^\n]\n");

        fscanf(fptr,"%lf", &stress_data[str_no][1]);
        fscanf(fptr,"%lf", &stress_data[str_no][3]);
        fscanf(fptr,"%lf", &stress_data[str_no][4]);
        fscanf(fptr, "%*[^\n]\n");

        fscanf(fptr,"%lf", &stress_data[str_no][2]);
        fscanf(fptr,"%lf", &stress_data[str_no][4]);
        fscanf(fptr,"%lf", &stress_data[str_no][5]);
        fscanf(fptr, "%*[^\n]\n");
        if (str_no == nstr-1){
            break;
        }
    } 
}
