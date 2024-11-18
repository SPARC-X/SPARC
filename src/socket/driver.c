/**
 * @file driver.c
 * @brief Driver mode for SPARC (socket communication)
 *        This file contains both the socket communication methods and a socket-aware main loop
 * @authors T.Tian <alchem0x2a@gmail.com;tian.tian@gatech.edu>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <time.h>
#include <mpi.h>
#include <math.h>
#include "driver.h"
#include "isddft.h"
#include "libinetsocket.h"
#include "libunixsocket.h"

#include "initialization.h"
/* #include "orbitalElecDensInit.h" */
#include "electronicGroundState.h"
#include "relax.h"
/* #include "md.h" */

/* Tools to be used in socket interface. Don't need #ifdef switches */

int split_socket_name(const char *str, char *host, int *port, int *inet)
{
  /**
   * @brief split socket name by delimiter ":"
   *        examples:
   *        "localhost:1234" --> host = "localhost", port = 1234, inet = 1
   *        "unix_socket:UNIX" --> host = "unix_socket", port = 0, inet = 0
   *        ":1234"       --> host = "localhost", port = 1234, inet = 1
   *        "unix_socket:"    --> invalid
   *        "localhost:" --> invalid
   * "unix_socket:UNIX"
   * @param str: socket name
   * @param host: host name
   * @param port: port number
   * @param inet: 1 for inet socket, 0 for unix socket
   **/
  const char *delim_pos = strchr(str, ':');
  if (delim_pos == NULL || strchr(delim_pos + 1, ':') != NULL)
    {
      printf("Error: string must have format host:port or unix_socket:UNIX\n");
      return 1;
    }

  int delim_index = delim_pos - str;

  if (delim_index > 0)
    {
      strncpy(host, str, delim_index);
      host[delim_index] = '\0';
    }
  else
    {
      host[0] = '\0'; // Empty string
    }
  /* If the length of host is zero, then use localhost instead */
  if (strlen(host) == 0)
    {
      strcpy(host, "localhost");
    }

  const char *port_str = &str[delim_index + 1];
  char *endptr;
  long int value = strtol(port_str, &endptr, 10);
  if (*endptr == '\0')
    {
      // The port string was successfully converted to an integer
      // value == 0 may come from "localhost:0" or empty string,
      // both are not allowed
      if (value > 0 && value <= 65535)
        {
	  *port = (int)value;
        }
      else
        {
	  printf("Error: port number must be a positive integer no larger than 65535\n");
	  return 1;
        }
    }
  else if (strcasecmp(port_str, "UNIX") == 0)
    {
      // The port string is "UNIX" (case insensitive)
      *port = 0;
    }
  else
    {
      // The port string is neither a valid integer nor "UNIX"
      printf("Error: port must be an integer or 'UNIX'\n");
      return 1;
    }

  if (*port == 0)
    {
      *inet = 0;
    }
  else
    {
      *inet = 1;
    }

  return 0;
}

int initialize_Socket(SPARC_OBJ *pSPARC)
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (pSPARC->SocketFlag != 1)
    {
      if (rank == 0)
        {
	  printf("You're using initializing the socket with SocketFlag != 0! Exit");
        }
      exit(EXIT_FAILURE);
    }
  // Initialize the flags compatible with SocketFlag
  // TODO: maybe we should move to another location, like copy_sparc_input
  pSPARC->MDFlag = 0;
  pSPARC->RelaxFlag = 0;
  pSPARC->PrintAtomPosFlag = 1;
  pSPARC->PrintForceFlag = 1;
  pSPARC->SocketSCFCount = 0;

#ifdef DEBUG
  if (rank == 0){
    printf("##########################Initializing socket##########################\n");
    printf("SPARC socket_inet type? %d\n", pSPARC->socket_inet);
  }
#endif
  // Create a fd according to the socket type
  int socket_fd = -1;
  // For some reason the inet socket may die out earlier. For first stage testing use unix socket only
  // TODO: should fix this later when server side code is done
  if (rank != 0)
    {
      return 0;
    }

  // Only init on rank 0
  if (pSPARC->socket_inet == 1)
    {
      // Convert port number to string
      char service[L_STRING];
      sprintf(service, "%d", pSPARC->socket_port);
#ifdef DEBUG
      printf("This is your service: %s\n", service);
      printf("This is your service: %s\n", pSPARC->socket_host);
#endif // DEBUG
      socket_fd = create_inet_stream_socket(pSPARC->socket_host, service, LIBSOCKET_IPv4, 0);
      if (socket_fd == -1)
        {
	  printf("Failed to create or communicate with the inet socket %s:%s, Exiting...\n",
		 pSPARC->socket_host, service);
	  exit(EXIT_FAILURE);
        }
      pSPARC->socket_fd = socket_fd;
    }
  else if (pSPARC->socket_inet == 0)
    {
      // Use unix socket
      // TODO: change the actual unix socket path under /tmp/
      // TODO: unix socket must have server side up, otherwise will die immediately
      socket_fd = create_unix_stream_socket(pSPARC->socket_host, 0);
      if (socket_fd == -1)
        {
	  printf("Failed to create or communicate with the unix socket %s, Exiting...\n",
		 pSPARC->socket_host);
	  exit(EXIT_FAILURE);
        }
      pSPARC->socket_fd = socket_fd;
    }
  else
    {
      printf("Incorrect socket type, exiting...\n");
      printf("The code is buggy! Exiting...\n");
      exit(EXIT_FAILURE);
    }
#ifdef DEBUG
  printf("Created socket fd %i\n", pSPARC->socket_fd);
#endif
  return 0;
}

/**
 * TODO: complete
 */
int close_Socket(SPARC_OBJ *pSPARC)
{
  int socket_fd = pSPARC->socket_fd;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0)
    {
      return 0;
    }

  // rank 0
  int close_res = close(socket_fd);
  if (close_res != 0)
    {
      perror("Failed to close socket!");
      exit(EXIT_FAILURE);
    }
  else
    {
#ifdef DEBUG
      printf("@Driver mode: closed socker %d.\n", pSPARC->socket_fd);
#endif
    }
  pSPARC->socket_fd = -1;
  return 0;
}

/**
 *@brief  General purpose read function. Will read until the fd until buffer len is filled
 **/
int readBuffer(int fd, void *data, int len)
{
  int n, nr;
  printf("Waiting to read.......\n");
  n = nr = read(fd, data, len);
  printf("After reading, there are n %d\n", n);
  while (nr > 0 && n < len)
    {
      printf("Read %d bytes", nr);
      nr = read(fd, data + n, len - n);
      n += nr;
    }
  if (n == 0)
    {
      perror("Error reading from socket: server has quit or connection broke");
      return -1;
    }
  return 0;
}

/**
 * TODO: complete
 */
// int writeBuffer(SPARC_OBJ *pSPARC, const char *data, int len){
//     return 0;
// }

/**
 * @brief  Write functions
           All writeBuffer_* functions are meant to be used on rank == 0!
 **/
int writeBuffer_double(SPARC_OBJ *pSPARC, const double data_d)
{
  int sockfd = pSPARC->socket_fd;
  int ret = write(sockfd, &data_d, sizeof(double));
  if (ret < 0)
    fprintf(stderr, "Failed to write to socket fd %d, exiting...\n", sockfd);
  return ret;
}

int writeBuffer_int(SPARC_OBJ *pSPARC, const int data_i)
{
  int sockfd = pSPARC->socket_fd;
  int ret = write(sockfd, &data_i, sizeof(int));
  if (ret < 0)
    fprintf(stderr, "Failed to write to socket fd %d, exiting...\n", sockfd);
  return ret;
}

int writeBuffer_char(SPARC_OBJ *pSPARC, const char c)
{
  int sockfd = pSPARC->socket_fd;
  int ret = write(sockfd, &c, sizeof(char));
  if (ret < 0)
    fprintf(stderr, "Failed to write to socket fd %d, exiting...\n", sockfd);
  return ret;
}

int writeBuffer_double_vec(SPARC_OBJ *pSPARC, const double *data_dv, int len)
{
  int sockfd = pSPARC->socket_fd;
  int ret = write(sockfd, data_dv, sizeof(double) * len);
  if (ret < 0)
    fprintf(stderr, "Failed to write to socket fd %d, exiting...\n", sockfd);
  return ret;
}

int writeBuffer_int_vec(SPARC_OBJ *pSPARC, const int *data_iv, int len)
{
  int sockfd = pSPARC->socket_fd;
  int ret = write(sockfd, data_iv, sizeof(int) * len);
  if (ret < 0)
    fprintf(stderr, "Failed to write to socket fd %d, exiting...\n", sockfd);
  return ret;
}

int writeBuffer_string(SPARC_OBJ *pSPARC, const char *str, int len)
{
  int sockfd = pSPARC->socket_fd;
  int ret = write(sockfd, str, sizeof(char) * len);
  if (ret < 0)
    fprintf(stderr, "Failed to write to socket fd %d, exiting...\n", sockfd);
  return ret;
}

/**
 * @brief  Read functions. All readBuffer_* functions are meant to be used
                           at rank == 0
 * **/

int readBuffer_double(SPARC_OBJ *pSPARC, double *data_d)
{
  int sockfd = pSPARC->socket_fd;
  int ret = readBuffer(sockfd, data_d, sizeof(double));
  return ret;
}

int readBuffer_int(SPARC_OBJ *pSPARC, int *data_i)
{
  int sockfd = pSPARC->socket_fd;
  int ret = readBuffer(sockfd, data_i, sizeof(int));
  return ret;
}

int readBuffer_char(SPARC_OBJ *pSPARC, char *c)
{
  int sockfd = pSPARC->socket_fd;
  int ret = readBuffer(sockfd, c, sizeof(char));
  return ret;
}

int readBuffer_double_vec(SPARC_OBJ *pSPARC, double *data_dv, int len)
{
  int sockfd = pSPARC->socket_fd;
  int ret = readBuffer(sockfd, data_dv, sizeof(double) * len);
  return ret;
}

int readBuffer_int_vec(SPARC_OBJ *pSPARC, int *data_iv, int len)
{
  int sockfd = pSPARC->socket_fd;
  int ret = readBuffer(sockfd, data_iv, sizeof(int) * len);
  return ret;
}

int readBuffer_string(SPARC_OBJ *pSPARC, char *str, int len)
{
  int sockfd = pSPARC->socket_fd;
  int ret = readBuffer(sockfd, str, sizeof(char) * len);
  return ret;
}

/**@brief Function to reinitialize the mesh for SPARC object after changing the lattice
          pSPARC->LatVec should already be assigned
 **/
void reinit_mesh(SPARC_OBJ *pSPARC){
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  double range_x, range_y, range_z;
  /* double old_range_x, old_range_y, old_range_z; */
  
  // TODO: should we disable latvec_scale in socket mode?
  range_x = sqrt(pSPARC->LatVec[0] * pSPARC->LatVec[0] + pSPARC->LatVec[1] * pSPARC->LatVec[1] + pSPARC->LatVec[2] * pSPARC->LatVec[2]);
  range_y = sqrt(pSPARC->LatVec[3] * pSPARC->LatVec[3] + pSPARC->LatVec[4] * pSPARC->LatVec[4] + pSPARC->LatVec[5] * pSPARC->LatVec[5]);
  range_z = sqrt(pSPARC->LatVec[6] * pSPARC->LatVec[6] + pSPARC->LatVec[7] * pSPARC->LatVec[7] + pSPARC->LatVec[8] * pSPARC->LatVec[8]);
  /* old_range_x = pSPARC->range_x; */
  pSPARC->range_x = range_x;
  /* old_range_y = pSPARC->range_y; */
  pSPARC->range_y = range_y;
  /* old_range_z = pSPARC->range_z; */
  pSPARC->range_z = range_z;

  // Update the lattice matrix information like Jacbdet, LatUVec, etc
  Cart2nonCart_transformMat(pSPARC);
  

  // Update finite difference delta
  pSPARC->delta_x = pSPARC->range_x/(pSPARC->numIntervals_x);
  pSPARC->delta_y = pSPARC->range_y/(pSPARC->numIntervals_y);
  pSPARC->delta_z = pSPARC->range_z/(pSPARC->numIntervals_z);
  pSPARC->dV = pSPARC->delta_x * pSPARC->delta_y * pSPARC->delta_z * pSPARC->Jacbdet;

  int FDn = pSPARC->order / 2;
  int p, i;
  // 1st derivative weights including mesh
  double dx_inv, dy_inv, dz_inv;
  dx_inv = 1.0 / pSPARC->delta_x;
  dy_inv = 1.0 / pSPARC->delta_y;
  dz_inv = 1.0 / pSPARC->delta_z;
  for (p = 1; p < FDn + 1; p++) {
    pSPARC->D1_stencil_coeffs_x[p] = pSPARC->FDweights_D1[p] * dx_inv;
    pSPARC->D1_stencil_coeffs_y[p] = pSPARC->FDweights_D1[p] * dy_inv;
    pSPARC->D1_stencil_coeffs_z[p] = pSPARC->FDweights_D1[p] * dz_inv;
  }

    
  // 2nd derivative weights including mesh
  double dx2_inv, dy2_inv, dz2_inv;
  dx2_inv = 1.0 / (pSPARC->delta_x * pSPARC->delta_x);
  dy2_inv = 1.0 / (pSPARC->delta_y * pSPARC->delta_y);
  dz2_inv = 1.0 / (pSPARC->delta_z * pSPARC->delta_z);

  // Stencil coefficients for mixed derivatives
  if (pSPARC->cell_typ == 0) {
    for (p = 0; p < FDn + 1; p++) {
      pSPARC->D2_stencil_coeffs_x[p] = pSPARC->FDweights_D2[p] * dx2_inv;
      pSPARC->D2_stencil_coeffs_y[p] = pSPARC->FDweights_D2[p] * dy2_inv;
      pSPARC->D2_stencil_coeffs_z[p] = pSPARC->FDweights_D2[p] * dz2_inv;
    }
  } else if (pSPARC->cell_typ > 10 && pSPARC->cell_typ < 20) {
    for (p = 0; p < FDn + 1; p++) {
      pSPARC->D2_stencil_coeffs_x[p] = pSPARC->lapcT[0] * pSPARC->FDweights_D2[p] * dx2_inv;
      pSPARC->D2_stencil_coeffs_y[p] = pSPARC->lapcT[4] * pSPARC->FDweights_D2[p] * dy2_inv;
      pSPARC->D2_stencil_coeffs_z[p] = pSPARC->lapcT[8] * pSPARC->FDweights_D2[p] * dz2_inv;
      pSPARC->D2_stencil_coeffs_xy[p] = 2 * pSPARC->lapcT[1] * pSPARC->FDweights_D1[p] * dx_inv; // 2*T_12 d/dx(df/dy)
      pSPARC->D2_stencil_coeffs_xz[p] = 2 * pSPARC->lapcT[2] * pSPARC->FDweights_D1[p] * dx_inv; // 2*T_13 d/dx(df/dz)
      pSPARC->D2_stencil_coeffs_yz[p] = 2 * pSPARC->lapcT[5] * pSPARC->FDweights_D1[p] * dy_inv; // 2*T_23 d/dy(df/dz)
      pSPARC->D1_stencil_coeffs_xy[p] = 2 * pSPARC->lapcT[1] * pSPARC->FDweights_D1[p] * dy_inv; // d/dx(2*T_12 df/dy) used in d/dx(2*T_12 df/dy + 2*T_13 df/dz)
      pSPARC->D1_stencil_coeffs_yx[p] = 2 * pSPARC->lapcT[1] * pSPARC->FDweights_D1[p] * dx_inv; // d/dy(2*T_12 df/dx) used in d/dy(2*T_12 df/dx + 2*T_23 df/dz)
      pSPARC->D1_stencil_coeffs_xz[p] = 2 * pSPARC->lapcT[2] * pSPARC->FDweights_D1[p] * dz_inv; // d/dx(2*T_13 df/dz) used in d/dx(2*T_12 df/dy + 2*T_13 df/dz)
      pSPARC->D1_stencil_coeffs_zx[p] = 2 * pSPARC->lapcT[2] * pSPARC->FDweights_D1[p] * dx_inv; // d/dz(2*T_13 df/dx) used in d/dz(2*T_13 df/dz + 2*T_23 df/dy)
      pSPARC->D1_stencil_coeffs_yz[p] = 2 * pSPARC->lapcT[5] * pSPARC->FDweights_D1[p] * dz_inv; // d/dy(2*T_23 df/dz) used in d/dy(2*T_12 df/dx + 2*T_23 df/dz)
      pSPARC->D1_stencil_coeffs_zy[p] = 2 * pSPARC->lapcT[5] * pSPARC->FDweights_D1[p] * dy_inv; // d/dz(2*T_23 df/dy) used in d/dz(2*T_12 df/dx + 2*T_23 df/dy)
    }
  }
    
  // maximum eigenvalue of -0.5 * Lap (only with periodic boundary conditions)
  if(pSPARC->cell_typ == 0) {
    pSPARC->MaxEigVal_mhalfLap = pSPARC->D2_stencil_coeffs_x[0]
      + pSPARC->D2_stencil_coeffs_y[0]
      + pSPARC->D2_stencil_coeffs_z[0];
    double scal_x, scal_y, scal_z;
    scal_x = (pSPARC->Nx - pSPARC->Nx % 2) / (double) pSPARC->Nx;
    scal_y = (pSPARC->Ny - pSPARC->Ny % 2) / (double) pSPARC->Ny;
    scal_z = (pSPARC->Nz - pSPARC->Nz % 2) / (double) pSPARC->Nz;
    for (int p = 1; p < FDn + 1; p++) {
      pSPARC->MaxEigVal_mhalfLap += 2.0 * (pSPARC->D2_stencil_coeffs_x[p] * cos(M_PI*p*scal_x)
					   + pSPARC->D2_stencil_coeffs_y[p] * cos(M_PI*p*scal_y)
					   + pSPARC->D2_stencil_coeffs_z[p] * cos(M_PI*p*scal_z));
    }
    pSPARC->MaxEigVal_mhalfLap *= -0.5;
  }

  double h_eff = 0.0;
  if (fabs(pSPARC->delta_x - pSPARC->delta_y) < 1E-12 &&
      fabs(pSPARC->delta_y - pSPARC->delta_z) < 1E-12) {
    h_eff = pSPARC->delta_x;
  } else {
    // find effective mesh s.t. it has same spectral width
    h_eff = sqrt(3.0 / (dx2_inv + dy2_inv + dz2_inv));
  }

  // find Chebyshev polynomial degree based on max eigenvalue (spectral width)
  if (pSPARC->ChebDegree < 0) {
    pSPARC->ChebDegree = Mesh2ChebDegree(h_eff);
  } else {
  }

  // default Kerker tolerance
  if (pSPARC->TOL_PRECOND < 0.0) { // kerker tol not provided by user
    pSPARC->TOL_PRECOND = (h_eff * h_eff) * 1e-3;
  }


  Calculate_kpoints(pSPARC);
  if (pSPARC->Nkpts >= 1 && pSPARC->kptcomm_index != -1) {
    Calculate_local_kpoints(pSPARC);
  }
  Calculate_PseudochargeCutoff(pSPARC);

  
  if (rank == 0){
    write_output_reinit(pSPARC);
  }
}

/** @brief Function to map the atomic coordinates to the domain within LatVec.
 *         will throw error if the positions are outside domain 
 **/
void map_atom_coord(SPARC_OBJ *pSPARC){
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   int i = 0;
  // map positions back into the domain if necessary, the function is taken from initialization.c
  if (pSPARC->BCx == 0) {
    double x;
    for (i = 0; i < pSPARC->n_atom; i++) {
      x = *(pSPARC->atom_pos+3*i);
      if (x < 0 || x > pSPARC->range_x)
	{
	  x = fmod(x,pSPARC->range_x);
	  *(pSPARC->atom_pos+3*i) = x + (x<0.0)*pSPARC->range_x;
	}
    }
  } else if (rank == 0) {
    // for Dirichlet BC, exit if atoms goes out of the domain
    double x;
    for (i = 0; i < pSPARC->n_atom; i++) {
      x = *(pSPARC->atom_pos+3*i);
      if (x < pSPARC->xin || x > pSPARC->range_x + pSPARC->xin)
	{
	  printf("\nERROR: position of atom # %d is out of the domain with Dirichlet BC!\n",i+1);
	  exit(EXIT_FAILURE);
	}
    }
  }

  if (pSPARC->BCy == 0) {
    double y;
    for (i = 0; i < pSPARC->n_atom; i++) {
      y = *(pSPARC->atom_pos+1+3*i);
      if (y < 0 || y > pSPARC->range_y)
	{
	  y = fmod(y,pSPARC->range_y);
	  *(pSPARC->atom_pos+1+3*i) = y + (y<0.0)*pSPARC->range_y;
	}
    }
  } else if (rank == 0) {
    // for Dirichlet BC, exit if atoms goes out of the domain
    double y;
    for (i = 0; i < pSPARC->n_atom; i++) {
      y = *(pSPARC->atom_pos+1+3*i);
      if (y < 0 || y > pSPARC->range_y)
	{
	  printf("\nERROR: position of atom # %d is out of the domain with Dirichlet BC!\n",i+1);
	  exit(EXIT_FAILURE);
	}
    }
  }

  if (pSPARC->BCz == 0) {
    double z;
    for (i = 0; i < pSPARC->n_atom; i++) {
      z = *(pSPARC->atom_pos+2+3*i);
      if (z < 0 || z > pSPARC->range_z)
	{
	  z = fmod(z,pSPARC->range_z);
	  *(pSPARC->atom_pos+2+3*i) = z + (z<0.0)*pSPARC->range_z;
	}
    }
  } else if (rank == 0){
    // for Dirichlet BC, exit if atoms goes out of the domain
    double z;
    for (i = 0; i < pSPARC->n_atom; i++) {
      z = *(pSPARC->atom_pos+2+3*i);
      if (z < 0 || z > pSPARC->range_z)
	{
	  printf("\nERROR: position of atom # %d is out of the domain with Dirichlet BC!\n",i+1);
	  exit(EXIT_FAILURE);
	}
    }
  }
}

/**
 * @brief  Function that reassign atomic positions / lattice vectors etc to SPARC_OBJ
 *         This function will be used at each initialization / single point loop
 *         This function is MPI safe
 *         TODO: check if need strict atoms size check
 * **/



void reassign_atoms_info(SPARC_OBJ *pSPARC, int natoms, double *atom_pos, double *lattice, double *reci_lattice)
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  // For now only support natoms == n_atom
  if (natoms != pSPARC->n_atom)
    {
      if (rank == 0)
	printf("SPARC's socket interface currently does not support changing number of atoms yet! Exit...\n");
      exit(EXIT_FAILURE);
    }
  
  // TODO: fix the cell type check


  pSPARC->atom_pos = (double *)malloc(sizeof(double) * 3 * natoms);
  memcpy(pSPARC->atom_pos, atom_pos, sizeof(double) * 3 * natoms);
  memcpy(pSPARC->LatVec, lattice, sizeof(double) * 9);
  map_atom_coord(pSPARC);
  reinit_mesh(pSPARC);
}

/**
 * @brief  Read message from sparc and return a status code
 *         Use status defined in socket.h.
           This is a function for ALL RANKS
 **/
int read_socket_header(SPARC_OBJ *pSPARC, int *status)
{
  int rank, size, rank0_finish;
  double sleep_interval = 0.1;	// Checking every 0.5 seconds
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  rank0_finish = 0;		// Helper for other ranks to check
  int ret = 0;
  char header[IPI_HEADERLEN];
  int sockfd = pSPARC->socket_fd;
  MPI_Request request;

  if (rank == 0)
    {
      ret = readBuffer_string(pSPARC, header, IPI_HEADERLEN);
      rank0_finish = 1;
#ifdef DEBUG
      printf("@Driver mode: get raw header %s\n", header);
#endif
      // Status code should broadcast to
      if (ret < 0)
        {
	  fprintf(stderr, "Failed to read from socket fd %d, exiting...\n", sockfd);
        }
      // Trim the string and compare with upper case letters
      else if (strncasecmp(header, "STATUS", strlen("STATUS")) == 0)
        {
	  *status = IPI_MSG_STATUS;
        }
      else if (strncasecmp(header, "POSDATA", strlen("POSDATA")) == 0)
        {
	  *status = IPI_MSG_POSDATA;
        }
      else if (strncasecmp(header, "GETFORCE", strlen("GETFORCE")) == 0)
        {
	  *status = IPI_MSG_GETFORCE;
        }
      /* Extra keywords for extended SPARC protocol */
      else if (strncasecmp(header, "SETPARAM", strlen("SETPARAM")) == 0)
        {
	  *status = SPARC_MSG_SETPARAM;
        }
      else
        {
	  *status = IPI_MSG_OTHER;
        }
      /* Notify other ranks */
      for (int i = 1; i < size; i++){
	MPI_Isend(&rank0_finish, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &request);
      }
    }
  else {
    MPI_Irecv(&rank0_finish, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &request);
    int ready = 0;
    while (rank0_finish == 0){
      MPI_Test(&request, &ready, MPI_STATUS_IGNORE);
      if (ready == 0)
	sleep(sleep_interval);
      else if (rank0_finish == 0){
	printf("Received rank0_finish %d at rank %d, this is a bug.\n", rank0_finish, rank);
	exit(1);
      }
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Bcast(&ret, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(status, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return ret;
}

/**
 * @brief Get positions from socket and reassign to SPARC_OBJ
 *        MPI broadcast is done in this function
          This is a function for ALL RANKS
 *
 **/
int read_atoms_position_fom_socket(SPARC_OBJ *pSPARC, int init)
{
  int rank;
  int ret = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  // TODO: check if status is in IPI_MSG_POSDATA
  int sockfd = pSPARC->socket_fd;
  int natoms;
  double ipi_cell[9], ipi_inv_cell[9];
  // The read process will need to get the forces, although we don't pass them back to SPARC
  double *ipi_atom_pos, *ipi_forces;
  // TODO: do read at rank0
  if (rank == 0){
    ret |= readBuffer_double_vec(pSPARC, ipi_cell, 9);
    ret |= readBuffer_double_vec(pSPARC, ipi_inv_cell, 9);
    ret |= readBuffer_int(pSPARC, &natoms);
  }
  MPI_Bcast(&ret, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (ret < 0){
    perror("Error receiving Cell and Natoms information from socket! Will exit");
    return ret;
  }
  // Bcast info to each rank
  MPI_Bcast(&natoms, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(ipi_cell, 9, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(ipi_inv_cell, 9, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // on each rank
  ipi_atom_pos = (double *)malloc(sizeof(double) * 3 * natoms);
  ipi_forces = (double *)malloc(sizeof(double) * 3 * natoms);

  // Read real positions, now rank 0
  if (rank == 0)
    {
      ret |= readBuffer_double_vec(pSPARC, ipi_atom_pos, 3 * natoms);
    }
  MPI_Bcast(&ret, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (ret < 0){
    perror("Error receiving Pos information from socket! Will exit");
    return ret;
  }
  
  MPI_Bcast(ipi_atom_pos, 3 * natoms, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  // TODO: make sure the values are broadcast
  // TODO: wrap cell for the positions
#ifdef DEBUG
  if (rank == 0)
    {
      printf("Starting socket communication\n");
      printf("Received from socket the following position data:\n");
      printf("natoms: %d\n", natoms);
      printf("cell: \t%f %f %f \n\t%f %f %f \n\t%f %f %f\n", ipi_cell[0], ipi_cell[1], ipi_cell[2], ipi_cell[3], ipi_cell[4], ipi_cell[5], ipi_cell[6], ipi_cell[7], ipi_cell[8]);
      printf("inverse cell \t%f %f %f \n\t%f %f %f \n\t%f %f %f\n", ipi_inv_cell[0], ipi_inv_cell[1], ipi_inv_cell[2], ipi_inv_cell[3], ipi_inv_cell[4], ipi_inv_cell[5], ipi_inv_cell[6], ipi_inv_cell[7], ipi_inv_cell[8]);
    }
#endif // DEBUG

  // This should work on each rank
  reassign_atoms_info(pSPARC, natoms, ipi_atom_pos, ipi_cell, ipi_inv_cell);

  free(ipi_atom_pos);
  free(ipi_forces);
  return ret;
}

/**
 * @brief Convert stress 6 vector to 9 matrix virial
 *        SPARC's stress information (in PBC) is [xx, xy, xz, yy, yz, zz] in Ha/Bohr^3, different from the common voigt stress
 *        Virial will be [xx, xy, xz, xy, yy, yz, xz, yz, zz] * -volume in Ha
 *        Cell volume is given by: pSPARC->Jacbdet * pSPARC->range_x * pSPARC->range_y * pSPARC->range_z
 **/
void stress_to_virial(SPARC_OBJ *pSPARC, double *virial)
{
  // TODO: may need to be a bit looseen
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  double volCell = pSPARC->Jacbdet * pSPARC->range_x * pSPARC->range_y * pSPARC->range_z;
  
  // This is the general form for the converted virial notation of the stress. For PBC (pSPARC->BC == 2) it's Ha/Bohr^3
  // In other BCs, we should do post processing
  double virial_calc[9] = {
    -pSPARC->stress[0] * volCell, -pSPARC->stress[1] * volCell, -pSPARC->stress[2] * volCell,
    -pSPARC->stress[1] * volCell, -pSPARC->stress[3] * volCell, -pSPARC->stress[4] * volCell,
    -pSPARC->stress[2] * volCell, -pSPARC->stress[4] * volCell, -pSPARC->stress[5] * volCell
  };

  // Stress not calculated, return zero-tensor
  if (pSPARC->Calc_stress == 0){
    if (rank == 0)
      printf("Stress is not calculated in this system. Will return zero-stress tensor to the socket server.\n");
    for (int i; i < 9; i++){
      virial_calc[i]= 0.0;
    }
  }
  // 0D case. make sure all components are 0
  else if (pSPARC->BC == 1){
    if (rank == 0)
      printf("You're simulating an isolated system, no stress will be calculated. Zero stress tensor will be send to the socket server\n");
    for (int i; i < 9; i++){
      virial_calc[i]= 0.0;
    }
  }
  // 2D stress cases
  else if (pSPARC->BC == 3){
    if (rank == 0)
      printf("The simulation has 2D stress tensor in Ha/Bohr**2. The equivalent value in periodic system will be returned!\n");
    // Periodic in xy-
    if (pSPARC->BCx == 0 && pSPARC->BCy == 0){
      for (int i; i < 9; i++){
	virial_calc[i] /= pSPARC->range_z;
      }}
    // Periodic in xz-
    else if (pSPARC->BCx == 0 && pSPARC->BCz == 0){
      for (int i; i < 9; i++){
	virial_calc[i] /= pSPARC->range_y;
      }}
    // Periodic in yz-
    else if (pSPARC->BCy == 0 && pSPARC->BCz == 0){
      for (int i; i < 9; i++){
	virial_calc[i] /= pSPARC->range_x;
      }
    }
  }
  // 1D stress cases
  else if (pSPARC->BC == 4){
    if (rank == 0)
      printf("The simulation has 1D stress tensor in Ha/Bohr. The equivalent value in periodic system will be returned!\n");
    // Periodic in x-
    if (pSPARC->BCx == 0){
      for (int i; i < 9; i++){
	virial_calc[i] /= (pSPARC->range_y * pSPARC->range_z);
      }
    }
    // Periodic in y-
    else if (pSPARC->BCy == 0){
      for (int i; i < 9; i++){
	virial_calc[i] /= (pSPARC->range_x * pSPARC->range_z);
      }
    }
    else if (pSPARC->BCx == 0){
      for (int i; i < 9; i++){
	virial_calc[i] /= (pSPARC->range_x * pSPARC->range_y);
      }
    }
  }
  /* Cyclic or helix systems, copy from stress.c */
  else if (pSPARC->BC >= 5 && pSPARC->BC <= 7){
    if (rank == 0)
      printf("The simulation has 1D stress tensor in Ha/Bohr (cyclic or helix boundaries). The equivalent value in periodic system will be returned!\n");
    for (int i; i < 9; i++){
      virial_calc[i] /= (M_PI * ( pow(pSPARC->xout,2.0) - pow(pSPARC->xin,2.0) ) ) ;
    }
  }
#ifdef DEBUG
  if (rank == 0)
    {
      printf("SPARC's electronic stress information is (Ha/Bohr^3): %f %f %f %f %f %f\n", pSPARC->stress[0], pSPARC->stress[1], pSPARC->stress[2], pSPARC->stress[3], pSPARC->stress[4], pSPARC->stress[5]);
      printf("Virial matrix is (Ha): \t%f %f %f \n\t%f %f %f \n\t%f %f %f\n", virial_calc[0], virial_calc[1], virial_calc[2], virial_calc[3], virial_calc[4], virial_calc[5], virial_calc[6], virial_calc[7], virial_calc[8]);
    }
#endif // DEBUG
  memcpy(virial, virial_calc, sizeof(double) * 9);
}

int write_forces_to_socket(SPARC_OBJ *pSPARC)
{
  // TODO: check if status is in IPI_MSG_POSDATA
  int rank;
  int ret = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int sockfd = pSPARC->socket_fd;
  int natoms = pSPARC->n_atom;
  double ipi_potential;
  double ipi_virial[9];
  // The read process will need to get the forces, although we don't pass them back to SPARC
  double ipi_atom_pos[3 * natoms], ipi_forces[3 * natoms];
  // Check if we need to substract entropy
  ipi_potential = pSPARC->Etot;
  memcpy(ipi_forces, pSPARC->forces, sizeof(double) * 3 * natoms);
  stress_to_virial(pSPARC, ipi_virial);
  /* If any return value is -1, it will make final ret 0 */
  ret &= (write_message_to_socket(pSPARC, "FORCEREADY") + 1);
  if (rank == 0){
    ret &= (writeBuffer_double(pSPARC, ipi_potential) + 1);
    ret &= (writeBuffer_int(pSPARC, natoms) + 1);
    ret &= (writeBuffer_double_vec(pSPARC, ipi_forces, 3 * natoms) + 1);
    ret &= (writeBuffer_double_vec(pSPARC, ipi_virial, 9) + 1);
    // No more message to send
    // SPARC serialization should be handled by the SPARC protocol in Python API
    ret &= (writeBuffer_int(pSPARC, 0) + 1);
  }
  ret -= 1;
  if ((ret < 0) && (rank == 0))
    perror("Error writing potential / forces / stress to socket server! Exit...\n");
  MPI_Bcast(&ret, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return ret;
}

/* Wrapper for writer message. This is a function for all ranks */
int write_message_to_socket(SPARC_OBJ *pSPARC, char *message)
{
  int rank;
  int ret = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int sockfd = pSPARC->socket_fd;
  char clean_msg[IPI_HEADERLEN + 1];
  // memset(clean_msg, ' ', IPI_HEADERLEN + 1);
  if (strlen(message) > IPI_HEADERLEN)
    {
      fprintf(stderr, "Message header %s is too long to be sent through socket! Please contact the developers.\n", message);
      exit(EXIT_FAILURE);
    }

  strncpy(clean_msg, message, strlen(message));
  memset(clean_msg + strlen(message), ' ', IPI_HEADERLEN - strlen(message));
  clean_msg[IPI_HEADERLEN] = '\0';
  if (rank == 0){
    ret = writeBuffer_string(pSPARC, clean_msg, IPI_HEADERLEN);
#ifdef DEBUG
    printf("@Driver mode: Sending message to socket: %s###\n", clean_msg);
#endif // DEBUG
  }
  MPI_Bcast(&ret, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return ret;
}

/**
 *
 *
 * @brief  Print the atomic positions and lattice to the static file
 *         this is an updated version from write_output_init.
 *         we skip the first step since it's already initialized
 */
void socket_static_print_atom_pos(SPARC_OBJ *pSPARC)
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank != 0)
    return;

#ifdef DEBUG
  printf("SocketSCFCOUNT is %d \n", pSPARC->SocketSCFCount);
#endif
  // Do not write header for socket mode at init stage
  if ((pSPARC->SocketSCFCount == 0) && (pSPARC->SocketFlag == 1))
    return;

  // SocketFlag = 0 but SocketSCFCount != 0 cannot happen!
  if ((pSPARC->SocketSCFCount != 0) && (pSPARC->SocketFlag == 0)){
    perror("Internal error! SocketFlag = 0 but there is a non-zero SocketSCFCount, a bug may exist in the code. Please contact the developer.");
    exit(-1);
  }
				       

  if ((pSPARC->PrintAtomPosFlag == 1 || pSPARC->PrintForceFlag == 1) && pSPARC->MDFlag == 0 && pSPARC->RelaxFlag == 0)
    {
      FILE *static_fp = fopen(pSPARC->StaticFilename, "a");
      if (static_fp == NULL)
        {
	  printf("\nCannot open file \"%s\"\n", pSPARC->StaticFilename);
	  exit(EXIT_FAILURE);
        }

      // print atoms in either old static fashion or socket fashion
      if (pSPARC->PrintAtomPosFlag == 1)
        { 
	  if (pSPARC->SocketFlag == 1){
	    fprintf(static_fp, "***************************************************************************\n");
	    fprintf(static_fp, "                        Atom positions (socket step %d)                    \n", pSPARC->SocketSCFCount);
	    fprintf(static_fp, "***************************************************************************\n");
	  }
	  else{
	    fprintf(static_fp, "***************************************************************************\n");
	    fprintf(static_fp,"                            Atom positions                                 \n");
	    fprintf(static_fp, "***************************************************************************\n");
	  }

	  int count = 0;
	  for (int i = 0; i < pSPARC->Ntypes; i++)
	    {
	      fprintf(static_fp, "Fractional coordinates of %s:\n", &pSPARC->atomType[L_ATMTYPE * i]);
	      for (int j = 0; j < pSPARC->nAtomv[i]; j++)
		{
		  fprintf(static_fp, "%18.10f %18.10f %18.10f\n",
			  pSPARC->atom_pos[3 * count] / pSPARC->range_x,
			  pSPARC->atom_pos[3 * count + 1] / pSPARC->range_y,
			  pSPARC->atom_pos[3 * count + 2] / pSPARC->range_z);
		  count++;
		}
	    }

	  // Step 2: print the LATVEC information (required for each)
	  if (pSPARC->SocketFlag == 1){
            fprintf(static_fp, "Lattice (Bohr):\n");
            fprintf(static_fp, "%18.10E %18.10E %18.10E \n", pSPARC->LatVec[0], pSPARC->LatVec[1], pSPARC->LatVec[2]);
            fprintf(static_fp, "%18.10E %18.10E %18.10E \n", pSPARC->LatVec[3], pSPARC->LatVec[4], pSPARC->LatVec[5]);
            fprintf(static_fp, "%18.10E %18.10E %18.10E \n", pSPARC->LatVec[6], pSPARC->LatVec[7], pSPARC->LatVec[8]);
	  }
        }
	
      fclose(static_fp);
    }
}

/* Print error information for premature socket closure to output file */
void print_socket_error_info(SPARC_OBJ *pSPARC)
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank != 0)
    return;

  FILE *output_fp = fopen(pSPARC->OutFilename,"w");
  if (output_fp == NULL) {
    printf(stderr, "\nCannot open file \"%s\"\n",pSPARC->OutFilename);
    exit(EXIT_FAILURE);
  }

  fprintf(output_fp, "***************************************************************************\n");
  fprintf(output_fp,"                              !!!WARNING!!!                              \n");
  fprintf(output_fp, "*            Communication between SPARC and the socket server            *\n");
  fprintf(output_fp, "*            was dropped prematurely. Some calculation results            *\n");
  fprintf(output_fp, "*              may be incomplete. Please check your results.              *\n");
  fprintf(output_fp, "***************************************************************************\n");
  fclose(output_fp);

}

/**
 Below are functions used for extended SPARC protocol controls
 Most of them are still in proof-of-concept stage, to showcase
 that message passing is doable
 **/

/**@brief Test function to set parameter in SPARC. For now, only demonstrate read and write are possible
 **/
int read_setparam(SPARC_OBJ *pSPARC)
{
  int rank;
  int ret = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  // TODO: check if status is in IPI_MSG_POSDATA
  int sockfd = pSPARC->socket_fd;
  int msglen = 0;
  char *msg;
  if (rank == 0)
    {
      printf("Receiving param name \n");
      readBuffer_int(pSPARC, &msglen);
      printf("Will receive a msg with length %d\n", msglen);
      msg = (char *)malloc(sizeof(char) * (msglen + 1));
      readBuffer_string(pSPARC, msg, msglen);
      msg[msglen] = '\0';
      printf("Received param %s\n", msg);
      free(msg);

      printf("Receiving param value \n");
      readBuffer_int(pSPARC, &msglen);
      printf("Will receive a msg with length %d\n", msglen);
      msg = (char *)malloc(sizeof(char) * (msglen + 1));
      readBuffer_string(pSPARC, msg, msglen);
      msg[msglen] = '\0';
      printf("Received value %s\n", msg);
      free(msg);
    }
  // TODO: when it's a real function, make return values correct
  return 0;
}

/** End SPARC protocol functions **/

/**
 *
 *
 * @brief  Main function to implement socket communication. It should be independent of other main_Functions
 */
void main_Socket(SPARC_OBJ *pSPARC)
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#ifdef DEBUG
  if (rank == 0)
    printf("Starting socket communication\n");
#endif // DEBUG
  int sfd = pSPARC->socket_fd;
  int status, init, hasdata, retcode, print_socket_err;
  status = -1; // -1: not initialized
  init = 1;
  hasdata = 0;
  print_socket_err = 0;
#ifdef DEBUG
  if (rank == 0)
    printf("Status: socket_max_niter %d\n", pSPARC->socket_max_niter);
#endif // DEBUG
  // TODO: option to specify N_MAXSTEPS (or directly taken from MD / relax?)
  while (pSPARC->SocketSCFCount <= pSPARC->socket_max_niter)
    {
      retcode = read_socket_header(pSPARC, &status);
      // We can safely close the socket in this case.
      if (retcode != 0) break;
      if (status == IPI_MSG_STATUS)
        {
	  // TODO: may want to implement the NEEDINIT method
	  if (hasdata == 1)
            {
	      retcode = write_message_to_socket(pSPARC, "HAVEDATA");
            }
	  else
            {
	      retcode = write_message_to_socket(pSPARC, "READY");
            }
	  // For write functions, retcode >= 0 is success
	  if (retcode < 0)
	    {
	      print_socket_err = 1;
	      break;
	    }
        }
      else if (status == IPI_MSG_NEEDINIT)
        {
	  if (rank == 0)
	    perror("NEEDINIT is not implemented yet!\n");
	  exit(EXIT_FAILURE);
        }
      else if (status == IPI_MSG_POSDATA)
        {
	  // Need to put the SocketSCFCount first due to print
	  pSPARC->SocketSCFCount++;
	  retcode = read_atoms_position_fom_socket(pSPARC, init);
	  if (retcode < 0)
	    {
	      print_socket_err = 1;
	      break;
	    }
	  socket_static_print_atom_pos(pSPARC);
	  if (init == 1)
            {
	      init = 0;
            }
	  // TODO: implement the electron density extrapolation method here
	  if (pSPARC->elecgs_Count > 0)
            {
	      // elecDensExtrapolation(pSPARC);
	      // Check_atomlocation(pSPARC);
            }
    Calculate_Properties(pSPARC);
	  //Calculate_electronicGroundState(pSPARC);
#ifdef DEBUG
	  if (rank == 0)
            {
	      printf("@Driver mode: single point #%d, Total energy: %f\n", pSPARC->elecgs_Count, pSPARC->Etot);
	      printf("@Driver mode: total energy in eV unit: %f\n", pSPARC->Etot * CONST_EH);
            }
#endif // DEBUG
	  pSPARC->elecgs_Count++;
	  hasdata = 1;
        }
      else if (status == IPI_MSG_GETFORCE)
        {
	  retcode = write_forces_to_socket(pSPARC);
	  if (retcode < 0)
	    {
	      print_socket_err = 1;
	      break;
	    }
	  hasdata = 0;
        }
      /* All extended SPARC protocol states should be listed here! */
      else if (status == SPARC_MSG_SETPARAM)
	{
	  // The message should be received after status READY
	  retcode = read_setparam(pSPARC);
	}
      /* END of SPARC protocol settings */
      else if (status == IPI_MSG_OTHER)
        {
	  if (rank == 0)
	    perror("Getting an unknown message from server, exiting...\n");
	  exit(EXIT_FAILURE);
        }
      else if (status == IPI_MSG_EXIT)
        {
#ifdef DEBUG
	  if (rank == 0)
	    printf("Server requesting SPARC to exit. Break the loop.\n");
#endif // DEBUG
	  break;
        }
    }
  if (print_socket_err != 0)
    print_socket_error_info(pSPARC);
  close_Socket(pSPARC);
}
