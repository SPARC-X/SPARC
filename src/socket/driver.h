#ifndef DRIVER_H
#define DRIVER_H
#include "isddft.h"
#include "libinetsocket.h"
#include "libunixsocket.h"

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
int split_socket_name(const char *str, char *host, int *port, int *inet);

/**
 * @brief   Initialize a socket file descriptor for communication with the client.
 *          the FD at pSPARC->socket_fd is created according to  pSPARC->socket_inet
 **/
int initialize_Socket(SPARC_OBJ *pSPARC);

/**
 * @brief   Destroy the socket file descriptor
 **/
int close_Socket(SPARC_OBJ *pSPARC);

/**
 * @brief   Read buffer from the socket fd into string data and length of len
 **/
// int readBuffer(SPARC_OBJ *pSPARC, char *data, int len);

/**
 * @brief   Write buffer of size len to the socket fd
 **/
// int writeBuffer(SPARC_OBJ *pSPARC, const char *data, int len);


/**
 * @brief   Main function with socket control
 **/
void main_Socket(SPARC_OBJ *pSPARC);

// IPI constants

#define IPI_HEADERLEN 12
#define IPI_MSG_NEEDINIT 0
#define IPI_MSG_STATUS 1
#define IPI_MSG_POSDATA 2
#define IPI_MSG_GETFORCE 3
#define IPI_MSG_OTHER 4

/*
  Below are extended MSG states for extended SPARC protocol.
  New protocol keywords should be defined starting from 100
 */
#define SPARC_MSG_SETPARAM 100

/* Exit status */
#define IPI_MSG_EXIT 999

// Hartree to eV conversions
#define HARTREE_TO_EV 27.211386024367243
#define HARTREE_BOHR_TO_EV_ANGSTROM 51.422067090480645
#define HARTREE_BOHR3_TO_EV_ANGSTROM3 183.6315353072019


#endif // DRIVER_H
