#ifndef EXCHANGECORRELATIONATOM_H
#define EXCHANGECORRELATIONATOM_H

#include "isddftAtom.h"

void Calculate_Vxc_atom(SPARC_ATOM_OBJ *pSPARC_ATOM);

/**
* @brief add core electron density if needed
**/
void add_rho_core_atom(SPARC_ATOM_OBJ *pSPARC_ATOM, double *rho_in, double *rho_out, int ncol);

/**
 * @brief   slater exchange
 */
void slater(int DMnd, double *rho, double *ex, double *vx);


/**
 * @brief   pw correlation
 *          J.P. Perdew and Y. Wang, PRB 45, 13244 (1992)
 */
void pw(int DMnd, double *rho, double *ec, double *vc);


/**
 * @brief   pz correlation
 *          J.P. Perdew and A. Zunger, PRB 23, 5048 (1981).
 */
void pz(int DMnd, double *rho, double *ec, double *vc);


/**
 * @brief   pbe exchange
 *
 * @param   iflag=1  J.P.Perdew, K.Burke, M.Ernzerhof, PRL 77, 3865 (1996)
 * @param   iflag=2  PBEsol: J.P.Perdew et al., PRL 100, 136406 (2008)
 * @param   iflag=3  RPBE: B. Hammer, et al., Phys. Rev. B 59, 7413 (1999)
 * @param   iflag=4  Zhang-Yang Revised PBE: Y. Zhang and W. Yang., Phys. Rev. Lett. 80, 890 (1998)
 */
void pbex(int DMnd, double *rho, double *sigma, int iflag, double *ex, double *vx, double *v2x);


/**
* @brief the function to compute the potential and energy density of PW86 GGA exchange
*/
void rPW86x(int DMnd, double *rho, double *sigma, double *vdWDFex, double *vdWDFVx1, double *vdWDFVx2);


/**
 * @brief   pbe correlation
 *
 * @param   iflag=1  J.P.Perdew, K.Burke, M.Ernzerhof, PRL 77, 3865 (1996)
 * @param   iflag=2  PBEsol: J.P.Perdew et al., PRL 100, 136406 (2008)
 * @param   iflag=3  RPBE: B. Hammer, et al., Phys. Rev. B 59, 7413 (1999)
 */
void pbec(int DMnd, double *rho, double *sigma, int iflag, double *ec, double *vc, double *v2c);


/**
 * @brief   slater exchange - spin polarized 
 */
void slater_spin(int DMnd, double *rho, double *ex, double *vx);


/**
 * @brief   pw correlation - spin polarized 
 *          J.P. Perdew and Y. Wang, PRB 45, 13244 (1992)
 */
void pw_spin(int DMnd, double *rho, double *ec, double *vc);


/**
 * @brief   pz correlation - spin polarized 
 *          J.P. Perdew and A. Zunger, PRB 23, 5048 (1981).
 */
void pz_spin(int DMnd, double *rho, double *ec, double *vc);


/**
 * @brief   pbe exchange - spin polarized
 *
 * @param   iflag=1  J.P.Perdew, K.Burke, M.Ernzerhof, PRL 77, 3865 (1996)
 * @param   iflag=2  PBEsol: J.P.Perdew et al., PRL 100, 136406 (2008)
 * @param   iflag=3  RPBE: B. Hammer, et al., Phys. Rev. B 59, 7413 (1999)
 * @param   iflag=4  Zhang-Yang Revised PBE: Y. Zhang and W. Yang., Phys. Rev. Lett. 80, 890 (1998)
 */
void pbex_spin(int DMnd, double *rho, double *sigma, int iflag, double *ex, double *vx, double *v2x);


/**
* @brief the function to compute the potential and energy density of PW86 GGA exchange, spin-polarized case
*/
void rPW86x_spin(int DMnd, double *rho, double *sigma, double *vdWDFex, double *vdWDFVx1, double *vdWDFVx2);


/**
 * @brief   pbe correlation - spin polarized
 *
 * @param   iflag=1  J.P.Perdew, K.Burke, M.Ernzerhof, PRL 77, 3865 (1996)
 * @param   iflag=2  PBEsol: J.P.Perdew et al., PRL 100, 136406 (2008)
 * @param   iflag=3  RPBE: B. Hammer, et al., Phys. Rev. B 59, 7413 (1999)
 */
void pbec_spin(int DMnd, double *rho, double *sigma, int iflag, double *ec, double *vc, double *v2c);

#endif