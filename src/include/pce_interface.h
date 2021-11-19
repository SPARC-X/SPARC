#ifndef PCE_INTERFACE_H
#define PCE_INTERFACE_H

#include "isddft.h"
#include <libpce.h>

void SPARC2NONLOCAL_interface(const SPARC_OBJ *pSPARC, const Hybrid_Decomp* hd, NonLocal_Info *nl, device_type device);

#endif
