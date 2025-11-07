import sys
import numpy as np
import math
import os
import re

np.set_printoptions(threshold=sys.maxsize)

''' 
@brief    READPSEUDOPOT reads the pseudopotential file (psp8 format).

@param ityp       Element type index.
@param psdfname   The pseudopotential filename, with suffix.
@param element    Element type.

@authors  Qimen Xu <qimenxu@gatech.edu>
          Abhiraj Sharma <asharma424@gatech.edu>
          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>

@copyright (c) 2019 Material Physics & Mechanics Group, Georgia Tech
'''



def read_pseudopotential_file(
    psp_dir_path : str, 
    psp_file_name: str, 
    print_debug  : bool = False):

    pseudopotential_filename = os.path.join(psp_dir_path, psp_file_name)

    with open(pseudopotential_filename,"r") as psp:
        # Read all lines from the file
        lines = psp.readlines()
        
        # Process each line: split by the separator pattern (,\s+)+(\+\S)
        # This pattern matches: comma followed by whitespace, then a plus sign with something
        separator_pattern = r'(,\s+)+(\+\S)'
        l1_list = []
        for line in lines:
            # Split by the separator pattern
            parts = re.split(separator_pattern, line)
            # Keep only the first part (before the separator) and filter out empty strings
            if parts:
                l1_list.append([parts[0].strip()])
        
        # Now split each string by comma or whitespace
        pattern = r',|\s+'
        l1_split = [re.split(pattern, string[0]) for string in l1_list]
        # Filter out empty strings from each split result
        l1_split = [[item for item in split_list if item] for split_list in l1_split]
   
        Zatom = float(l1_split[1][0])
        Z = float(l1_split[1][1])
       
        pspxc = float(l1_split[2][1])
          
        lmax = int(l1_split[2][2])
        lloc = float(l1_split[2][3])
        mmax = int(l1_split[2][4])
        
        fchrg = float(l1_split[3][1])
        
        nproj = [int(l1_split[4][i]) for i in range(int(lmax+1))]
       
        extension_switch = float(l1_split[5][0])
        pspsoc = 0   # indicating if the psp file including spin-orbit coupling
       
        if extension_switch == 2 or extension_switch == 3:
            if print_debug:
                print("This psp8 includes spin-orbit coupling.\n")
            pspsoc = 1
            nprojso = [float(l1_split[6][i]) for i in range(int(lmax))]

       
        Pot   = [dict({'gamma_Jl' : np.zeros((int(nproj[i]),1)), 'proj' : (int(nproj[i]),mmax) }) for i in range(lmax+1)]
        Potso = [dict({'gamma_Jl' : np.zeros((int(nproj[i]),1)), 'proj' : (int(nproj[i]),mmax) }) for i in range(lmax+1)]
       
        l_read = float(l1_split[6][0])
  
        l_count = 0
        lc_count = 6
        for l in range(int(lmax+1)):
           if l != lloc:
               Pot[l]['gamma_Jl'] = [float(l1_split[6 + l*mmax+l][i]) for i in range(1,nproj[l]+1)]
               sz1 = (mmax,2+nproj[l])
               A1 = [float(l1_split[j+l*mmax][k]) if l==0 else float(l1_split[j+l*mmax + l][k]) for j in range(7 ,mmax+7) for k in range(2+nproj[l])]
               y1 = np.reshape(np.array(A1), sz1)
               r = y1[:,1]                            
               Pot[l]['proj'] = y1[:,2:]         
               Pot[l]['proj'][1:,:] = Pot[l]['proj'][1:,:]/np.reshape(r[1:],(-1,1))             
               Pot[l]['proj'][0,:] = Pot[l]['proj'][1,:]          
           else:           
                A2 = [float(l1_split[j+l*mmax][k]) if l==0 else float(l1_split[j+l*mmax + l][k]) for j in range(7 ,mmax+7) for k in range(3)]
                y2 = np.reshape(np.array(A2), (mmax, 3))
                r = y2[:,1]           
                Vloc = y2[:,2]
                  
           l_read = float(l1_split[6 + (l+1)*mmax+l+1][0])
      
           l_count = l_count+1
           if l ==0:
               lc_count = mmax+lc_count
           else:
               lc_count = mmax+lc_count+1
    
        
        if lloc > lmax or l_read ==4:    
           A3 = [float(l1_split[j+(l_count)*mmax+l_count][k]) for j in range(7 ,mmax+7) for k in range(3)]
           y3 = np.reshape(np.array(A3), (mmax, 3))
           r = y3[:,1]
           Vloc = y3[:,2]  
           l_count = l_count+1   
           lc_count = mmax+lc_count+1
             
        '''read spin-orbit projectors'''
        if pspsoc == 1: 
           for l in range(1,lmax+1):    
               Potso[l]['gamma_Jl'] = [float(l1_split[6 + (l_count)*mmax][i]) for i in range(1,nproj[l]+1)]       
               sz = (mmax,2+nprojso[l])        
               A4 = [float(l1_split[j+(l_count)*mmax+l_count][k]) for j in range(7 ,mmax+7) for k in range(2+nprojso[l])]   
               y4 = np.reshape(np.array(A4), sz)       
               r = y4[:,1]                          
               Potso[l]['proj'] = y4[:,2:]      
               Potso[l]['proj'][1:,:] = Potso[l]['proj'][1:,:]/np.reshape(r[1:],(-1,1))           
               Potso[l]['proj'][0,:] = Potso[l]['proj'][1,:]              
               lc_count = mmax + lc_count
            
        '''read core density'''
        if fchrg > 0:   
            Atilde = [float(l1_split[lc_count+1+j][k]) for j in range(mmax) for k in range(7)]         
            y4 = np.reshape(np.array(Atilde), (mmax, 7))       
            uu = y4[:,2]/(4*math.pi)       
            rho_tilde = uu        
            rTilde = y4[:,1]        
            lc_count = mmax + lc_count
            
        else:  
            rTilde = r       
            rho_tilde= np.zeros((np.size(r),1))
              
        uu = np.zeros((mmax)) 
        A5 = [float(l1_split[lc_count+1+j][k]) for j in range(mmax) for k in range(5)]
        y5 = np.reshape(np.array(A5), (mmax,5))
        uu[:] = y5[:,2]/(4*math.pi) 
        rho_isolated_guess = uu
        
        rc = 0

        rc_max_list = np.zeros((lmax+1))
        for l in range(lmax+1):
            r_core_read = float(l1_split[0][l+3])
            rc_max = r_core_read
            if l != lloc:
                ''' % check if r_core is large enough s.t. |proj| < 1E-8'''
                r_indx_all = np.where(r < r_core_read)
                r_indx = r_indx_all[0][-1]
                for i in range(np.shape(Pot[l]['proj'])[1]):   
                    try:
                        rc_temp = r[r_indx +np.where(np.absolute(Pot[l]['proj'][r_indx+1:,i])<(1e-8))[0][0] - 1]
                    except:
                        rc_temp = r[-1]
                    if rc_temp>rc_max:
                        rc_max = rc_temp
                      
                    rc_max_list[l] = rc_max
                if print_debug:
                    print("atom type {first}, l = {second}, r_core_read {third}, change to rmax where |UdV| < (1e-8), {fourth} \n".format(first = 1, second = l, third = r_core_read, fourth = rc_max)) 
        if rc_max > rc: 
            rc = rc_max
                  
        r_grid_vloc = r
        r_grid_rho = r       
       
        XC = pspxc
        r_grid_rho_Tilde = rTilde
        
        return [Z, Zatom, XC, Vloc, r_grid_vloc, rc, Pot, lmax, lloc, nproj, r_grid_rho, rho_isolated_guess, rho_tilde, r_grid_rho_Tilde, pspsoc, Potso, rc_max_list]

