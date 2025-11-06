


import sys
import numpy as np


'''
Input: Atomic Number Z
Output: Occ matrix containing "n" quantum number list in first row, "l"
quantum number in the second row, the corresponding spin-up occupation 
in the third row and spin-down occupation in the fourth row.
'''

def Occ_states(Z):
  
        if Z == 1:
            n_quantum = np.array([1])
            l_quantum = np.array([0])
            s_quantum_up = np.array([1])
            s_quantum_down = np.array([0])
        elif Z == 2:
            n_quantum = np.array([1])
            l_quantum = np.array([0])
            s_quantum_up = np.array([1])
            s_quantum_down = np.array([1])
        elif Z == 3:
            n_quantum = np.array([1,2])
            l_quantum = np.array([0,0])
            s_quantum_up = np.array([1,1])
            s_quantum_down = np.array([1,0])
        elif Z == 4:
            n_quantum = np.array([1,2])
            l_quantum = np.array([0,0])
            s_quantum_up = np.array([1,1])
            s_quantum_down = np.array([1,1])
        elif Z == 5:
            n_quantum = np.array([1,2,2])
            l_quantum = np.array([0,0,1])
            s_quantum_up = np.array([1,1,1])
            s_quantum_down = np.array([1,1,0])
        elif Z == 6:
            n_quantum = np.array([1,2,2])
            l_quantum = np.array([0,0,1])
            s_quantum_up = np.array([1,1,2])
            s_quantum_down = np.array([1,1,0])
        elif Z == 7:
            n_quantum = np.array([1,2,2])
            l_quantum = np.array([0,0,1])
            s_quantum_up = np.array([1,1,3])
            s_quantum_down = np.array([1,1,0])
        elif Z == 8:
            n_quantum = np.array([1,2,2])
            l_quantum = np.array([0,0,1])
            s_quantum_up = np.array([1,1,3])
            s_quantum_down = np.array([1,1,1])
        elif Z == 9:
            n_quantum = np.array([1,2,2])
            l_quantum = np.array([0,0,1])
            s_quantum_up = np.array([1,1,3])
            s_quantum_down = np.array([1,1,2])
        elif Z == 10:
            n_quantum = np.array([1,2,2])
            l_quantum = np.array([0,0,1])
            s_quantum_up = np.array([1,1,3])
            s_quantum_down = np.array([1,1,3])
        elif Z == 11:
            n_quantum = np.array([1,2,2,3])
            l_quantum = np.array([0,0,1,0])
            s_quantum_up = np.array([1,1,3,1])
            s_quantum_down = np.array([1,1,3,0])
        elif Z == 12:
            n_quantum = np.array([1,2,2,3])
            l_quantum = np.array([0,0,1,0])
            s_quantum_up = np.array([1,1,3,1])
            s_quantum_down = np.array([1,1,3,1])
        elif Z == 13:
            n_quantum = np.array([1,2,2,3,3])
            l_quantum = np.array([0,0,1,0,1])
            s_quantum_up = np.array([1,1,3,1,1])
            s_quantum_down = np.array([1,1,3,1,0])
        elif Z == 14:
            n_quantum = np.array([1,2,2,3,3])
            l_quantum = np.array([0,0,1,0,1])
            s_quantum_up = np.array([1,1,3,1,2])
            s_quantum_down = np.array([1,1,3,1,0])
        elif Z == 15:
            n_quantum = np.array([1,2,2,3,3])
            l_quantum = np.array([0,0,1,0,1])
            s_quantum_up = np.array([1,1,3,1,3])
            s_quantum_down = np.array([1,1,3,1,0])
        elif Z == 16:
            n_quantum = np.array([1,2,2,3,3])
            l_quantum = np.array([0,0,1,0,1])
            s_quantum_up = np.array([1,1,3,1,3])
            s_quantum_down = np.array([1,1,3,1,1])
        elif Z == 17:
            n_quantum = np.array([1,2,2,3,3])
            l_quantum = np.array([0,0,1,0,1])
            s_quantum_up = np.array([1,1,3,1,3])
            s_quantum_down = np.array([1,1,3,1,2])
        elif Z == 18:
            n_quantum = np.array([1,2,2,3,3])
            l_quantum = np.array([0,0,1,0,1])
            s_quantum_up = np.array([1,1,3,1,3])
            s_quantum_down = np.array([1,1,3,1,3])
        elif Z == 19:
            n_quantum = np.array([1,2,2,3,3,4])
            l_quantum = np.array([0,0,1,0,1,0])
            s_quantum_up = np.array([1,1,3,1,3,1])
            s_quantum_down = np.array([1,1,3,1,3,0])
        elif Z == 20:
            n_quantum = np.array([1,2,2,3,3,4])
            l_quantum = np.array([0,0,1,0,1,0])
            s_quantum_up = np.array([1,1,3,1,3,1])
            s_quantum_down = np.array([1,1,3,1,3,1])
        elif Z == 21:
            n_quantum = np.array([1, 2, 2, 3, 3, 3, 4])
            l_quantum = np.array([0, 0, 1, 0, 1, 2, 0])
            s_quantum_up = np.array([1, 1, 3, 1, 3, 1, 1])
            s_quantum_down = np.array([1, 1, 3, 1, 3, 0, 1])
        elif Z == 22:
            n_quantum = np.array([1, 2, 2, 3, 3, 3, 4])
            l_quantum = np.array([0, 0, 1, 0, 1, 2, 0])
            s_quantum_up = np.array([1, 1, 3, 1, 3, 2, 1])
            s_quantum_down = np.array([1, 1, 3, 1, 3, 0, 1])
        elif Z == 23:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 3, 1 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 0, 1 ])
        elif Z == 24:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 0, 0 ])
        elif Z == 25:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 0, 1 ])
        elif Z == 26:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 1, 1 ])
        elif Z == 27:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 2, 1 ])
        elif Z == 28:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 3, 1 ])
        elif Z == 29:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 0 ])
        elif Z == 30:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1 ])
        elif Z == 31:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 1 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 0 ])
        elif Z == 32:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 2 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 0 ])
        elif Z == 33:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 0 ])
        elif Z == 34:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 1 ])
        elif Z == 35:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 2 ])
        elif Z == 36:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3 ])
        elif Z == 37:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 5 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 0 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 1 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 0 ])
        elif Z == 38:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 5 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 0 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 1 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 1 ])
        elif Z == 39:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 5 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 0 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 1, 1 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 0, 1 ])
        elif Z == 40:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 5 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 0 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 2, 1 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 0, 1 ])
        elif Z == 41:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 5 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 0 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 4, 1 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 0, 0 ])
        elif Z == 42:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 5 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 0 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 1 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 0, 0 ])
        elif Z == 43:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 5 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 0 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 1 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 0, 1 ])
        elif Z == 44:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 5 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 0 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 1 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 2, 0 ])
        elif Z == 45:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 5 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 0 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 1 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 3, 0 ])
        elif Z == 46:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5 ])
        elif Z == 47:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 5 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 0 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 1 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 0 ])
        elif Z == 48:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 5 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 0 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 1 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 1 ])
        elif Z == 49:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 0, 1 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 1, 1 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 1, 0 ])
        elif Z == 50:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 0, 1 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 1, 2 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 1, 0 ])
        elif Z == 51:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 0, 1 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 1, 3 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 1, 0 ])
        elif Z == 52:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 0, 1 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 1, 3 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 1, 1 ])
        elif Z == 53:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 0, 1 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 1, 3 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 1, 2 ])
        elif Z == 54:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 0, 1 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 1, 3 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 1, 3 ])
        elif Z == 55:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 0, 1, 0 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 1, 3, 1 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 1, 3, 0 ])
        elif Z == 56:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 0, 1, 0 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 1, 3, 1 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 1, 3, 1 ])
        elif Z == 57:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 1, 3, 1, 1 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 1, 3, 0, 1 ])
        elif Z == 58:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 1, 1, 3, 1, 1 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 0, 1, 3, 0, 1 ])
        elif Z == 59:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 0 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 3, 1, 3, 1 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 0, 1, 3, 1 ])
        elif Z == 60:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 0 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 4, 1, 3, 1 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 0, 1, 3, 1 ])
        elif Z == 61:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 0 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 5, 1, 3, 1 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 0, 1, 3, 1 ])
        elif Z == 62:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 0 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 6, 1, 3, 1 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 0, 1, 3, 1 ])
        elif Z == 63:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 0 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 1 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 0, 1, 3, 1 ])
        elif Z == 64:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 1, 1 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 0, 1, 3, 0, 1 ])
        elif Z == 65:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 0 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 1 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 2, 1, 3, 1 ])
        elif Z == 66:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 0 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 1 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 3, 1, 3, 1 ])
        elif Z == 67:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 0 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 1 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 4, 1, 3, 1 ])
        elif Z == 68:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 0 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 1 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 5, 1, 3, 1 ])
        elif Z == 69:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 0 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 1 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 6, 1, 3, 1 ])
        elif Z == 70:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 0 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 1 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 1 ])
        elif Z == 71:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 1, 1 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 0, 1 ])
        elif Z == 72:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 2, 1 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 0, 1 ])
        elif Z == 73:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 3, 1 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 0, 1 ])
        elif Z == 74:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 4, 1 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 0, 1 ])
        elif Z == 75:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 0, 1 ])
        elif Z == 76:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 1, 1 ])
        elif Z == 77:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 2, 1 ])
        elif Z == 78:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 4, 0 ])
        elif Z == 79:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 0 ])
        elif Z == 80:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1 ])
        elif Z == 81:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0, 1 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1, 1 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1, 0 ])
        elif Z == 82:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0, 1 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1, 2 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1, 0 ])
        elif Z == 83:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0, 1 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1, 3 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1, 0 ])
        elif Z == 84:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0, 1 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1, 3 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1, 1 ])
        elif Z == 85:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0, 1 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1, 3 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1, 2 ])
        elif Z == 86:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0, 1 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1, 3 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1, 3 ])
        elif Z == 87:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 7 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0, 1, 0 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1, 3, 1 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1, 3, 0 ])
        elif Z == 88:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 7 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0, 1, 0 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1, 3, 1 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1, 3, 1 ])
        elif Z == 89:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 0 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1, 3, 1, 1 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1, 3, 0, 1 ])
        elif Z == 90:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 0 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1, 3, 2, 1 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 1, 3, 0, 1 ])
        elif Z == 91:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 7 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 0 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 2, 1, 3, 1, 1 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 0, 1, 3, 0, 1 ])
        elif Z == 92:
            n_quantum = np.array([ 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 7 ])
            l_quantum = np.array([ 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 0 ])
            s_quantum_up = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 3, 1, 3, 1, 1 ])
            s_quantum_down = np.array([ 1, 1, 3, 1, 3, 5, 1, 3, 5, 7, 1, 3, 5, 0, 1, 3, 0, 1 ])
        else:
            raise ValueError("ERROR: Z should be between 1 and 92, but got {} instead".format(Z))
        
        return n_quantum, l_quantum, s_quantum_up, s_quantum_down 



class OccupationInfo:
    """
    Occupation information for atomic states.
    """
    z_valence                  : int         # Valence charge (for pseudopotential)
    z_nuclear                  : int         # True nuclear charge of the atom
    all_electron_flag          : bool        # Whether to use all-electron or pseudopotential
    occ_n                      : np.ndarray  # Principal quantum number n for each orbital
    occ_l                      : np.ndarray  # Angular momentum quantum number l for each orbital
    occ_spin_up                : np.ndarray  # Spin-up occupation for each orbital
    occ_spin_down              : np.ndarray  # Spin-down occupation for each orbital
    occ_spin_up_plus_spin_down : np.ndarray  # Total occupation (spin-up + spin-down)


    def __init__(self, 
        z_nuclear         : int,   # True nuclear charge (atomic number)
        z_valence         : int,   # Valence charge (for pseudopotential Coulomb tail)
        all_electron_flag : bool,  # Whether to use all-electron or pseudopotential
        ):
        """
        Initialize occupation information.
        
        Parameters
        ----------
        z_nuclear : int
            True nuclear charge of the atom (atomic number)
        z_valence : int
            Valence charge for pseudopotential calculations
        all_electron_flag : bool
            True for all-electron, False for pseudopotential
        """
        assert isinstance(z_nuclear, int), "z_nuclear must be an integer"
        assert isinstance(z_valence, int), "z_valence must be an integer"
        assert isinstance(all_electron_flag, bool), "all_electron_flag must be a boolean"
        assert 0 < z_nuclear <= 92, "z_nuclear must be between 1 and 92"

        self.z_nuclear = z_nuclear
        self.z_valence = z_valence
        self.all_electron_flag = all_electron_flag

        n_quantum, l_quantum, s_quantum_up, s_quantum_down = Occ_states(z_nuclear)

        if all_electron_flag:
            self.occ_n = n_quantum
            self.occ_l = l_quantum
            self.occ_spin_up   = s_quantum_up
            self.occ_spin_down = s_quantum_down
        else:
            # For pseudopotential: only valence electrons
            n_core_electrons = z_nuclear - z_valence
            orbital_occupation_numbers = s_quantum_up + s_quantum_down
            cumulative_occupation = np.cumsum(orbital_occupation_numbers)
            valence_orbitals_indices = np.where(cumulative_occupation > n_core_electrons)[0]
            self.occ_n = n_quantum[valence_orbitals_indices]
            self.occ_l = l_quantum[valence_orbitals_indices]
            self.occ_spin_up   = s_quantum_up[valence_orbitals_indices]
            self.occ_spin_down = s_quantum_down[valence_orbitals_indices]
            
        self.occ_spin_up_plus_spin_down = self.occ_spin_up + self.occ_spin_down
    
    @property
    def occupations(self) -> np.ndarray:
        """
        Total occupation numbers (spin-up + spin-down) for each orbital.
        Alias for occ_spin_up_plus_spin_down for cleaner API.
        """
        return self.occ_spin_up_plus_spin_down
    
    @property
    def l_values(self) -> np.ndarray:
        """
        Angular momentum quantum numbers for each orbital.
        Alias for occ_l for cleaner API.
        """
        return self.occ_l
    
    @property
    def n_values(self) -> np.ndarray:
        """
        Principal quantum numbers for each orbital.
        Alias for occ_n for cleaner API.
        """
        return self.occ_n
    
    @property
    def unique_l_values(self) -> np.ndarray:
        """Get unique angular momentum quantum numbers present in occupied states."""
        return np.unique(self.occ_l)
    

    @property
    def n_states(self) -> int:
        """Get total number of occupied states."""
        return len(self.occ_n)


    def n_states_for_l(self, l: int) -> int:
        """
        Get number of occupied states for a given angular momentum quantum number.
        
        Parameters
        ----------
        l : int
            Angular momentum quantum number
        
        Returns
        -------
        n_states : int
            Number of states with this l value
        """
        return np.sum(self.occ_l == l)


    
    

    def print_info(self):
        print("=" * 60)
        print("\t\t OCCUPATION INFORMATION")
        print("=" * 60)
        print(f"\t z_valence (valence charge) : {self.z_valence}")
        print(f"\t z_nuclear (nuclear charge) : {self.z_nuclear}")
        print(f"\t all_electron_flag          : {self.all_electron_flag}")
        print(f"\t occ_n                      : {self.occ_n}")
        print(f"\t occ_l                      : {self.occ_l}")
        print(f"\t occ_spin_up                : {self.occ_spin_up}")
        print(f"\t occ_spin_down              : {self.occ_spin_down}")
        print(f"\t occ_spin_up_plus_spin_down : {self.occ_spin_up_plus_spin_down}")
        print()



if __name__ == "__main__":
    Zatom = 13
    occupation_info = OccupationInfo(Zatom)